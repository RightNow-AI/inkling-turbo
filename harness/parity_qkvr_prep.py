#!/usr/bin/env python3
"""Parity: fused_qkvr_prep vs PyTorch reference (Inkling prep kernel).

Semantics derived from vllm/models/inkling/nvidia/ops/qkvr_prep.py (fork base
850295881), verified against the kernel body, not the docstring:

- Q rows (qkvr_prep.py:244-260): q slice -> fp32 RMSNorm (eps inside rsqrt,
  mean over head_dim) * q_norm_weight; WITH log_scaling the normed value is
  rounded to out dtype FIRST, then multiplied by log_scaling[token] in fp32
  (:254-256), then stored rounded.
- rel rows (:261-282): projected = sum_d r[d] * rel_proj[d, :] in fp32; the
  fp32 sum is rounded to out dtype, back to fp32 (:275), THEN scaled by
  log_scaling (:276-277), stored rounded. r sits at column
  Q_WIDTH + 2*KV_WIDTH + head*16 (:264).
- KV rows (:283-402): raw k/v (pre-conv) are written to the conv cache at
  conv_slot (:300-316); conv taps 0..W-1 read source_position = pos-(W-1)+tap:
  intra-batch rows (source_row >= query_start[token]) come from qkvr itself
  (:324,:327-336), earlier positions come from the conv cache via the per-
  request conv block table (:325,:338-363), negative positions contribute 0.
  Depthwise weights are indexed (head*head_dim+dim)*W + tap (:364-369).
  k_out = round_bf16(acc_k + k_raw) (residual, :373); v_out likewise (:374).
  The VALUE cache stores v_out as-is; the KEY cache stores
  round_bf16(RMSNorm_fp32(k_out) * k_norm_weight) computed on the ROUNDED
  k_out (:375-393). Attention cache layout (block, page_off, head, dim).
- Two launch paths share semantics: fused-small (< 128 tokens,
  :840-868) and tiled (>= 128, :870-917). Both are tested.

Run (WSL): cd ~/inkling-turbo/vllm && source .venv/bin/activate && \
  python $REPO/harness/parity_qkvr_prep.py
"""

from __future__ import annotations

import torch

HQ, HKV, D, D_REL, W = 8, 2, 128, 16, 4
QW, KVW = HQ * D, HKV * D
CONV_BS = 16  # conv cache tokens per block
PAGE = 16  # attention cache page size
OFF_K, OFF_V = 0, 128  # K/V stream dims inside the conv cache


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * rstd * w


def reference(qkvr, kw, vw, qnw, knw, proj, eps, conv_cache0, conv_table,
              positions, seq_idx, query_start, conv_slots, attn_slots,
              log_scaling):
    """Pure fp32 reference with the kernel's exact bf16 rounding points.

    Returns (q_out, rel_out, kc_writes, vc_writes, cc_writes) where *_writes
    map slot -> per-head tensors.
    """
    T = qkvr.shape[0]
    x = qkvr.float()
    ext = proj.shape[1]
    q_out = torch.empty(T, HQ, D)
    rel_out = torch.empty(T, HQ, ext)
    for t in range(T):
        tau = log_scaling[t].item() if log_scaling is not None else None
        for h in range(HQ):
            q = x[t, h * D:(h + 1) * D]
            n = rms(q, qnw.float(), eps)
            if tau is not None:
                n = bf16(n) * tau
            q_out[t, h] = bf16(n)
            r = x[t, QW + 2 * KVW + h * D_REL: QW + 2 * KVW + (h + 1) * D_REL]
            p = bf16(r @ proj.float())
            if tau is not None:
                p = p * tau
            rel_out[t, h] = bf16(p)

    cc = conv_cache0.clone().float()
    kc_writes, vc_writes = {}, {}
    # pass 1: raw inserts (only prior-request positions are ever read back,
    # so insert order vs read order cannot alias within one call)
    for t in range(T):
        if conv_slots[t] < 0:
            continue
        b, o = divmod(int(conv_slots[t]), CONV_BS)
        for h in range(HKV):
            cc[b, h, o, OFF_K:OFF_K + D] = x[t, QW + h * D: QW + (h + 1) * D]
            cc[b, h, o, OFF_V:OFF_V + D] = \
                x[t, QW + KVW + h * D: QW + KVW + (h + 1) * D]
    # pass 2: conv + residual + cache writes
    for t in range(T):
        pos, req, qs = int(positions[t]), int(seq_idx[t]), int(query_start[t])
        for h in range(HKV):
            k_raw = x[t, QW + h * D: QW + (h + 1) * D]
            v_raw = x[t, QW + KVW + h * D: QW + KVW + (h + 1) * D]
            acc_k = torch.zeros(D)
            acc_v = torch.zeros(D)
            for tap in range(W):
                sp = pos - (W - 1) + tap
                sr = t - (W - 1) + tap
                if sp < 0:
                    continue
                if sr >= qs:
                    sk = x[sr, QW + h * D: QW + (h + 1) * D]
                    sv = x[sr, QW + KVW + h * D: QW + KVW + (h + 1) * D]
                else:
                    pb = int(conv_table[req, sp // CONV_BS])
                    sk = cc[pb, h, sp % CONV_BS, OFF_K:OFF_K + D]
                    sv = cc[pb, h, sp % CONV_BS, OFF_V:OFF_V + D]
                ch = slice(h * D, (h + 1) * D)
                acc_k += sk * kw.float()[ch, tap]
                acc_v += sv * vw.float()[ch, tap]
            k_out = bf16(acc_k + k_raw)
            v_out = bf16(acc_v + v_raw)
            k_norm = bf16(rms(k_out, knw.float(), eps))
            if attn_slots[t] >= 0:
                kc_writes[(int(attn_slots[t]), h)] = k_norm
                vc_writes[(int(attn_slots[t]), h)] = v_out
    return q_out, rel_out, kc_writes, vc_writes, cc


def run_case(name, seq_lens, first_positions, ext, use_log, seed,
             prefill_cache_tokens=0):
    """prefill_cache_tokens: per-seq count of pre-existing positions whose raw
    K/V must already sit in the conv cache (decode / chunked continuation)."""
    from vllm.models.inkling.nvidia.ops.qkvr_prep import fused_qkvr_prep

    torch.manual_seed(seed)
    dev = "cuda"
    T = sum(seq_lens)
    n_req = len(seq_lens)
    eps = 1e-6
    width = QW + 2 * KVW + HQ * D_REL
    qkvr = torch.randn(T, width, dtype=torch.bfloat16, device=dev)
    kw = torch.randn(HKV * D, W, dtype=torch.bfloat16, device=dev) * 0.3
    vw = torch.randn(HKV * D, W, dtype=torch.bfloat16, device=dev) * 0.3
    qnw = torch.rand(D, dtype=torch.bfloat16, device=dev) + 0.5
    knw = torch.rand(D, dtype=torch.bfloat16, device=dev) + 0.5
    proj = torch.randn(D_REL, ext, dtype=torch.bfloat16, device=dev) * 0.5

    positions, seq_idx, query_start = [], [], []
    row = 0
    for s, (L, p0) in enumerate(zip(seq_lens, first_positions)):
        positions += list(range(p0, p0 + L))
        seq_idx += [s] * L
        query_start += [row] * L
        row += L
    positions = torch.tensor(positions, dtype=torch.int64, device=dev)
    seq_idx = torch.tensor(seq_idx, dtype=torch.int32, device=dev)
    query_start = torch.tensor(query_start, dtype=torch.int32, device=dev)

    # Non-identity conv block table: request s, logical block l -> physical
    max_pos = max(p0 + L for L, p0 in zip(seq_lens, first_positions))
    max_blocks = (max_pos + CONV_BS - 1) // CONV_BS + 1
    n_conv_blocks = n_req * max_blocks + 3
    perm = torch.randperm(n_conv_blocks - 1, device=dev) + 1  # avoid block 0
    conv_table = perm[: n_req * max_blocks].reshape(n_req, max_blocks).to(
        torch.int32)
    conv_cache = torch.zeros(n_conv_blocks, HKV, CONV_BS, 256,
                             dtype=torch.bfloat16, device=dev)

    conv_slots = torch.tensor(
        [int(conv_table[int(seq_idx[i]), int(positions[i]) // CONV_BS])
         * CONV_BS + int(positions[i]) % CONV_BS for i in range(T)],
        dtype=torch.int32, device=dev)

    # Pre-existing conv state for continuation cases: raw K/V of the last
    # prefill_cache_tokens positions before each seq's first_position.
    prior = {}
    if prefill_cache_tokens:
        for s in range(n_req):
            for pp in range(max(0, first_positions[s] - prefill_cache_tokens),
                            first_positions[s]):
                pk = torch.randn(HKV, D, dtype=torch.bfloat16, device=dev)
                pv = torch.randn(HKV, D, dtype=torch.bfloat16, device=dev)
                pb = int(conv_table[s, pp // CONV_BS])
                conv_cache[pb, :, pp % CONV_BS, OFF_K:OFF_K + D] = pk
                conv_cache[pb, :, pp % CONV_BS, OFF_V:OFF_V + D] = pv
                prior[(s, pp)] = (pk, pv)

    n_pages = (T + PAGE - 1) // PAGE + 2
    pperm = torch.randperm(n_pages * PAGE, device=dev)[:T].to(torch.int32)
    attn_slots = pperm.clone()
    key_cache = torch.zeros(n_pages, PAGE, HKV, D, dtype=torch.bfloat16,
                            device=dev)
    value_cache = torch.zeros_like(key_cache)

    log_scaling = (torch.rand(T, dtype=torch.float32, device=dev) * 0.3 + 1.0
                   if use_log else None)

    conv_cache0 = conv_cache.clone()
    q_out, rel_out = fused_qkvr_prep(
        qkvr, kw, vw, qnw, knw, proj, eps, HQ, HKV, D, D_REL,
        conv_cache, key_cache, value_cache, positions, conv_table, seq_idx,
        conv_slots, query_start, attn_slots, OFF_K, OFF_V, CONV_BS,
        log_scaling)
    torch.cuda.synchronize()

    rq, rrel, kcw, vcw, rcc = reference(
        qkvr.cpu(), kw.cpu(), vw.cpu(), qnw.cpu(), knw.cpu(), proj.cpu(), eps,
        conv_cache0.cpu(), conv_table.cpu(), positions.cpu(), seq_idx.cpu(),
        query_start.cpu(), conv_slots.cpu(), attn_slots.cpu(),
        log_scaling.cpu() if log_scaling is not None else None)

    tol = 2e-2
    errs = []
    dq = (q_out.view(T, HQ, D).float().cpu() - rq).abs().max().item()
    if dq > tol:
        errs.append(f"q max_diff={dq:.3e}")
    dr = (rel_out.float().cpu() - rrel).abs().max().item()
    if dr > tol:
        errs.append(f"rel max_diff={dr:.3e}")
    for (slot, h), ref_k in kcw.items():
        got = key_cache[slot // PAGE, slot % PAGE, h].float().cpu()
        d = (got - ref_k).abs().max().item()
        if d > tol:
            errs.append(f"kcache slot{slot} h{h} diff={d:.3e}")
            break
    for (slot, h), ref_v in vcw.items():
        got = value_cache[slot // PAGE, slot % PAGE, h].float().cpu()
        d = (got - ref_v).abs().max().item()
        if d > tol:
            errs.append(f"vcache slot{slot} h{h} diff={d:.3e}")
            break
    dcc = (conv_cache.float().cpu() - rcc).abs().max().item()
    if dcc > tol:
        errs.append(f"conv cache max_diff={dcc:.3e}")
    return errs


def main() -> None:
    print(f"device: {torch.cuda.get_device_name(0)}")
    cases = [
        # (name, seq_lens, first_positions, ext, log_scaling, prior_tokens)
        ("prefill_single_local", [12], [0], 512, False, 0),
        ("prefill_multi_global", [5, 9, 3], [0, 0, 0], 1024, True, 0),
        ("decode_steps_local", [1, 1], [7, 22], 512, False, W),
        ("chunked_continue_global", [6, 4], [10, 3], 1024, True, W),
        ("tiled_path_160tok", [80, 80], [0, 0], 512, False, 0),
    ]
    failures = 0
    for name, lens, p0s, ext, lg, prior in cases:
        try:
            errs = run_case(name, lens, p0s, ext, lg,
                            seed=hash(name) % (2**31),
                            prefill_cache_tokens=prior)
        except Exception as exc:  # noqa: BLE001
            errs = [f"EXCEPTION {type(exc).__name__}: {exc}"]
        if errs:
            failures += 1
            print(f"[{name}] FAIL: {'; '.join(errs)}")
        else:
            print(f"[{name}] OK")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
