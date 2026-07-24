#!/usr/bin/env python3
"""Parity: FA4 relative attention vs PyTorch reference (Inkling semantics).

Reference semantics (verified against vllm fa4_rel_attention.py score_mod +
tml-fa4 ShearingBias, journal/day0-implementation.md):
  scores[i,j] = (q_i . k_j) * softmax_scale + bias(i,j)
  bias(i,j)   = rel_logits[i, h, d]  if 0 <= d < rel_extent else 0.0,  d = i - j
  causal; softmax_scale = 1/head_dim (q/k are unit-RMS in the real model).
Local (SWA) mode adds window masking: attend iff 0 <= i - j <= window_left.

Run (WSL): cd ~/inkling-turbo/vllm && source .venv/bin/activate && \
  python $REPO/harness/parity_fa4_rel.py
"""

from __future__ import annotations

import argparse

import torch


def reference_rel_attention(
    q: torch.Tensor,  # (T, Hq, D)
    k: torch.Tensor,  # (T, Hkv, D)
    v: torch.Tensor,  # (T, Hkv, D)
    rel_logits: torch.Tensor,  # (T, Hq, rel_extent)
    softmax_scale: float,
    window_left: int | None = None) -> torch.Tensor:
    T, Hq, D = q.shape
    Hkv = k.shape[1]
    rel_extent = rel_logits.shape[-1]
    rep = Hq // Hkv
    kf = k.repeat_interleave(rep, dim=1).float()
    vf = v.repeat_interleave(rep, dim=1).float()
    qf = q.float()

    scores = torch.einsum("ihd,jhd->hij", qf, kf) * softmax_scale

    dist = torch.arange(T).view(-1, 1) - torch.arange(T).view(1, -1)  # i - j
    in_range = (dist >= 0) & (dist < rel_extent)
    d_idx = dist.clamp(0, rel_extent - 1).to(q.device)
    # bias[h, i, j] = rel_logits[i, h, d] masked to the valid range
    bias = rel_logits.float().permute(1, 0, 2).gather(
        2, d_idx.unsqueeze(0).expand(Hq, T, T)
    )
    scores += bias * in_range.to(q.device)

    causal = dist.to(q.device) >= 0
    if window_left is not None:
        causal &= dist.to(q.device) <= window_left
    scores.masked_fill_(~causal, float("-inf"))

    out = torch.einsum("hij,jhd->ihd", torch.softmax(scores, dim=-1), vf)
    return out.to(q.dtype)


def run_case(
    T: int, Hq: int, Hkv: int, D: int, rel_extent: int, window_left: int | None,
    seed: int) -> dict:
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(T, Hq, D, dtype=torch.bfloat16, device=dev) / (D**0.25)
    k = torch.randn(T, Hkv, D, dtype=torch.bfloat16, device=dev) / (D**0.25)
    v = torch.randn(T, Hkv, D, dtype=torch.bfloat16, device=dev)
    # One truth for every backend: rel_logits derived from (r, proj) exactly
    # as qkvr_prep does (fp32 dot, bf16 round). Backends 1-2 + the reference
    # consume rel_logits; backend 3 (relproj_v1) consumes (r, proj) raw.
    r_small = torch.randn(T, Hq, 16, dtype=torch.bfloat16, device=dev) * 0.4
    proj = torch.randn(16, rel_extent, dtype=torch.bfloat16, device=dev) * 0.3
    rel_logits = (r_small.float() @ proj.float()).to(torch.bfloat16)
    scale = 1.0 / D

    ref = reference_rel_attention(q, k, v, rel_logits, scale, window_left)

    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    window = (None, None) if window_left is None else (window_left, 0)

    results = {}

    # Backend 1: tml-fa4 sheared rel_bias (sm_100/110 kernel; on sm_120 the
    # interface asserts tile_n == 128, expected unsupported locally).
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func as fa4_sheared

    try:
        out = fa4_sheared(
            q=q, k=k, v=v,
            rel_bias=rel_logits,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=T, max_seqlen_k=T,
            softmax_scale=scale, causal=True, window_size=window)
        if isinstance(out, tuple):
            out = out[0]
        diff = (out.float() - ref.float()).abs()
        results["tml_fa4_rel_bias"] = (diff.max().item(), diff.mean().item())
    except AssertionError:
        results["tml_fa4_rel_bias"] = "SKIP: sheared path unsupported on this arch"
    except Exception as exc:  # noqa: BLE001 - report per-backend failures
        results["tml_fa4_rel_bias"] = f"FAILED: {type(exc).__name__}: {exc}"

    # Backend 2: score_mod gather, vLLM's actual sm_120/Hopper path
    # (vllm/models/inkling/nvidia/ops/fa4_rel_attention.py else-branch).
    try:
        from vllm.models.inkling.nvidia.ops.fa4_rel_attention import _get_score_mod
        from vllm.vllm_flash_attn.cute import (
            flash_attn_varlen_func as fa_score_mod)

        cute_window = (None, None) if window_left is None else window
        out = fa_score_mod(
            q=q, k=k, v=v,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=T, max_seqlen_k=T,
            softmax_scale=scale, causal=True, window_size=cute_window,
            score_mod=_get_score_mod(rel_extent),
            aux_tensors=[rel_logits.contiguous()])
        if isinstance(out, tuple):
            out = out[0]
        diff = (out.float() - ref.float()).abs()
        results["score_mod"] = (diff.max().item(), diff.mean().item())
    except Exception as exc:  # noqa: BLE001
        results["score_mod"] = f"FAILED: {type(exc).__name__}: {exc}"

    # Backend 3: U2-Hopper Design B V1, register-resident r-projection bias
    # (kernels/relproj_score_mod.py). Consumes (r, proj) instead of the
    # materialized rel_logits; rel_logits used by ref/backends 1-2 is
    # derived from the same (r, proj) below, so all backends see one truth.
    try:
        import sys as _sys
        from pathlib import Path as _Path

        _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
        from kernels.relproj_score_mod import get_relproj_score_mod
        from vllm.vllm_flash_attn.cute import (
            flash_attn_varlen_func as fa_relproj)

        cute_window = (None, None) if window_left is None else window
        out = fa_relproj(
            q=q, k=k, v=v,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=T, max_seqlen_k=T,
            softmax_scale=scale, causal=True, window_size=cute_window,
            score_mod=get_relproj_score_mod(rel_extent),
            aux_tensors=[r_small.contiguous(), proj.contiguous()])
        if isinstance(out, tuple):
            out = out[0]
        diff = (out.float() - ref.float()).abs()
        results["relproj_v1"] = (diff.max().item(), diff.mean().item())
    except Exception as exc:  # noqa: BLE001
        results["relproj_v1"] = f"FAILED: {type(exc).__name__}: {exc}"

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=2e-2)
    args = ap.parse_args()

    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"capability {torch.cuda.get_device_capability(0)}")

    # (name, T, Hq, Hkv, D, rel_extent, window_left), real Inkling head geometry,
    # reduced head count / seqlen to fit quick runs.
    cases = [
        ("global_short", 128, 8, 1, 128, 1024, None),
        ("global_beyond_extent", 1536, 8, 1, 128, 1024, None),
        ("swa_512", 1536, 8, 2, 128, 512, 511),
    ]
    failures = 0
    for name, T, Hq, Hkv, D, ext, wl in cases:
        res = run_case(T, Hq, Hkv, D, ext, wl, seed=hash(name) % (2**31))
        for backend, r in res.items():
            if isinstance(r, str):
                print(f"[{name}] {backend}: {r}")
                failures += 1
            else:
                mx, mean = r
                ok = mx <= args.tol
                failures += 0 if ok else 1
                print(f"[{name}] {backend}: max_diff={mx:.4e} mean={mean:.4e} "
                      f"{'OK' if ok else f'FAIL(tol={args.tol})'}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    import sys as _s
    if "--debug" not in _s.argv:
        main()


def debug_dump(T: int = 128, Hq: int = 8, Hkv: int = 1, ext: int = 1024) -> None:
    """sm_90 mapping diagnostic: per-position |err| structure of the bias
    path vs reference, dumped as JSON heatmap summaries."""
    import json
    from pathlib import Path

    torch.manual_seed(7)
    dev = "cuda"
    D = 128
    q = torch.randn(T, Hq, D, dtype=torch.bfloat16, device=dev) / (D**0.25)
    k = torch.randn(T, Hkv, D, dtype=torch.bfloat16, device=dev) / (D**0.25)
    v = torch.randn(T, Hkv, D, dtype=torch.bfloat16, device=dev)
    r = torch.randn(T, Hq, 16, dtype=torch.bfloat16, device=dev) * 0.4
    proj = torch.randn(16, ext, dtype=torch.bfloat16, device=dev) * 0.3
    rel = (r.float() @ proj.float()).to(torch.bfloat16)
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    ref = reference_rel_attention(q, k, v, rel, 1.0 / D, None)

    import os
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func

    out = flash_attn_varlen_func(
        q=q, k=k, v=v, rel_bias=rel, cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=T, max_seqlen_k=T, softmax_scale=1.0 / D, causal=True)
    if isinstance(out, tuple):
        out = out[0]

    if os.environ.get("U2_DEBUG_SENTINEL") == "1":
        ref_plain = reference_rel_attention(q, k, v, torch.zeros_like(rel), 1.0 / D, None)
        d_plain = (out.float() - ref_plain.float()).abs().max().item()
        print("DIAG SENTINEL: out max:", round(out.float().abs().max().item(), 3),
              "| vs plain max:", round(d_plain, 4),
              "(if >> 2.29 => apply RUNS & writes propagate)")
        import sys; sys.exit(0)
    if os.environ.get("U2_DEBUG_ZEROBIAS") == "1":
        ref_plain = reference_rel_attention(q, k, v, torch.zeros_like(rel), 1.0 / D, None)
        e = (out.float() - ref_plain.float()).abs()
        print("DIAG ZEROBIAS: kernel-vs-plain max:", round(e.max().item(), 4),
              "mean:", round(e.mean().item(), 6),
              "(GREEN => scale/wiring correct)")
        import sys; sys.exit(0)
    if os.environ.get("U2_DEBUG_DISTBIAS") == "1":
        ii = torch.arange(T).view(T, 1, 1).float()
        dd = torch.arange(ext).view(1, 1, ext).float()
        rb2 = dd.expand(T, Hq, ext).contiguous()  # rb2[i,h,d]=d => bias(i,j)=i-j=dist
        ref_d = reference_rel_attention(q, k, v, rb2.to(torch.bfloat16).to(dev), 1.0 / D, None)
        e = (out.float() - ref_d.float()).abs()
        print("DIAG DISTBIAS: kernel-vs-distref max:", round(e.max().item(), 4),
              "mean:", round(e.mean().item(), 6),
              "(GREEN => dist=row-kv correct => bug is tensor content)")
        import sys; sys.exit(0)
    if os.environ.get("U2_DEBUG_COLBIAS") == "1":
        ii = torch.arange(T).view(T, 1, 1).float()
        dd = torch.arange(ext).view(1, 1, ext).float()
        rb2 = (ii - dd).expand(T, Hq, ext).contiguous()  # rb2[i,h,d]=i-d => bias(i,j)=j
        ref_col = reference_rel_attention(q, k, v, rb2.to(torch.bfloat16).to(dev), 1.0 / D, None)
        e = (out.float() - ref_col.float()).abs()
        print("DIAG COLBIAS: kernel-vs-colref max:", round(e.max().item(), 4),
              "mean:", round(e.mean().item(), 6),
              "(GREEN => column derivation correct)")
        import sys; sys.exit(0)
    if os.environ.get("U2_DEBUG_ROWBIAS") == "1":
        # synthetic bias(i,j) = i (row index): kernel adds Float32(row_g)
        row_bias = torch.arange(T, device=dev).view(T, 1, 1).expand(T, Hq, ext).to(torch.bfloat16).float()
        # reference with bias(i,j)=i for all valid j
        rb2 = torch.zeros(T, Hq, ext)
        rb2[:] = torch.arange(T).view(T, 1, 1).float()
        ref_syn = reference_rel_attention(q, k, v, rb2.to(torch.bfloat16).to(dev), 1.0 / D, None)
        e = (out.float() - ref_syn.float()).abs()
        print("DIAG ROWBIAS synthetic: kernel-vs-rowbias-ref max:",
              round(e.max().item(), 4), "mean:", round(e.mean().item(), 5))
        import sys; sys.exit(0)
    # decisive: compare kernel vs BIASED ref and vs PLAIN ref (no bias)
    ref_plain = reference_rel_attention(q, k, v, torch.zeros_like(rel), 1.0 / D, None)
    err = (out.float() - ref.float()).abs().amax(dim=-1)  # (T, Hq)
    err_plain = (out.float() - ref_plain.float()).abs().amax(dim=-1)
    print("DIAG kernel-vs-biased max:", round(err.max().item(), 3),
          "| kernel-vs-plain max:", round(err_plain.max().item(), 3))
    print("DIAG rows closer to PLAIN (bias not applied):",
          int((err_plain < err)[:, 0].sum().item()), "of", T)
    print("DIAG rows closer to BIASED (bias applied ~right):",
          int((err < err_plain)[:, 0].sum().item()), "of", T)
    per_row = err[:, 0]
    rows_bad = (per_row > 2e-2).nonzero().flatten().tolist()
    d = {
        "rows_bad_h0": rows_bad,
        "per_row_err_h0": [round(x, 4) for x in per_row.tolist()],
        "per_head_maxerr": [round(x, 4) for x in err.amax(0).tolist()],
        "err_row_blocks_of_16": [
            round(per_row[i:i + 16].max().item(), 4) for i in range(0, T, 16)
        ],
    }
    p = Path.home() / "u2_debug_dump.json"
    p.write_text(json.dumps(d, indent=1))
    print("DEBUG rows_bad count:", len(rows_bad), "of", T)
    print("DEBUG per-16-row-block max err:", d["err_row_blocks_of_16"])
    print("DEBUG per-head max err:", d["per_head_maxerr"])
    print("saved:", p)


if __name__ == "__main__" and "--debug" in __import__("sys").argv:
    debug_dump()
