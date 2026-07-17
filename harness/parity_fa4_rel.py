#!/usr/bin/env python3
"""Parity: FA4 relative attention vs PyTorch reference (Inkling semantics).

Reference semantics (verified against vllm fa4_rel_attention.py score_mod +
tml-fa4 ShearingBias, journal/day0-implementation.md):
  scores[i,j] = (q_i . k_j) * softmax_scale + bias(i,j)
  bias(i,j)   = rel_logits[i, h, d]  if 0 <= d < rel_extent else 0.0,  d = i - j
  causal; softmax_scale = 1/head_dim (q/k are unit-RMS in the real model).
Local (SWA) mode adds window masking: attend iff 0 <= i - j <= window_left.

Run (WSL): cd ~/inkling-turbo/vllm && source .venv/bin/activate && \
  python /mnt/c/Users/jaber/RightNow-Full/inkling/harness/parity_fa4_rel.py
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
    window_left: int | None = None,
) -> torch.Tensor:
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
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(T, Hq, D, dtype=torch.bfloat16, device=dev) / (D**0.25)
    k = torch.randn(T, Hkv, D, dtype=torch.bfloat16, device=dev) / (D**0.25)
    v = torch.randn(T, Hkv, D, dtype=torch.bfloat16, device=dev)
    rel_logits = 0.5 * torch.randn(T, Hq, rel_extent, dtype=torch.bfloat16, device=dev)
    scale = 1.0 / D

    ref = reference_rel_attention(q, k, v, rel_logits, scale, window_left)

    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    window = (None, None) if window_left is None else (window_left, 0)

    results = {}

    # Backend 1: tml-fa4 sheared rel_bias (sm_100/110 kernel; on sm_120 the
    # interface asserts tile_n == 128 — expected unsupported locally).
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func as fa4_sheared

    try:
        out = fa4_sheared(
            q=q, k=k, v=v,
            rel_bias=rel_logits,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=T, max_seqlen_k=T,
            softmax_scale=scale, causal=True, window_size=window,
        )
        if isinstance(out, tuple):
            out = out[0]
        diff = (out.float() - ref.float()).abs()
        results["tml_fa4_rel_bias"] = (diff.max().item(), diff.mean().item())
    except AssertionError:
        results["tml_fa4_rel_bias"] = "SKIP: sheared path unsupported on this arch"
    except Exception as exc:  # noqa: BLE001 - report per-backend failures
        results["tml_fa4_rel_bias"] = f"FAILED: {type(exc).__name__}: {exc}"

    # Backend 2: score_mod gather — vLLM's actual sm_120/Hopper path
    # (vllm/models/inkling/nvidia/ops/fa4_rel_attention.py else-branch).
    try:
        from vllm.models.inkling.nvidia.ops.fa4_rel_attention import _get_score_mod
        from vllm.vllm_flash_attn.cute import (
            flash_attn_varlen_func as fa_score_mod,
        )

        cute_window = (None, None) if window_left is None else window
        out = fa_score_mod(
            q=q, k=k, v=v,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=T, max_seqlen_k=T,
            softmax_scale=scale, causal=True, window_size=cute_window,
            score_mod=_get_score_mod(rel_extent),
            aux_tensors=[rel_logits.contiguous()],
        )
        if isinstance(out, tuple):
            out = out[0]
        diff = (out.float() - ref.float()).abs()
        results["score_mod"] = (diff.max().item(), diff.mean().item())
    except Exception as exc:  # noqa: BLE001
        results["score_mod"] = f"FAILED: {type(exc).__name__}: {exc}"

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=2e-2)
    args = ap.parse_args()

    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"capability {torch.cuda.get_device_capability(0)}")

    # (name, T, Hq, Hkv, D, rel_extent, window_left) — real Inkling head geometry,
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
    main()
