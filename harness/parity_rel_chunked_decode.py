#!/usr/bin/env python3
"""Gate: relative-bias attention when seqlen_q != seqlen_k.

WHY THIS FILE EXISTS

`harness/parity_fa4_rel.py` has three cases and all three pass
`cu_seqlens_q == cu_seqlens_k`, so every query sequence is the same length as
its key sequence. That is full prefill. It is the one family of shapes where the
sm_90 sheared-bias reader happened to be correct, and it was the only family
anything tested.

Everything else, chunked prefill and every decode step, has seqlen_k > seqlen_q.
On those shapes `flash_fwd_sm90.py` computed its shear shift from
`128 * (m_block + 1)` instead of `128 * n_block_max`. Since
`apply_rel_bias_sm90` guards on `0 <= tile_idx < bias_num_tiles`, the effect was
not a small numeric drift: at batch-1 decode with 64K of KV the shift came out
+9 instead of -502, so exactly one KV block received a bias tile, it was the
OLDEST block, and the other 511 received none. The model's learned
relative-position term was absent from decode.

Three things had to be simultaneously true for that to go unnoticed, and they
were:
  parity_fa4_rel only ever built seqlen_q == seqlen_k;
  the decode microbenchmarks time the kernel and never look at its output;
  the full-model gate ran `max_tokens=0, echo=True`, so it compared prompt
  logprobs and never decoded a token.

This gate closes the first hole. It is deliberately a separate file rather than
three more cases in parity_fa4_rel.py, because the reference here has to handle
a query that starts partway into the sequence, which the existing reference does
not.

WHAT IT CHECKS

For a request with `ctx` cached keys followed by `T_q` scheduled query tokens,
query index i sits at absolute position `ctx + i` and may attend to keys
`0 .. ctx + i`. The relative-position term applies at causal distance
`d = (ctx + i) - j` when `0 <= d < rel_extent`. That is the same semantics as
the prefill reference, with the query's absolute position offset by `ctx`.

Cases cover the shapes the old formula got wrong, plus one control that it got
right, so a pass here is informative in both directions.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch

D = 128
DEV = "cuda"


def reference(q, k, v, rel_logits, softmax_scale, ctx, window_left=None):
    """(T_q, Hq, D) queries at absolute positions ctx .. ctx + T_q - 1."""
    T_q, Hq, _ = q.shape
    T_k, Hkv, _ = k.shape
    rel_extent = rel_logits.shape[-1]
    rep = Hq // Hkv
    kf = k.repeat_interleave(rep, dim=1).float()
    vf = v.repeat_interleave(rep, dim=1).float()

    scores = torch.einsum("ihd,jhd->hij", q.float(), kf) * softmax_scale

    q_pos = torch.arange(T_q, device=q.device).view(-1, 1) + ctx
    k_pos = torch.arange(T_k, device=q.device).view(1, -1)
    dist = q_pos - k_pos                                    # (T_q, T_k)

    in_range = (dist >= 0) & (dist < rel_extent)
    d_idx = dist.clamp(0, rel_extent - 1)
    bias = rel_logits.float().permute(1, 0, 2).gather(
        2, d_idx.unsqueeze(0).expand(Hq, T_q, T_k)
    )
    scores += bias * in_range

    keep = dist >= 0
    if window_left is not None:
        keep &= dist <= window_left
    scores.masked_fill_(~keep, float("-inf"))

    out = torch.einsum("hij,jhd->ihd", torch.softmax(scores, dim=-1), vf)
    return out.to(q.dtype)


def run_case(name, T_q, ctx, Hq, Hkv, rel_extent, window_left, seed):
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func

    torch.manual_seed(seed)
    T_k = ctx + T_q
    q = torch.randn(T_q, Hq, D, dtype=torch.bfloat16, device=DEV) / (D**0.25)
    k = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=DEV) / (D**0.25)
    v = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=DEV)

    # Same construction as parity_fa4_rel: rel_logits from (r, proj) the way
    # qkvr_prep builds them, so the bias is a realistic tensor and not noise.
    r_small = torch.randn(T_q, Hq, 16, dtype=torch.bfloat16, device=DEV) * 0.4
    proj = torch.randn(16, rel_extent, dtype=torch.bfloat16, device=DEV) * 0.3
    rel_logits = (r_small.float() @ proj.float()).to(torch.bfloat16)

    scale = 1.0 / D
    ref = reference(q, k, v, rel_logits, scale, ctx, window_left)

    cu_q = torch.tensor([0, T_q], dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0, T_k], dtype=torch.int32, device=DEV)
    window = (None, None) if window_left is None else (window_left, 0)

    out = flash_attn_varlen_func(
        q=q, k=k, v=v,
        rel_bias=rel_logits,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=T_q, max_seqlen_k=T_k,
        softmax_scale=scale, causal=True, window_size=window,
    )
    if isinstance(out, tuple):
        out = out[0]

    diff = (out.float() - ref.float()).abs()
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "shape": {"T_q": T_q, "ctx": ctx, "T_k": T_k, "Hq": Hq, "Hkv": Hkv,
                  "rel_extent": rel_extent, "window_left": window_left},
    }


# Tolerances match parity_fa4_rel's observed bf16 behaviour: the three prefill
# cases there land at max 7.8125e-03, which is the bf16 quantum at these
# magnitudes. A dropped or misplaced bias tile is orders of magnitude larger,
# so this threshold separates the two cleanly rather than finely.
TOL_MAX = 0.05
TOL_MEAN = 0.005

CASES = [
    # The control: seqlen_q == seqlen_k, which the old formula got RIGHT.
    # If this fails, something unrelated to the shear shift is broken.
    ("control_full_prefill", dict(T_q=1536, ctx=0, Hq=8, Hkv=1,
                                  rel_extent=1024, window_left=None)),
    # Chunked prefill: a short scheduled chunk on top of cached context.
    ("chunked_global_128_on_1408", dict(T_q=128, ctx=1408, Hq=8, Hkv=1,
                                        rel_extent=1024, window_left=None)),
    ("chunked_global_256_on_768", dict(T_q=256, ctx=768, Hq=8, Hkv=1,
                                       rel_extent=512, window_left=None)),
    # Decode: one query token against a long cache. This is the shape the old
    # formula degraded most, and the shape serving spends most of its time in.
    ("decode_global_ctx2047", dict(T_q=1, ctx=2047, Hq=8, Hkv=1,
                                   rel_extent=1024, window_left=None)),
    ("decode_global_ctx4095", dict(T_q=1, ctx=4095, Hq=8, Hkv=1,
                                   rel_extent=1024, window_left=None)),
    # Sliding window, where the extent and the window interact.
    ("decode_swa_ctx4095", dict(T_q=1, ctx=4095, Hq=8, Hkv=2,
                                rel_extent=512, window_left=511)),
    ("chunked_swa_128_on_1408", dict(T_q=128, ctx=1408, Hq=8, Hkv=2,
                                     rel_extent=512, window_left=511)),
]


def main() -> None:
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"capability {torch.cuda.get_device_capability(0)}")
    print(f"tolerance: max <= {TOL_MAX}, mean <= {TOL_MEAN}")
    print()

    results = {}
    failures = 0
    for name, kw in CASES:
        try:
            r = run_case(name, seed=abs(hash(name)) % (2**31), **kw)
            ok = r["max_abs_diff"] <= TOL_MAX and r["mean_abs_diff"] <= TOL_MEAN
            r["pass"] = ok
            print(f"[{name}] {'OK' if ok else 'FAIL'}: "
                  f"max={r['max_abs_diff']:.4e} mean={r['mean_abs_diff']:.4e}")
            if not ok:
                failures += 1
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            r = {"error": f"{type(exc).__name__}: {exc}", "pass": False}
            print(f"[{name}] FAIL: {type(exc).__name__}: {exc}")
            failures += 1
        results[name] = r

    cc = torch.cuda.get_device_capability(0)
    out = Path(__file__).with_name(
        f"parity_rel_chunked_decode_sm{cc[0]}{cc[1]}.json"
    )
    out.write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tol_max": TOL_MAX,
        "tol_mean": TOL_MEAN,
        "cases": results,
        "passed": len(CASES) - failures,
        "total": len(CASES),
    }, indent=2), encoding="utf-8")

    print()
    print(f"{len(CASES) - failures}/{len(CASES)} cases within tolerance")
    print(f"saved: {out}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
