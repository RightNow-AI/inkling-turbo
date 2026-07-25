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


def reference(q, k, v, rel_logits, softmax_scale, ctx, window_left=None,
              with_bias=True):
    """(T_q, Hq, D) queries at absolute positions ctx .. ctx + T_q - 1.

    with_bias=False drops the relative-position term entirely. That variant is
    not a second oracle, it is the yardstick for how much the bias moves this
    shape's output at all. See SIGNAL_MARGIN below.
    """
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

    if with_bias:
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
    ref_nobias = reference(q, k, v, rel_logits, scale, ctx, window_left,
                           with_bias=False)

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
    # signal_mean is how far the bias moves this shape's output, measured as
    # mean|ref - ref_nobias|. It is an upper bound on what a kernel that DROPS
    # the bias would score against ref, so it is the number that says whether
    # TOL_MEAN can see a failure on this shape at all. Recorded per case so a
    # later reader can check the gate had power on the shape it passed.
    signal_mean = float((ref.float() - ref_nobias.float()).abs().mean())
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "signal_mean": signal_mean,
        "signal_over_tol": signal_mean / TOL_MEAN,
        "ref_absmax": float(ref.float().abs().max()),
        "shape": {"name": name, "T_q": T_q, "ctx": ctx, "T_k": T_k, "Hq": Hq,
                  "Hkv": Hkv, "rel_extent": rel_extent,
                  "window_left": window_left, "seed": seed},
    }


# Tolerances CALIBRATED against a deliberately broken control, not guessed.
#
# CAVEAT ON REPRODUCING THIS TABLE: it was measured while the per-case seed was
# abs(hash(name)), which PYTHONHASHSEED randomises per process, so these exact
# values came from tensors that no rerun could draw again. The seeds are fixed
# constants now (see CASES). The separation these numbers establish is a
# property of the defect and not of the draw, so the calibration conclusion
# stands, but the individual figures should be refreshed on the next H100 run so
# that they are reproducible from the file as it now stands.
#
# Both runs are on one H100. "fixed" is the n_block_max form, "broken" is the
# 128*(m_block+1) form put back on purpose to prove this gate is a gate:
#
#   case                          fixed max / mean      broken max / mean
#   control_full_prefill          7.81e-03 / 6.86e-05   7.81e-03 / 6.86e-05
#   chunked_global_128_on_1408    4.88e-04 / 3.73e-05   7.14e-02 / 1.05e-02
#   chunked_global_256_on_768     9.77e-04 / 4.67e-05   1.11e-01 / 1.43e-02
#   decode_global_ctx2047         2.44e-04 / 2.89e-05   3.56e-02 / 6.44e-03
#   decode_global_ctx4095         2.44e-04 / 2.18e-05   1.81e-02 / 3.28e-03
#   decode_swa_ctx4095            9.77e-04 / 6.96e-05   8.68e-02 / 1.69e-02
#   chunked_swa_128_on_1408       9.77e-04 / 6.39e-05   1.34e-01 / 1.79e-02
#
# The control is identical in both, which is the point: the defect only touches
# seqlen_q != seqlen_k, so a change there would mean the swap did something else.
#
# THE MEAN IS THE DISCRIMINATOR, and the first version of this file got that
# wrong. With TOL_MAX = 0.05 the broken decode_global_ctx4095 case passed at
# 1.81e-02, so 5 of 6 defective cases were caught and one was certified as
# correct. Max cannot separate cleanly because the legitimate prefill control
# sits at the bf16 quantum 7.81e-03, only 2.3x below the tightest broken value.
#
# Mean separates with no overlap at all: worst legitimate 6.96e-05 against best
# defective 3.28e-03, a factor of 47. TOL_MEAN = 5e-4 sits in that gap with 7.2x
# headroom above every passing case and 6.6x below every failing one.
#
# TOL_MAX stays loose on purpose. It is a backstop for a gross failure such as
# NaN or a dropped output, not the primary signal.
TOL_MAX = 0.05
TOL_MEAN = 5e-4

# TOL_MEAN IS ABSOLUTE, AND THAT PUTS A CEILING ON HOW DEEP A GLOBAL DECODE CASE
# MAY BE. A decode output is a near-uniform average of the attended value
# vectors, so both its magnitude and the size of any perturbation to it fall off
# like 1 / sqrt(number of attended keys). Dropping the bias entirely, computed
# from the reference alone (no kernel, no GPU, global causal, rel_extent 1024):
#
#   ctx      mean|ref_nobias - ref|      vs TOL_MEAN 5e-4
#   1023     1.26e-02                    12.6x   catchable
#   4095     3.48e-03                     7.0x   catchable, deepest case here
#   8191     1.45e-03                     2.9x   thin
#   16383    8.95e-04                     1.8x   very thin
#   32767    5.47e-04                     1.1x   effectively blind
#   65535    2.24e-04                     0.4x   BLIND: bias absent still passes
#
# So this file's cases stop at ctx = 4095 deliberately. Do NOT add a deeper
# global decode case here expecting it to mean anything: at 64K of KV, which is
# the shape the withdrawn decode figures were measured on, a kernel with no
# relative bias at all scores 2.24e-04 and passes. Sliding window is exempt,
# because the window bounds the number of attended keys and stops the dilution:
# at window 511 the same experiment gives 1.47e-02 at ctx 4095 and 1.55e-02 at
# ctx 65535. Deep global decode needs a reference-free instrument instead, of
# the kind in harness/parity_rel_bias_coverage.py.
#
# SIGNAL_MARGIN enforces that ceiling instead of trusting a comment. Every case
# must satisfy mean|ref - ref_nobias| >= SIGNAL_MARGIN * TOL_MEAN, otherwise a
# kernel that drops the bias could not have failed and a pass certifies nothing.
# A case that violates it is reported as a defective CASE, and the fix is to
# change the case or add a reference-free probe, never to lower TOL_MEAN.
SIGNAL_MARGIN = 4.0

# Seeds are explicit constants. They used to be abs(hash(name)), and str hashing
# is salted by PYTHONHASHSEED, so every process drew a different tensor and none
# of the max / mean values in the calibration table above could be reproduced by
# rerunning. A gate whose evidence cannot be regenerated is not evidence.
CASES = [
    # The control: seqlen_q == seqlen_k, which the old formula got RIGHT.
    # If this fails, something unrelated to the shear shift is broken.
    ("control_full_prefill", dict(T_q=1536, ctx=0, Hq=8, Hkv=1,
                                  rel_extent=1024, window_left=None,
                                  seed=1001)),
    # Chunked prefill: a short scheduled chunk on top of cached context.
    ("chunked_global_128_on_1408", dict(T_q=128, ctx=1408, Hq=8, Hkv=1,
                                        rel_extent=1024, window_left=None,
                                        seed=1002)),
    ("chunked_global_256_on_768", dict(T_q=256, ctx=768, Hq=8, Hkv=1,
                                       rel_extent=512, window_left=None,
                                       seed=1003)),
    # Decode: one query token against a long cache. This is the shape the old
    # formula degraded most, and the shape serving spends most of its time in.
    # ctx stops at 4095 on purpose: see the dilution table above TOL_MEAN.
    ("decode_global_ctx2047", dict(T_q=1, ctx=2047, Hq=8, Hkv=1,
                                   rel_extent=1024, window_left=None,
                                   seed=1004)),
    ("decode_global_ctx4095", dict(T_q=1, ctx=4095, Hq=8, Hkv=1,
                                   rel_extent=1024, window_left=None,
                                   seed=1005)),
    # Sliding window, where the extent and the window interact. The window caps
    # the attended key count, so these keep their signal at any depth.
    ("decode_swa_ctx4095", dict(T_q=1, ctx=4095, Hq=8, Hkv=2,
                                rel_extent=512, window_left=511, seed=1006)),
    ("chunked_swa_128_on_1408", dict(T_q=128, ctx=1408, Hq=8, Hkv=2,
                                     rel_extent=512, window_left=511,
                                     seed=1007)),
]


def main() -> None:
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"capability {torch.cuda.get_device_capability(0)}")
    print(f"tolerance: max <= {TOL_MAX}, mean <= {TOL_MEAN}")
    print()

    results = {}
    failures = 0
    blind = 0
    for name, kw in CASES:
        try:
            r = run_case(name, **kw)
            within = (r["max_abs_diff"] <= TOL_MAX
                      and r["mean_abs_diff"] <= TOL_MEAN)
            informative = r["signal_over_tol"] >= SIGNAL_MARGIN
            r["within_tolerance"] = within
            r["informative"] = informative
            r["pass"] = bool(within and informative)
            if not informative:
                blind += 1
            print(f"[{name}] {'OK' if r['pass'] else 'FAIL'}: "
                  f"max={r['max_abs_diff']:.4e} mean={r['mean_abs_diff']:.4e} "
                  f"signal={r['signal_mean']:.3e} "
                  f"({r['signal_over_tol']:.1f}x TOL_MEAN)"
                  + ("" if informative else
                     f"  <- CASE CANNOT FAIL: a kernel with no bias at all "
                     f"would score about {r['signal_mean']:.3e}, under "
                     f"TOL_MEAN={TOL_MEAN}. Fix the case, not the tolerance."))
            if not r["pass"]:
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
        "signal_margin": SIGNAL_MARGIN,
        "cases": results,
        "passed": len(CASES) - failures,
        "total": len(CASES),
        "uninformative": blind,
    }, indent=2), encoding="utf-8")

    print()
    print(f"{len(CASES) - failures}/{len(CASES)} cases pass "
          f"(within tolerance AND able to have failed)")
    if blind:
        print(f"WARNING: {blind} case(s) could not have failed at "
              f"TOL_MEAN={TOL_MEAN}. They certify nothing. Change the case or "
              f"use a reference-free probe; do not lower the tolerance.")
    print(f"saved: {out}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
