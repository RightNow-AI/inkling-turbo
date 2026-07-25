#!/usr/bin/env python3
"""Gate: relative-bias attention with more than one sequence in the batch.

WHY THIS FILE EXISTS

Every attention parity case in this repository that is checked against a
float32 oracle builds exactly ONE sequence:

  harness/parity_fa4_rel.py          cu = [0, T],  one sequence, seqlen_q == seqlen_k
  harness/parity_rel_chunked_decode  cu_q = [0, T_q], cu_k = [0, T_k], one sequence
  harness/tune_sm80.py::parity_ok    cu = [0, T],  one sequence, Hq == Hkv

With one sequence, `seqlen.offset_q` is identically 0. The sm_90 bias reader
computes its per-sequence base as

    mBias_cur = cute.domain_offset((seqlen.offset_q, 0), mBias[None, None, head])

so a wrong offset_q term, a wrong per-sequence seqlen_k, or a batch index
crossed with a sequence index is INVISIBLE to every oracle-checked case in the
repo. That is structurally the same blindness that hid the shear shift defect:
there, a suite that only built seqlen_q == seqlen_k made `128 * (m_block + 1)`
and `128 * n_block_max` numerically equal, so the wrong one passed.

`harness/parity_shear_fusion.py::case_attention_consumes` does build two
sequences, but it compares our `bias=` path against our own `rel_bias=` path.
Both read the same offset_q arithmetic, so a shared error cancels. It is an
equivalence check, not a correctness check.

Two more axes are closed here, both of which every existing oracle case also
misses:

  seqlen_k not a multiple of the KV tile. Every oracle-checked case so far has
  seqlen_k in {128, 512, 1024, 1536, 2048, 4096}, all multiples of 128, so the
  partial last KV block never met the sheared bias.

  a partial m tail that follows at least one full m block. parity_fa4_rel uses
  128 and 1536, parity_rel_chunked_decode uses 1, 128, 256 and 1536. All are
  either whole multiples of 128 or smaller than 128. A seqlen_q of 200 is
  neither, and it is an ordinary chunked-prefill size.

WHAT IT CHECKS

For request r with `ctx[r]` cached keys and `T_q[r]` scheduled query tokens,
query i of that request sits at absolute position `ctx[r] + i` and attends keys
`0 .. ctx[r] + i` of that request. The relative term applies at causal distance
`d = (ctx[r] + i) - j` when `0 <= d < rel_extent`. This is FlashAttention's
bottom-right causal alignment, which is also what the day-0 score_mod computes
(`rel_dist = (q_idx + seqlen_k - seqlen_q) - kv_idx`, vllm
models/inkling/nvidia/ops/fa4_rel_attention.py line 46), so one convention
covers the reference, the sheared kernel and the callback baseline.

`rel_bias` is indexed by the GLOBAL query row, `offset_q + i`, exactly as the
score_mod does with `global_q_idx`. The oracle below slices it per request, so
a kernel that used the wrong row base fails here and cannot fail silently.

DEFECT SEPARATION, and why every case reports it

A tolerance can only fail if correct and defective outputs are further apart
than the tolerance, and at long context they are not. The relative term touches
only the most recent `rel_extent` keys, so with attention spread near-uniformly
over tens of thousands of keys, deleting the bias barely moves the output.
`harness/parity_rel_chunked_decode.py` records the dilution measured from the
reference alone, global causal, rel_extent 1024, against its TOL_MEAN of 5e-4:

    ctx 1023   mean|ref_nobias - ref| = 1.26e-02   25.2x   catchable
    ctx 4095                            3.48e-03    7.0x   catchable
    ctx 8191                            1.45e-03    2.9x   thin
    ctx 32767                           5.47e-04    1.1x   effectively blind
    ctx 65535                           2.24e-04    0.4x   BLIND

Sliding window is exempt, because the window caps the attended key count and
stops the dilution.

Every case here therefore computes the defective reference as well, and records
`signal_mean`, `signal_over_tol` and `can_detect_dropped_bias`. Each case also
declares a `role`:

  coverage  the case exists to prove the bias arrives. If it is not
            informative, the CASE is defective and the run fails, because a pass
            would certify nothing. The fix is to change the case or add a
            reference-free probe, never to lower the tolerance.
  control   the case exists to isolate one variable, for example offset_q with
            everything else held in the family that already worked. A control
            may legitimately have no bias signal at all, and it is exempt.

The `bias_gain` knob buys signal without changing what is compared, because the
same tensor goes to the kernel and to the oracle. It does NOT buy it in
proportion: past a point the biased keys take most of the softmax mass and the
difference stops growing. Measured on CPU at full shape, the mixed global decode
batch scores 7.0x TOL_MEAN at gain 1 and 5.9x at gain 16, while the sliding
window batch goes from 17.0x to 540x. So gain helps a lot where the window bounds
the key count and barely at all in deep global decode, which is exactly where the
problem is. `harness/parity_rel_bias_coverage.py` attacks that case from the
other side, with a probe that needs no reference.

NOT RUN ON A GPU. Written from the code, never executed. Every number it would
report is absent until it runs on real hardware.

Run (WSL): cd ~/inkling-turbo/vllm && source .venv/bin/activate && \
  python $REPO/harness/parity_rel_varlen_batch.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch

D = 128
DEV = "cuda"

# Tolerances are NOT independently chosen here. They are the ones
# harness/parity_rel_chunked_decode.py calibrated on an H100 by running the
# defect on purpose: worst legitimate mean 6.96e-05 against best defective mean
# 3.28e-03, a factor of 47, with TOL_MEAN sitting in that gap. The mean is the
# discriminator; TOL_MAX is a loose backstop for a gross failure such as a NaN
# or a dropped output, because the legitimate bf16 quantum at 7.81e-03 max sits
# only 2.3x below the tightest defective max and cannot separate cleanly.
#
# That calibration was measured at Hq=8, Hkv=1 or 2, one sequence. The cases
# here reach Hq=64 and several sequences, which those numbers do not cover, so
# the FIRST run on real hardware must be read for headroom rather than assumed
# to inherit it. If a legitimate case lands within 4x of TOL_MEAN, recalibrate
# and say so; do not loosen the tolerance to make a run green.
TOL_MAX = 0.05
TOL_MEAN = 5e-4

# A case is informative only if a kernel that dropped the bias would have failed
# it. Same rule and same margin as parity_rel_chunked_decode.py::SIGNAL_MARGIN.
SIGNAL_MARGIN = 4.0


KV_TILE = 128  # tile_n is forced to 128 for the sm_90 bias path (interface.py)


def reference_one_seq(q, k, v, rel_rows, softmax_scale, ctx, window_left,
                      bias_only_oldest_tile=False):
    """One request. q is (T_q, Hq, D) at absolute positions ctx .. ctx+T_q-1,
    k/v are (T_k, Hkv, D) for that request only, rel_rows is that request's
    slice of the global rel_bias, (T_q, Hq, rel_extent).

    bias_only_oldest_tile reproduces the observed sm_90 defect: the relative
    term reaches KV tile 0 and no other tile. It is not a second reference to
    test against, it is used to measure how far apart correct and defective
    OUTPUTS are at each shape, which is the only way to know whether a
    tolerance can fail at all. See DEFECT SEPARATION in the module docstring.
    """
    T_q, Hq, _ = q.shape
    T_k, Hkv, _ = k.shape
    rel_extent = rel_rows.shape[-1]
    rep = Hq // Hkv
    kf = k.repeat_interleave(rep, dim=1).float()
    vf = v.repeat_interleave(rep, dim=1).float()

    scores = torch.einsum("ihd,jhd->hij", q.float(), kf) * softmax_scale

    q_pos = torch.arange(T_q, device=q.device).view(-1, 1) + ctx
    k_pos = torch.arange(T_k, device=q.device).view(1, -1)
    dist = q_pos - k_pos

    in_range = (dist >= 0) & (dist < rel_extent)
    if bias_only_oldest_tile:
        in_range = in_range & ((k_pos // KV_TILE) == 0)
    d_idx = dist.clamp(0, rel_extent - 1)
    bias = rel_rows.float().permute(1, 0, 2).gather(
        2, d_idx.unsqueeze(0).expand(Hq, T_q, T_k)
    )
    scores += bias * in_range

    keep = dist >= 0
    if window_left is not None:
        keep &= dist <= window_left
    scores.masked_fill_(~keep, float("-inf"))

    out = torch.einsum("hij,jhd->ihd", torch.softmax(scores, dim=-1), vf)
    return out.to(q.dtype)


def run_case(name, T_q_list, ctx_list, Hq, Hkv, rel_extent, is_local, seed,
             bias_gain=1.0):
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func

    assert len(T_q_list) == len(ctx_list)
    torch.manual_seed(seed)

    T_k_list = [c + t for c, t in zip(ctx_list, T_q_list)]
    total_q = sum(T_q_list)
    total_k = sum(T_k_list)

    q = torch.randn(total_q, Hq, D, dtype=torch.bfloat16, device=DEV) / (D**0.25)
    k = torch.randn(total_k, Hkv, D, dtype=torch.bfloat16, device=DEV) / (D**0.25)
    v = torch.randn(total_k, Hkv, D, dtype=torch.bfloat16, device=DEV)

    # rel_logits built from (r, proj) the way qkvr_prep builds them, so the bias
    # is a realistic tensor rather than noise. Same construction as
    # parity_fa4_rel and parity_rel_chunked_decode, so the three are comparable.
    r_small = torch.randn(total_q, Hq, 16, dtype=torch.bfloat16, device=DEV) * 0.4
    proj = torch.randn(16, rel_extent, dtype=torch.bfloat16, device=DEV) * 0.3
    rel_logits = ((r_small.float() @ proj.float()) * bias_gain).to(torch.bfloat16)

    def cumulative(lengths: list[int]) -> list[int]:
        acc, out_list = 0, [0]
        for n in lengths:
            acc += n
            out_list.append(acc)
        return out_list

    cu_q = torch.tensor(cumulative(T_q_list), dtype=torch.int32, device=DEV)
    cu_k = torch.tensor(cumulative(T_k_list), dtype=torch.int32, device=DEV)

    # Production convention: window_left = rel_extent - 1 with window_right = 0,
    # which is what the interface requires when the window is set alongside a
    # relative bias (interface.py rel_extent guard) and what
    # parity_shear_fusion.py uses.
    window_left = rel_extent - 1 if is_local else None
    window = (None, None) if window_left is None else (window_left, 0)
    scale = 1.0 / D

    out = flash_attn_varlen_func(
        q=q, k=k, v=v,
        rel_bias=rel_logits,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=max(T_q_list), max_seqlen_k=max(T_k_list),
        softmax_scale=scale, causal=True, window_size=window,
    )
    if isinstance(out, tuple):
        out = out[0]

    per_seq = []
    worst_max, worst_mean, worst_n = 0.0, 0.0, 0
    for r, (T_q, T_k, ctx) in enumerate(zip(T_q_list, T_k_list, ctx_list)):
        qs, qe = int(cu_q[r]), int(cu_q[r + 1])
        ks, ke = int(cu_k[r]), int(cu_k[r + 1])
        ref = reference_one_seq(
            q[qs:qe], k[ks:ke], v[ks:ke], rel_logits[qs:qe], scale, ctx,
            window_left)
        diff = (out[qs:qe].float() - ref.float()).abs()
        mx, mean = float(diff.max()), float(diff.mean())
        # What the defect would look like at this shape, on these inputs.
        defective = reference_one_seq(
            q[qs:qe], k[ks:ke], v[ks:ke], rel_logits[qs:qe], scale, ctx,
            window_left, bias_only_oldest_tile=True)
        gap = (ref.float() - defective.float()).abs()
        sep = float(gap.mean())
        per_seq.append({"seq": r, "T_q": T_q, "ctx": ctx, "T_k": T_k,
                        "offset_q": qs, "max_abs_diff": mx,
                        "mean_abs_diff": mean,
                        "signal_mean": sep,
                        "signal_max": float(gap.max()),
                        "signal_over_tol": sep / TOL_MEAN,
                        "kv_tiles": (T_k + KV_TILE - 1) // KV_TILE,
                        "seqlen_k_mod_128": T_k % 128,
                        "seqlen_q_mod_128": T_q % 128})
        worst_max = max(worst_max, mx)
        # mean over all compared elements, not a mean of means
        worst_mean += mean * diff.numel()
        worst_n += diff.numel()

    _measurable = [x for x in per_seq if x["kv_tiles"] > 1]
    return {
        "max_abs_diff": worst_max,
        "mean_abs_diff": worst_mean / worst_n if worst_n else None,
        "num_seqs": len(T_q_list),
        "bias_gain": bias_gain,
        # Only sequences with more than one KV tile can express this defect at
        # all: with a single tile, "bias reached only the oldest tile" is the
        # correct answer, so such a sequence has no signal by construction and
        # must be excluded rather than dragging the batch minimum to zero. The
        # weakest MEASURABLE sequence decides, because a batch containing one
        # blind measurable sequence certifies nothing about that sequence.
        "measurable_seqs": len(_measurable),
        "signal_mean": (min(x["signal_mean"] for x in _measurable)
                        if _measurable else 0.0),
        "signal_over_tol": (min(x["signal_over_tol"] for x in _measurable)
                            if _measurable else 0.0),
        "can_detect_dropped_bias": bool(
            _measurable
            and min(x["signal_over_tol"] for x in _measurable)
            >= SIGNAL_MARGIN),
        "per_seq": per_seq,
        "shape": {"T_q": T_q_list, "ctx": ctx_list, "T_k": T_k_list,
                  "Hq": Hq, "Hkv": Hkv, "rel_extent": rel_extent,
                  "window_left": window_left},
    }


# role: "coverage" means the case exists to prove the bias arrives, and a
# non-informative coverage case FAILS the run because a pass would certify
# nothing. "control" means the case isolates one variable and may legitimately
# carry no bias signal.
#
# Seeds are explicit constants. abs(hash(name)) is salted by PYTHONHASHSEED, so
# a gate keyed on it draws different tensors in every process and none of its
# recorded numbers can be regenerated. parity_rel_chunked_decode.py hit exactly
# that and had to annotate its own calibration table as unreproducible.
CASES = [
    # Control. One sequence, seqlen_q == seqlen_k, the family everything
    # already covered. If this fails, the failure is not about batching.
    ("control_single_full_prefill", "control",
     dict(T_q_list=[512], ctx_list=[0], Hq=8, Hkv=1, rel_extent=512,
          is_local=False, seed=2001)),
    # offset_q alone. Every sequence is a full prefill, so the shear shift is in
    # the family it always got right, and the ONLY new thing is offset_q != 0
    # for sequences 1 and 2. A wrong row base fails here and nowhere else. It is
    # a control: the third sequence has 64 keys, one KV tile, so there is no
    # bias signal to measure there and none is expected.
    ("multi_prefill_offset_only", "control",
     dict(T_q_list=[256, 137, 64], ctx_list=[0, 0, 0], Hq=8, Hkv=1,
          rel_extent=512, is_local=False, seed=2002)),
    # A KV tail on its own: one sequence, seqlen_k not a multiple of 128.
    ("single_kv_tail_decode", "coverage",
     dict(T_q_list=[1], ctx_list=[1000], Hq=8, Hkv=1, rel_extent=512,
          is_local=False, seed=2003)),
    # A partial m tail that follows a full m block, on its own.
    ("single_m_tail_chunked", "coverage",
     dict(T_q_list=[200], ctx_list=[1000], Hq=8, Hkv=1, rel_extent=512,
          is_local=False, seed=2004)),
    # The serving shape. A mixed decode batch: one query token per request,
    # different context length per request, none of them tile aligned. ctx stops
    # at 4095 for the same dilution reason parity_rel_chunked_decode.py stops
    # there. Deeper global decode belongs to the reference-free probe.
    ("mixed_decode_batch_global", "coverage",
     dict(T_q_list=[1, 1, 1, 1], ctx_list=[2047, 1000, 137, 4095],
          Hq=8, Hkv=1, rel_extent=1024, is_local=False, seed=2005)),
    ("mixed_decode_batch_swa", "coverage",
     dict(T_q_list=[1, 1, 1, 1], ctx_list=[2047, 1000, 137, 4095],
          Hq=8, Hkv=2, rel_extent=512, is_local=True, seed=2006)),
    # Continuous batching in one call: a prefill chunk, two decodes and a
    # short chunk, which is what the scheduler actually hands the kernel.
    ("mixed_chunked_and_decode_global", "coverage",
     dict(T_q_list=[200, 1, 137, 64], ctx_list=[0, 900, 1408, 63],
          Hq=8, Hkv=1, rel_extent=512, is_local=False, seed=2007)),
    ("mixed_chunked_and_decode_swa", "coverage",
     dict(T_q_list=[200, 1, 137, 64], ctx_list=[0, 900, 1408, 63],
          Hq=8, Hkv=2, rel_extent=512, is_local=True, seed=2008)),
    # Inkling global head geometry at TP1 is 64 q heads over 8 kv heads. The
    # sm_90 interface force-disables pack_gqa when a bias is present, so the
    # ratio decides which code path runs and is not a free parameter.
    ("mixed_decode_batch_hq64", "coverage",
     dict(T_q_list=[1, 1, 1], ctx_list=[4095, 2000, 129], Hq=64, Hkv=8,
          rel_extent=1024, is_local=False, seed=2009)),
    # The same shapes with the bias scaled up. These keep their signal at any
    # depth, so they stay informative if the shapes above are ever deepened.
    ("mixed_decode_batch_global_gain16", "coverage",
     dict(T_q_list=[1, 1, 1, 1], ctx_list=[2047, 1000, 137, 4095],
          Hq=8, Hkv=1, rel_extent=1024, is_local=False, bias_gain=16.0,
          seed=2010)),
    ("mixed_decode_batch_swa_gain16", "coverage",
     dict(T_q_list=[1, 1, 1, 1], ctx_list=[2047, 1000, 137, 4095],
          Hq=8, Hkv=2, rel_extent=512, is_local=True, bias_gain=16.0,
          seed=2011)),
    ("mixed_chunked_and_decode_global_gain16", "coverage",
     dict(T_q_list=[200, 1, 137, 64], ctx_list=[0, 900, 1408, 63],
          Hq=8, Hkv=1, rel_extent=512, is_local=False, bias_gain=16.0,
          seed=2012)),
]


def main() -> None:
    cc = torch.cuda.get_device_capability(0)
    device = torch.cuda.get_device_name(0)
    print(f"device: {device}, capability {cc}")
    print(f"tolerance: max <= {TOL_MAX}, mean <= {TOL_MEAN}; a coverage case "
          f"must also carry signal >= {SIGNAL_MARGIN}x TOL_MEAN")
    print()

    results = {}
    failures = 0
    blind = 0
    for name, role, kw in CASES:
        try:
            r = run_case(name, **kw)
            r["role"] = role
            within = (r["max_abs_diff"] <= TOL_MAX
                      and r["mean_abs_diff"] <= TOL_MEAN)
            informative = r["can_detect_dropped_bias"]
            r["within_tolerance"] = within
            r["informative"] = informative
            # A control is exempt from the signal requirement by design. A
            # coverage case is not: with no signal it cannot fail, so passing it
            # is not a result.
            r["pass"] = bool(within and (informative or role == "control"))
            if role == "coverage" and not informative:
                blind += 1
            note = ("" if informative else
                    "  <-- NO POWER: cannot fail on a dropped bias")
            print(f"[{name}] {'OK' if r['pass'] else 'FAIL'} ({role}): "
                  f"max={r['max_abs_diff']:.4e} mean={r['mean_abs_diff']:.4e} "
                  f"signal={r['signal_mean']:.3e} "
                  f"({r['signal_over_tol']:.1f}x TOL_MEAN){note}")
            if not r["pass"]:
                failures += 1
                for sq in r["per_seq"]:
                    print(f"    seq {sq['seq']} offset_q={sq['offset_q']} "
                          f"T_q={sq['T_q']} ctx={sq['ctx']} T_k={sq['T_k']}: "
                          f"max={sq['max_abs_diff']:.4e} "
                          f"mean={sq['mean_abs_diff']:.4e} "
                          f"signal={sq['signal_mean']:.3e}")
        except AssertionError as exc:
            r = {"role": role, "skip": f"AssertionError: {exc}", "pass": None}
            print(f"[{name}] SKIP ({role}): AssertionError: {exc}")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            r = {"role": role, "error": f"{type(exc).__name__}: {exc}",
                 "pass": False}
            print(f"[{name}] FAIL ({role}): {type(exc).__name__}: {exc}")
            failures += 1
        results[name] = r

    ran = [r for r in results.values() if r.get("pass") is not None]
    detectors = [r for r in ran if r.get("informative")]
    out = Path(__file__).with_name(
        f"parity_rel_varlen_batch_sm{cc[0]}{cc[1]}.json")
    out.write_text(json.dumps({
        "device": device,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tol_max": TOL_MAX,
        "tol_mean": TOL_MEAN,
        "signal_margin": SIGNAL_MARGIN,
        "kv_tile": KV_TILE,
        "cases": results,
        "passed": len(ran) - failures,
        "ran": len(ran),
        "total": len(CASES),
        "coverage_cases_without_power": blind,
        "cases_able_to_detect_dropped_bias": len(detectors),
        "detection_note": (
            "signal_mean is mean|correct - bias_reaching_only_the_oldest_kv_"
            "tile| at this shape, which is what a kernel that dropped the bias "
            "would score against the reference. Below tol_mean the case cannot "
            "fail on a missing bias, only on a numerically wrong one. Count the "
            "detectors, not the passes."),
    }, indent=2), encoding="utf-8")

    print()
    print(f"{len(ran) - failures}/{len(ran)} cases pass "
          f"({len(CASES) - len(ran)} skipped)")
    print(f"{len(detectors)}/{len(ran)} cases could have failed if the bias "
          f"were dropped entirely")
    if blind:
        print(f"WARNING: {blind} coverage case(s) had no power. Fix the case or "
              f"move the shape to harness/parity_rel_bias_coverage.py. Do not "
              f"lower TOL_MEAN.")
    print(f"saved: {out}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
