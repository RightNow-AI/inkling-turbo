#!/usr/bin/env python3
"""Gate: relative-bias attention over a PAGED KV cache, which is the only call
shape vLLM ever makes.

WHY THIS FILE EXISTS

This is the largest ungated axis in the repository, and unlike the shear shift
defect it is not a subtle specialisation. It is a whole argument that no gate
has ever passed.

`vllm/models/inkling/nvidia/ops/fa4_rel_attention.py::inkling_fa4_rel_attention`
calls the kernel like this, and this is what serving runs on every step:

    flash_attn_varlen_func(
        q=q, k=key_cache, v=value_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,          <-- not cu_seqlens_k
        max_seqlen_q=max_seqlen_q,        <-- max_seqlen_k NOT passed
        page_table=block_table,           <-- paged K and V
        num_splits=num_splits,
        rel_bias=rel_logits, ...)

Every harness in harness/ calls it like this instead:

    flash_attn_varlen_func(q, k, v, rel_bias=..., cu_seqlens_q=..,
                           cu_seqlens_k=.., max_seqlen_q=.., max_seqlen_k=..)

so `page_table` and `seqused_k` are never once exercised, on any architecture,
by any gate. `kernels/tml_fa4_modified/interface.py` even makes the two mutually
exclusive: `assert cu_seqlens_k is None` when a page table is given, so the two
shapes cannot accidentally overlap. The consequences are concrete:

  the paged K and V load path in flash_fwd_sm90.py (`mPageTable is not None`,
  both the TMA variant and `paged_kv_manager.load_page_table`) has no gate;

  `seqused_k` is where the kernel and the ShearingBias writer learn each
  sequence's KV length, and that length is the input to the shear shift that
  was just wrong. The fix was validated, when it is validated, only on the
  cu_seqlens_k route;

  with `max_seqlen_k` unset the interface sets it to `num_pages * page_size`,
  the WHOLE cache, not the longest live sequence. That value is what
  ShearingBias is handed. Nothing has ever checked that the writer and the
  reader agree under that substitution.

The page table here is a random permutation of physical pages, never the
identity, because an identity table makes a page-table indexing bug invisible
in exactly the way a single sequence makes an offset_q bug invisible.

Both page sizes matter and both are covered. `interface.py` computes
`paged_kv_non_tma = page_size not in [None, tile_n]` with tile_n forced to 128
for the sm_90 bias path, so page_size 128 takes the TMA paged path and any other
page size takes a different, non-TMA path. vLLM's default attention block size
is not 128, so production is on the non-TMA branch.

WHAT IT CHECKS

The reference gathers each request's pages into a contiguous K and V, then runs
the same float32 oracle as harness/parity_rel_varlen_batch.py: query i of
request r sits at absolute position `ctx[r] + i` where `ctx[r] = L[r] - T_q[r]`,
attends keys `0 .. ctx[r] + i`, and takes the relative term at distance
`d = (ctx[r] + i) - j` for `0 <= d < rel_extent`. If the gathered reference and
the kernel disagree, either the page walk or the bias placement is wrong, and
the per-sequence breakdown says which sequence.

The split-KV cases are compared against the SAME float32 oracle, deliberately.
`scripts/modal_e2e_bench.py::splitkv_case` compares num_splits > 1 against
num_splits == 1 of our own kernel, which cannot detect an error both share, and
during the shear shift defect both shared one.

DEFECT SEPARATION

Each case computes what the observed defect would look like at its own shape and
records `signal_mean`, `signal_over_tol` and `can_detect_dropped_bias`. At long
context the relative term touches only the most recent rel_extent keys, so
deleting it barely moves a near-uniform decode output and the signal falls under
the tolerance. parity_rel_chunked_decode.py measures the dilution, global causal
with rel_extent 1024, against TOL_MEAN 5e-4: 3.48e-03 at ctx 4095 (7.0x),
1.45e-03 at ctx 8191 (2.9x), 2.24e-04 at ctx 65535 (0.4x, blind). Sliding window
is exempt because the window caps the attended key count.

So each case declares a `role`. A "coverage" case without signal FAILS the run,
because passing it would certify nothing; a "control" case is exempt because it
exists to isolate one variable. A sequence with only one KV tile is excluded from
the signal minimum, since "the bias reached only the oldest tile" is the correct
answer there and it has no signal by construction.

bias_gain buys signal but not in proportion: past a point the biased keys take
most of the softmax mass. Measured on CPU at full shape, paged16 global decode
goes from 7.1x TOL_MEAN at gain 1 to 28.5x at gain 16, while the deep split case
goes to 1248x. `harness/parity_rel_bias_coverage.py` covers deep global decode
with no reference at all. Count the detectors in the artifact, not the passes.

NOT RUN ON A GPU. Written from the code, never executed. On sm_80 and sm_120 the
interface raises `paged KV not supported on generic path`, so those arches are
expected to report SKIP with that message rather than a number.

Run (WSL): cd ~/inkling-turbo/vllm && source .venv/bin/activate && \
  python $REPO/harness/parity_rel_paged.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch

D = 128
DEV = "cuda"

# Tolerances are inherited, not invented. harness/parity_rel_chunked_decode.py
# calibrated them on an H100 by putting the defect back on purpose: worst
# legitimate mean 6.96e-05 against best defective mean 3.28e-03. The mean is the
# discriminator; TOL_MAX is a loose backstop for a gross failure such as a NaN or
# a dropped output, since the legitimate bf16 max quantum of 7.81e-03 sits only
# 2.3x below the tightest defective max.
#
# That calibration used contiguous KV, one sequence, Hq=8. Paged KV with several
# sequences is not covered by it, so read the FIRST real run for headroom instead
# of assuming it carries over. If a legitimate case lands within 4x of TOL_MEAN,
# recalibrate and record it; do not loosen the tolerance to make a run green.
TOL_MAX = 0.05
TOL_MEAN = 5e-4

# A coverage case is only a result if a kernel that dropped the bias would have
# failed it. Same rule and margin as parity_rel_chunked_decode.py.
SIGNAL_MARGIN = 4.0


KV_TILE = 128  # tile_n is forced to 128 for the sm_90 bias path (interface.py)


def reference_one_seq(q, k, v, rel_rows, softmax_scale, ctx, window_left,
                      bias_only_oldest_tile=False):
    """One request, with k and v ALREADY gathered out of the paged cache.

    bias_only_oldest_tile reproduces the observed sm_90 defect and is used only
    to measure how far apart correct and defective output are at this shape.
    See DEFECT SEPARATION in the module docstring: at long context that distance
    is smaller than the tolerance, so a passing case is not automatically an
    informative case.
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


def gather_pages(cache, block_table_row, length, page_size):
    """Rebuild one request's contiguous KV from the paged cache, the way the
    kernel is supposed to walk it. cache is (num_pages, page_size, H, D)."""
    n_pages = (length + page_size - 1) // page_size
    chunks = [cache[int(block_table_row[j])] for j in range(n_pages)]
    return torch.cat(chunks, dim=0)[:length]


def run_case(name, T_q_list, ctx_list, Hq, Hkv, rel_extent, is_local,
             page_size, num_splits, seed, bias_gain=1.0):
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func

    assert len(T_q_list) == len(ctx_list)
    torch.manual_seed(seed)

    L_list = [c + t for c, t in zip(ctx_list, T_q_list)]
    B = len(T_q_list)
    total_q = sum(T_q_list)
    max_pages_per_seq = max((L + page_size - 1) // page_size for L in L_list)

    # Physical pages: more than the requests need, so the table is sparse in the
    # cache and a walk that ignores the table lands on unrelated data.
    num_pages = B * max_pages_per_seq + 3
    perm = torch.randperm(num_pages, device=DEV)[: B * max_pages_per_seq]
    block_table = perm.reshape(B, max_pages_per_seq).to(torch.int32)

    key_cache = (torch.randn(num_pages, page_size, Hkv, D,
                             dtype=torch.bfloat16, device=DEV) / (D**0.25))
    value_cache = torch.randn(num_pages, page_size, Hkv, D,
                              dtype=torch.bfloat16, device=DEV)

    q = torch.randn(total_q, Hq, D, dtype=torch.bfloat16, device=DEV) / (D**0.25)

    r_small = torch.randn(total_q, Hq, 16, dtype=torch.bfloat16, device=DEV) * 0.4
    proj = torch.randn(16, rel_extent, dtype=torch.bfloat16, device=DEV) * 0.3
    rel_logits = ((r_small.float() @ proj.float()) * bias_gain).to(torch.bfloat16)

    acc, cu = 0, [0]
    for n in T_q_list:
        acc += n
        cu.append(acc)
    cu_q = torch.tensor(cu, dtype=torch.int32, device=DEV)
    seqused_k = torch.tensor(L_list, dtype=torch.int32, device=DEV)

    window_left = rel_extent - 1 if is_local else None
    window = (None, None) if window_left is None else (window_left, 0)
    scale = 1.0 / D

    # Mirrors inkling_fa4_rel_attention exactly, including NOT passing
    # max_seqlen_k. That omission is part of what is under test: the interface
    # then hands ShearingBias num_pages * page_size as max_seqlen_k.
    out = flash_attn_varlen_func(
        q=q, k=key_cache, v=value_cache,
        rel_bias=rel_logits,
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        max_seqlen_q=max(T_q_list),
        page_table=block_table,
        softmax_scale=scale, causal=True, window_size=window,
        num_splits=num_splits,
    )
    if isinstance(out, tuple):
        out = out[0]

    per_seq = []
    worst_max, sum_mean_weighted, n_elem = 0.0, 0.0, 0
    for r, (T_q, L, ctx) in enumerate(zip(T_q_list, L_list, ctx_list)):
        qs, qe = int(cu_q[r]), int(cu_q[r + 1])
        k_r = gather_pages(key_cache, block_table[r], L, page_size)
        v_r = gather_pages(value_cache, block_table[r], L, page_size)
        ref = reference_one_seq(q[qs:qe], k_r, v_r, rel_logits[qs:qe], scale,
                               ctx, window_left)
        diff = (out[qs:qe].float() - ref.float()).abs()
        mx, mean = float(diff.max()), float(diff.mean())
        defective = reference_one_seq(q[qs:qe], k_r, v_r, rel_logits[qs:qe],
                                      scale, ctx, window_left,
                                      bias_only_oldest_tile=True)
        gap = (ref.float() - defective.float()).abs()
        sep = float(gap.mean())
        per_seq.append({"seq": r, "T_q": T_q, "ctx": ctx, "seqlen_k": L,
                        "offset_q": qs, "pages": (L + page_size - 1) // page_size,
                        "max_abs_diff": mx, "mean_abs_diff": mean,
                        "signal_mean": sep,
                        "signal_max": float(gap.max()),
                        "signal_over_tol": sep / TOL_MEAN,
                        "kv_tiles": (L + KV_TILE - 1) // KV_TILE})
        worst_max = max(worst_max, mx)
        sum_mean_weighted += mean * diff.numel()
        n_elem += diff.numel()

    _measurable = [x for x in per_seq if x["kv_tiles"] > 1]
    return {
        "max_abs_diff": worst_max,
        "mean_abs_diff": sum_mean_weighted / n_elem if n_elem else None,
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
        "shape": {"T_q": T_q_list, "ctx": ctx_list, "seqlen_k": L_list,
                  "Hq": Hq, "Hkv": Hkv, "rel_extent": rel_extent,
                  "window_left": window_left, "page_size": page_size,
                  "num_pages": num_pages, "num_splits": num_splits,
                  "page_table_is_identity": False},
    }


# role: "coverage" must be able to fail on a dropped bias, "control" need not.
# Seeds are explicit constants: abs(hash(name)) is salted by PYTHONHASHSEED, so a
# gate keyed on it cannot reproduce its own recorded numbers.
CASES = [
    # page_size 128 takes the TMA paged path (page_size == tile_n).
    ("paged128_decode_batch_global", "coverage",
     dict(T_q_list=[1, 1, 1, 1], ctx_list=[2047, 1000, 137, 4095],
          Hq=8, Hkv=1, rel_extent=1024, is_local=False, page_size=128,
          num_splits=1, seed=3001)),
    ("paged128_decode_batch_swa", "coverage",
     dict(T_q_list=[1, 1, 1, 1], ctx_list=[2047, 1000, 137, 4095],
          Hq=8, Hkv=2, rel_extent=512, is_local=True, page_size=128,
          num_splits=1, seed=3002)),
    ("paged128_chunked_mixed_global", "coverage",
     dict(T_q_list=[200, 1, 137, 64], ctx_list=[0, 900, 1408, 63],
          Hq=8, Hkv=1, rel_extent=512, is_local=False, page_size=128,
          num_splits=1, seed=3003)),
    # page_size 16 takes the non-TMA paged path, which is the branch a vLLM
    # deployment with the default attention block size actually runs.
    ("paged16_decode_batch_global", "coverage",
     dict(T_q_list=[1, 1, 1, 1], ctx_list=[2047, 1000, 137, 4095],
          Hq=8, Hkv=1, rel_extent=1024, is_local=False, page_size=16,
          num_splits=1, seed=3004)),
    ("paged16_decode_batch_swa", "coverage",
     dict(T_q_list=[1, 1, 1, 1], ctx_list=[2047, 1000, 137, 4095],
          Hq=8, Hkv=2, rel_extent=512, is_local=True, page_size=16,
          num_splits=1, seed=3005)),
    ("paged16_chunked_mixed_global", "coverage",
     dict(T_q_list=[200, 1, 137, 64], ctx_list=[0, 900, 1408, 63],
          Hq=8, Hkv=1, rel_extent=512, is_local=False, page_size=16,
          num_splits=1, seed=3006)),
    # Production head geometry through the paged path.
    ("paged16_decode_batch_hq64", "coverage",
     dict(T_q_list=[1, 1, 1], ctx_list=[4095, 2000, 129], Hq=64, Hkv=8,
          rel_extent=1024, is_local=False, page_size=16, num_splits=1,
          seed=3007)),
    # Split-KV against the ORACLE, not against num_splits=1. The serving router
    # caps splits at 1 for capability 9 today, so these are the gate that has to
    # pass before that cap is lifted, and the gate that
    # scripts/modal_e2e_bench.py::splitkv_case cannot be. ctx stays at 4095 or
    # below for signal; the gain cases cover deeper.
    ("paged128_decode_splits4_global", "coverage",
     dict(T_q_list=[1, 1], ctx_list=[4095, 2047], Hq=8, Hkv=1,
          rel_extent=1024, is_local=False, page_size=128, num_splits=4,
          seed=3008)),
    ("paged128_decode_splits8_swa", "coverage",
     dict(T_q_list=[1], ctx_list=[8191], Hq=8, Hkv=2, rel_extent=512,
          is_local=True, page_size=128, num_splits=8, seed=3009)),
    # Detector cases: same shapes, bias scaled up so a dropped or misplaced bias
    # is far outside the tolerance rather than inside it. The deep split case
    # lives here because at ctx 8191 global it would otherwise be blind.
    ("paged16_decode_batch_global_gain16", "coverage",
     dict(T_q_list=[1, 1, 1, 1], ctx_list=[2047, 1000, 137, 4095],
          Hq=8, Hkv=1, rel_extent=1024, is_local=False, page_size=16,
          num_splits=1, bias_gain=16.0, seed=3010)),
    ("paged128_decode_batch_global_gain16", "coverage",
     dict(T_q_list=[1, 1, 1, 1], ctx_list=[2047, 1000, 137, 4095],
          Hq=8, Hkv=1, rel_extent=1024, is_local=False, page_size=128,
          num_splits=1, bias_gain=16.0, seed=3011)),
    ("paged128_decode_splits8_global_gain16", "coverage",
     dict(T_q_list=[1], ctx_list=[8191], Hq=8, Hkv=1, rel_extent=1024,
          is_local=False, page_size=128, num_splits=8, bias_gain=16.0,
          seed=3012)),
]


def main() -> None:
    cc = torch.cuda.get_device_capability(0)
    device = torch.cuda.get_device_name(0)
    print(f"device: {device}, capability {cc}")
    print(f"tolerance: max <= {TOL_MAX}, mean <= {TOL_MEAN}; a coverage case "
          f"must also carry signal >= {SIGNAL_MARGIN}x TOL_MEAN")
    print("call shape mirrors inkling_fa4_rel_attention: page_table + "
          "seqused_k, no cu_seqlens_k, no max_seqlen_k")
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
                          f"seqlen_k={sq['seqlen_k']} pages={sq['pages']}: "
                          f"max={sq['max_abs_diff']:.4e} "
                          f"mean={sq['mean_abs_diff']:.4e} "
                          f"signal={sq['signal_mean']:.3e}")
        except AssertionError as exc:
            # sm_80 and sm_120 route the bias through the generic kernel, which
            # refuses a page table. That is a real limitation, reported as a skip
            # with the kernel own words rather than hidden.
            r = {"role": role, "skip": f"AssertionError: {exc}", "pass": None}
            print(f"[{name}] SKIP ({role}): AssertionError: {exc}")
        except NotImplementedError as exc:
            r = {"role": role, "skip": f"NotImplementedError: {exc}",
                 "pass": None}
            print(f"[{name}] SKIP ({role}): NotImplementedError: {exc}")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            r = {"role": role, "error": f"{type(exc).__name__}: {exc}",
                 "pass": False}
            print(f"[{name}] FAIL ({role}): {type(exc).__name__}: {exc}")
            failures += 1
        results[name] = r

    ran = [r for r in results.values() if r.get("pass") is not None]
    detectors = [r for r in ran if r.get("informative")]
    out = Path(__file__).with_name(f"parity_rel_paged_sm{cc[0]}{cc[1]}.json")
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
            "fail on a missing bias. Count the detectors, not the passes."),
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
