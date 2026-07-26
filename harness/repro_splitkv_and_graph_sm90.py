#!/usr/bin/env python3
"""One Hopper run that closes the two remaining sm_90 unknowns.

WHY THIS FILE EXISTS

Two things in this repository are unvalidated, and both of them need a GPU with
compute capability 9.0 that the author does not have. Each one on its own is too
small to justify renting an H100, so they are packed into a single script that
answers both in one boot.

  1. SPLIT-KV DECODE HAS NEVER EXECUTED CORRECTLY ON ANY HARDWARE.
     `kernels/patches/u2_splitkv_notes.md` says so in its first line. Its one
     attempt, session 26 on one H100, died before the kernel ran: every case
     came back with the `n_block` scoping defect written up in
     `journal/regression-sm90-n-block.md`, which is a trace-time
     `UnboundLocalError` in `flash_fwd_sm90.py`, not a numerical result. That
     defect is fixed. Nobody has retried. So the entire sm_90 split-KV
     mechanism, the leading-to-trailing split-mode layout transpose, the empty
     split, the negative-trip-count clamps, the bf16 partials, and the combine
     kernel reading them, is code that has been reasoned about carefully and
     never once run.

  2. CUDA GRAPH CAPTURE IS PROVEN ON THE GENERIC KERNEL AND UNTESTED ON sm_90.
     `harness/repro_cuda_graph_capture.py` passed on a 5090 (sm_120) on
     2026-07-25. sm_120 and sm_80 run `flash_fwd.py`. sm_90 runs
     `flash_fwd_sm90.py`, a different forward kernel with a different epilogue
     and a different scheduler. The `ShearingBias` pre-kernel and the
     interface-level `torch.empty` allocations are shared, so part of the
     question is already answered, but the kernel that does the attention is
     not the same kernel, and production vLLM serves with graphs on by default.
     That is the last production-blocking unknown for Hopper.

HOW IT AVOIDS INVENTING A SECOND SOURCE OF TRUTH

Nothing here defines a new oracle, new tolerances, or new tensors.

  * The float32 reference, TOL_MAX, TOL_MEAN, SIGNAL_MARGIN and the case shapes
    all come from `harness/parity_rel_chunked_decode.py`, whose tolerances were
    calibrated against a deliberately broken kernel and whose per-shape blind
    spots are documented there.
  * `draw()` and the entire CUDA graph section come from
    `harness/repro_cuda_graph_capture.py`. The graph sections are that file's
    `run_single` and `run_shared_pool` called directly, not reimplemented, so
    an sm_90 result from this script is directly comparable to the sm_120
    result already in the record instead of being a separate experiment that
    happens to have a similar name.

WHAT THE SPLIT-KV SECTION ACTUALLY DRIVES

Read out of `kernels/tml_fa4_modified/interface.py`:

  * `num_splits` is a plain caller argument on `flash_attn_varlen_func`
    (line 1577), forwarded to `_flash_attn_fwd` (line 1617). It defaults to 1.
    A value < 1 means "ask `num_splits_heuristic`" (line 611). A value > 1 is
    taken literally and sets `is_split_kv` (line 625). This file passes literal
    values, because a heuristic value depends on `multi_processor_count` and
    would make the same script exercise a different number of splits on
    different machines.
  * On sm_90 the interface then forces `disable_scheduler_metadata = True`
    (line 647), because the sm_90 kernel ignores `num_splits_dynamic_ptr`, and
    stores the partial outputs in the KERNEL's dtype rather than fp32
    (line 634). Both are automatic. The caller does not, and must not, ask.
  * On sm_90 with `rel_bias` the interface also pins `pack_gqa = False`
    (line 499) and `tile_mn = (128, 128)` (line 541). Nothing here passes
    either one, so the call is byte-identical in intent to the one the parity
    gate and the graph gate already make, plus `num_splits`.

That pinned tile is what makes the block arithmetic below exact rather than a
guess: with `tile_n = 128`, `BlockInfo.get_n_block_min_max` gives

    n_blocks_per_split = ceil((n_block_max - n_block_min) / num_splits)
    n_block_min       += split_idx * n_blocks_per_split
    n_block_max        = min(n_block_min + n_blocks_per_split, n_block_max)

so the four cases below are chosen to cover four different split geometries:

    section                                 blocks  per  non-empty  empty
    splitkv_decode_global_ctx4095_s4          32      8      4         0
    splitkv_decode_global_ctx4095_s8          32      4      8         0
    splitkv_chunked_global_128_on_1408_s5     12      3      4         1
    splitkv_decode_swa_ctx4095_s16             4      1      4        12

The last two are the ones that matter most. `_s5` does not divide evenly, and
`_s16` asks for four times more splits than there are blocks, so twelve of its
sixteen splits own an empty KV range. Empty splits are item 5 of the design
notes, the path where the consumer has to emit `lse = -inf` and `O = 0` and the
combine kernel has to drop them, and it is the path most likely to be wrong.
`_s16` is also sliding-window, which is item 6, the negative trip count clamp.

TWO NEGATIVE CONTROLS, so that a green split run means something

A parity number on its own cannot distinguish "split-KV is correct" from
"num_splits was quietly reset to 1 somewhere and we timed the ordinary kernel".
Both failure modes are covered:

  split_active     `interface._flash_attn_fwd_combine` is wrapped with a
                   counting spy for the duration of the section. That function
                   is called from exactly one place, `interface.py:1478`, under
                   `if is_split_kv:`. If it fires once, with an `out_partial`
                   whose leading extent equals the requested `num_splits`, then
                   the split path really ran. The section fails if it does not.
                   The spy also records the partial dtype, which is the one
                   place the sm_90 port deviates from upstream (bf16 instead of
                   fp32 partials).

  coverage_signal  the score a kernel that silently LOST one split's worth of
                   KV would post against the oracle. It is computed from the
                   reference alone by truncating K and V to
                   `T_k - n_blocks_per_split * 128` keys, which for every case
                   here is exactly the key range the last split owns. If that
                   number is not at least SIGNAL_MARGIN times TOL_MEAN, a
                   dropped split could not have failed the gate and the section
                   reports itself uninformative rather than passing.

  bias_signal      the same `mean|ref - ref_nobias|` power check the parity gate
                   uses, carried over unchanged, so the shear reader is still
                   under test inside the split.

Two further numbers are recorded but do not gate: `nosplit_vs_ref`, the same
call at `num_splits=1` in the same process, which says whether a split failure
is attributable to splitting or to something wrong with the shape generally,
and `split_vs_nosplit`, which isolates the split machinery from the base
kernel's own error and from the bf16-partial precision cost.

PRE-FLIGHT, MEASURED ON THE LOCAL 5090 (sm_120) ON 2026-07-26

The two power numbers come out of the oracle, not out of the kernel, so they can
be measured anywhere. They were, before any Hopper time was booked, so that
nobody rents an H100 for a gate that turns out to be unable to fail:

  section                                bias drop   losing one split's KV
  splitkv_decode_global_ctx4095_s4       7.5x        16.7x
  splitkv_decode_global_ctx4095_s8       7.5x        11.4x
  splitkv_chunked_global_128_on_1408_s5  17.9x       24.5x
  splitkv_decode_swa_ctx4095_s16         32.5x       46.3x

all as multiples of TOL_MEAN, against a required SIGNAL_MARGIN of 4.0x. The
narrowest is `_s8` at 11.4x, which is the expected shape of it: at eight splits
the last one owns only 512 of 4096 keys. Every case can fail.

The same run recorded `num_splits=1` parity on all four shapes at mean 2.2e-05
to 6.3e-05 against TOL_MEAN 5e-04, so the shapes themselves are healthy on the
generic kernel and a split failure on Hopper will not be confounded by a shape
that was already marginal. Those are sm_120 numbers from `flash_fwd.py` and say
nothing about `flash_fwd_sm90.py`.

TOLERANCES ARE THE IMPORTED ONES, DELIBERATELY UNCHANGED

The design notes predict roughly one extra bf16 rounding from the bf16 partials,
that is on the order of 2x the non-split output error. The sm_120 and H100
non-split decode cases sit at mean 2.2e-05 to 2.9e-05 against TOL_MEAN 5e-04
(table in `parity_rel_chunked_decode.py`), so 2x still clears the tolerance by
more than an order of magnitude. If a split case fails on `mean` by a factor
near 2 and passes `split_vs_nosplit`, suspect the bf16 partials and read the
"Deviation from upstream" section of `kernels/patches/u2_splitkv_notes.md`
before touching the tolerance. Do not touch the tolerance.

THE ADVISORY SECTION

`graph_splitkv_decode_global_ctx4095_s4` captures a split-KV call in a CUDA
graph, which is the combination production would actually use if split-KV were
ever turned on for serving. It is marked advisory and does NOT gate the exit
code, because neither gap requires it and a failure there is a third finding,
not a regression in either of the two things this script exists to settle. It
costs one extra subprocess and answers a question that would otherwise need a
second rental.

ARCHITECTURE GATING

sm_90 conclusions are drawn only on compute capability 9.0. On anything else the
script still runs, so it can be smoke-tested for free, and says in the banner, in
every section verdict, in the JSON, and in the last line that the result is not
an sm_90 result. `--require-sm90` turns "not on Hopper" into an immediate
non-zero exit, which is what a remote runner should pass so that a mis-scheduled
instance fails loudly instead of producing a green log about the wrong GPU.

On sm_120 and sm_80 the split call is refused by the interface itself
(`interface.py:1305` and `interface.py:1181`, both "SplitKV not supported"). That
refusal is recorded as a SKIP, not a failure, and only for the exact documented
assertion. Everything the section computed before the refusal, the two power
signals and the non-split parity, is still recorded, so a local run does
usefully pre-flight the gate's power before any money is spent.

WHAT ONE H100 RUN OF THIS WILL AND WILL NOT ESTABLISH

Will: that sm_90 split-KV produces the right answer at four split geometries
including empty splits and a non-dividing split count, with proof that the split
path executed; and that the sm_90 relative-bias forward can be captured and
replayed in a CUDA graph, including with a mutated bias buffer and in a shared
graph pool, at four shapes.

Will not: anything about performance, which is `scripts/modal_e2e_bench.py`'s
job; anything about multi-sequence varlen batches, since every case here is one
sequence, matching the file the shapes come from; anything about paged KV,
`pack_gqa=True`, `learnable_sink`, or fp8 KV under split, all of which are either
asserted off or untested; and anything about global decode deeper than
ctx = 4095, where `parity_rel_chunked_decode.py` shows this oracle goes blind.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parity_rel_chunked_decode import (  # noqa: E402
    CASES as PARITY_CASES,
    D,
    DEV,
    SIGNAL_MARGIN,
    TOL_MAX,
    TOL_MEAN,
    reference,
)
import repro_cuda_graph_capture as rcg  # noqa: E402

CASE_KW = dict(PARITY_CASES)

# Pinned by interface.py:541 for sm_90 + rel_bias ("bias tiles must never be
# partial"), and by the sm_90 bias path forcing pack_gqa=False at line 499. Used
# only to PREDICT the split geometry for the report and to derive the
# coverage-loss control; nothing here passes tile_mn to the kernel.
TILE_M = 128
TILE_N = 128

# (parity case name, num_splits). Geometry for each is in the module docstring.
SPLIT_SPECS = {
    "splitkv_decode_global_ctx4095_s4": ("decode_global_ctx4095", 4),
    "splitkv_decode_global_ctx4095_s8": ("decode_global_ctx4095", 8),
    "splitkv_chunked_global_128_on_1408_s5": ("chunked_global_128_on_1408", 5),
    "splitkv_decode_swa_ctx4095_s16": ("decode_swa_ctx4095", 16),
}

# Graph sections are repro_cuda_graph_capture's own, run by that file's code.
GRAPH_SECTIONS = list(rcg.GRAPH_CASES) + ["shared_pool_multishape"]

# Advisory only, does not gate the exit code. See the docstring.
ADVISORY_SPECS = {
    "graph_splitkv_decode_global_ctx4095_s4": ("decode_global_ctx4095", 4),
}

SPLIT_SECTIONS = list(SPLIT_SPECS)
ADVISORY_SECTIONS = list(ADVISORY_SPECS)
# Split first: it is the thing that has never executed, so if a run dies early
# the money still bought the new information.
SECTIONS = SPLIT_SECTIONS + GRAPH_SECTIONS + ADVISORY_SECTIONS

MARK = "@@RESULT@@"
REFUSAL_TEXT = "SplitKV not supported"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def is_splitkv_refusal(exc) -> bool:
    """True only for the interface's own documented "no split here" assertion.

    interface.py:1181 (generic / sm_80 path) and interface.py:1305 (sm_120)
    both raise AssertionError with this text. Any OTHER AssertionError is a
    real failure and must not be swallowed as a skip.
    """
    return isinstance(exc, AssertionError) and REFUSAL_TEXT in str(exc)


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def n_block_geometry(T_q, T_k, window_left, num_splits,
                     tile_m=TILE_M, tile_n=TILE_N):
    """Mirror of BlockInfo.get_n_block_min_max for a single m block.

    Every split case here has T_q <= tile_m, so m_block is 0 and the whole
    request is one m block. Asserted rather than assumed, because the formula
    below silently becomes wrong for a multi-block query.

    Returns the predicted geometry AND the number of newest keys that the last
    split owns, which is what the coverage-loss control removes.
    """
    assert T_q <= tile_m, (
        f"n_block_geometry assumes a single m block, got T_q={T_q} > {tile_m}")
    # causal, or local with window_size_right == 0: both take the same branch.
    n_block_max = min(ceil_div(T_k, tile_n),
                      ceil_div(tile_m + T_k - T_q, tile_n))
    if window_left is None:
        n_block_min = 0
    else:
        n_block_min = max((T_k - T_q - window_left) // tile_n, 0)
    n_blocks = n_block_max - n_block_min
    per = ceil_div(n_blocks, num_splits) if n_blocks > 0 else 0
    non_empty = ceil_div(n_blocks, per) if per > 0 else 0
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "n_block_min": n_block_min,
        "n_block_max": n_block_max,
        "n_blocks": n_blocks,
        "n_blocks_per_split": per,
        "non_empty_splits": non_empty,
        "empty_splits": num_splits - non_empty,
        "keys_owned_by_last_split": per * tile_n,
    }


def install_combine_observer():
    """Count calls to interface._flash_attn_fwd_combine.

    That function is invoked from exactly one site, interface.py:1478, guarded
    by `if is_split_kv:`. Counting it is therefore a direct observation that
    the split path executed, which no output comparison can provide.

    THE ATTRIBUTE COPY IS LOAD BEARING. The original body looks up its own JIT
    cache through the module global (`_flash_attn_fwd_combine.compile_cache`,
    interface.py:1819), so once the global points at the wrapper, the original
    reads the WRAPPER's attributes. Without carrying them across, the first
    split call dies with AttributeError inside code that has nothing to do with
    this harness. The cache dict object is shared, not copied, so compiled
    kernels are still cached normally.
    """
    import vllm.third_party.tml_fa4.interface as iface

    obs = {"combine_calls": 0, "partial_shape": None, "partial_dtype": None}
    orig = iface._flash_attn_fwd_combine

    def spy(out_partial, *args, **kwargs):
        obs["combine_calls"] += 1
        obs["partial_shape"] = list(out_partial.shape)
        obs["partial_dtype"] = str(out_partial.dtype)
        return orig(out_partial, *args, **kwargs)

    for attr, value in vars(orig).items():
        setattr(spy, attr, value)
    iface._flash_attn_fwd_combine = spy
    obs["observer_carries_compile_cache"] = hasattr(
        iface._flash_attn_fwd_combine, "compile_cache")
    return obs


def diff_stats(a, b):
    d = (a.float() - b.float()).abs()
    return float(d.max()), float(d.mean())


# --------------------------------------------------------------------------
# section: split-KV parity against the imported float32 oracle
# --------------------------------------------------------------------------

def run_split(section):
    case_name, num_splits = SPLIT_SPECS[section]
    kw = CASE_KW[case_name]
    T_q, ctx = kw["T_q"], kw["ctx"]
    T_k = ctx + T_q
    Hq, Hkv = kw["Hq"], kw["Hkv"]
    rel_extent = kw["rel_extent"]
    window_left = kw["window_left"]
    seed = kw["seed"]
    scale = 1.0 / D
    window = (None, None) if window_left is None else (window_left, 0)

    geom = n_block_geometry(T_q, T_k, window_left, num_splits)
    r = {
        "section": section,
        "section_kind": "split",
        "gating": True,
        "parity_case": case_name,
        "num_splits": num_splits,
        "geometry": geom,
        "shape": {"name": case_name, "T_q": T_q, "ctx": ctx, "T_k": T_k,
                  "Hq": Hq, "Hkv": Hkv, "rel_extent": rel_extent,
                  "window_left": window_left, "seed": seed},
    }

    # Identical draw to parity_rel_chunked_decode and repro_cuda_graph_capture
    # for the same case name, because it IS their function.
    q, k, v, rel = rcg.draw(T_q, ctx, Hq, Hkv, rel_extent, seed)
    cu_q = torch.tensor([0, T_q], dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0, T_k], dtype=torch.int32, device=DEV)

    ref = reference(q, k, v, rel, scale, ctx, window_left)
    ref_nobias = reference(q, k, v, rel, scale, ctx, window_left,
                           with_bias=False)

    # Power check 1: would a kernel that dropped the relative bias fail?
    bias_max, bias_mean = diff_stats(ref, ref_nobias)
    r["bias_signal_mean"] = bias_mean
    r["bias_signal_over_tol"] = bias_mean / TOL_MEAN

    # Power check 2: would a kernel that lost the LAST split's KV range fail?
    # The last split owns the newest keys_owned_by_last_split keys, so
    # truncating K and V to a prefix reproduces that loss exactly. That is only
    # true while the block grid ends exactly at T_k, which every case here does
    # (4096 = 32 * 128, 1536 = 12 * 128). Asserted so that a case added later
    # with a ragged tail cannot quietly turn this control into a different
    # experiment.
    assert geom["n_block_max"] * TILE_N == T_k, (
        f"coverage control assumes the kv block grid ends at T_k, but "
        f"n_block_max={geom['n_block_max']} * {TILE_N} != T_k={T_k}")
    keep = T_k - geom["keys_owned_by_last_split"]
    r["coverage_control_keys_kept"] = keep
    if keep <= 0:
        r["coverage_signal_mean"] = None
        r["coverage_signal_over_tol"] = 0.0
        r["coverage_control_valid"] = False
    else:
        ref_lost = reference(q, k[:keep], v[:keep], rel, scale, ctx,
                             window_left)
        finite = bool(torch.isfinite(ref_lost).all().item())
        cov_max, cov_mean = diff_stats(ref, ref_lost)
        r["coverage_signal_mean"] = cov_mean
        r["coverage_signal_over_tol"] = cov_mean / TOL_MEAN
        # A truncation that leaves some query row with no attendable key would
        # make the control NaN and its "signal" meaningless.
        r["coverage_control_valid"] = finite

    r["informative"] = bool(
        r["bias_signal_over_tol"] >= SIGNAL_MARGIN
        and r["coverage_control_valid"]
        and r["coverage_signal_over_tol"] >= SIGNAL_MARGIN)

    obs = install_combine_observer()
    r["observer_carries_compile_cache"] = obs["observer_carries_compile_cache"]

    def call(splits):
        from vllm.third_party.tml_fa4 import flash_attn_varlen_func
        out = flash_attn_varlen_func(
            q=q, k=k, v=v,
            rel_bias=rel,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=T_q, max_seqlen_k=T_k,
            softmax_scale=scale, causal=True, window_size=window,
            num_splits=splits,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out

    # Baseline first. On a non-sm_90 GPU this still runs, so the local smoke
    # test verifies the shape, the oracle and both power signals for free.
    nosplit = call(1).clone()
    torch.cuda.synchronize()
    r["combine_calls_after_nosplit"] = obs["combine_calls"]
    ns_max, ns_mean = diff_stats(nosplit, ref)
    r["nosplit_vs_ref_max"] = ns_max
    r["nosplit_vs_ref_mean"] = ns_mean
    r["nosplit_matches_ref"] = bool(ns_max <= TOL_MAX and ns_mean <= TOL_MEAN)

    try:
        out = call(num_splits).clone()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        if is_splitkv_refusal(exc):
            r["status"] = "skip"
            r["pass"] = None
            r["split_refused"] = True
            r["split_refusal_text"] = str(exc)
            return r
        r["status"] = "fail"
        r["pass"] = False
        r["split_error"] = f"{type(exc).__name__}: {exc}"
        r["split_traceback"] = traceback.format_exc()
        return r

    r["split_refused"] = False
    r["combine_calls"] = obs["combine_calls"]
    r["partial_shape"] = obs["partial_shape"]
    r["partial_dtype"] = obs["partial_dtype"]
    # Exactly one combine, for the split call and not the baseline, over a
    # partial tensor whose leading extent is the num_splits we asked for.
    r["split_active"] = bool(
        r["combine_calls_after_nosplit"] == 0
        and obs["combine_calls"] == 1
        and obs["partial_shape"] is not None
        and obs["partial_shape"][0] == num_splits)

    r["output_finite"] = bool(torch.isfinite(out).all().item())

    sp_max, sp_mean = diff_stats(out, ref)
    r["split_vs_ref_max"] = sp_max
    r["split_vs_ref_mean"] = sp_mean
    r["split_matches_ref"] = bool(sp_max <= TOL_MAX and sp_mean <= TOL_MEAN)

    d_max, d_mean = diff_stats(out, nosplit)
    r["split_vs_nosplit_max"] = d_max
    r["split_vs_nosplit_mean"] = d_mean

    r["pass"] = bool(
        r["informative"]
        and r["split_active"]
        and r["output_finite"]
        and r["nosplit_matches_ref"]
        and r["split_matches_ref"])
    r["status"] = "pass" if r["pass"] else "fail"
    return r


# --------------------------------------------------------------------------
# section: CUDA graph capture, run by repro_cuda_graph_capture's own code
# --------------------------------------------------------------------------

def run_graph(section):
    if section == "shared_pool_multishape":
        r = rcg.run_shared_pool()
    else:
        r = rcg.run_single(section)
    # `kind` is left exactly as repro_cuda_graph_capture set it ("single" or
    # "shared_pool"). Its own describe() branches on that value, and this file
    # reuses that describe(), so overwriting it would silently render a
    # shared-pool result with the single-case formatter.
    r["section_kind"] = "graph"
    r["gating"] = True
    r["status"] = "pass" if r.get("pass") else "fail"
    return r


class SplitCase(rcg.Case):
    """repro_cuda_graph_capture.Case with num_splits on the call.

    Only the call changes. Buffers, draw, reference and redraw are inherited,
    so a split graph result is comparable to the non-split graph result for the
    same case name.
    """

    def __init__(self, name, num_splits):
        super().__init__(name)
        self.num_splits = num_splits

    def call(self):
        from vllm.third_party.tml_fa4 import flash_attn_varlen_func
        out = flash_attn_varlen_func(
            q=self.q, k=self.k, v=self.v,
            rel_bias=self.rel,
            cu_seqlens_q=self.cu_q, cu_seqlens_k=self.cu_k,
            max_seqlen_q=self.T_q, max_seqlen_k=self.T_k,
            softmax_scale=self.scale, causal=True, window_size=self.window,
            num_splits=self.num_splits,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out

    def shape_dict(self):
        d = super().shape_dict()
        d["num_splits"] = self.num_splits
        return d


def run_graph_split(section):
    """Advisory: capture a split-KV call.

    `rcg.run_single` builds its case with a module-global `Case(name)` lookup,
    so rebinding that global for the duration is what lets the already-verified
    capture, replay, stability and mutated-bias logic be reused verbatim
    instead of copied. Restored in a finally, though the section owns its own
    process anyway.
    """
    case_name, num_splits = ADVISORY_SPECS[section]
    r = {"section": section, "section_kind": "graph_split", "gating": False,
         "advisory": True, "parity_case": case_name,
         "num_splits": num_splits}
    saved = rcg.Case
    rcg.Case = lambda name, _ns=num_splits: SplitCase(name, _ns)
    try:
        inner = rcg.run_single(case_name)
    except Exception as exc:  # noqa: BLE001
        if is_splitkv_refusal(exc):
            r["status"] = "skip"
            r["pass"] = None
            r["split_refused"] = True
            r["split_refusal_text"] = str(exc)
            return r
        r["status"] = "fail"
        r["pass"] = False
        r["harness_error"] = f"{type(exc).__name__}: {exc}"
        r["traceback"] = traceback.format_exc()
        return r
    finally:
        rcg.Case = saved
    inner.update(r)
    inner["status"] = "pass" if inner.get("pass") else "fail"
    return inner


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def describe_split(r):
    name = r["section"]
    g = r["geometry"]
    head = (f"[{name}] num_splits={r['num_splits']} on {r['parity_case']} "
            f"(T_q={r['shape']['T_q']}, T_k={r['shape']['T_k']}, "
            f"window_left={r['shape']['window_left']})")
    lines = [head]
    lines.append(
        f"    predicted geometry: {g['n_blocks']} kv blocks of {g['tile_n']}, "
        f"{g['n_blocks_per_split']} per split, "
        f"{g['non_empty_splits']} non-empty, {g['empty_splits']} EMPTY")
    lines.append(
        f"    power: bias drop would score mean={r['bias_signal_mean']:.3e} "
        f"({r['bias_signal_over_tol']:.1f}x TOL_MEAN); losing one split's KV "
        f"would score mean="
        + (f"{r['coverage_signal_mean']:.3e} "
           f"({r['coverage_signal_over_tol']:.1f}x TOL_MEAN)"
           if r.get("coverage_signal_mean") is not None else "INVALID"))
    if not r.get("informative"):
        lines.append(
            "    <- UNINFORMATIVE: this case could not have failed. "
            "Change the case, never the tolerance.")
    lines.append(
        f"    num_splits=1 vs oracle: max={r['nosplit_vs_ref_max']:.4e} "
        f"mean={r['nosplit_vs_ref_mean']:.4e} "
        f"{'ok' if r['nosplit_matches_ref'] else 'BAD'}")

    if r.get("status") == "skip":
        lines.insert(1, "    SKIP: this GPU's interface refuses SplitKV: "
                        f"{r['split_refusal_text']}")
        lines.append("    (the two power numbers and the num_splits=1 parity "
                     "above are real; nothing about split-KV was tested)")
        return "\n".join(lines)

    if r.get("split_error"):
        lines.insert(1, f"    FAIL: the split call raised: {r['split_error']}")
        return "\n".join(lines)

    lines.append(
        f"    split executed: combine called {r['combine_calls']}x on "
        f"out_partial{r['partial_shape']} dtype={r['partial_dtype']} "
        f"{'ok' if r['split_active'] else 'NO: THE SPLIT PATH DID NOT RUN'}")
    lines.append(
        f"    split vs oracle:  max={r['split_vs_ref_max']:.4e} "
        f"mean={r['split_vs_ref_mean']:.4e} "
        f"{'ok' if r['split_matches_ref'] else 'BAD'}")
    lines.append(
        f"    split vs nosplit: max={r['split_vs_nosplit_max']:.4e} "
        f"mean={r['split_vs_nosplit_mean']:.4e}  (not gating; isolates the "
        f"split machinery and the bf16-partial cost)")
    lines.insert(1, f"    {'OK' if r['pass'] else 'FAIL'}")
    return "\n".join(lines)


def describe(r):
    """Dispatch on the SECTION NAME, not on any field the payload carries.

    A subprocess that died before printing its marker line, or that fell into
    child()'s catch-all, produces a payload with almost nothing in it. Handling
    that first means no formatter ever indexes a key that a crashed section
    could not have written.
    """
    name = r.get("section", "?")
    if r.get("harness_error"):
        return f"[{name}] FAIL: harness error: {r['harness_error']}"
    if name in ADVISORY_SPECS:
        if r.get("status") == "skip":
            return (f"[{name}] SKIP (advisory, not gating): "
                    f"{r.get('split_refusal_text', '')}")
        return "ADVISORY, NOT GATING\n" + rcg.describe(r)
    if name in SPLIT_SPECS:
        return describe_split(r)
    return rcg.describe(r)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run_section(name):
    if name in SPLIT_SPECS:
        return run_split(name)
    if name in ADVISORY_SPECS:
        return run_graph_split(name)
    return run_graph(name)


def child(name):
    try:
        r = run_section(name)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        r = {"section": name, "section_kind": "unknown",
             "gating": name not in ADVISORY_SPECS,
             "status": "fail", "pass": False,
             "harness_error": f"{type(exc).__name__}: {exc}"}
    cc = torch.cuda.get_device_capability(0)
    r["compute_capability"] = f"{cc[0]}.{cc[1]}"
    r["device"] = torch.cuda.get_device_name(0)
    r["sm90_authoritative"] = bool(cc == (9, 0))
    # flush=True is not cosmetic. stdout to a pipe is block buffered, and a
    # section that has just recorded a CUDA fault can abort during interpreter
    # or CUDA teardown, taking an unflushed buffer with it. The parent would
    # then see "subprocess produced no result" instead of the finding the child
    # had already written down.
    print(MARK + json.dumps(r), flush=True)
    sys.stdout.flush()
    return 0 if r.get("status") in ("pass", "skip") else 1


def parent(sections, require_sm90):
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device. This harness needs a GPU.")
        raise SystemExit(2)

    cc = torch.cuda.get_device_capability(0)
    authoritative = cc == (9, 0)
    kernel_path = "flash_fwd_sm90.py" if authoritative else "flash_fwd.py"

    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"capability {cc[0]}.{cc[1]}")
    print(f"forward kernel exercised: {kernel_path}")
    print(f"tolerance vs oracle: max <= {TOL_MAX}, mean <= {TOL_MEAN}, "
          f"signal margin {SIGNAL_MARGIN}x "
          f"(all imported from parity_rel_chunked_decode)")
    print()
    if authoritative:
        print("ARCH GATE: compute capability 9.0. Results below ARE sm_90 "
              "results.")
    else:
        print("=" * 74)
        print("ARCH GATE: this is NOT compute capability 9.0.")
        print("NOTHING printed below is an sm_90 result. The split-KV sections")
        print("will be refused by the interface itself and reported as SKIP.")
        print("The CUDA graph sections exercise the GENERIC kernel")
        print("(flash_fwd.py), which is a different kernel from the one the")
        print("open question is about. A green run here is a smoke test of")
        print("this harness and of the generic path, and nothing more.")
        print("=" * 74)
        if require_sm90:
            print()
            print("--require-sm90 was passed and this is not an H100-class "
                  "device. Refusing to run.")
            raise SystemExit(2)
    print()

    if os.environ.get("U2_SM90_GENERIC") == "1":
        msg = ("U2_SM90_GENERIC=1 is set. On sm_90 that routes the relative-"
               "bias call through the generic kernel, where SplitKV is "
               "asserted off (interface.py:1181), so the split sections could "
               "not test anything. Unset it and rerun.")
        if authoritative:
            print(f"REFUSING TO RUN: {msg}")
            raise SystemExit(2)
        print(f"NOTE: {msg}")
        print()

    print("each section runs in its own subprocess: an illegal address or a "
          "stuck capture poisons a CUDA context for the rest of the process "
          "(journal/regression-sm120-varlen-illegal-address.md)")
    print()

    results = {}
    failures = 0
    skipped = 0
    advisory_failures = 0
    for name in sections:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--run", name],
            capture_output=True, text=True)
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith(MARK):
                payload = json.loads(line[len(MARK):])
        if payload is None:
            tail = proc.stderr.strip().splitlines()
            payload = {
                "section": name,
                "section_kind": "unknown",
                "gating": name not in ADVISORY_SPECS,
                "status": "fail",
                "pass": False,
                "harness_error": (
                    f"subprocess produced no result (exit {proc.returncode}); "
                    f"last stderr: {tail[-1] if tail else 'none'}"),
                "stderr_tail": "\n".join(tail[-40:]),
            }
        results[name] = payload
        print(describe(payload))
        print()
        status = payload.get("status")
        if status == "skip":
            skipped += 1
        elif status != "pass":
            if payload.get("gating", True):
                failures += 1
            else:
                advisory_failures += 1

    out = Path(__file__).with_name(
        f"{Path(__file__).stem}_sm{cc[0]}{cc[1]}.json")
    out.write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "sm90_authoritative": authoritative,
        "kernel_path": kernel_path,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tol_max": TOL_MAX,
        "tol_mean": TOL_MEAN,
        "signal_margin": SIGNAL_MARGIN,
        "tile_m_assumed": TILE_M,
        "tile_n_assumed": TILE_N,
        "u2_sm90_generic_env": os.environ.get("U2_SM90_GENERIC"),
        "sections": results,
        "gating_sections": [s for s in sections if s not in ADVISORY_SPECS],
        "advisory_sections": [s for s in sections if s in ADVISORY_SPECS],
        "gating_failures": failures,
        "advisory_failures": advisory_failures,
        "skipped": skipped,
        "total": len(sections),
        "scope": (
            "sm_90 split-KV and sm_90 CUDA graph capture. Valid for compute "
            f"capability {cc[0]}.{cc[1]} only."
            if authoritative else
            f"NOT an sm_90 result. Ran on compute capability {cc[0]}.{cc[1]} "
            f"({kernel_path}). The split-KV sections were refused by the "
            f"interface and the graph sections exercised the generic kernel."),
    }, indent=2), encoding="utf-8")

    split_done = [n for n in sections
                  if n in SPLIT_SPECS
                  and results[n].get("status") == "pass"]
    graph_done = [n for n in sections
                  if n in GRAPH_SECTIONS
                  and results[n].get("status") == "pass"]
    print(f"gating sections: {len(sections) - len(ADVISORY_SECTIONS)}, "
          f"failures {failures}, skipped {skipped}")
    if advisory_failures:
        print(f"advisory failures: {advisory_failures} (do not affect the "
              f"exit code)")
    print(f"split-KV sections that passed: {len(split_done)}/"
          f"{len(SPLIT_SECTIONS)}")
    print(f"CUDA graph sections that passed: {len(graph_done)}/"
          f"{len(GRAPH_SECTIONS)}")
    print(f"saved: {out}")
    print()
    if authoritative:
        print("SCOPE: compute capability 9.0, flash_fwd_sm90.py. This run "
              "covers sm_90 split-KV decode and sm_90 CUDA graph capture.")
    else:
        print(f"SCOPE: compute capability {cc[0]}.{cc[1]}, {kernel_path}. "
              "THIS IS NOT AN sm_90 RESULT. Split-KV was never executed and "
              "the graph sections exercised the generic kernel. Neither open "
              "question is answered by this run.")
    print("PASS" if failures == 0 else "FAIL")
    raise SystemExit(1 if failures else 0)


def main():
    ap = argparse.ArgumentParser(
        description="sm_90 split-KV parity and sm_90 CUDA graph capture, in "
                    "one run.")
    ap.add_argument("--run", default=None,
                    help="internal: run one section in this process")
    ap.add_argument("--list", action="store_true",
                    help="print section names and exit")
    ap.add_argument("--only", default=None,
                    help="comma separated subset of section names")
    ap.add_argument("--require-sm90", action="store_true",
                    help="exit non-zero immediately unless this is compute "
                         "capability 9.0. Pass this on a rented H100 so a "
                         "mis-scheduled instance fails loudly.")
    args = ap.parse_args()

    if args.list:
        for s in SECTIONS:
            tag = "advisory" if s in ADVISORY_SPECS else "gating"
            print(f"{s}\t{tag}")
        return
    if args.run:
        raise SystemExit(child(args.run))

    sections = SECTIONS
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in SECTIONS]
        if unknown:
            raise SystemExit(f"unknown section(s): {unknown}. "
                             f"Use --list.")
        sections = [s for s in SECTIONS if s in wanted]
    parent(sections, args.require_sm90)


if __name__ == "__main__":
    main()
