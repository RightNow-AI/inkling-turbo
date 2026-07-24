# Design: shear inside the attention kernel, not before it

Status: **design, with the cost model validated against measurement.** Not
implemented. This is the successor to `u2_shear_fusion.py`, which is implemented,
correct, and measured to be a loss.

## The problem, stated as arithmetic

Our sliding-window prefill loses to day-0 `score_mod`: 1221 us against 863 us on
an H100. The whole deficit is one kernel. `ShearingBias` rewrites the
relative-bias buffer from its natural `(T, H, ext)` layout into the sheared
`(T + 128, H, ext + 256)` layout the attention kernel reads, and it costs 461 us
of that 1221.

Remove it and sliding-window prefill lands near 760 us, which beats 863. Global
prefill lands near 2480 against 4841, which is 1.95x rather than 1.46x. That is
the entire remaining gap in this project's kernel story.

## Why it cannot be optimised in place

`ShearingBias` moves, per launch:

| shape | read natural | write sheared | total | measured | achieved |
|---|---|---|---|---|---|
| prefill 8K, sliding window | 537 MB | 818 MB | 1355 MB | 460.9 us | **2.94 TB/s** |
| prefill 8K, global | 1074 MB | 1363 MB | 2437 MB | 827.2 us | **2.95 TB/s** |

H100 HBM3 peak is 3.35 TB/s, so this kernel is at **88% of roofline on both
shapes**. It is not badly written and there is no tuning left in it. A pure data
movement pass that already runs at 88% of peak can only be made cheaper by not
running it.

That also explains why `u2_shear_fusion.py` failed. It moved the same movement
into `qkvr_prep`, where the write became a per-row scatter into a padded buffer
rather than a streaming write, and the writer went from 218 us to 1252 us on the
sliding-window shape. Relocating a roofline-bound pass does not help; the pass
has to disappear.

## The design

Have the attention kernel read the **natural** buffer and perform the shear as
part of staging bias into shared memory. No padded buffer is ever materialized,
and there is no second launch.

For score tile `(m_block, n_block)` and head `h`, row `i` needs bias at causal
distances `d = i - k` for the 128 values of `k` in the tile. So row `i` needs

    rel[i, h, (i - k_end) : (i - k_start) + 1]      reversed

which is **128 contiguous elements of the natural buffer**, 256 bytes in bf16,
at a per-row offset that advances by one per row. That diagonal band is the
shear. Nothing needs to be computed that `ShearingBias` was not already
computing; it just happens in registers on the way to smem instead of in HBM.

Load contiguous from gmem, write reversed into smem. The reversal is free on the
smem side, which has no coalescing requirement, only bank conflicts to avoid.

## The traffic this saves

The attention kernel today reads roughly 805 MB of bias tiles for the
sliding-window prefill shape (64 m_blocks x about 6 n_blocks inside a 511 window
x 64 heads x 32 KB). Under this design it reads the same tile count and the same
byte count, from the natural buffer instead of the padded one. So:

    before:  ShearingBias (537 read + 818 write)  +  attention reads 805
    after:                                            attention reads ~805

The saving is the whole `ShearingBias` pass, 1355 MB, which at the measured
2.94 TB/s is the whole 461 us.

## What this costs, honestly

1. **TMA is lost for the bias operand.** A diagonal band is not a box, so the
   bias load drops from TMA to `cp.async` with a per-row address. Q, K and V
   keep TMA. This costs pipelining efficiency and adds address arithmetic, so
   the realistic saving is a large fraction of 461 us rather than all of it.
2. **Redundant reads across n_blocks.** Adjacent `n_block`s for the same
   `m_block` read overlapping distance ranges of the same rows. L2 should absorb
   most of it, but this is the main risk to the estimate and it is the thing to
   measure first.
3. **Bank conflicts on the reversed smem write** need a swizzle or a padded smem
   stride. Standard, but it must be got right or the staging becomes the new
   bottleneck.
4. It is a change to the bias load path in `flash_fwd_sm90.py`, which is the
   part of this kernel that took seventeen sessions to get right the first time.
   The `partition_C` insight that made the apply side correct still holds and is
   untouched; only the staging changes.

## Why this is worth attempting anyway

The alternative is shipping a kernel that loses one of the two prefill cases,
where that case is 55 of Inkling's 66 layers. And the upside is not marginal: it
turns 1.46x into about 1.95x on global prefill and turns a loss into a win on
sliding window, on the strength of a cost model that already predicts two
measured numbers to within 1%.

## Validation plan

`harness/parity_shear_fusion.py` already contains the right oracle. Its 14
writer cases define the sheared buffer bit-exactly, and its two
`attention_consumes_*` cases run the FA4 kernel against a buffer built by the
stock `ShearingBias`. An in-kernel shear must produce output matching the
`rel_bias=` path within that kernel's own measured run-to-run noise, on the same
cases, before any timing is reported.

Order:

1. Local `sm_120` parity first, because it is free.
2. One H100, `parity_fa4_rel` plus `microbench_attn_day0`, about $0.40. The
   number that decides it is `prefill_swa_8k` against 863 us and
   `prefill_global_8k` against 4841 us, from
   `journal/remote/validate_s26b_h100x1_route/`.
3. No claim until that run exists. The last projection in this area was wrong by
   the opposite sign, and it was wrong because it was arithmetic that nobody had
   checked on hardware.
