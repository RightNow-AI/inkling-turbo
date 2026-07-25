# U2 in-kernel shear: what was built, what was checked, what was not

Status: **written, anchors verified against the real tree, index math verified
offline over 229,773,824 score positions, NEVER EXECUTED.** No GPU was
available to this lane. Nothing in this file is a performance claim; no
performance number for this path exists on any hardware.

Deliverable: `kernels/patches/u2_inkernel_shear.py`, 13 anchored edits over 2
files, idempotent, gate `INKLING_TURBO_INKERNEL_SHEAR` **default off**.

* `vllm/third_party/tml_fa4/flash_fwd_sm90.py` (4 edits)
* `vllm/third_party/tml_fa4/interface.py` (9 edits)

Nothing else is touched. With the environment variable unset, both files
behave exactly as they do today: every new code path is inside
`if const_expr(self.inkernel_shear)` or `if inkernel_shear`, and
`inkernel_shear` is False unless the variable is set.

## What it does

`INKLING_TURBO_INKERNEL_SHEAR=1` makes `_flash_attn_fwd` pass `rel_bias`
straight through as the attention kernel's bias operand:

* no `torch.empty` of the `(total_q + 128, H, rel_extent + 256)` buffer,
* no `ShearingBias` compile and no `ShearingBias` launch,
* no `CuSeqlensToBlocksKernel` and no `CuBlocksToBatchKernel` (they exist only
  to schedule `ShearingBias`),

and `FlashAttentionForwardSm90` performs the shear itself, in the address it
forms for the bias tile.

sm_90 only. Every other architecture, and sm_90 with `U2_SM90_GENERIC=1`,
keeps the pre-sheared buffer.

## Deviation from the design note, with the evidence

`kernels/patches/u2_inkernel_shear_notes.md` specifies "perform the shear as
part of staging bias into shared memory". **There is no shared-memory staging
of bias in this kernel to change.** Two places say so directly:

* `flash_fwd_sm90.py` `_get_shared_storage_cls`: `sBias` is a `MemRange` of
  length **0**, with the comment "bias is read gmem->rmem through a
  tiled_copy_C partition; no smem stage, so no shared-memory cost".
* `kernel()`: `sBias = None` is passed into `mma()`, and `apply_rel_bias_sm90`
  reads `thr_mma_qk.partition_C(gBias_tile)` where `gBias_tile` is a **gmem**
  tile.

So the shear was applied where the bias address is formed instead. That is a
strict subset of the designed work, and it retires two of the note's four
listed costs:

| note's cost | status here |
|---|---|
| 1. "TMA is lost for the bias operand" | **void.** Bias never used TMA on this kernel. Q/K/V/O keep theirs, untouched. |
| 2. Redundant reads across n_blocks | **applies unchanged.** Still the main risk to the estimate, still the first thing to measure. |
| 3. Bank conflicts on the reversed smem write | **void.** There is no smem write. No swizzle, no padded smem stride, no extra 32 KB of shared memory, so `num_stages` does not have to move. |
| 4. It touches the bias path that took seventeen sessions | **applies.** Mitigated by not touching the pairing rule at all: see below. |

The set of gmem addresses read per score tile is the same count and nearly the
same shape as today (128 rows x 128 contiguous bf16 elements). What changes is
that the 128-element run starts one element later on each successive row, so
the tile is a diagonal band instead of a box. That costs alignment, not
volume: a 256-byte row band spans 9 32-byte sectors instead of 8, about 12
percent more sectors for the bias operand, against deleting 1355 MB of
`ShearingBias` traffic per launch.

## The mechanism

`ShearingBias` places natural column `d` of row `row` at sheared column
`col = bias_idx_right(row) - 1 - d`, with

    bias_idx_right(row) = n_idx_right(row) + rel_extent_padded
                          - 128 * n_block_max(m_block(row))

so a reader that wants key index `kv` reads sheared column
`kv + padded - 128 * n_block_max`, and the natural column it lands on is

    d = n_idx_right(row) - 1 - kv

with `n_idx_right(row) = row + 1 + (seqlen_k - seqlen_q) + window_right`
(`BlockInfo.get_n_idx_left_right`, causal or local-with-right branch, which is
the only form `ShearingBias` accepts). For element `(i, j)` of score tile
`(m_block, n_block)`:

    d(i, j) = i - j + shear_k
    shear_k = m_block * tile_m - n_block * tile_n
              + (seqlen_k - seqlen_q) + window_right

which is the plain relative distance the model means, the same quantity
`harness/parity_fa4_rel.py`'s reference calls `dist = i - j`.

The natural address of `(i, j)` is

    (row0 + i) * s_r + d(i, j) * s_c
      = [row0 * s_r + shear_k * s_c] + i * (s_r + s_c) - j * s_c

so **a cute layout of shape `(tile_m, tile_n)` and stride
`(s_r + s_c, -s_c)` over the natural buffer IS the shear**, and the bracket is
one `domain_offset` with coord `(row0, row0 - shear_k)` (that coord solves the
bracket for any strides, so the code does not assume a contiguous last
dimension). `s_r` and `s_c` are read from the tensor, so a non-contiguous
`rel_bias` still works.

### Why this does not reintroduce the seventeen-session bug

The rule that made the apply side correct is untouched: the bias tile is
handed to `thr_mma_qk.partition_C`, the same partitioner that produced
`acc_S`, and paired with `acc_S` by flat index. The patch changes only *which
tensor* is handed to `partition_C`. No coordinate arithmetic enters the
address.

Coordinates are used for one thing: the range predicate, read out of
`thr_mma_qk.partition_C(cute.make_identity_tensor(...))`, which is exactly
what `mask.apply_mask` and `apply_score_mod` already do with the same
partitioner. Coordinates never derive an address.

### Validity reduces to a loop-invariant bound

`ShearingBias` writes a real value only where `0 <= d < num_bias_vals(row)`,
and otherwise writes `-inf` for `d < 0` and `left_pad` for
`d >= num_bias_vals`, with `left_pad = -inf` when `window_size_left` is set
and `0.0` otherwise (`shearing_bias.py:88-89`; the two edge fixups at
`shearing_bias.py:463-466` pin the per-element boundary to exactly this).

    num_bias_vals(row) = n_idx_right - max(n_idx_left, n_idx_right - rel_extent)

`d < num_bias_vals` is `d < rel_extent` AND `kv >= n_idx_left`, and the second
condition collapses onto `d`:

* `n_idx_left == 0` (not local, or no `window_size_left`): vacuous.
* `n_idx_left == max(row + ctx - wl, 0)`: unclamped,
  `n_idx_right - n_idx_left == wl + wr + 1` exactly, so the condition is
  `d < wl + wr + 1`. Clamped (`row + ctx < wl`), the condition is
  `d < n_idx_right`, which is vacuous, and `d <= row + ctx + wr < wl + wr`
  there anyway, so applying the unclamped bound rejects nothing.

So for every configuration `ShearingBias` accepts,

    valid(i, j)  <=>  0 <= d < d_max
    d_max = rel_extent                          if window_size_left is None
            min(rel_extent, wl + wr + 1)        otherwise

`d_max` is computed once per work tile. Per element the cost is two integer
comparisons, and for tiles wholly inside or wholly outside the band it is
zero: the apply function branches at tile granularity into

1. `d_hi < 0`: whole tile right of the band, add `-inf`, no load;
2. `d_lo >= d_max`: whole tile past the band, add `left_pad`, no load. This is
   the tile the pre-sheared path skipped via `tile_idx < 0`;
3. `d_lo >= 0 and d_hi < d_max and tile_m <= row_limit`: every element valid,
   unpredicated loads, the same inner loop shape as today;
4. otherwise: per-element predicate.

For global 8K prefill at `rel_extent=512` that is one band-edge tile at the
diagonal, three fast tiles, one band-edge tile at distance 4, and nothing at
distance 5 or more, per m_block. For the sliding-window shape it is one
band-edge tile at each end and fast tiles between.

**Every load is provably in bounds**: a load happens only when
`0 <= d < d_max <= rel_extent` and the row is `< seqlen_q`, in both the fast
and the predicated branch. The `(T + 128)` row pad and the 256-column pad of
the old buffer are not needed, and rows past `seqlen_q`, which the pre-sheared
path read as uninitialised memory and masked afterwards, are not read at all.

## What was verified offline

`py -m py_compile` on both patched files, and the patch applied to a scratch
copy of the deployed layout (`kernels/tml_fa4_modified/*` placed at
`vllm/third_party/tml_fa4/`, which is what the deploy step produces):

```
flash_fwd_sm90.py: 4 edits applied
interface.py: 9 edits applied
total: 13 edits over 2 files
```

every anchor matched exactly once, both files compile, and a second run prints
`already applied, nothing to do` and exits 0. `vllm/` was never mutated.

Then the index math, by brute force against `ShearingBias`'s own expressions
and `BlockInfo`'s own expressions, transcribed from
`tml-fa4/flash_attn/cute/shearing_bias.py:357-476` and
`tml-fa4/flash_attn/cute/block_info.py:26-93`. 120 configurations:
`rel_extent` in {128, 512, 1024} x `(seqlen_q, seqlen_k)` in
{(200,200), (8192,8192), (1,8192), (32,8192), (128,8192), (256,300), (300,256),
(4096,4096)} x {causal; causal + window (ext-1, 0); causal + window (127, 0);
local + shifted diagonal (ext-257, 256); causal + window wider than
rel_extent}. For every `(m_block, n_block, i, j)` the kernel actually
processes, 229,773,824 positions and 82,625,350 of them loads:

* **A.** `d(i, j) = i - j + shear_k` equals `n_idx_right(row) - 1 - kv`.
  Zero mismatches.
* **B.** `0 <= d < num_bias_vals(row)` equals `0 <= d < d_max`, that is, the
  reduction to a loop-invariant bound is exact and not merely sufficient.
  Zero mismatches.
* **C.** the closed form `bias_idx_right = n_idx_right + padded - 128 *
  n_block_max` equals the writer's clamped expression, and the
  `max(..., 0)` clamp in `bias_block_idx_right` never fires in any of the 120
  configurations (0 clamps in 184,870 rows). This reconfirms, independently,
  the closed form recorded in `u2_shear_fusion_notes.md`.
* **D.** every position the patch loads satisfies `0 <= d < rel_extent`, so
  every load is inside the natural buffer.

The script is not committed (it is a throwaway; it lives in this session's
scratchpad). Re-deriving it is a 100-line exercise from the two source files
named above, and the four assertions are stated precisely enough to redo.

**What this does NOT prove:** that the code compiles under CuTe DSL, that the
band layout partitions the way the arithmetic says, or that a single element
lands where it should on hardware. It proves only that the arithmetic the code
implements is the arithmetic `ShearingBias` implements.

## The specialisation bug this uncovered, and its fix

**Status: FIXED.** `eb1e487` fixed `flash_fwd_sm90.py`; `9b63979` fixed
`flash_fwd.py`, the sm_80 / sm_120 generic kernel, where the same defect was
live. `harness/parity_rel_chunked_decode.py` is the gate that covers it and
scores 7/7 on an H100 at `e9857de`, with a deliberately broken control run
scoring 2/7 on the same gate as the artifact records it, and 1/7 on a replay
under the tightened `TOL_MEAN` of 5e-4. Write-up:
`journal/regression-sm90-bias-shift.md`. Artifacts:
`journal/remote/validate_s27_decodefix/` and
`journal/remote/validate_s27_brokencontrol/`.

This section is kept as the record of how the defect was found and how wrong it
was. It no longer describes live code.

`flash_fwd_sm90.py`'s bias reader USED to compute

    bias_tile_shift = padded // tile_n - (128 * (m_block + 1)) // tile_n

which hardcodes `n_block_max(m_block) == m_block + 1`, the
`seqlen_q == seqlen_k` specialisation that constraint 3 of this lane's brief
warns about. Measuring `n_block_max(m_block) != m_block + 1` over the same
120 configurations, at `rel_extent=512`:

| shape | rows where the old shift was wrong | worst error |
|---|---|---|
| 200 / 200 causal, 8192 / 8192 causal, 4096 / 4096 causal, all with window_right = 0 | 0 | 0 tiles |
| 300 / 256 causal | 44 of 300 | 1 tile |
| 256 / 300 causal | 256 of 256 | 1 tile |
| 128 / 8192 causal | 128 of 128 | 63 tiles |
| 32 / 8192 causal | 32 of 32 | 63 tiles |
| 1 / 8192 causal | 1 of 1 | 63 tiles |
| 8192 / 8192, shifted diagonal (window_right = 256) | 8064 of 8192 | 2 tiles |

`n_block_max == m_block + 1` holds identically when `seqlen_q == seqlen_k` and
`window_right` is 0 or absent, which was exactly the family the repository's
correctness gates covered, so this was consistent with the recorded sm_90
parity 3/3. It failed for every chunked-prefill and every decode call. At
`seqlen_q=1, seqlen_k=8192, rel_extent=512` the correct shift is
`padded/128 - 64` and the code used `padded/128 - 1`, so `tile_idx` was in
range only for `n_block == 0` and the relative bias was applied to the oldest
KV block instead of the newest, with the wrong values.

Nothing in the repository would have caught it at the time:

* `harness/parity_fa4_rel.py` has three cases, all `seqlen_q == seqlen_k`.
* `harness/microbench_attn_day0.py`'s five decode cases do run
  `seqlen_q != seqlen_k`, but they are timings and check no output.
* `scripts/gate_logit_parity.py`'s 32/32 token match calls the completions API
  with `max_tokens=0` and `echo=True`, so it compares **prompt** logprobs. It
  never decodes. The one check in it that does decode
  (`BATCH_MAX_TOKENS = 32`) compares batched against batch-1 *within* a build,
  and the README records that check as failing on the stock build too.
* `_use_sheared_bias()` after `u2_serving_route.py` returns True for
  capability 9 on **every** shape, so serving decode does route here.

The first of those holes is now closed by
`harness/parity_rel_chunked_decode.py`, seven cases of which six have
`seqlen_k > seqlen_q` and one is the `seqlen_q == seqlen_k` control.

**What this patch does about it.** The gate-off branch of the `mma` edit is a
verbatim copy of the fixed block, lifted programmatically out of
`kernels/tml_fa4_modified/flash_fwd_sm90.py` rather than retyped, so applying
this patch cannot reintroduce the defect. That matters because an earlier draft
of this patch DID reproduce the pre-fix arithmetic in that branch: the patch and
the fix landed in the same commit (`eb1e487`), so the patch was born stale, its
`mma` anchor matched 0 times, and making the anchor match without also fixing
the body would have silently undone the fix on the DEFAULT path. Two mechanical
checks guard it now, both listed under "How to check it, in order", step 0.

Both paths now use the general form, so:

* gate-on and gate-off should agree on every shape this patch's constraints
  allow, `seqlen_q == seqlen_k` and `seqlen_k > seqlen_q` alike, to within bf16
  rounding of the same values (they read the same numbers through different
  addresses);
* a disagreement on any shape is therefore a defect in one of the two paths. It
  is no longer an expected difference, which is what it was while the
  pre-sheared reader was still specialised.

## How to check it, in order

There is **no free local gate for this change.** Everything that differs is
inside `flash_fwd_sm90.py`, and `FlashAttentionForwardSm90.__init__` asserts
`sm_90 <= arch <= sm_90a`, so an RTX 5090 cannot execute it. On sm_120 the
interface gate evaluates to False and the patch is inert. The design note's
"local sm_120 parity first, because it is free" does not apply here. The first
execution of this code is on an H100. Plan the session so one boot covers
everything below.

### 0. Free, no GPU

```
python kernels/patches/u2_inkernel_shear.py $VLLM
```
Expect `flash_fwd_sm90.py: 4 edits applied`, `interface.py: 9 edits applied`,
`total: 13 edits over 2 files`. A second run prints
`already applied, nothing to do` and exits 0. Order relative to
`u2_serving_route.py` and `u3_fp8_kv.py` does not matter: they touch
`fa4_rel_attention.py` and `qkvr_prep.py`, neither of which this patch reads.
It must **not** be combined with `u2_shear_fusion.py`, which makes `qkvr_prep`
emit the sheared layout that this patch removes the need for.

Then two checks that the gate-off path still carries the `n_block_max` fix.
Both are free, and both must hold with the environment variable **unset**:

```
grep -c -- "- (128 \* n_block_max_bias) // self.tile_n" \
  $VLLM/vllm/third_party/tml_fa4/flash_fwd_sm90.py     # expect 1
grep -c -- "- (128 \* (m_block + 1)) // self.tile_n" \
  $VLLM/vllm/third_party/tml_fa4/flash_fwd_sm90.py     # expect 0
```

The second one is not paranoia. The first draft of this patch failed it, for
the reason recorded in "The specialisation bug this uncovered, and its fix".

Re-apply after every bootstrap: `interface.py` and `flash_fwd_sm90.py` live in
the deployed tree and are overwritten by the deploy step. The `mma` anchor is
the fragile one: it is the block `eb1e487` rewrote, so any further change to
the pre-sheared reader will make this patch abort with
`anchor 'mma: build the in-kernel-shear score_mod_fn' matched 0 times`. When
that happens, re-lift the block from `kernels/tml_fa4_modified/flash_fwd_sm90.py`
into BOTH `SM90_MMA_OLD` and the `else:` body of `SM90_MMA_NEW`, never just the
first.

### 1. H100, baseline first, gate OFF

```
python harness/parity_fa4_rel.py
```
Expect the unchanged 3/3 for backend `tml_fa4_rel_bias`. **Write the three
`max_diff` values down.** They are the comparison for the next step, and they
also prove the deploy is sane before the gate is blamed for anything.

### 2. H100, gate ON. This is the gate.

```
INKLING_TURBO_INKERNEL_SHEAR=1 python harness/parity_fa4_rel.py
```
Expect 3/3 again, with `max_diff` for `tml_fa4_rel_bias` **the same order of
magnitude as step 1**, not merely under `--tol`. Both runs compare against the
same fp32 reference and read the same bf16 bias values, so a real difference
in the diffs means the two paths are reading different numbers.

What each case exercises:

| case | shape | which branches |
|---|---|---|
| `global_short` | T=128, ext=1024, causal | one tile, band-edge branch only (`d` in [-127, 127]), `left_pad = 0.0` |
| `global_beyond_extent` | T=1536, ext=1024, causal | 12 m_blocks: the fast path on interior tiles, the band edge at the diagonal and at distance 8, and the no-load `d_lo >= d_max` early-out beyond that |
| `swa_512` | T=1536, ext=512, window_left=511 | local path: `left_pad = -inf`, `d_max = min(512, 511 + 0 + 1) = 512`, `n_block_min > 0` |

Reading a failure:

* **Trace-time error inside `flash_fwd_sm90.py`** (`make_layout`,
  `domain_offset`, or `partition_C` on the band): go to step 3.
* **`max_diff` around the magnitude of the bias itself** on every case: the
  column mapping is off. Suspect `shear_k`. The `+ window_right` term and the
  `seqlen_k - seqlen_q` term are the two places a sign can go wrong, and at
  `seqlen_q == seqlen_k` with `window_right = 0` both vanish, so a failure
  here means `m_block * tile_m` or `n_block * tile_n` and not the ctx terms.
* **`max_diff` equal to the difference between the reference and a no-bias
  reference**: the bias is being dropped entirely. Check that `has_bias` is
  still True, that is, that `bias = rel_bias` really happened
  (`interface.py`, the `rel_bias if inkernel_shear else torch.empty` line).
* **`swa_512` fails while both `global_*` pass**: `d_max` or `left_pad`. The
  local branch is the only one that sets `left_pad = -inf` and the only one
  that clamps `d_max`.
* **`global_short` fails while `global_beyond_extent` passes**: the band-edge
  branch. `global_short` has no fast-path tile at all.
* **`global_beyond_extent` fails while `global_short` passes**: the fast path,
  or the `d_lo >= d_max` early-out adding `left_pad` where the old path added
  nothing. Those agree only because `left_pad` is 0.0 globally.

### 3. H100, only if step 2 fails at trace time

```
INKLING_TURBO_INKERNEL_SHEAR=coord python harness/parity_fa4_rel.py
```
Same arithmetic, but the bias element is fetched as
`mBiasNat[row, d]` per element instead of through the band layout, so it
never builds a strided view and never forms a pointer outside the buffer. It
is slower by construction and exists only to split "the arithmetic is wrong"
from "the DSL will not express this layout". If `coord` passes and `1` fails,
the three lines to look at are the `cute.make_layout(..., stride=(nat_stride_row
+ nat_stride_col, -nat_stride_col))`, the `cute.domain_offset((row_abs0,
row_abs0 - shear_k), ...)` around it, and the `partition_C` on the result, all
inside `apply_rel_bias_inkernel_shear_sm90`.

### 4. H100, same session, timings

```
python harness/microbench_attn_day0.py
mv harness/microbench_attn_day0.json \
   journal/remote/microbench_attn_day0_sessionNN_h100.json
INKLING_TURBO_INKERNEL_SHEAR=1 python harness/microbench_attn_day0.py
mv harness/microbench_attn_day0.json \
   journal/remote/microbench_attn_inkernel_shear_sessionNN_h100.json
```
Run **both**, in the same session, on the same box. The session-25 numbers
came from a different file state and the regression in
`journal/regression-sm90-n-block.md` is why comparing across sessions is not
safe here.

Expect in the gate-on artifact: `ShearingBias`, `CuSeqlensToBlocksKernel` and
`CuBlocksToBatchKernel` **absent** from `kernels_us` in every case, and one
fewer kernel launch per attention call. The numbers that decide the design are
`prefill_swa_8k` total against day-0's 863 us and `prefill_global_8k` total
against day-0's 4841 us, from
`journal/remote/validate_s26b_h100x1_route/`. The number that decides whether
the note's cost 2 is real is the attention entry itself: if it grew a lot, the
redundant reads across `n_block` are not being absorbed by L2, and that is the
finding, not a failure.

Also worth reading in the same artifact: the five decode cases. They run
`seqlen_q != seqlen_k`, so they are the only thing in the repository that
executes the general form at all, even though they check no output. A crash or
an illegal memory access there is a real signal.

### 5. The `seqlen_k > seqlen_q` gate, which now exists

```
python harness/parity_rel_chunked_decode.py
INKLING_TURBO_INKERNEL_SHEAR=1 python harness/parity_rel_chunked_decode.py
```

This is the gate that did not exist when this patch was first written. Seven
cases, six with `seqlen_k > seqlen_q`, one `seqlen_q == seqlen_k` control. It
scores 7/7 gate-off at `e9857de`. **Gate-on must also be 7/7**, and this is now
the most informative single run for the in-kernel shear, because it is the only
harness that exercises the `seqlen_k - seqlen_q` term of `shear_k` against a
checked reference. At `seqlen_q == seqlen_k` that term vanishes, so
`parity_fa4_rel.py` cannot see a sign error in it and this one can.

A gate-on score below 7/7 while gate-off is 7/7 localises the defect to the
in-kernel path. Both below 7/7 means the deploy is not the fixed kernel, so
re-check step 0's two greps before blaming the gate.

## Residual risks, ranked

1. **The band layout may not survive CuTe DSL.** `cute.make_layout` with a
   dynamic stride plus a negated dynamic stride, then `partition_C` on it. The
   individual ingredients are all used in the tree (`cute.make_tensor(t.iterator,
   cute.make_layout(shape, stride=...))` in `pack_gqa.py:54` and
   `cute_dsl_utils.py:59`, `.iterator` on a slice in
   `blackwell_helpers.py:144`, `partition_C` on a `domain_offset`ed gmem
   tensor in the existing bias path), but never in this combination and never
   with a negative stride. Cheap to discover, and `=coord` isolates it.
2. **Redundant bias reads across n_block.** Adjacent `n_block`s for one
   `m_block` read overlapping distance ranges of the same rows. The design
   note calls this the main risk to the estimate and this patch does not
   change that. It is measured in step 4, not predicted here.
3. **Tile base pointer outside the buffer.** The band's origin is
   `row0 * s_r + shear_k * s_c`, and `shear_k` can be negative (bounded below
   by about `-tile_n`, so at most about 127 elements before the buffer) or
   larger than `rel_extent`. Every actual load is in bounds (proof D above),
   but the out-of-range base is *computed*. This is the same pattern every
   predicated FA tile uses, and `=coord` avoids forming it at all.
4. **False alignment claim on the natural operand.** The band's row stride is
   `s_r + s_c`, odd whenever `s_c == 1`, so pairs of adjacent columns are not
   4-byte aligned in general. The patch drops the bias operand's alignment
   claim to the element size on this path (`to_cute_tensor(..., assumed_align=2)`
   in the interface, `assume_tensor_aligned(mBias, align=element_size)` in
   `__call__`) so nothing can legally merge two 2-byte loads into a
   misaligned 4-byte one. If a load-vectorizer merge shows up in the SASS
   anyway, that is the thing to look at.
5. **Code size in the apply function.** Four traced branches, three of them
   containing a fully unrolled 64-element loop, and the fourth containing a
   3-way dynamic branch per element. If register pressure or I-cache pressure
   shows up as a slowdown with correct output, the cheapest cut is to drop
   branch 1 (`d_hi < 0`), which is unreachable for causal and local work tiles
   anyway: a tile entirely above the diagonal is never scheduled.
6. **`pack_gqa`.** Asserted off in two places (the interface gate and the
   kernel constructor) rather than assumed. `interface.py` already forces it
   off for arch 9 with `rel_bias`; if that ever changes, this fails loudly
   instead of mis-shearing.
7. **Split-KV.** `inkling_fa4_num_splits` returns 1 for capability 9, so
   serving never splits on Hopper, and `d` depends only on the absolute row
   and key index, so splitting the `n_block` range does not change it. Not
   asserted, not exercised.
8. **Deployment fragility.** Both targets are deployed copies of
   `kernels/tml_fa4_modified/`, overwritten by every bootstrap. The patch must
   be re-applied after each one, and its idempotence check will tell you
   whether it already has been.

## Things deliberately not done

* The pre-sheared reader was not changed by this patch. It no longer needs
  changing: `eb1e487` corrected `128 * (m_block + 1)` to `128 * n_block_max`
  with `absolute=True`, and this patch's gate-off branch carries that corrected
  block through verbatim. See "The specialisation bug this uncovered, and its
  fix".
* `harness/parity_fa4_rel.py` was not modified. It does not need a
  `seqlen_q != seqlen_k` case any more either;
  `harness/parity_rel_chunked_decode.py` covers that family, and step 5 runs it
  under both gate states.
* No shared-memory staging was added. See "Deviation from the design note".
* `flash_fwd.py` (the sm_80/sm_120 generic kernel) was not touched, so
  `U2_SM90_GENERIC=1` keeps working as the A/B reference and is explicitly
  excluded from the gate.
* Nothing was committed or pushed.
