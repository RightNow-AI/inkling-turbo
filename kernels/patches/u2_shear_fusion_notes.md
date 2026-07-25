# U2 shear fusion: folding ShearingBias into qkvr_prep

> **Correction banner added 2026-07-25. This document is the design record and it
> is now out of date in two ways that matter, so read this first and then read the
> rest as history.**
>
> **1. The central projection was measured and it is refuted.** The headline below
> says the fusion should turn the sliding-window prefill loss into a win, from
> 1223.0 us to roughly 759.4 us. Session 26 measured it on an H100 and the fused
> writer is a **net loss on both prefill shapes**: global prefill 1312.1 to 2336.1,
> so +1019.4 us, and sliding-window prefill 685.9 to 1251.6, so +561.1 us. It
> saves 4.7 us on batch-32 decode. The writer has to emit `rel_extent + 256`
> columns into a larger buffer, and that costs more than the `ShearingBias` launch
> it removes. The feature ships **off** by default. The projection is left in place
> below rather than deleted, per the rule in
> [docs/METHODOLOGY.md](../../docs/METHODOLOGY.md#failure-records), and it is
> recorded as a correction in [LEDGER.md](../../LEDGER.md). Measured numbers:
> `journal/remote/validate_s26_h100x1/`.
>
> **2. Three ratios in the projection table are withdrawn, for a separate
> reason.** The `today vs score_mod` column carries 2.45x, 2.80x and 2.73x on the
> three decode rows, and the `projected vs score_mod` column carries 2.66x, 2.83x
> and 2.74x. Every one of those divides a kernel that was applying the
> relative-position bias to one KV block instead of ten by a baseline that gathered
> every score. **The decode ratios in both columns are withdrawn**, along with every
> other pre-fix decode ratio in this repository. The absolute timings stand as
> timings. The two prefill rows are unaffected, because at `seqlen_q == seqlen_k`
> the defective shift is an identity.
> [Full account](../../journal/regression-sm90-bias-shift.md).
>
> **3. The `sm_90` status line below is stale.** The gate has since run on Hopper
> and scored **14/16**, not "never run": all 14 writer cases are bit-exact and both
> attention cases failed on an unbound `n_block` in `flash_fwd_sm90.py`, a defect
> since fixed with the gate not re-run. The `sm_120` 16/16 remains
> **journal-only**; its only record is commit `7375849`.

Status, as originally written and kept for the record: written, compiles, index
math verified by simulation, and **correctness-gated on `sm_120` only**:
`harness/parity_shear_fusion.py` returned 16/16 cases on an RTX 5090. It has
**never run on `sm_90` or `sm_100`**, and **no performance number below has been
measured on any hardware**. Every performance number below is projected and
labelled as such. Points 1 to 3 above supersede the second and third sentences of
that paragraph.

The gate now writes `harness/parity_shear_fusion_sm<cc>.json`. The RTX 5090 run
predates that change, so the JSON artifact for it is being regenerated and is
not in the repository yet.

## Verdict

Feasible. The shear is a per-row permutation whose only row-dependent quantity
is a single integer offset, and that offset is computable from data qkvr_prep
already has (plus one small per-request tensor). A token-indexed kernel can
emit the sheared layout in one pass, with no second pass and no extra
synchronisation.

## The measured problem

Session 25, one H100 SXM5, torch 2.11/cu130, us per iter.
Source: `journal/remote/microbench_attn_day0_session25_h100.json` (ours),
`journal/remote/microbench_attn_scoremod_session25_h100.json` (day-0).

| workload | total | attention | ShearingBias | scheduler | score_mod |
|---|---|---|---|---|---|
| prefill_global_8k | 3308.8 | 2479.0 | 827.2 | 2.6 | 4799.4 |
| prefill_swa_8k | 1223.0 | 759.4 | 460.9 | 2.7 | 956.5 |
| decode_b32_global_kv8k | 124.1 | 114.3 | 7.0 | 2.8 | 304.4 |
| decode_b32_global_kv64k | 854.8 | 845.1 | 6.9 | 2.8 | 2391.2 |
| decode_b1_global_kv64k | 852.6 | 849.1 | 3.5 | 0.0 | 2326.6 |

"scheduler" is `CuSeqlensToBlocksKernel` + `CuBlocksToBatchKernel`. Those two
exist only to schedule ShearingBias (`interface.py:723-794`, inside the
`rel_bias is not None` block), so they disappear with it.

The prefill_swa_8k and prefill_global_8k rows are different shapes, not the
same shape in two modes: the kernel signatures in the JSON differ in head
count, which is why the shear costs 827.2 in one and 460.9 in the other. Do
not treat 460.9 and 827.2 as two measurements of the same work.

## Layout contract

`journal/u2-hopper-design.md:244-258`, "Layout contract COMPLETE
(machine-verified, 2026-07-19)", states verbatim:

> col(i, k) = k + padded - 128 * (m_block(i) + 1),  m_block(i) = i // 128

with `padded = rel_extent + 256`, verified at 20,100/20,100 positions for
T=200, ext=512. `harness/parity_shear_writer.py` produced it by decoding
encoded values out of a real ShearingBias run on sm_120.

That form is the **seqlen_q == seqlen_k specialisation**. Re-deriving from the
writer (`shearing_bias.py:357-476`) gives the general form. The writer places
relative index `d` at

    col(i, d) = bias_idx_right(i) - 1 - d

and its own arithmetic reduces (unclamped, see below) to

    bias_idx_right(i) = n_idx_right(i) + rel_extent_padded
                        - 128 * n_block_max(m_block(i))

so with the reversed index `c = rel_extent - 1 - d` the destination is
contiguous and increasing in `c`:

    col(i, c) = base(i) + c
    base(i)   = n_idx_right(i) + 256 - 128 * n_block_max(m_block(i))

    n_idx_right(i) = i_local + 1 + (seqlen_k - seqlen_q) + window_right
    n_block_max(m) = min(ceil(seqlen_k / 128),
                         ceil((128*(m+1) + seqlen_k - seqlen_q
                               + window_right) / 128))

At seqlen_q == seqlen_k this collapses to the journal's line: `n_block_max`
becomes `m_block + 1`, and `col = k + padded - 128*(m_block+1)`. Checked in
code, all positions, before anything was written.

Three properties that make the fusion work, all derived from the writer's own
expressions and checked over 6906 row configurations (rel_extent 128/512/1024,
prefill / chunked / decode, causal and SWA):

1. `base(i)` is in `[1, 256]`, always. The 256-column pad is exactly the room
   the shear needs, no more.
2. The `max(..., 0)` clamp in `bias_block_idx_right` never fires for
   `rel_extent >= 128`. `n_block_max - n_block_for_rel0` is 1 or 2 while
   `rel_extent_padded/128 >= 3`. If it ever fired, the shift would stop being
   uniform within a 128-row block and the whole tile-fetch contract would
   break, not just this patch.
3. `bias_idx_left = bias_idx_right - num_bias_vals` exactly; the `max(0, ...)`
   never fires either.

## Row layout the fused writer produces

Per valid row, covering all `rel_extent + 256` columns exactly once:

    [0, base)                     left_pad
    [base, base + rel_extent)     value if c >= rel_extent - num_bias_vals
                                  else left_pad
    [base + rel_extent, padded)   -inf

    num_bias_vals = n_idx_right - max(n_idx_left, n_idx_right - rel_extent)
    n_idx_left    = max(i_local + ctx - window_left, 0)  if local, else 0
    left_pad      = -inf if window_size_left is not None else 0.0
                    (shearing_bias.py:88-89)

The `c >= rel_extent - num_bias_vals` mask is not padding hygiene. It is real
semantics: for a row whose causal history is shorter than `rel_extent`, the
projection produces a value at every `d`, and the writer deliberately replaces
the ones beyond the history with `left_pad`. Dropping that mask would leave
finite bias on positions the model must not see.

## The hard parts

### (a) Can a token-indexed kernel emit this directly?

Yes. The only row-dependent quantity is the scalar `base(i)`, and everything it
needs is per-token:

* `i_local = token - cu_seqlens_q[req]`
* `seqlen_q = cu_seqlens_q[req+1] - cu_seqlens_q[req]`
* `seqlen_k = seq_lens[req]` (FA4's `seqused_k`)
* `req = seq_idx[token]`, already passed to qkvr_prep

The shear moves values only *within* a row, never across rows, so token
parallelism is preserved. The 128-row block only enters through
`m_block = i_local // 128` inside `n_block_max`, which is integer arithmetic on
`i_local`, not a cross-row dependency.

The other half of the answer is the reversal. `col(i,d) = bias_idx_right - 1 -
d` runs *backwards* in `d`, which would turn a contiguous store into a
stride -1 store. Reversing the projection column index instead
(`proj_col = rel_extent - 1 - col`) moves the reversal onto a tiny, fully
cached weight read (`rel_proj` is 16 x rel_extent) and leaves the output store
contiguous and per-row-offset. `rel_proj` is untouched on the host, so nothing
else that reads it is affected.

`seq_lens` is the one new tensor. It could be avoided by deriving `ctx` from
`positions[t] - i_local`, but only the first term of `n_block_max`,
`ceil(seqlen_k/128)`, genuinely needs `seqlen_k`, and it binds on the last
(partial) m_block of every request. Passing the same `seq_lens` FA4 receives is
both cheaper to reason about and impossible to get out of sync with the reader.

### (b) Padding, and who zeroes it

Nobody zeroes it: the fused writer covers every column. The value window is
`rel_extent` wide at offset `base`; the two wings are `[0, base)` and
`[base + rel_extent, padded)`, of widths `base` and `256 - base`, both at most
256. So the wings always live inside columns `[0, 256)` and
`[rel_extent, rel_extent + 256)`, which is why the pad tile is a fixed 512-wide
index space regardless of `rel_extent`.

The three masks are disjoint by construction, so no column is written twice and
no ordering between the value store and the pad store is required. When
`rel_extent == 128` the two wing tiles overlap in columns `[128, 256)`; the
masks (`col < base` and `col >= base + 128`) are still disjoint there.

Work placement: each column program owns one slice of the 512 pad columns
(`PAD_BLOCK = next_pow2(512 / num_column_programs)`), so the pad write is spread
evenly instead of landing entirely on program 0, and the launch grid is
unchanged from today.

Rows `[num_tokens, num_tokens + 128)` of the buffer stay uninitialised. That is
what happens today: `interface.py:711-718` allocates the bias with
`torch.empty` and `bias_total_q_padded = total_q + tile_m`, and ShearingBias's
`store_m_idx_in_bounds` guard never writes those rows. The attention kernel
reads them in the last m-tile and masks the result. The fused path preserves
this exactly rather than "improving" it.

### (c) Packed GQA

**The fusion requires `pack_gqa=False`, and the patch asserts it.**

With `pack_gqa`, ShearingBias reindexes rows through `pack_gqa_layout` so the
row space is packed rows (`q_idx * qhead_per_kvhead + head_offset`), and both
`m_block` and `m_idx_actual` are computed in that space. The *addresses* do not
move (the buffer is still `(total_q, h, padded)`), but `base()` becomes
head-dependent: two heads of the same token can sit in different 128-row
blocks and get shifts differing by 128. A writer that computes
`m_block = i_local // 128` would silently place one of them wrong.

This is not a new restriction. `interface.py` already forces `pack_gqa=False`
for arch 9 with `rel_bias`, with the session-24 root cause recorded in comment
form: packed tile rows interleave 8 GQA q-heads per sequence position, which
breaks both the bias head-slice and the 128-row shear contract
(`journal/upstream/04-pack-gqa-row-semantics.md`). The fusion inherits that
constraint and extends it to Blackwell, where `pack_gqa` can otherwise be on.
The interface patch asserts rather than silently mis-shearing.

Extending to packed GQA is mechanical but not free: `_shear_row_geometry` would
take `qhead_per_kvhead`, compute the packed row index from `(token, head)`, and
use it for `m_block` while keeping `i_local` for `n_idx_right`. It should not
be attempted before the unpacked path is green on silicon.

### (d) Varlen and cu_seqlens

The fused writer is varlen-native: it indexes `cu_seqlens_q` and `seq_lens`
per token, which is the only form vLLM produces. ShearingBias also accepts the
batched `(b, s_q, h, rel_extent)` form; a batched problem is a varlen problem
with uniform `cu_seqlens`, and the per-row geometry is identical because
`SeqlenInfoQK` resolves `seqlen_q`/`seqlen_k` to the same numbers either way.

`harness/parity_shear_fusion.py` proves that rather than asserting it:
`batched_form_global` and `batched_form_swa` run ShearingBias in the **batched**
form and compare its output against the fused **varlen** output row for row.
Both forms of the reference are therefore exercised.

## What was verified offline, and what was not

Verified (no GPU needed, all re-runnable arithmetic):

* the closed form for `base(i)` against the writer's own
  `bias_idx_right` / `bias_idx_left` / `num_bias_vals` expressions, 6906 row
  configurations, zero mismatches;
* the closed form against the journal's machine-verified contract on the case
  the journal verified, zero mismatches;
* a literal simulation of the writer's inner loops, including the vec_size=2
  pair loads, both `is_even` branches and both edge fixups
  (`shearing_bias.py:463-466`), against a literal simulation of the Triton code
  in this patch: 10,116 rows over rel_extent 128/512/1024, prefill / chunked /
  decode, causal and SWA, zero mismatches, no unwritten columns on either side;
* patch idempotence and anchor uniqueness on copies of the real tree, with the
  full chain `u2_serving_route -> u3_fp8_kv -> u2_shear_fusion`, all four
  patched files compiling;
* `compare()` and `diagnose_shift()` in the harness, against synthetic rows
  mutated by +-1 and +-128 column shifts, an unwritten column, and a masking
  error: each is reported as its own distinct failure class.

Verified on silicon, `sm_120` (RTX 5090), 16/16 cases of
`harness/parity_shear_fusion.py`. One recorded run; the only record is commit
`7375849`, which wrote no artifact. On `sm_90` the same gate scores **14/16**,
see "What sm_90 says" below:

* **Triton codegen.** The device functions, the `tl.where` broadcasts, the
  `tl.broadcast_to` in the pad store, and the negative-index guard on the
  reversed projection load compile and run. No shape or type fixes were needed.
* **The real ShearingBias vs the simulation of it.** The 14 writer cases
  compare against the real CuTe kernel's output, not against the simulation, on
  global, sliding-window, varlen, batched, prefill, chunked and decode shapes.
* **Bit-equality.** The 14 writer cases are bit-exact, so the reversed operand
  layout did not change Triton's dot lowering. No tolerance was added.
* **Plumbing.** The two `attention_consumes_*` cases ran the FA4 kernel on the
  fused buffer inside the kernel's own measured noise floor, so the buffer
  reaches the kernel and `has_bias` stays on.

Still not verified, and the reason each matters:

* **`sm_90` and `sm_100`.** The gate has run on one arch. `sm_120` is not
  Blackwell and does not stand in for `sm_100`, and the Hopper path is the one
  this patch exists to speed up. Nothing here is validated for the arch we
  serve on.
* **Every performance number below.** No fused-shear timing exists on any
  hardware, `sm_120` included. The `sm_120` run was a correctness gate and was
  not timed.

## Projected performance

**PROJECTED, NOT MEASURED.** Assumes only that the ShearingBias and scheduler
kernels vanish and the attention kernel is unchanged, which is what the fusion
does to the *attention* microbench. It is an upper bound on the win, because
qkvr_prep's rel writer gets more expensive and that cost is not in these
totals.

| workload | today | projected | today vs score_mod | projected vs score_mod |
|---|---|---|---|---|
| prefill_swa_8k | 1223.0 | ~~759.4~~ **REFUTED, measured 1251.6** | 0.78x (we lose) | ~~1.26x~~ **REFUTED** |
| prefill_global_8k | 3308.8 | ~~2479.0~~ **REFUTED, the writer alone costs +1019.4** | 1.45x | ~~1.94x~~ **REFUTED** |
| decode_b32_global_kv8k | 124.1 | 114.3 | ~~2.45x~~ **WITHDRAWN** | ~~2.66x~~ **WITHDRAWN** |
| decode_b32_global_kv64k | 854.8 | 845.1 | ~~2.80x~~ **WITHDRAWN** | ~~2.83x~~ **WITHDRAWN** |
| decode_b1_global_kv64k | 852.6 | 849.1 | ~~2.73x~~ **WITHDRAWN** | ~~2.74x~~ **WITHDRAWN** |

The headline was the SWA prefill case: ~~**projected 759.4us against score_mod's
956.5us, turning the one workload we lose into a win.**~~ **That headline is
refuted.** An H100 measured the fused writer at 1251.6 us against 685.9 us for
the writer plus `ShearingBias` it replaces, so the fusion makes this case worse
by 561.1 us rather than better by 464 us. The arithmetic below was sound and its
premise was not: it assumed the attention kernel unchanged and the pre-kernel
simply gone, and did not price the extra 256 columns the fused writer has to
emit. Point 1 of the banner at the top of this file has the measured numbers. The
projection is kept because a refuted projection that is still legible is what
makes the refutation checkable.

The cost that is *not* in that table: at rel_extent=512 the fused rel store
writes 768 columns per (token, head) row instead of 512, so qkvr_prep's rel
kernel writes 50% more bytes. Counting the whole rel path in bf16 elements per
row:

    today:  512 write (qkvr) + 512 read + 768 write (ShearingBias) = 1792
    fused:  768 write (qkvr)                                       =  768

a 57% reduction in rel-path traffic, one fewer kernel launch per attention
layer, and two fewer scheduler launches. The direction is not in doubt; the
magnitude is, because **the qkvr_prep rel kernel has never been microbenched**.
Until it is, the honest claim is "ShearingBias goes to zero and qkvr_prep gets
somewhat more expensive", not a number.

## Validation sequence on an H100

Run in this order and stop at the first red step. Steps 1 and 2 have been run
on `sm_120`; **no step in this list has been run on an H100**, which is what
this section is for.

1. Deploy `kernels/tml_fa4_modified`, then
   `python kernels/patches/u2_serving_route.py $VLLM`,
   `python kernels/patches/u3_fp8_kv.py $VLLM`,
   `python kernels/patches/u2_shear_fusion.py $VLLM`.
   Expect: `15 / 3 / 6 / 4 applied`. A second run of this patch reports
   `already applied, nothing to do` and exits 0.
   RESULT: both observed, on copies of the tree.
   Order matters only if you want u3 as well: u3 anchors on the stock form of
   text this patch rewrites, so u3 goes first. u3 is not a prerequisite.
   Applying `u2_shear_fusion.py` alone to a clean tree also gives `15 / 3 / 6 /
   4`, compiles, and introduces no u3 symbols.

2. `python harness/parity_shear_fusion.py`
   Expect: `16/16 cases bit-exact`, exit 0.
   RESULT on `sm_120` (RTX 5090): `16/16`, one recorded run, no artifact.
   RESULT on `sm_90` (H100, session 26): **`14/16`**. The 14 writer cases are
   bit-exact. Both `attention_consumes_*` cases fail with
   `NameError: cannot access local variable 'n_block'` inside
   `flash_fwd_sm90.py`, so on Hopper the pre-sheared `bias=` path does not
   execute at all. Artifact: `journal/remote/validate_s26_h100x1/`.
   The gate now writes `harness/parity_shear_fusion_sm<cc>.json`; the RTX 5090
   run predates that.
   This is the gate. If a case fails, read the diagnosis line first:
   * "PURE COLUMN SHIFT ... +-1" -> `base()` off by one, look at
     `n_idx_right` (the `+1` and the `window_right` term).
   * "PURE COLUMN SHIFT ... +-128" -> `n_block_max` wrong; almost certainly
     the `min(ceil(seqlen_k/128), ...)` term, which only binds on the last
     partial m-block of a request.
   * "not a pure shift" -> `num_bias_vals` or the left-pad mask, not `base()`.
   * failure only in `chunked_*` / `decode_*` while `prefill_*` passes ->
     something hard-coded the seqlen_q == seqlen_k form.
   * failure only in `throughput_path_global` -> the throughput kernel's
     in-loop geometry, not the mapping.
   * failure only in `fused_small_*` -> the `_qkvr_qkv_kernel` copy.
   * failure only in `attention_consumes_*` -> the buffer is right and the
     plumbing is wrong. A *large* output diff means the pre-sheared bias was
     dropped entirely (has_bias off, upstream finding 01); check that the
     `rel_bias_presheared` alias in `_flash_attn_fwd` lands before the
     `has_bias=rel_bias is not None` gate at `interface.py:485`.

3. `python harness/parity_shear_writer.py` unchanged, to confirm the reference
   kernel still behaves as the contract says on this machine before trusting
   step 2's reference.

4. `python harness/parity_fa4_rel.py`.
   Expect: unchanged 3/3 green. This harness calls `flash_attn_varlen_func`
   directly with `rel_bias=`, so it does **not** exercise the fused path and
   `INKLING_TURBO_FUSED_SHEAR` has no effect on it; it is here purely to
   confirm the interface edits did not disturb the existing path. The fused
   path's end-to-end check is `attention_consumes_*` inside step 2.

5. `python harness/parity_qkvr_prep.py`.
   Expect: unchanged 5/5. The default path must be untouched; every fused code
   path is behind `rel_shear is not None`.

6. Microbench, both with and without `INKLING_TURBO_FUSED_SHEAR=1`:
   `python harness/microbench_attn_day0.py`. Expect the ShearingBias and
   `Cu*Kernel` entries to be **absent** from `kernels_us` in the fused run, and
   the attention entry to be unchanged within noise. Save as
   `journal/remote/microbench_attn_fusedshear_sessionNN_h100.json`.

7. **The measurement that does not exist yet**: time the qkvr_prep rel kernel
   itself, natural vs sheared, at the prefill shapes. Without it there is no
   defensible end-to-end number, only the attention-side projection above.
   A `torch.profiler` run over `qkvr_rel_proj` alone at
   (T=8192, H as in the microbench, ext=512) is enough.

8. Only then: `scripts/gate_logit_parity.py` and the e2e serving benchmark.

## Risks, ranked

1. **Triton compile errors on first run.** Cheap, obvious, fix in place.
2. **`seq_lens` is not the tensor I think it is.** The fused writer reads
   `fa_md.seq_lens`, which is exactly what the same call passes to FA4 as
   `seqused_k`, so writer and reader cannot disagree. But if a sliding-window
   KV-cache group ever hands FA4 a *windowed* `seq_lens`, both sides move
   together and the shear stays self-consistent; what changes is which values
   the attention masks. Not a fusion bug, but check `fa_md.seq_lens` on the SWA
   layers during step 4.
3. **Padding tokens.** With CUDA-graph padding, `tokens > num_actual_tokens`
   and `seq_idx` for the pad tokens is not meaningful. The fused writer stops
   at `num_actual_tokens` rows for exactly this reason; the natural path still
   computes all rows. If a future change reads rel for padded rows, this breaks
   quietly. The `rel_rows` guard in `_qkvr_qkv_kernel` is the only thing
   standing there.
4. **`pack_gqa` turning on under us on Blackwell.** Asserted in the interface,
   so it fails loudly rather than mis-shearing.
5. **int32 index overflow** in `row * OUT_ROW_STRIDE`. The stride grows from
   `rel_extent` to `rel_extent + 256`, so headroom shrinks by a third. At
   65536 rows x 768 that is 5e7, far from 2^31, and the existing kernel has the
   same exposure. Worth remembering if `max_num_batched_tokens` ever gets large.
6. **The pad wings cost more than the shear saved** at small `rel_extent`.
   At rel_extent=128 the fused writer moves 384 columns where the natural one
   moved 128, a 3x growth on a kernel that was cheap. The saving is still
   positive overall (384 vs 128+128+384=640) but the balance is worst there.
   Inkling ships rel_extent=512, where it is 768 vs 1792.
7. **Deployment fragility.** `interface.py` lives in the deployed tree and is
   overwritten by every bootstrap, so this patch must be re-applied after each
   one. The source of truth for that file is
   `kernels/tml_fa4_modified/interface.py`, which this lane does not own.

## Files

* `kernels/patches/u2_shear_fusion.py` - the patch (28 anchored edits over 4
  files, idempotent, asserts every anchor is unique).
* `harness/parity_shear_fusion.py` - the gate (16 cases, bit-exact, names the
  off-by-one; 14 layout cases plus 2 that run the attention kernel on the
  fused buffer).
* `kernels/patches/u2_shear_fusion_notes.md` - this file.
