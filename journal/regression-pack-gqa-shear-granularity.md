# The shear writer packs GQA heads and the generic reader does not

Found and fixed 2026-07-25. Third member of the shear-shift family, after
[regression-sm90-bias-shift.md](regression-sm90-bias-shift.md) and the generic
port recorded in [regression-ampere-tile-sweep.md](regression-ampere-tile-sweep.md).
Second time `pack_gqa` has broken row semantics in this project; the first is
[upstream/04-pack-gqa-row-semantics.md](upstream/04-pack-gqa-row-semantics.md).

The shear expression itself was correct by this point. What was wrong is that the
writer and the reader evaluated it at two different `m_block` granularities.

## Root cause

`ShearingBias` ran with `pack_gqa=True` while the generic attention kernel that
reads its output was constructed with `pack_gqa=False`.

- `kernels/tml_fa4_modified/interface.py:1172` and `:1296` hardcode
  `pack_gqa=False` for the `sm_80` and `sm_120` kernels, commented "generic path
  never packs tensors".
- `interface.py:870` passed `pack_gqa`, the heuristic value, to `ShearingBias`.
  `_pack_gqa_heuristic` at `:234` returns True for any `qhead_per_kvhead > 1`.

The reader's `m_block` therefore spans 128 query **tokens** and it derives the
shear offset from `n_block_max` at that granularity
(`kernels/tml_fa4_modified/flash_fwd.py:961-992`). The writer's `m_block` spans
128 packed **rows**, which is `128 / qhead_per_kvhead` tokens
(`tml-fa4/flash_attn/cute/shearing_bias.py:358-373`, where `attn_m_block` is
`m_idx // 128` on a packed row index), so it writes a different `n_block_max`
per token sub-group. The two agree only on the last sub-group of each 128-token
block. Every earlier sub-group is sheared by one 128-column block too few, so the
bias value for causal distance `d` lands at distance `d - 128`.

## The predicate

With `qhpk = Hq / Hkv`, `G = 128 / qhpk`, `mb = t // 128`, `g = (t % 128) // G`
and `NB_K = ceil(T_k / 128)`, query token `t` is mis-sheared if and only if

```
min(NB_K, ceil((mb*128 + G*(g+1) + ctx)/128)) != min(NB_K, ceil((mb*128 + 128 + ctx)/128))
```

Fitted to 35 measured shapes and correct on all 35. It is fitted to that data,
so it is a description and not an independent prediction. Its corollaries all
held:

- **`ctx % 128 == 0` is never affected.** That includes every
  `seqlen_q == seqlen_k` case and all pure prefill.
- **`qhpk == 1` is never affected.**
- **Small `T_q` is protected by the `NB_K` cap**, which is why every `T_q = 1`
  decode case passed.

Those three corollaries are the entire reason this survived. Every gated shape in
this repository had `seqlen_k` a multiple of 128, and every parity case in
`harness/tune_sm80.py` had `Hq == Hkv == 8`.

## Localisation on the reported case

`T_q=200, ctx=1000, Hq=8, Hkv=1, rel_extent=512`: query tokens 0..15 and
128..143, 32 of 200, have their whole bias row displaced by exactly +128 columns.
The other 168 rows are bit-exact against the contract. Established two
independent ways on a local 5090:

1. A readout probe. `q = 0`, `rel_bias` one-hot at distance `d0`, and `v`
   encoding the key index in two channels, so softmax collapses onto whichever
   key received the bias. For `d0 = 128, 200, 511` the 32 bad rows put the bias
   on key `expected + 128`. For `d0 = 0, 1, 127` they applied no bias at all,
   because the implied distance `d0 - 128` is negative. Both consistent with a
   pure +128 column shift and with nothing else.
2. Capturing the sheared buffer directly, by wrapping the compiled
   `ShearingBias` in `_flash_attn_fwd.compile_cache_shear_bias.cache`, and
   checking `bias_idx_right = n_idx_right + padded - 128 * n_block_max` per row.
   The writer's `n_block_max` is 8 on tokens 0..15 where the reader uses 9, and
   9 on tokens 128..143 where the reader uses 10.

## Boundary table

Pre-fix, local 5090 (`sm_120`), tolerances of `harness/parity_rel_varlen_batch.py`.
Geometry is `Hq=8, Hkv=1, ext=512, causal` unless stated.

| shape | max | mean | verdict | predicted bad tokens |
|---|---|---|---|---|
| T_q=1 / ctx=1000 | 4.9e-04 | 4.0e-05 | PASS | 0 |
| T_q=16, 17, 20, 24 / ctx=1000 | 4.9e-04 | ~4.3e-05 | PASS | 0 |
| **T_q=25 / ctx=1000** | 7.7e-02 | 8.3e-03 | **FAIL** | 16 |
| T_q=26, 32, 40, 64, 127, 128, 129 / ctx=1000 | 7.7e-02 | 1.6e-03 to 8.0e-03 | FAIL | 16 |
| T_q=192, 200, 256 / ctx=1000 | 8.7e-02 | 1.6e-03 to 2.1e-03 | FAIL | 32 |
| T_q=200 / ctx=0 | 7.8e-03 | 1.7e-04 | PASS | 0 |
| T_q=200 / ctx=128 | 2.0e-03 | 9.5e-05 | PASS | 0 |
| **T_q=200 / ctx=129** | 3.3e-01 | **2.0e-02** | FAIL | 112 |
| T_q=200 / ctx=1012, 1016, 1023, 1024 | <1.0e-03 | ~4.1e-05 | PASS | 0 |
| T_q=200 / ctx=1100 | 9.0e-02 | 5.3e-03 | FAIL | 96 |
| T_q=200 / ctx=1000, qhpk <= 4 | 4.9e-04 | ~4.2e-05 | PASS | 0 |
| T_q=200 / ctx=1000, Hq=16 Hkv=1 | 8.3e-02 | 2.9e-03 | FAIL | 48 |
| T_q=200 / ctx=1000, **Hq=64 Hkv=8, production TP1** | 1.0e-01 | 2.0e-03 | FAIL | 32 |
| T_q=200 / ctx=1000, ext=1024 | 1.1e-01 | 2.7e-03 | FAIL | 32 |
| T_q=200 / ctx=1000, swa win=511 | 9.8e-04 | 6.3e-05 | PASS | 0 |
| T_q=200 / ctx=1100, swa win=511 | 2.3e-01 | 1.1e-02 | FAIL | 64 |
| T_q=[200,137] / ctx=[1000,1408] | 8.7e-02 | 1.2e-03 | FAIL | 32 |
| T_q=[200,1,137,64] / ctx=[0,900,1408,63] | 7.8e-03 | 1.2e-04 | PASS | 0 |

14 of 29 shapes failed. The worst is 39x `TOL_MEAN`. **It reaches the real
Inkling head geometry**, `Hq=64` over `Hkv=8`, which is the TP1 shape, and the
project's whole gate suite caught exactly one of the 14.

## Confirmed on Ampere, as a failure

The A100 session on the same day
([remote/validate_a100x1_s31](remote/validate_a100x1_s31/)) ran the code
**without** this fix and scored 11 of 12 on `parity_rel_varlen_batch`, failing
the same `single_m_tail_chunked` case at mean 1.9666e-03 against the 5090's
1.9674e-03. So the defect is observed on `sm_80` and not merely inferred from a
shared code path. **The fix has not run on `sm_80`.** Ampere and `sm_120` share
`flash_fwd.py` and the identical heuristic, so the same guard covers both, but
that is an argument and not a measurement.

## The fix

One guard in `kernels/tml_fa4_modified/interface.py`, extending the `sm_90`
precedent immediately above it:

```python
if arch // 10 in [8, 12] and rel_bias is not None:
    pack_gqa = False
```

`qhead_per_kvhead_packgqa` feeds only the bias sub-system, the prepare-blocks
kernel, `total_group_blocks_max`, the shear compile key and `ShearingBias`,
because `seqlen_q_packgqa` multiplies by `qhead_per_kvhead` unconditionally. So
`num_m_blocks`, `total_mblocks`, `num_splits` and `max_m_blocks_leq_one` are
untouched, and the split-KV path is not reachable here since `num_splits`
defaults to 1. `sm_90` already forced `pack_gqa=False` whenever `rel_bias` is
present. Blackwell keeps packing, because its native reader handles a packed
bias.

## Evidence for the fix

Applied to the deployed tree first, then to the repository and re-verified after
redeploy, with `diff` empty between the two for all three kernel files.

- `parity_rel_varlen_batch`: **12/12**, was 11/12. `single_m_tail_chunked` max
  0.103515625 to 4.8828e-04, mean 1.9674e-03 to 4.1567e-05.
- `parity_rel_chunked_decode`: **7/7**, and all seven means and maxes
  **bit-identical** to the committed `remote/local_sm120_s29` artifact.
- `parity_rel_bias_coverage`: **6/6**, probes unchanged, the 64K case still
  touches tiles 504..511.
- `parity_fa4_rel`: `tml_fa4_rel_bias` 3/3, `score_mod` 3/3.
  `parity_qkvr_prep`: 5/5.
- Across all 18 recorded cases, **every previously passing case is
  bit-identical**. Only the failing one moved.
- **Control run.** With the guard removed, `parity_rel_varlen_batch` returns to
  11/12 with `single_m_tail_chunked` back at max 1.0352e-01, and
  `T_q=25 / ctx=1000` flips FAIL to PASS across the two builds. The gate has now
  been seen failing on the defect it catches, which is the standard this project
  holds tolerances to.
- Writer contract re-checked post-fix: 0 rows off contract on all eight
  previously broken shapes, including the sliding-window one.

**No perf claim.** The same 4096-token prefill shape timed 808 and 6293 us/iter
in two runs minutes apart on the laptop, so nothing is measurable there.
Structurally the shear kernel writes the same `num_head * total_q` rows either
way and only the (block, head) factorisation of its grid changes.

## Effect on published numbers

None. `sm_90` already forced `pack_gqa=False` with `rel_bias`, so every Hopper
number is unaffected, and Blackwell is untouched. The `sm_80` and `sm_120`
tile-sweep percentages were already withdrawn for a different reason.

## Lessons

1. **A gate has a head geometry, not just a shape.** The sequence-length hole in
   `harness/tune_sm80.py` was found and closed on 2026-07-25, and the same file
   still checked `Hq == Hkv == 8` while timing `Hq=64` over `Hkv=8`. The defect
   here is invisible at `qhpk == 1` by construction. Closing one axis of a
   coverage hole reads like closing the hole. It is not. That file now derives
   its parity geometry from its own `CASES` list, so the two cannot diverge
   again on any axis.
2. **`ctx % 128 == 0` was load-bearing across the entire suite and nobody chose
   it.** Both shear defects found so far are exact on that condition, and every
   gated shape in the repository satisfied it, because round numbers are what
   people type. Real serving passes `seqused_k` from `cache_seqlens`, which is
   whatever the sequence happens to be. There is now a `ctx = 1000` probe, 104
   mod 128, in the tile-sweep gate.
3. **Two components can each be correct about a contract and still disagree
   about its units.** The reader and the writer both implemented
   `padded - 128 * n_block_max` faithfully. They disagreed about what one
   `m_block` contains. The `sm_90` path was saved from this by a line written
   for an unrelated reason, `pack_gqa=False` for bias on arch 9.
