# Regression: the sm_90 shear shift dropped the bias at decode

Found 2026-07-25, the same day as [the `n_block` regression](regression-sm90-n-block.md)
and in the same file. This is the worse of the two. `3b78fc6` broke every sm_90
attention call loudly, at trace time, and was public for a few hours. This defect
was silent, it sat in the code path of every sm_90 attention call from the day the
Hopper bias reader was written, it was wrong in every one of them where
`seqlen_q != seqlen_k`, which is all decode and all chunked prefill, and it is the
reason the decode speedup figures on the front page are withdrawn.

Fixed on branch `fix/sm90-bias-shift-decode`, commit `eb1e487`, and validated on
an H100 the same day in `e9857de`: the new chunked and decode parity gate passes
7 of 7 on the fix, a deliberately broken control run shows that same gate failing
on the defect, and the decode microbenchmarks have been re-taken against the
day-0 baseline in one container. **The decode claim survives re-measurement at a
smaller number**, and the numbers published before the fix stay withdrawn because
they were never a like-for-like comparison. Both sets are below, with artifacts.

## Symptom

There was none. That is the entire problem.

No crash, no `NaN`, no tolerance failure, no visible numeric drift. Every gate
that bears on it was green: per-op parity 3/3 on `sm_90`, the full-model gate 32
of 32 on tokens, and the microbenchmarks producing a number on every attention
case. The defect does not perturb a result, it silently omits
work: Inkling's learned relative-position term was effectively absent from decode
and from chunked prefill on Hopper, while the timing harness recorded how fast
that was.

## Mechanism

The reader as published, `kernels/tml_fa4_modified/flash_fwd_sm90.py` at
`de7c795`, the tip of public `main`:

```
L1223   padded_bias = mBias_cur.shape[1]
L1230   bias_tile_shift = (
L1231       padded_bias // self.tile_n
L1232       - (128 * (m_block + 1)) // self.tile_n
L1233   )
L1234   bias_num_tiles = padded_bias // self.tile_n
```

and the consumer, `apply_rel_bias_sm90`, same file:

```
L1710   tile_idx = n_block + bias_tile_shift
L1718   if tile_idx >= 0 and tile_idx < bias_num_tiles:
L1719       gBias_tile = gBias_tiles[None, None, tile_idx]
            ...                                     acc = acc * scale + bias
        else:
            ...                                     acc = acc * scale
```

The layout that `ShearingBias` actually writes is defined by

```
bias_idx_right(row) = n_idx_right(row) + rel_extent_padded
                      - 128 * n_block_max(m_block(row))
```

which is derived, and separately re-verified against `ShearingBias`'s own
expressions, in [`kernels/patches/u2_inkernel_shear.md`](../kernels/patches/u2_inkernel_shear.md)
lines 72 to 80 and 196 to 200. At tile granularity that makes the shift
`padded // tile_n - (128 * n_block_max) // tile_n`. `tile_n` is 128 for this
path and `padded % 128 == 0`, so a tile is either wholly inside the shear range
or wholly outside it, and the tile-granular form is exact rather than rounded.

`128 * (m_block + 1)` is the `seqlen_q == seqlen_k` **specialisation** of
`128 * n_block_max`. The two agree for every `m_block` only when
`seqlen_q == seqlen_k` and `window_right` is zero or absent. Outside that family
they can still coincide for some query tiles and not others, which is why 300/256
causal is wrong on 44 of its 300 rows rather than all of them. At decode they
never coincide once the cache is longer than one KV block, because `m_block` is 0
while `n_block_max` is the whole block count.

Because the consumer guards on `0 <= tile_idx < bias_num_tiles`, a wrong shift
does not read out of range. It falls into the `else` branch and adds no bias at
all. The failure mode is omission, which is why no tolerance was violated
anywhere it was measured: no measurement looked at decode output.

### Which KV blocks got a bias tile

Real Inkling geometry from `harness/microbench_attn_day0.py:126-132`:
`rel_extent = 1024`, so `padded = rel_extent + 256 = 1280`
(`kernels/tml_fa4_modified/interface.py:713`) and
`bias_num_tiles = 1280 // 128 = 10`. Ten tiles is the whole extent, so ten KV
blocks per query tile is the maximum that can ever carry bias.

| shape | `n_block_max` | shift used | shift required | blocks that got bias | should have been |
|---|---|---|---|---|---|
| prefill 8K, `m_block` 0 | 1 | +9 | +9 | 1 of 1 | 1 of 1, correct |
| prefill 8K, `m_block` 63 | 64 | -54 | -54 | 10 of 64 | 10 of 64, correct |
| chunked prefill, 128 q on 8192 KV | 64 | +9 | -54 | 1 of 64, the OLDEST | 10 of 64, the newest |
| decode, batch 1, 8K KV | 64 | +9 | -54 | 1 of 64, the OLDEST | 10 of 64, the newest |
| decode, batch 1, 64K KV | 512 | +9 | -502 | 1 of 512, the OLDEST | 10 of 512, the newest |

The one block that did pass the guard did not get the right numbers either. With
`m_block = 0` the shift is +9, so `n_block` 0 reads sheared tile 9, which is the
band belonging to the newest KV block, and it is applied to the oldest one. Bias
on the wrong block, with the wrong values, and none anywhere else.
`u2_inkernel_shear.md:238-241` states the same conclusion independently.

Decode is the shape serving spends most of its time in, and it was the shape
that lost the most: at 64K of KV, 511 of 512 blocks got nothing.

### It is not only a decode defect

The same document brute-forced `n_block_max(m_block) != m_block + 1` across 120
configurations (`u2_inkernel_shear.md:224-232`, at `rel_extent = 512`):

| shape | rows where the published shift is wrong | worst error |
|---|---|---|
| 200/200, 8192/8192, 4096/4096 causal, `window_right = 0` | 0 | 0 tiles |
| 300/256 causal | 44 of 300 | 1 tile |
| 256/300 causal | 256 of 256 | 1 tile |
| 128/8192 causal | 128 of 128 | 63 tiles |
| 32/8192 causal | 32 of 32 | 63 tiles |
| 1/8192 causal | 1 of 1 | 63 tiles |
| 8192/8192, shifted diagonal (`window_right = 256`) | 8064 of 8192 | 2 tiles |

The last row matters for the shape of the mistake: even at
`seqlen_q == seqlen_k`, a non-zero `window_right` breaks the specialisation.
Inkling does not use one, so that row is a latent case rather than a served one,
but it shows the identity being relied on was narrower than "full prefill".

## How it was found

Not by a failing gate. By deriving the layout contract in general form while
designing the in-kernel shear (`kernels/patches/u2_inkernel_shear.md`), which
brute-forced the index arithmetic against
`tml-fa4/flash_attn/cute/shearing_bias.py:357-476` and
`block_info.py:26-93` over 120 configurations, 229,773,824 positions, and
recorded the specialisation as a side finding it deliberately did not fix,
because changing a green path was not that lane's decision to make
(`u2_inkernel_shear.md:213-270`). The brute-force script is not committed, and
that document says so.

This is the exact inverse of the `n_block` regression, which is worth holding
next to it. There, static reasoning was useless and one small GPU settled it in
minutes. Here, every GPU we ran was green and only reading the arithmetic
settled it. Neither method is sufficient on its own, and choosing between them
by habit is what lets a defect live.

## Why three separate gates all missed it

Three gaps lined up, and each one alone would have been enough to catch this.

1. **The parity suite tests one shape family.** `harness/parity_fa4_rel.py` has
   three cases (`:163-166`) and all three pass `cu_seqlens_q == cu_seqlens_k`
   (`:88`, `:110`, `:138`). That is precisely the family in which
   `n_block_max == m_block + 1` is an identity, so the suite lived entirely
   inside the one family the defect got right. Three green cases certified one
   shape family, and were read as certifying the kernel.
2. **The decode benchmarks never look at output.** The five decode cases in
   `harness/microbench_attn_day0.py:128-132` do run `seqlen_q != seqlen_k`, so
   they executed the defect on every iteration. They time the kernel and inspect
   nothing. A timing harness cannot notice missing work; that is what makes
   timing a wrong-output path dangerous rather than merely useless.
3. **The full-model gate never decoded a token.**
   `scripts/gate_logit_parity.py` calls the completions API with `max_tokens=0`
   and `echo=True` (`:402-403`, `:418`), so all 2369 compared positions are
   echoed **prompt** logprobs. Its own docstring says a token mismatch in the
   echoed prompt region "is impossible with one tokenizer" (`:439-441`). The
   recorded `tokens_compared: 2369, tokens_match_all: true` in
   [gate_logit_parity_8xh100.json](remote/gate_logit_parity_8xh100.json) is
   therefore a tautology, not evidence about the kernel. The one sub-gate that
   does generate tokens (`batch_max_tokens: 32`, 348 positions per side)
   compares batched against batch-1 **within** one build, and it failed on the
   stock build too, which is already recorded.

One thing to be careful about in that third point, because it is easy to
overclaim. What is established is that the gate generated no tokens, so it
exercised no decode call: `max_tokens=0`. Whether it also avoided
`seqlen_q != seqlen_k` entirely is not established. The prompts are short, the
longest contributing 1212 of the 2369 positions, which is inside any plausible
prefill chunk budget, but the artifact records neither the scheduler's
`max_num_batched_tokens` nor whether prefix caching produced partial hits on the
repeated prompts, and either could have produced a chunked call. So the correct
statement is that the gate could not have caught a decode defect, not that it
never touched the defective shape family.

## The fix

`kernels/tml_fa4_modified/flash_fwd_sm90.py:1256-1262` on the fix branch:

```python
_, n_block_max_bias = block_info.get_n_block_min_max(
    seqlen, m_block, split_idx, batch_idx, absolute=True
)
bias_tile_shift = (
    padded_bias // self.tile_n
    - (128 * n_block_max_bias) // self.tile_n
)
```

`absolute=True` is load bearing. `ShearingBias` runs once per sequence and knows
nothing about KV splits, so the layout it wrote is defined by the full block
count. Passing a split's own `n_block_max` would shift the bias by a per-split
amount against a buffer that was never sheared that way, which would turn a
decode defect into a split-KV defect.

**The stock Blackwell reader does this correctly.** In a stock tml-fa4 checkout
at the pinned commit `13374f0` ("Fix forward argument handling on pre-Blackwell
GPUs"), `flash_attn/cute/flash_fwd_sm100.py:2384-2400` fetches the block bounds
twice, once normally and once with `absolute=True`, and derives its bias index
offset from the difference:

```
L2385   n_block_min, n_block_max = block_info.get_n_block_min_max(
L2386       seqlen, m_block, split_idx, batch_idx,
L2387   )
L2388   _, n_block_max_abs = block_info.get_n_block_min_max(
L2389       seqlen, m_block, split_idx, batch_idx, absolute=True,
L2390   )
L2399   bias_idx_offset1 = n_block_max_abs - n_block_max
```

That file has no counterpart in this repository, which ships only
`flash_fwd.py`, `flash_fwd_sm90.py` and `interface.py` under
`kernels/tml_fa4_modified/`, so verifying it means grepping `absolute=True` in a
stock checkout at that commit. The general form was sitting in the reference
implementation we ported from, split-awareness included, and we specialised it by
hand on the way across. **This was our porting error, not an upstream defect**,
and none of the five upstream findings in [journal/upstream/](upstream/) depends
on it.

Every already-validated shape is unchanged by the fix, because `n_block_max`
really does equal `m_block + 1` there. The published parity runs and the prefill
microbenchmarks compute the same shift before and after, so the fix is not
expected to move any number that currently stands.

New gate: `harness/parity_rel_chunked_decode.py`, seven cases with
`seqlen_k > seqlen_q` covering chunked prefill, decode and sliding window, plus a
`seqlen_q == seqlen_k` control so that a pass is informative in both directions.
Its reference reduces bit-identically to `parity_fa4_rel.py`'s at `ctx = 0`.

## Validated on an H100, and the gate was shown to fail on the defect

Two runs on one H100, in the shipping configuration (our kernels plus
`u2_serving_route`, u3 and the shear fusion asserted absent). Artifacts:
[validate_s27_decodefix](remote/validate_s27_decodefix/) and
[validate_s27_brokencontrol](remote/validate_s27_brokencontrol/).

Run 1, the fix. `parity_rel_chunked_decode` **7 of 7**, `parity_fa4_rel` still
3 of 3, `parity_qkvr_prep` still green. That is the first correctness result the
`sm_90` decode path has ever had.

Run 2, a control with `128 * (m_block + 1)` deliberately put back, so the new
gate could be observed failing on the defect it was written for. A gate only ever
seen passing is not known to be a gate. Per-case mean absolute difference against
the float32 oracle, from the two `parity_rel_chunked_decode_*.json` files:

| case | fixed, mean | broken, mean |
|---|---|---|
| `control_full_prefill` (`seqlen_q == seqlen_k`) | 6.86e-05 | 6.86e-05, identical |
| `chunked_global_128_on_1408` | 3.73e-05 | 1.05e-02 |
| `chunked_global_256_on_768` | 4.67e-05 | 1.43e-02 |
| `decode_global_ctx2047` | 2.89e-05 | 6.44e-03 |
| `decode_global_ctx4095` | 2.18e-05 | 3.28e-03 |
| `decode_swa_ctx4095` | 6.96e-05 | 1.69e-02 |
| `chunked_swa_128_on_1408` | 6.39e-05 | 1.79e-02 |

The control row being bit-identical in both runs is the point: the defect touches
only `seqlen_q != seqlen_k`, so any movement there would have meant the swap did
something else as well.

That control also found a hole in the new gate. As stored in the artifacts the
tolerances were `TOL_MAX = 0.05`, `TOL_MEAN = 0.005`, and under those the broken
`decode_global_ctx4095` case **passed** at max 1.81e-02 and mean 3.28e-03, so the
gate caught 5 of the 6 defective cases and certified one defective case as
correct. Max cannot separate cleanly, because the legitimate full-prefill control
sits at the bf16 quantum of 7.81e-03. The mean separates with no overlap: worst
legitimate 6.96e-05 against best defective 3.28e-03, a factor of 47. `TOL_MEAN`
is now `5e-4`, which sits in that gap with 7.2x headroom above every passing case
and 6.6x below every failing one, and replaying the two artifacts against it
gives 7 of 7 on the fix and 1 of 7 on the broken kernel, the 1 being the control.
Every number in that paragraph is re-derivable from the per-case `mean_abs_diff`
fields in the two JSON files.

## The same specialisation is still live in the generic kernel

Found while writing this file, and **not fixed**.
`kernels/tml_fa4_modified/flash_fwd.py`, the generic path that serves `sm_80`
and `sm_120`, carries the identical expression in two places:

- `:917-919`, the smem-staged reader, with `bias_k_min_tile = -bias_tile_shift`
  at `:921` and a **lower-bound-only** guard at `:1233`
  (`if n_block >= bias_k_min_tile`), which is not the two-sided guard the
  `sm_90` path has;
- `:1359`, the direct-gmem reader, `shift = n_block * self.tile_n + padded - 128 * (m_block + 1)`,
  guarded two-sided per element at `:1366` (`col >= 0 and col < padded`), so
  that one omits bias the same way `sm_90` did.

What this touches, stated conservatively:

- The `sm_80` and `sm_120` **parity** results are unaffected. Every case in
  `harness/parity_fa4_rel.py` is `seqlen_q == seqlen_k`, where the shift is
  correct, so 3/3 green remains a true statement about that family, and remains
  silent about every other family, exactly as on Hopper.
- The A100 **tile-sweep timings** are decode shapes (`harness/tune_sm80.py:48-49`
  time `T_q = 1` against `T_k = 65536`) while that harness's parity gate runs
  `cu_seqlens_q == cu_seqlens_k` (`:105-106`). So the 10.1%, 18.2% and 18.7%
  tile-tuning figures were measured with the bias gather largely absent on both
  sides of the comparison. They are self-comparisons rather than comparisons
  against a differently-behaving baseline, so they are not "kernel doing less
  work than its baseline" the way the Hopper decode ratios were. What they need
  is re-checking, because a tile size chosen while most of the bias work was
  being skipped is not necessarily the tile size that wins once it is not.
- The `sm_120` relative timings carry the same caveat, and they were already
  labelled relative-only and journal-only.
- The behaviour of the smem-staged reader under a wrong shift is **not**
  established here. Its guard is one-sided, so it may index past the tile domain
  rather than skip. That is a question for an A100 or a 5090 with the new
  chunked and decode parity harness, not something to conclude from reading.
- `harness/parity_rel_chunked_decode.py` has now run on `sm_90` only. No
  `seqlen_q != seqlen_k` correctness result exists on `sm_80` or `sm_120`, and
  running that harness there is cheap compared with what this defect cost.

## What this means for the published numbers

Withdrawn, and marked as withdrawn in place in
[README.md](../README.md#what-is-measured) and [LEDGER.md](../LEDGER.md):

| withdrawn claim | as published | why |
|---|---|---|
| decode, batch 1, 64K KV | ours 853 / 860 vs `score_mod` 2327 / 2412, "2.7x to 2.8x faster" | ours biased 1 KV block of 512 where 10 should have been biased, and biased the wrong one, while `score_mod` gathered every score correctly |
| decode, batch 32, 64K KV | ours 855 / 866 vs 2391 / 2383, "2.8x faster" | same |
| decode, batch 32, 8K KV | ours 124 / 124 vs 304 / 304, "2.5x faster" | ours biased 1 KV block of 64 where 10 should have been biased |
| the cost of correct bias at decode | "15.8% over plain attention, 853 us against 736 us" | 853 us is the same defective measurement, so the figure understates the cost of doing the work properly |
| session 24 decode | "decode b1 kv64k 905.6 us vs 2375, 2.6x" | same defect, earlier box |

The numbers were measured, not invented. Both sides of each ratio are real
timings of real kernels on the same box in the same run. What is wrong is the
comparison: our kernel was doing a small fraction of the bias work that the
baseline was doing, so the ratio measures the omission, not the design. It is
not like for like, whatever the re-measurement then says, so it stays withdrawn
rather than being quietly replaced.

### What the like-for-like measurement says

From [validate_s27_decodefix](remote/validate_s27_decodefix/), fixed kernel and
day-0 `score_mod` timed in the same container, us/iter:

| case | ours, fixed | day-0 `score_mod` | ratio | was published as |
|---|---|---|---|---|
| decode, batch 1, 64K KV | 894.7 | 2375.6 | **2.66x faster** | 2.7x to 2.8x |
| decode, batch 32, 64K KV | 867.6 | 2389.4 | **2.75x faster** | 2.8x |
| decode, batch 32, 8K KV | 146.3 | 307.7 | **2.10x faster** | 2.5x |
| prefill 8K, global | 3353.7 | 4820.8 | **1.44x faster** | 1.45x to 1.46x |
| prefill 8K, sliding window | 1224.5 | 965.4 | **1.27x slower** | 1.28x to 1.41x slower |

So the decode claim survives at a smaller number rather than collapsing. The
projection recorded before the run, that the ratio could only shrink, held.

What the correct bias costs, taking the broken control run as the before and the
fix as the after. Two containers of the same box class rather than one process, so
treat these as percentages good to about a point:

| case | broken | fixed | cost |
|---|---|---|---|
| decode, batch 1, 64K KV | 852.3 | 894.7 | +5.0% |
| decode, batch 32, 64K KV | 851.8 | 867.6 | +1.9% |
| decode, batch 32, 8K KV | 122.2 | 146.3 | +19.7% |
| prefill 8K, global | 3329.7 | 3353.7 | +0.7% |
| prefill 8K, sliding window | 1210.6 | 1224.5 | +1.1% |

The prefill rows are the interesting confirmation. The shift is arithmetically
identical there, so the +0.7% and +1.1% are the cost of the extra
`get_n_block_min_max` call and nothing else, and they are the reason the standing
global-prefill ratio moves from 1.45x to 1.44x. The decode rows are the cost of
actually gathering the bias, and at batch 32 with 8K of KV that cost is 19.7%,
which is where the 2.5x became 2.10x.

Against biasless attention in the same run, correct bias at batch-1 64K decode
costs 894.7 against 736.0, so **21.6%**. That figure replaces the withdrawn 15.8%,
which was the same comparison made with the gather mostly skipped.

Two limits on all of this. The decode ratios rest on **one** post-fix container,
where the prefill ratios now have three runs behind them, so they are good to one
decimal and are not yet reproduced on a second machine. And the before-and-after
cost table crosses containers, so it is weaker than the single-process comparison
used for the shear fusion.

Decode-side profiler figures were taken on the defective kernel too. "64 CTAs on
132 SMs" is structural and stands, since it is a launch geometry rather than a
measurement. The decode DRAM percentage and occupancy read off Nsight Compute in
session 24 were measured with most of the bias traffic absent, and are marked
accordingly in the ledger.

**Unaffected, and still standing.** Everything at `seqlen_q == seqlen_k`:

- `prefill_global_8k`, ours 3308.8, 3306.9 and 3353.7 against `score_mod` 4799.4,
  4841.2 and 4820.8, so 1.44x to 1.46x faster across three runs, the 1.44x being
  the fixed kernel
  ([session 25](remote/microbench_attn_day0_session25_h100.json),
  [session 26b](remote/validate_s26b_h100x1_route/),
  [the decode-fix run](remote/validate_s27_decodefix/));
- `prefill_swa_8k`, ours 1223.0, 1221.4 and 1224.5 against 956.5, 863.2 and
  965.4, so 1.27x to 1.41x slower, which is the case we lose;
- per-op parity 3/3 on `sm_90`, `sm_80` and `sm_120`, within its family;
- the `sm_80` support claim, which is that no day-0 path runs there at all;
- the 8x H100 memory recipe and the fact that the model serves;
- the shear-fusion result, which is a writer-side comparison in which attention
  consumes an identical buffer either way, so its decode row is a comparison
  between two writers rather than a statement about the reader.

## Lessons

1. **A parity suite that covers one shape family certifies one shape family.**
   Three green cases at `cu_seqlens_q == cu_seqlens_k` said nothing whatever
   about the shapes the kernel spends production time in. The rule now lives in
   [docs/METHODOLOGY.md](../docs/METHODOLOGY.md#parity-oracle-discipline), and it
   was learned the expensive way: the public front page carried a wrong
   comparison for days.
2. **A timing harness is not a correctness harness, and running one on a shape
   no oracle covers is worse than not running it.** The decode microbenchmarks
   did exercise the defect, thousands of times, and their output was a number.
3. **A gate whose docstring says the comparison cannot fail is not a gate.**
   `gate_logit_parity.py` recorded 2369 matching tokens and said in its own
   comments that a mismatch there was impossible. That should have been read as
   a design bug in the gate on the day it was written.
4. **Prefer the general form even when the specialisation is provably equal.**
   `128 * (m_block + 1)` was correct for every shape anyone had tested when it
   was written. `128 * n_block_max` costs one extra call and is correct for every
   shape, and the reference implementation we ported from already did it that
   way.
5. **When a defect is found by reading, look for its siblings by reading too.**
   The same expression was in the generic kernel the whole time, two lines of
   grep away, and it is still there.
6. **A gate that has only ever been seen passing is not known to be a gate.** The
   new harness was run a second time against a deliberately re-broken kernel, and
   that run is what turned it from an assertion into a check. It also paid for
   itself immediately: it exposed a tolerance under which one defective case
   passed, which would have left the same class of defect able to hide again.
7. **A withdrawal is not undone by a re-measurement.** The re-measured decode
   ratios are 2.66x, 2.75x and 2.10x, close enough to the withdrawn 2.7x, 2.8x
   and 2.5x that it would be tempting to say the old numbers were roughly right
   and move on. They were not right, they were unfounded, and being nearly equal
   to a sound number by luck is not the same as being sound. The struck rows stay
   struck.
