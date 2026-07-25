# Withdrawn: the Ampere tile-sweep percentages

Companion to [regression-sm90-bias-shift.md](regression-sm90-bias-shift.md),
written 2026-07-25. That file records a shear-shift defect found on `sm_90` and
fixed there. This file records what the same defect did to the one set of numbers
this repository published off the **generic** kernel: the A100 tile-sweep
percentages.

They are withdrawn. Not corrected, not re-derived, withdrawn, because the tile
size they selected was selected under a kernel whose relative-position bias was
addressing the wrong place at exactly the shapes being timed, and the harness's
own parity gate was checking a different shape family from the one it timed.

The `sm_80` **support** claim is not affected and is not withdrawn. It never
depended on a percentage. It does pick up a caveat, stated at the end.

## What is withdrawn

| withdrawn | as published | evidence class | where it appeared |
|---|---|---|---|
| `tile_n=32` against untuned `tile_n=64`, batch-1 decode 64K KV | 5350.1 us against 5953.7 us, ~~"10.1% faster"~~ | strong, JSON | README, LEDGER, upstream report 05, `interface.py` comment |
| same, 32-sequence decode 64K KV | 60801.4 us against 74356.6 us, ~~"18.2% faster"~~ | strong, JSON | README, LEDGER, upstream report 05, `interface.py` comment |
| post-deploy re-run, 32-sequence decode 64K KV | 60977.5 us against 75013.4 us, ~~"18.7% faster"~~ | **journal-only**, session 27, no JSON | README, LEDGER, METHODOLOGY, `journal/remote/README.md`, upstream report 05 |
| the code comment that summarised them | ~~"decode-shaped calls run 10-18 percent faster at `tile_n=32`"~~ | derived from the above | `kernels/tml_fa4_modified/interface.py`, the `arch // 10 == 8` branch |

Artifact: [remote/tune_sm80_a100.json](remote/tune_sm80_a100.json). It is not
deleted and not edited. The four timings per configuration are real timings of a
real kernel, and they stay in the file. What is withdrawn is every **decode**
comparison derived from them.

### What survives from the same sweep

The withdrawal is narrower than the artifact, and saying which rows survive is
the whole point of stating the mechanism precisely.

- **The two prefill rows survive.** `prefill_global_8k` and `prefill_swa_8k` are
  both 8192 against 8192 (`harness/tune_sm80.py:46-47`), so `seqlen_q ==
  seqlen_k` with `window_right` absent, which is exactly the family in which the
  defective shift is an identity. Those timings were taken with the bias landing
  where it belongs. So `tile_n=64` beating `tile_n=32` on sliding-window prefill,
  9175.2 us against 10565.6 us, still stands, and so does `tile_n=32` on global
  prefill, 10712.7 against 11124.1.
- **The `tile_n=128` collapse survives, and the cleanest evidence for it is the
  prefill half.** `prefill_global_8k` goes from 10712.7 us at `tile_n=32` to
  362806.1 us at 128, a factor of 34 on a shape family the defect does not touch.
  The decode collapse is larger still, 131729.5 us against 5350.1 us at batch 1
  and 2865716.7 us on the 32-sequence case, but those two sit on defect-affected
  shapes, so quote the prefill figure. The cause recorded at the time was `sm_80`
  shared-memory pressure, no bias addressing error explains a factor of 34, and
  nothing here disturbs the conclusion that 128 must never be selected on that
  architecture.
- Consequence for the shipped selection rule, the `arch // 10 == 8` branch of
  `interface.py`, which picks
  `tile_n=32` when `max_seqlen_q <= 32` and 64 otherwise: **the "otherwise 64"
  half still has measured support** from the sliding-window prefill row. The
  decode half, the reason the rule exists at all, does not. The shipped default
  is left as it is, because no measurement supports changing it either, and the
  comment above it now says so.

## The mechanism, stated exactly, because the mechanism is the lesson

`harness/tune_sm80.py` is the file this repository cites, in
[CONTRIBUTING.md](../CONTRIBUTING.md) and in the README, as **the one place where
a code rule and not a habit enforces parity before a timing is reported**. It
does enforce it. It refuses to record a configuration's timings unless that
configuration's own parity run came back green (`:200-206`). The rule fired, it
went green three times out of three, and it was green about the wrong shapes.

The two halves of the file built their shapes independently.

**What it timed.** `CASES` at `:44-50`, of which two are decode:

```
:48   ("decode_b1_global_kv64k",     1,         65536, 64, 8, 1024, None),
:49   ("decode_32seqs_global_kv64k", "batched", 65536, 64, 8, 1024, None),
```

`make_case()` at `:54-85` turns those into calls where `seqlen_q != seqlen_k` by
construction. The single-sequence branch at `:73-74` builds
`cu_q = [0, T_q]` with `T_q = 1` against `cu_k = [0, T_k]` with `T_k = 65536`.
The batched branch at `:65-66` builds `cu_q = arange(B + 1)` and
`cu_k = arange(B + 1) * L`, so 32 sequences of one query row each against 65536
keys each.

**What it checked.** `parity_ok()` at `:88`, as it stood when the sweep ran, had
one shape: `T = PARITY_T`, and `PARITY_T = 512` (`:51`). It built **one**
`cu_seqlens` at `:102`,

```
:102  cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
:105  q=q, k=k, v=v, rel_bias=rel, cu_seqlens_q=cu, cu_seqlens_k=cu,
:106  max_seqlen_q=T, max_seqlen_k=T, ...
```

and passed it as both `cu_seqlens_q` and `cu_seqlens_k`. So the gate certified
512 query rows against 512 keys, twice, once global and once sliding window.

The generic kernel's bias reader computed its sheared tile shift as
`padded // tile_n - (128 * (m_block + 1)) // tile_n`
(`kernels/tml_fa4_modified/flash_fwd.py:917-919` before the fix). As
[regression-sm90-bias-shift.md](regression-sm90-bias-shift.md) derives,
`128 * (m_block + 1)` is the `seqlen_q == seqlen_k` specialisation of the layout
contract and is an identity on exactly the family `parity_ok()` built, and wrong
on exactly the family `make_case()` timed.

That is the whole failure in one sentence: **the gate that exists to stop a
timing being reported without green parity was checking 512-on-512 while the
timings it released were 1-on-65536, and the kernel was wrong on the second and
right on the first.** A code-enforced gate on the wrong shape family is worse
than no gate, because a repository cites the ones it enforces.

One more detail from the artifact, offered as an observation and not as proof.
All three configurations report the identical parity max difference, to every
digit: `0.008664369583129883`, at `tune_sm80_a100.json:4`, `:13` and `:22`. One
number, three different tile sizes. That is what a bf16 output quantum looks
like rather than a measurement that responds to what is being swept, and it was
sitting in the artifact the whole time.

## What the defect did on `sm_80`, and what is not established

The `sm_90` failure mode does not carry over unchanged, and it matters, so here
is the arithmetic rather than an analogy.

`sm_90` guards two-sided (`0 <= tile_idx < bias_num_tiles`), so a wrong shift
means the bias tile is **not fetched** and the score tile gets no bias. Omission.

The generic kernel's guard is **lower bound only**,
`if n_block >= bias_k_min_tile` (`flash_fwd.py:1264`), with
`bias_k_min_tile = -bias_tile_shift` (`:952`). When the defective shift came out
positive, `bias_k_min_tile` came out negative, the guard passed for every
`n_block`, and the copy at `:1267` fetched bias tile `n_block + shift` for every
KV block in the range. Line numbers here are as of 2026-07-25 and that file is
under concurrent edit, so anchor on the expressions rather than on the numbers.

At the swept decode shapes, `rel_extent = 1024` so `padded = 1280`
(`interface.py:730`). The correct shift comes from the writer's granularity,
which is 128 columns and not `tile_n` (see the next section):
`padded - 128 * n_block_max_128` columns, with `n_block_max_128 = 65536 / 128 =
512`, so `1280 - 65536 = -64256` columns.

| configuration | bias tiles | KV blocks at `T_k = 65536` | shift used | shift required | tiles read inside the bias domain |
|---|---|---|---|---|---|
| `tile_n=32` | 40 | 2048 | +36 | -2008 | `n_block` 0 to 3, so 4 of 2048 |
| `tile_n=64` | 20 | 1024 | +18 | -1004 | `n_block` 0 to 1, so 2 of 1024 |

Correct behaviour is that the newest 40 blocks of 2048 carry bias at
`tile_n=32`, and the newest 20 of 1024 at `tile_n=64`. What happened instead:
every KV block issued a bias tile copy, a handful of them landed inside the
40-tile or 20-tile bias domain and on the wrong band, and the rest addressed
past the end of that domain.

Two consequences follow, and both cut against the withdrawn numbers.

1. **This was not "the bias work was skipped".** On `sm_90` the two-sided guard
   made the defect cheap, which is why the withdrawn Hopper decode ratios
   flattered us. Here the copies all happened. The traffic was there and the
   addresses were wrong. So the direction the percentages would move under a
   correct kernel is not predictable from the `sm_90` story, and this file does
   not predict it.
2. **The two sides of the comparison were not doing the same wrong thing.** 4 of
   2048 reads landed in-domain at `tile_n=32` against 2 of 1024 at `tile_n=64`.
   A self-comparison between two configurations is only a fair comparison if both
   are doing the same work, and these were not.

**Not established, and not to be inferred from this file:** what fetching past
the end of the bias tile domain actually does. Whether it faults, returns zeros,
returns other memory, or is silently clamped by the copy atom is a hardware
question. No `sm_80` or `sm_120` GPU has ever run a `seqlen_q != seqlen_k`
relative-bias case. `regression-sm90-bias-shift.md` flagged the one-sided guard
and refused to conclude from reading; that refusal stands here.

Also not established: that `tile_n=32` is the wrong choice for decode. Nothing
here says the percentages were too high or too low. It says the evidence for
them does not stand.

## A blocking finding: the ported fix does not match the writer's contract

Found while writing this file, by reading. It is the reason the re-measurement
recipe below starts with a code question rather than a GPU booking. Two forms of
the ported fix exist in this branch's history, `9b63979` then `b5f0f7e`, and
neither is the contract in general. `b5f0f7e` is what ships now. What decides
between them, and what a correct third form would have to do, is stated at the end
of the section.

**Form A, committed in `9b63979`:**

```
flash_fwd.py, as committed
    bias_tile_shift = (
        padded_bias // self.tile_n
        - (128 * n_block_max) // self.tile_n
    )
```

`n_block_max` there is the one already in scope from `:844`, returned by
`block_info.get_n_block_min_max`. That function counts blocks in units of **its
own** `tile_n`: `n_block_max = ceil_div(seqlen_k, tile_n)` at
`tml-fa4/flash_attn/cute/block_info.py:63`. And `flash_fwd.py:832-841` builds its
`BlockInfo` with `self.tile_n`. So `128 * n_block_max` is the absolute KV column
end only when `self.tile_n == 128`.

The layout being read is not written at `tile_n` granularity. `ShearingBias`
builds its **own** `BlockInfo(128, 128, ...)` at
`tml-fa4/flash_attn/cute/shearing_bias.py:311-319` and computes its
`attn_n_block_max` from that at `:361`, so the shear is defined by a 128-column
KV block regardless of what the attention kernel later chooses for `tile_n`.
`128 * n_block_max_128` is the quantity the contract names, and
`128 * n_block_max_tile_n` is not it.

`sm_90` is safe. `interface.py:523-527` forces `tile_mn = (128, 128)` whenever
`rel_bias is not None` on arch 9, so there the two counts coincide and the
validated fix is correct as written.

The generic path never selects 128 for Inkling. `interface.py:558-561` picks 32
or 64 on arch 8. `interface.py:536-542` picks 64 on arch 12 for `head_dim > 64`,
and Inkling's head dim is 128. So on both architectures the generic kernel
actually serves, the ported fix is off by a factor of `128 / tile_n`.

Worked at the sweep's own parity shape, `T = 512`, `m_block = 3`, where the
writer's granularity gives `n_block_max_128 = 4` and the correct column shift is
`1280 - 512 = 768`:

| `tile_n` | contract tile shift | what form A computes | consequence, from reading only |
|---|---|---|---|
| 32 | 768 / 32 = **24** | `40 - (128*16)//32` = **-24** | `bias_k_min_tile` becomes +24, above every `n_block` in a 16-block range, so the lower-bound guard skips all of them and full prefill gets **no bias at all** |
| 64 | 768 / 64 = **12** | `20 - (128*8)//64` = **4** | shift wrong by 8 tiles, so the bias lands on the wrong band |
| 128 | 768 / 128 = **6** | `10 - (128*4)//128` = **6** | correct, and this is the `sm_90` case |

The specialisation that was removed, `128 * (m_block + 1)`, was `tile_n`
independent: with `tile_m = 128` it is the absolute row end, which at
`seqlen_q == seqlen_k` is the absolute column end at 128 granularity. That is why
`parity_fa4_rel.py` was 3/3 green on A100 and on the 5090 at `tile_n=64` before
the fix. Form A is not `tile_n` independent, so the prediction, from arithmetic
and not from silicon, is that form A would **fail** `parity_fa4_rel.py` on `sm_80`
and `sm_120` and that `tune_sm80.py` would suppress every timing it tried to
report. Form A's vestigial twin at `apply_rel_bias` has the same error, computing
`ceil_div(..., self.tile_n)` and multiplying by 128.

**Form B, committed in `b5f0f7e`** ("Fix a units error I introduced in the
shear-shift fix"), from a concurrent lane that found the same units error
independently. It supersedes form A at all three sites, including the `sm_90` one,
where it is a provable no-op because `tile_n == 128` is forced there:

```
    bias_tile_shift = padded_bias // self.tile_n - n_block_max
    shift = n_block * self.tile_n + padded - n_block_max * self.tile_n
```

That is the same quantity in key-index units: `n_block_max * tile_n`, the right
edge rounded **up to a multiple of `tile_n`**. It is a large improvement on form A
and it is still not the contract, because the writer rounds the right edge up to a
multiple of **128**.

**The invariant that decides it.** Let `x = min(seqlen_k, n_idx_right)`, the
unrounded right edge for the query block. The writer's column shift is
`padded - ceil_128(x)` and form B computes `padded - ceil_tile_n(x)`. For
`tile_n` dividing 128 those agree if and only if `ceil_tile_n(x)` is already a
multiple of 128, which for practical purposes means `x % 128 == 0`. Worked
counterexample, full prefill `seqlen_q = seqlen_k = 200` at `tile_n=32`,
`m_block=1`: the writer gives `attn_n_block_max = 2` so the shift is
`(1280 - 256)/32 = 32` tiles, the removed specialisation `128*(m_block+1)` also
gives 32, and form B gives `40 - ceil(200/32) = 40 - 7 = 33`. Form B is off by one
tile on a shape the original specialisation got right, and `200/200` is one of the
120 configurations the layout contract was brute-forced against
(`kernels/patches/u2_inkernel_shear.md`).

**No gate in this repository can tell the two forms apart.** Every gated shape has
`seqlen_k` a multiple of 128: `parity_fa4_rel.py` runs 128 and 1536 twice;
`parity_rel_chunked_decode.py` runs 1536, 1536, 1024, 2048, 4096, 4096, 1536;
`tune_sm80.py` runs 8192 and 65536. Real serving does not have that property,
because the paged path passes `seqused_k` from `cache_seqlens`, which is whatever
the sequence length happens to be.

**The contract form**, for whoever settles this: compute the block count at 128
granularity, not at `self.tile_n`, and keep the division by `tile_n` outside.
That is what the stock `sm_100` reader gets for free, because its bias index
offset is a difference of two block counts taken at the same granularity as the
writer's, and what `sm_90` gets for free, because `interface.py` forces
`tile_mn = (128, 128)` on that path.

Nothing in this section has been run. It is arithmetic against two source files,
the same class of evidence that found the original defect, and it is recorded
rather than acted on here because a kernel edit that no gate has exercised is
exactly what produced this incident twice already.

## What re-measurement requires

In order. Steps 1 and 2 are prerequisites, not formalities: skipping them gets
either a suppressed sweep or another number selected under a broken reader.

1. **Settle the granularity** in the generic reader's shift and in its vestigial
   twin, per the section above: the block count has to be taken at the writer's
   128 granularity, not at `self.tile_n`. Add a `seqlen_k % 128 != 0` case to a
   parity gate at the same time, because nothing in the repository currently
   distinguishes a correct shift from either of the two wrong ones.
2. **Run `harness/parity_fa4_rel.py` at each swept `tile_n`, not only at the
   default.** The previously published 3/3 was one configuration. A tile-size
   sweep needs the family it already claims to be correct on re-proved per
   configuration, or the sweep is again reporting timings for kernels no gate
   covered.
3. **`harness/parity_rel_chunked_decode.py`**, the first `seqlen_q != seqlen_k`
   gate this project has. It has run on `sm_90` only, 7 of 7, plus a broken
   control that the artifact records at **2 of 7**, one of the two passes being
   a defective case that slipped the then-current `TOL_MEAN` of 5e-3; under the
   tightened 5e-4 a replay of the same per-case numbers gives 1 of 7. On
   `sm_80` it has never run.
4. **The three gates written for families nothing has reached**, none of which
   has run on any GPU: `harness/parity_rel_bias_coverage.py`,
   `harness/parity_rel_paged.py`, `harness/parity_rel_varlen_batch.py`. The paged
   one matters most for serving, because paged KV is the only call shape vLLM
   ever makes and no gate has ever passed it.
5. **Re-run `harness/tune_sm80.py`.** Its `parity_ok()` now also covers the timed
   family, three cases with a real context at `:149-176`, reusing
   `parity_rel_chunked_decode`'s reference rather than duplicating it. Read that
   improvement precisely: those cases run `ctx` 4095 and 1408, **not** the 65535
   the timed cases use. `parity_rel_bias_coverage.py`'s docstring records the
   dilution that makes the difference matter, global causal at `rel_extent`
   1024 against `TOL_MEAN` 5e-4: 7.0x headroom at ctx 4095, 1.1x at ctx 32767,
   0.4x at ctx 65535, so an oracle gate at the timed depth cannot tell whether
   the bias arrived. A green `parity_ok()` now certifies the timed **family** and
   not the timed **depth**. The coverage probe, not the oracle, is the instrument
   for 64K.
6. **Re-derive the percentages from the new JSON.** Do not divide a new timing by
   a withdrawn one. Both sides come from the new run or there is no number.

**Cost: unpriced.** No A100 hourly rate exists anywhere in this repository to
quote. `scripts/grab_b200.py:35-37` prices B200 and H100 only;
`scripts/book_user_node.py:56` reads `price_cents_per_hour` from the provider at
launch and prints it, so the figure was never committed. Sessions 26 and 27 ran
on a founder-provided A100 node and recorded neither duration nor cost, which is
itself a gap against the session rule in
[docs/METHODOLOGY.md](../docs/METHODOLOGY.md#session-based-gpu-validation) step 8.
No estimate is invented here.

## The support claim survives, with one caveat it did not carry before

**Unchanged.** Inkling has a working attention kernel on Ampere and day-0 does
not. Every day-0 route raises `NotImplementedError` on SM8x, recorded with the
reproducer in
[upstream/05-no-sm8x-attention-path.md](upstream/05-no-sm8x-attention-path.md).
That is a capability claim, there is no baseline to divide by, and it never
rested on a tile percentage. Nothing in this file touches it.

**The caveat.** The parity evidence behind it is session 26's 3 of 3, max abs
diff 7.8e-3, 7.8e-3 and 1.56e-2 against a tolerance of 2e-2, and all three of
those cases are `seqlen_q == seqlen_k`. That is the family the defect got right.
So the honest form of the claim is: our kernel runs on Ampere, where nothing
else does, and it matches a float32 oracle **on the full-prefill family**.
Decode and chunked prefill on `sm_80` have no correctness result on any
hardware, and after the finding two sections up, full prefill on `sm_80` has no
correctness result for the code currently in the tree either. Both gaps close
with the same A100 session.

## Lessons

1. **A code-enforced parity gate is only as good as its shape coverage, and
   citing it makes the failure worse.** `tune_sm80.py` was named in
   CONTRIBUTING.md and in the README as the one place a rule was mechanical
   rather than cultural. It was mechanical. It was also checking 512-on-512 and
   releasing timings for 1-on-65536.
2. **When a timing harness and its gate live in the same file, one function
   should build both sets of shapes.** `make_case()` at `:54` and `parity_ok()`
   at `:88` constructed their tensors independently, and that is precisely where
   the two shape families diverged. Deriving the parity cases from `CASES` would
   have made the divergence impossible to write.
3. **Porting a fix is not the same as making the general form general.** The
   `sm_90` fix was derived, reviewed and validated on hardware. The same edit
   moved into the generic kernel introduced a new specialisation, `tile_n == 128`,
   in the same line that removed the old one. The second reading caught it. Only
   a GPU can confirm it.
4. **An identical gate number across every swept configuration is a signal.**
   Three tile sizes, one parity max difference to sixteen digits, sitting in the
   published artifact.
5. **Withdrawing is cheaper than defending.** These percentages were never the
   headline of this project, and the support claim they sat next to survives
   intact. The cost of withdrawing them is one section in a journal file. The
   cost of leaving them quotable is every other number in the repository.
