# Journal

This is the working record, kept in the order it happened. It includes the dead
ends, the wrong hypotheses, and the corrections. Entries are appended, not
rewritten, so a claim made on day two that turned out to be wrong on day four is
still there with the correction under it.

If you only want the result, read [README.md](../README.md). If you want to know
whether to trust the result, read this.

## Where to start

| File | What it is |
|---|---|
| [phase0.md](phase0.md) | The model configuration, pulled from primary sources. Hidden dims, head counts, the R projection, the global and sliding-window layer pattern. Every shape used anywhere else traces here. |
| [day0-implementation.md](day0-implementation.md) | A study of what vLLM shipped on day zero, read from the code rather than the blog post. This is the baseline the whole project is measured against. |
| [u2-hopper-design.md](u2-hopper-design.md) | The main record. Design, seventeen debugging sessions, the `pack_gqa` root cause, and every measured result. Long, and the interesting part is the failures. |
| [local-tier-bringup.md](local-tier-bringup.md) | Getting the day-0 path to run at all on `sm_120`, and the three upstream bugs that blocked it. |
| [remote/](remote/) | Raw artifacts from GPU sessions. See [remote/README.md](remote/README.md) for what each file backs. |
| [upstream/](upstream/) | Bug reports written against vLLM and tml-fa4. Drafted, dup-checked, not yet filed. |

## Reading u2-hopper-design.md

It is roughly 30 appended entries. The shape of the story:

1. **Design A and B.** Two candidate mechanisms, ranked by expected win.
2. **The correction.** Static analysis kills Design A: the `sm_90` kernel has no
   bias support at all, so there was no layout bug to fix. It was silently
   returning plain attention.
3. **V1, V1.5, v0.** Three measured dead ends. All parity green, all slower than
   the thing they replaced. Each one narrowed the cause: the cost is the
   per-element callback itself, not memory locality.
4. **The layout contract.** The shear writer's mapping, extracted by machine
   rather than derived by hand, and verified on 20,100 positions.
5. **Sessions 5 to 23.** The `sm_90` port. Correct on paper, wrong on silicon,
   seventeen times. Several plausible root causes ruled out by probe.
6. **Session 24.** `pack_gqa` was packing eight query heads into the rows of the
   score tile, so a row was never a sequence position. Every coordinate scheme
   had been wrong before it started. Parity 3/3 the same day.
7. **Sessions 25 to 28.** Reproduction on a second machine and a different
   software stack, A100 support, tile tuning, and the full-model gate on 8x H100.
   Session 27's tile-tuning percentages are **withdrawn**, and later refuted by
   measurement; the A100 support result is not. That result was narrowed to
   single-sequence calls on 2026-07-25 and then widened again the same day by
   sessions 31 and 32. [Why](regression-ampere-tile-sweep.md).
8. **Session 29, local and free.** The never-executed gates were run on the local
   `sm_120` 5090. Four came back green, which is the first time the generic
   kernel's corrected shear shift has run on any silicon, and the fifth found an
   illegal memory access on multi-sequence varlen batches. That is the production
   call shape, so it narrowed a published claim.
   [Write-up](regression-sm120-varlen-illegal-address.md).
9. **Session 30, local and free.** The `pack_gqa` shear-granularity defect,
   root-caused and fixed. The shear writer was packing GQA heads into its row
   dimension while the generic reader was not, so the two evaluated the same
   correct contract at two different `m_block` granularities. 14 of 29 measured
   shapes were wrong, including the production `Hq=64` over `Hkv=8` geometry, and
   the whole gate suite caught one of the 14. Multi-sequence varlen went 11/12 to
   **12/12**. [Write-up](regression-pack-gqa-shear-granularity.md),
   [artifacts](remote/local_sm120_s30/).
10. **Sessions 31 and 32, one A100 each, $0.99 and about $1.** The first
    correctness results on Ampere for anything other than single-sequence full
    prefill. Session 31 measured chunked prefill and decode at **7 of 7** with
    6.7x to 37.7x headroom, and observed the `pack_gqa` defect there. Session 32
    ran the fix: **12 of 12**. Session 31 also re-ran the tile sweep and
    **refuted** the withdrawn Ampere percentages rather than restoring them: the
    same configuration moved by up to 27.6% between runs while the configurations
    differ by at most 7.2%. [Artifacts](remote/validate_a100x1_s31/).

Four regression write-ups sit outside that sequence and are the shortest route to
how this project checks itself: [regression-sm90-bias-shift.md](regression-sm90-bias-shift.md),
the shear-shift defect and the three gates that missed it,
[regression-ampere-tile-sweep.md](regression-ampere-tile-sweep.md), what the same
defect did to the one set of numbers measured off the generic kernel and how a
repeat measurement then refuted them,
[regression-sm120-varlen-illegal-address.md](regression-sm120-varlen-illegal-address.md),
a crash on the production call shape that cost no number and narrowed a claim, and
[regression-pack-gqa-shear-granularity.md](regression-pack-gqa-shear-granularity.md),
the third member of the shear family and the one that reached real serving
geometry.

Read them in that order if you want the actual lesson, which is not about shear
layouts. Each of the four was found by a gate that did not exist the day before,
and each of those gates was written because the previous one turned out to be
aimed at the wrong shape.

The end of session 28 records a serving benchmark that was lost to a watchdog
race. That was our own orchestration bug. It is written up rather than quietly
retried, and the affected ledger rows are still `null`.

## Rules this journal follows

- A number appears here only if a tool produced it, and it says which class of
  record backs it. Most trace to an artifact under [remote/](remote/). The ones
  that do not are transcriptions with nothing re-parseable behind them, they are
  labelled **journal-only** at the point of use, and
  [remote/README.md](remote/README.md) holds the canonical list of them. An
  earlier version of this line claimed every number here has an artifact, which
  was not true.
- A timing from a kernel that failed parity is recorded as a failed kernel, never
  as a win.
- When a later session invalidates an earlier interpretation, the correction is
  appended and the original is left in place.
- Cost, capacity lost to our own bugs, and sessions that produced nothing are
  written down at the same size as the wins.

The full evidence rules are in [docs/METHODOLOGY.md](../docs/METHODOLOGY.md).
