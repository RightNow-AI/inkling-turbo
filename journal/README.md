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

The end of session 28 records a serving benchmark that was lost to a watchdog
race. That was our own orchestration bug. It is written up rather than quietly
retried, and the affected ledger rows are still `null`.

## Rules this journal follows

- A number appears here only if an artifact under [remote/](remote/) produced it.
- A timing from a kernel that failed parity is recorded as a failed kernel, never
  as a win.
- When a later session invalidates an earlier interpretation, the correction is
  appended and the original is left in place.
- Cost, capacity lost to our own bugs, and sessions that produced nothing are
  written down at the same size as the wins.

The full evidence rules are in [docs/METHODOLOGY.md](../docs/METHODOLOGY.md).
