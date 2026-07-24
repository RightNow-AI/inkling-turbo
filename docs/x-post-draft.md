# X post draft (release announcement)

## Main post

Introducing Inkling-turbo: open-source GPU kernels that make TML's Inkling
model dramatically faster to serve on vLLM.

Inkling's attention is nonstandard (no RoPE, learned relative-position bias,
alternating global/SWA). The day-0 stack handles it with a per-score callback.
We rebuilt it as a tile-level sheared-bias kernel.

Measured, same machine, same harness, parity-gated:

- H100: 2.7x to 8.4x faster than every day-0 production variant at batch-1
  64K-context decode. 3.2x to 4.6x at 8K prefill. Reproduced on two
  independent software stacks.
- A100: the day-0 path cannot run at all (score_mod is hard-blocked on SM8x
  upstream). Ours passes full parity. If you want Inkling attention on
  Ampere, this repo is currently the only way.
- RTX 5090: full parity, faster than the day-0 path on every case.

Every number in the repo is measured-or-null. Parity gates before timings,
fast-and-wrong counts as failure, and the journal documents every dead end
including 17 failed debug sessions and the pack_gqa root cause that
explained all of them.

B200/Blackwell numbers are next: the kernels already dispatch to sm_100,
and our capacity hunters are standing by to run the full architecture-local
gates plus end-to-end serving benchmarks the moment 8x B200 nodes free up.

Repo: github.com/jaberjaber23/Inkling-turbo

## Follow-up thread material (optional)

1. The bug that ate 17 debug sessions: pack_gqa packs 8 GQA query heads
   into the score-tile rows, so "row" never meant "sequence position" on
   sm_90. Every coordinate-based bias scheme was doomed before it started.
   The fix: partition the bias tile with the SAME partitioner that produces
   the accumulator. Zero coordinate arithmetic, cannot disagree with itself.

2. We found 5 upstream drift breaks along the way (cutlass 4.6 renames, an
   nvvm branch keyed to the wrong version signal, a regenerated wheel bucket
   that deleted cu12x wheels). All documented with repros in the repo,
   filing-ready.

3. Honest limits, in the README: no end-to-end serving claims yet, batch-1
   decode is parallelism-bound pending split-KV, and Blackwell is pending
   hardware. The ledger has a $18 entry for the B200 we lost to a Windows
   text-encoding bug in our own launcher. Keeping that in is the point.
