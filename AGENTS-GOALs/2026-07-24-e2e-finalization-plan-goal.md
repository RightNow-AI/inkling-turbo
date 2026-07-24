# 2026-07-24 — E2E finalization plan (execute when bench completes)

STANDING INSTRUCTION FROM FOUNDER: when the e2e benchmark finishes, document
everything and organize the whole repo fully, end to end.

## Current state (at plan time)
- 8x H100 gates box: 192.222.52.63, instance d73d4a49aa844759b5cd26e19d34732a
- e2e bench PID on box, launched standalone (gate_e2e_bench.sh), idempotent,
  `for BUILD in stock ours`. Does NOT auto-run summarizer (prints "next:").
- Serving recipe (proven): util 0.94, ctx 3072, --enforce-eager,
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, LD_LIBRARY_PATH compat.
- Logit gate DONE + banked (session 28, commit b267765): 32/32 greedy-token
  match stock vs ours; logprob tol fail-as-specified with controls also
  failing (platform nondeterminism documented).
- Main @ b267765, repo PRIVATE.

## Completion trigger
Bench process ends (monitor b9snspnxb watches). Verify all 60 run files:
`find ~/bench_results -name run*.json | wc -l` should be 60
(2 builds x 2 mixes x 3 conc x 5 runs).

## Finalization steps (in order)
1. On box: run `python ~/gate_summarize.py` -> ~/gate_summary.md.
2. Pull home: gate_summary.md -> journal/remote/, and the full
   bench_results/ tree -> journal/remote/bench_results_8xh100/.
3. Parse each build/mix/conc: median-of-5 + best for request_throughput,
   output_throughput, median_ttft_ms, median_tpot_ms, p99. Compute
   ours-vs-stock deltas. MEASURED-OR-NULL: any missing/failed run = null,
   never fabricated.
4. LEDGER.md: fill the 12 null e2e rows with the measured stock/ours/delta.
   Add the spend row for this 8x H100 session (hours x $31.92).
5. README.md: add an "End-to-end serving" headline row with the honest
   result (throughput delta at each concurrency, TTFT/TPOT), scope-limited
   to "8x H100, ctx 3072, W4A16 emulated NVFP4, per the memory recipe".
   If ours ~= stock (attention is a fraction of e2e), say so plainly - the
   e2e delta will be SMALL because MoE/GEMM dominate; the win is in the
   attention microbench + the sm_80 exclusivity, not necessarily e2e tok/s.
6. journal/u2-hopper-design.md: SESSION 29 with the full table + methodology
   (median-of-5, both mixes, same box/checkpoint/SLO, warnings noted:
   /tokenize skip, temperature default - both applied equally to both builds).
7. docs/METHODOLOGY.md: confirm the e2e section matches what was actually run.
8. Organize repo: ensure journal/, harness/, kernels/, scripts/, docs/ are
   coherent; remove stale scratch; verify .gitignore covers grab logs;
   final README pass so an NVIDIA/vLLM maintainer finds nothing to correct.
9. Commit + push. Then terminate the 8x box (stop billing) UNLESS the
   split-KV kernel still needs validation (it produced no code - skip).
10. Report to founder with the table; repo flips PUBLIC on their word only.

## Honesty guardrails
- e2e delta may be small/neutral; report it truthfully. The release story is
  already strong on attention microbench (2.7-8.4x), sm_80 exclusivity, and
  full-model token parity. Do NOT inflate e2e.
- If OOM/failures truncated any config, report which and leave null.
- Terminate the box after; do not leave it billing for finalization writing.
