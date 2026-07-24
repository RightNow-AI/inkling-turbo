# Remote artifacts

Raw output from GPU sessions. Nothing here is edited after the fact. If a run
failed, its artifact records the failure.

Every number in [README.md](../../README.md) and [LEDGER.md](../../LEDGER.md)
should be traceable to a file in this directory. If you find one that is not,
that is a bug and worth an issue.

## Naming, and one trap

The `b200_first_contact_*.log` files were produced by `scripts/grab_b200.py`,
which was written to hunt B200 capacity. **No B200 ever became available.** Every
one of these logs is an H100 session. The filename is the launcher's, kept so
each log traces to the run that wrote it, but do not read `b200` in a filename as
a claim about the hardware. Check the `=== GPU ===` line at the top of each log,
which is the actual device.

`microbench_attn_day0_*.json` is also easy to misread. That harness measures
whatever `vllm.third_party.tml_fa4` resolves to at runtime. Against a stock
checkout on `sm_90` it measures a path that silently drops the bias, which is
upstream finding 01. Against a deployed `kernels/tml_fa4_modified/` it measures
our kernel. The filename does not tell you which; the session column below does.

## Measurement artifacts

| File | Hardware | What it is | Backs |
|---|---|---|---|
| `microbench_attn_day0_session25_h100.json` | 1x H100 SXM5 | Our kernel, all cases | The "Ours" column of the README latency table |
| `microbench_attn_scoremod_session25_h100.json` | 1x H100 SXM5 | The day-0 `score_mod` baseline and biasless attention, same shapes, same box, same run. Also `relproj` and `relprojT`, which are OUR abandoned prototypes and not baselines. | The `score_mod` column of that table |
| `microbench_attn_day0_native_sm90_session24.json` | 1x H100 SXM5 | Our kernel, session 24, torch cu129 | The first native `sm_90` result, superseded as the headline by session 25 |
| `microbench_attn_scoremod_session24.json` | 1x H100 SXM5 | Day-0 baselines, session 24 | The session-24 comparison |
| `microbench_attn_scoremod_gpu_1x_h100_sxm5.json` | 1x H100 SXM5 | Byte-identical duplicate of the line above | Nothing. Redundant copy. |
| `tune_sm80_a100.json` | A100 SXM4 40GB | Parity-gated tile sweep, three configurations | The Ampere tile-tuning result |
| `gate_logit_parity_8xh100.json` | 8x H100, TP8 | Full-model gate: 32 prompts, stock build against ours, plus both same-build controls | The 32/32 token match, and the logprob gate recorded as a fail |
| `microbench_attn_day0_gpu_1x_h100_sxm5.json` | 1x H100 SXM5 | Session 1. Every attention case errored on a toolchain break; only `gate_select` produced a number | Nothing about attention. Kept as the record of the first drift failure. |
| `microbench_attn_day0_sm100.json` | 1x H100 SXM5, **not sm_100** | Session 3, the `rel_bias` path before we knew it was silently returning plain attention | **Nothing. These numbers are retracted.** The correction is in [u2-hopper-design.md](../u2-hopper-design.md#correction-2026-07-18-post-static-analysis-supersedes-design-a). The filename is wrong and predates the correction. |

## Session logs

Each log is one paid GPU session, timestamped `YYYYMMDD_HHMM` UTC at retrieval.
They are not duplicates of each other. The parity numbers drift between them, and
that drift is the debugging record: watch `tml_fa4_rel_bias` max_diff fall from
2.4 through 1.7 to 1.1 across the sessions of 2026-07-20 as wrong hypotheses got
eliminated, then read session 24 in the journal for what actually caused it.

| Logs | Session | Content |
|---|---|---|
| `20260718_1155` | H100 session 1 | First contact. Pipeline proven end to end. Attention cases fail on the CuTe DSL drift. |
| `20260718_1209` | H100 session 2 | Drift fixes applied. Fourth drift found. |
| `20260718_1222` | H100 session 3 | tml-fa4 direct path runs. Parity catches it returning wrong output. |
| `20260718_1237` | H100 session 4 | The honest `score_mod` baseline, and the 3.2x headroom measurement that motivated the whole unit. |
| `20260720_1320` | none | Four lines. A `set: pipefail` error killed the bootstrap before the GPU was touched. No evidence in it. |
| `20260720_1331` through `20260720_1857` | sessions 5 to 22 | The `sm_90` bias debugging flights. One log per hypothesis. |
| `20260720_2026` | session 23 | `sm_90` correctness via generic routing, parity 3/3, plus U3 FP8 KV parity 2/2. |
| `20260723_1052` | session 25, first attempt | Upstream regenerated its wheel bucket and deleted the pinned build. Install died. This is the environment-drift evidence behind the pinned-by-URL recipe in `scripts/bootstrap_b200.sh`. |
| `grab_b200_detached.log` | all | The launcher's own log across the capacity hunt. |

## Session records

- [h100-session1.md](h100-session1.md) covers sessions 1 through 4 in prose,
  including the post-session correction that retracts session 3's timings.
- [RUNBOOK.md](RUNBOOK.md) is the operational procedure for a remote session.
- Sessions 5 onward are recorded in
  [u2-hopper-design.md](../u2-hopper-design.md).

## Not in the repo

Nsight Compute reports (`.ncu-rep`) are gitignored. They are several megabytes
each. The metrics read off them are quoted in the session 24 ncu entry of
[u2-hopper-design.md](../u2-hopper-design.md), and `scripts/bootstrap_8x.sh`
regenerates them.
