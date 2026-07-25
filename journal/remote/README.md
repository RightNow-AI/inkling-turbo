# Remote artifacts

Raw output from GPU sessions. Nothing here is edited after the fact. If a run
failed, its artifact records the failure.

Most numbers in [README.md](../../README.md) and [LEDGER.md](../../LEDGER.md)
trace to a file in this directory. Five do not, and it is better to name them
than to state a rule that is false:

- the Nsight Compute percentages, because the `.ncu-rep` files are gitignored;
- the 18.7% post-deploy A100 re-run, journal session 27, no separate JSON.
  **Withdrawn 2026-07-25** along with the rest of the Ampere tile-tuning
  percentages, and kept on this list because a withdrawn number is not a deleted
  one: [regression-ampere-tile-sweep.md](../regression-ampere-tile-sweep.md);
- the `sm_120` relative timings, which were local and never packaged;
- the 8x H100 memory recipe and its 0.77GB-per-0.01 sensitivity, session 28
  prose;
- the `max 1.56e-2` per-op parity figure in LEDGER.md, journal session 25.
  `harness/parity_fa4_rel.py` prints its max error and writes no artifact, so
  there is nothing here to re-parse.

Those are journal-only: a number read off a tool and written down, with no
machine-readable record to re-parse. Everything else is here.

This list is the canonical one. No other document restates how many there are,
because every time one did, the count drifted. The **journal-only** label at the
point of use is the authority; `grep -rn "journal-only" LEDGER.md README.md` finds
them. If you find a number that is neither here nor labelled, that is a bug and
worth an issue.

Also journal-only, and not in the list above because it is a gate result rather
than a published number: the `sm_120` 16/16 run of
`harness/parity_shear_fusion.py`. Its only record is commit `7375849`; that run
predates the harness writing a file. The **sm_90** run of the same gate is strong
form, in `validate_s26_h100x1/`, and it scores 14/16 rather than 16/16.

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
| `validate_s27_decodefix/` | 1x H100 80GB HBM3 | The post-fix run: `parity_rel_chunked_decode` 7/7, `parity_fa4_rel` 3/3, and our kernel against the day-0 `score_mod` baseline in one container | **Every row of the README latency table.** The decode ratios 2.66x / 2.75x / 2.10x, the prefill 1.44x, and the 1.27x sliding-window loss |
| `validate_s27_brokencontrol/` | 1x H100 80GB HBM3 | The same payload with the `128*(m_block+1)` shift deliberately restored, so the new gate could be observed failing on the defect it was written for | That the gate is a gate, the tolerance choice (`TOL_MEAN` 5e-4), and the cost of correct decode bias measured as broken against fixed |
| `microbench_attn_day0_session25_h100.json` | 1x H100 SXM5 | Our kernel, all cases, **before the bias-shift fix** | The prefill rows of the README latency table. Its decode rows are **withdrawn**: the kernel was applying the bias to one KV block instead of ten. See [regression-sm90-bias-shift.md](../regression-sm90-bias-shift.md) |
| `microbench_attn_scoremod_session25_h100.json` | 1x H100 SXM5 | The day-0 `score_mod` baseline and biasless attention, same shapes, same box, same run. Also `relproj` and `relprojT`, which are OUR abandoned prototypes and not baselines. | The `score_mod` side of that comparison. The baseline itself was never defective; what is withdrawn is dividing by it |
| `microbench_attn_day0_native_sm90_session24.json` | 1x H100 SXM5 | Our kernel, session 24, torch cu129 | The first native `sm_90` result, superseded as the headline by session 25 |
| `microbench_attn_scoremod_session24.json` | 1x H100 SXM5 | Day-0 baselines, session 24 | The session-24 comparison |
| `microbench_attn_scoremod_gpu_1x_h100_sxm5.json` | 1x H100 SXM5 | Byte-identical duplicate of the line above | Nothing. Redundant copy. |
| `tune_sm80_a100.json` | A100 SXM4 40GB | Tile sweep, three configurations. Parity-gated, but the gate ran `seqlen_q == seqlen_k` while the decode cases were timed at `T_q=1` against `T_k=65536`, and the generic kernel was wrong on the second family | The **decode** percentages it backed are **withdrawn**: [regression-ampere-tile-sweep.md](../regression-ampere-tile-sweep.md). It still backs the two prefill rows, which are 8192-on-8192 and unaffected, and the `tile_n=128` collapse. Note that all three configurations report the identical parity max diff to sixteen digits, `:4`, `:13`, `:22` |
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
[u2-hopper-design.md](../u2-hopper-design.md), and that transcription is the
only form the evidence ships in.

**No script in this repository regenerates them.** The profiling was done by
hand on the session-24 box, against `harness/microbench_attn_day0.py`, and the
invocation was never committed. An earlier version of this file and of the
README claimed `scripts/bootstrap_8x.sh` regenerates them. It does not; that
script contains no `ncu` invocation. The claim was wrong and is corrected here
rather than deleted. Committing a CSV export of the section summaries is the
open item.
