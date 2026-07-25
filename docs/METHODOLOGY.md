# Methodology

Inkling-turbo treats correctness, performance, and deployment scope as separate gates. A result can pass one and fail another. Public claims include the artifact that establishes each gate and the hardware context in which it was measured.

## Claim classes

Not every contribution in this repository is a speed number, and the classes are not interchangeable. Each one has a different thing that has to exist before it can be stated.

| Claim class | Example | What establishes it | What it may not be turned into |
|---|---|---|---|
| Correctness | The kernel matches the float32 oracle on all three semantic cases | A parity run with recorded max and mean error, on the target architecture | Evidence that the kernel is fast |
| Capability | Inkling attention runs on `sm_80`, on single-sequence calls | A green parity run on that hardware, plus a recorded failure of every day-0 route on the same box | A speedup, when no baseline executes. **Nor a serving claim, when the parity run passes one sequence and serving batches many** |
| Defect | `rel_bias` is accepted and dropped on non-Blackwell arches | A reproducer, the file and line in a pinned tree, and an observation that separates the defect from a plausible alternative explanation | A claim about severity in production without a production observation |
| Deployment | This configuration serves the full checkpoint | A serving run that produced real tokens, plus the recorded failure of each rejected configuration | A performance result |
| Performance | 1.45x against the day-0 path at 8K global prefill | A timing artifact, the baseline that the real integration actually selects, and a green parity run for the same build **on the same shape family as the timing** | An end-to-end serving claim, or a claim about a shape family the parity run did not cover |

A capability claim is the strongest available statement on an architecture where no baseline runs, and it is also the weakest thing to overstate, because the temptation is to divide by something. On `sm_80` there is nothing to divide by. The correct statement is support, and the absolute timings are published with no ratio attached.

A capability claim also has a call-shape scope, which this project got wrong until 2026-07-25 and states here so the mistake is not repeatable. Every parity case behind the `sm_80` support claim used to pass a single `cu_seqlens` pair describing **one** sequence, while vLLM serving batches sequences into one varlen call on every step. A capability claim whose evidence is a call shape the integration never makes is a claim about the kernel and not about the deployment, and the two must be worded apart.

Both defects that hid in that gap are now fixed and the scope is narrower than it was. On `sm_80`, verified capability (8,0), multi-sequence varlen batching scores 11 of 12 and the illegal memory access does not reproduce ([validate_a100x1_s31](../journal/remote/validate_a100x1_s31/)); the twelfth case is a `pack_gqa` shear-granularity defect that reaches the production `Hq=64` over `Hkv=8` geometry, fixed to 12 of 12 on `sm_120` ([local_sm120_s30](../journal/remote/local_sm120_s30/)) and not yet re-run on Ampere. So the wording that survives is: our kernel is the only one that runs on Ampere, its parity evidence there now spans full prefill, chunked prefill, decode and bias extent, and multi-sequence batching is one case short of complete. [The crash](../journal/regression-sm120-varlen-illegal-address.md), [the shear defect](../journal/regression-pack-gqa-shear-granularity.md).

A defect claim carries an extra burden that a performance claim does not: it accuses someone else's code. Each of the five upstream findings therefore names the pinned commit, quotes the code, gives a runnable reproduction, and states what was observed rather than what was inferred. Where the evidence is static analysis only, the report says so. Finding 01 records that the measured silent-wrong-output evidence is `sm_90` only, and that the same static defect on `sm_80` and `sm_120` surfaces as a loud assertion failure first.

## Parity-oracle discipline

The parity oracle is an explicit PyTorch implementation of Inkling relative attention. It computes scaled QK scores, applies the learned relative-position value only when the causal distance is inside the configured extent, applies the global or sliding-window mask, then computes softmax and the value projection. The semantics were derived from the day-0 vLLM path and the sheared-bias writer, and are recorded in [the implementation study](../journal/day0-implementation.md#attention-path-nvidiaattentionpy-opsfa4_rel_attentionpy-u2u3u5-baseline).

Kernel and baseline backends consume inputs derived from the same seeded tensors. The current per-op suite covers three distinct semantic cases:

- a short global-attention case;
- a global case that crosses the relative-bias extent;
- a sliding-window case.

The local U2 gate is green only when all 3 cases pass the configured tolerance. The published `sm_120` result was 3/3 with maximum differences in the same range as the day-0 `score_mod` backend. [Recorded result](../journal/u2-hopper-design.md#v0-final-verdict-2026-07-19-parity-33-green-speed-failed-expected)

All three of those cases pass `cu_seqlens_q == cu_seqlens_k`, so that suite establishes the full-prefill family and nothing else. Rule 7 below exists because that limit was not stated for months and the suite was cited as though it covered the kernel. `harness/parity_rel_chunked_decode.py` covers the missing family, seven cases with `seqlen_k > seqlen_q` plus a `seqlen_q == seqlen_k` control. It passes 7 of 7 on `sm_90`, and it has been run against a deliberately broken kernel as well, which is the requirement in rule 9. On 2026-07-25 it also passed 7 of 7 on `sm_120`, with every case carrying a signal 7.5x to 37.8x above tolerance, which is the first `seqlen_q != seqlen_k` correctness result the generic kernel has ever had on any silicon. Those figures were **journal-only** at first, because the JSON that session committed is a zero-byte file; they are strong class from [local_sm120_s30](../journal/remote/local_sm120_s30/) onward, which re-ran the gates on the same laptop and wrote files that parse. Later the same day it passed **7 of 7 on `sm_80`** as well, on a Modal A100 with capability (8,0) asserted, each case carrying 6.7x to 37.7x headroom over `TOL_MEAN` ([validate_a100x1_s31](../journal/remote/validate_a100x1_s31/)), so decode-shape correctness now exists on Ampere. Every case in this gate, on all three architectures, still passes one sequence; `harness/parity_rel_varlen_batch.py` is the multi-sequence one.

The full integration gate is stricter than the per-op gate. It requires the fixed prompt suite and batched-output versus batch-1 consistency before a kernel is considered complete at the model level. That gate ran for U2 on 8x H100 against the real checkpoint. The token half passed 32/32, **on echoed prompt positions**: the stock-against-ours comparison ran `max_tokens=0` with `echo=True`, so it generated no tokens and exercised no decode call, which is a limitation of the gate rather than of the run and is recorded under rule 8. Scope that claim to the cross-build comparison and no wider: the batch-consistency controls in the same script did generate 32 tokens per prompt, 128 of their 348 compared positions per build, but they compare a build against itself, so no generated token was ever compared across builds. The two halves therefore do not even share a position mix, which is a second defect in that gate and not a refinement of the first. The logprob half is recorded as a fail, because the a-priori tolerance turned out to be tighter than the platform's own batch reproducibility, and the same-build control failed too. The control failure is reported, not waived. Fields that no gate has filled stay `null` in the [kernel ledger](../LEDGER.md#kernel-gates-ncu-90-of-binding-roofline).

Rules:

1. Compare against the oracle before comparing speed.
2. Use the same tensors, masks, scales, and output dtype for every backend.
3. Record maximum and mean error, not only a pass label.
4. Treat a skipped backend, compile failure, non-finite result, or tolerance failure as not green.
5. Do not use timings from a wrong-output path as evidence for an optimization.
6. When the oracle catches a defect, preserve the failing case and root-cause record. The silent non-Blackwell `rel_bias` omission is the canonical example. [Issue draft](../journal/upstream/01-rel-bias-silently-ignored-non-blackwell.md)
7. **A parity suite must cover every shape family the kernel dispatches to, and in particular it must contain cases where `seqlen_q != seqlen_k`.** A suite that tests one family certifies one family. Any shape whose index arithmetic differs is a separate family: full prefill, chunked prefill, decode, split-KV, and any windowing that shifts the diagonal. A timing taken on a family no oracle case covers is not admissible evidence, however green the suite is.
8. A number that no gate can contradict is not evidence. Before trusting a gate, state what result would have failed it. If the answer is "none", the gate is a description of the setup rather than a check on the kernel. **Where the gate compares against an oracle, this is measurable and not only arguable: record the margin between the signal the gate reads and the tolerance it reads it against, per case, and treat a margin below 1x as a failed gate rather than a passing one.** A gate that prints its own margin cannot silently become a tautology. [This has now been measured](#rule-8-has-now-been-measured-and-a-gate-failed-it-while-passing).
9. **A new gate is run against a deliberately broken kernel once, before its passing counts.** Restore the defect it was written for, confirm it fails, and keep that run as an artifact next to the passing one. A gate only ever seen passing has an unmeasured false-negative rate, and tolerances chosen without a defective sample are guesses.

### Rule 7 was learned the expensive way

On 2026-07-25 the `sm_90` kernel was found to have computed its sheared-bias
shift from `128 * (m_block + 1)`, the `seqlen_q == seqlen_k` specialisation of
`128 * n_block_max`. Every decode and chunked-prefill call therefore applied
Inkling's learned relative-position term to one KV block instead of ten, and to
the oldest one. It survived because all three cases in `harness/parity_fa4_rel.py`
pass `cu_seqlens_q == cu_seqlens_k`, so the entire suite lived inside the one
shape family the defect got right, and 3/3 green was read as a statement about
the kernel rather than about that family. The decode speedup ratios published off
the back of it are withdrawn, and the like-for-like re-measurement that replaced
them came in lower. They stay withdrawn rather than quietly corrected, because
the objection to them was never that they were imprecise.

Rule 9 comes from the same incident, and was paid for immediately: **a gate is
not known to be a gate until it has been observed failing on the defect it exists
for.** The new chunked-and-decode suite was run twice on an H100, once against
the fix and once against a kernel with the defect deliberately restored, and the
second run is what made the first one mean anything. It also found that the
suite's original mean tolerance certified one defective case as correct, which no
number of passing runs would ever have revealed.

Rule 8 comes from the same incident. `scripts/gate_logit_parity.py` reported
2369 matching tokens between two builds while running `max_tokens=0` with
`echo=True`, and its own docstring notes that a token mismatch in an echoed
prompt is impossible with one tokenizer. The gate could not have failed, so its
passing carried no information, and it was cited as evidence anyway. The full
account is in [journal/regression-sm90-bias-shift.md](../journal/regression-sm90-bias-shift.md).

### Rule 8 has now been measured, and a gate failed it while passing

The statement of rule 8 above is a piece of reasoning. It is correct and it is
derived: someone read `gate_logit_parity.py` and worked out that it could not
fail. On 2026-07-25 a gate reported the same property about itself as a number,
which is a sharper form of the rule and the form to prefer.

`harness/parity_rel_bias_coverage.py` exists to answer one question: does every
KV block that should receive a bias tile actually receive one, at depth. It ran
for the first time on 2026-07-25, on a local RTX 5090 Laptop (`sm_120`), and
scored 6 of 6. The 6 of 6 is not the finding.

**At production decode geometry that gate's oracle comparison is blind to a
completely dropped bias.** The separation it reads comes out at **0.5x of its own
tolerance**, so a kernel applying no relative-position bias whatsoever would have
passed that comparison. Under rule 8 that half of the gate is a description of
the setup and not a check on the kernel, and it says so with a measured ratio
rather than by inspection.

The half that discriminates is the probe that walks individual relative distances
and asserts that each one moves the output. It confirmed **13 of 13 distances
move it**, and it named the KV tiles touched at 64K as **`[504..511]`**, which are
the newest blocks and are exactly what the corrected shear shift is supposed to
select. So the gate is worth keeping; what is not worth trusting is its aggregate
oracle comparison at that geometry.

The arithmetic behind the blindness is the same arithmetic that let the
`128 * (m_block + 1)` defect survive three green gates: at 64K decode a single
bias tile out of 512 moves an aggregate error statistic by less than the bf16
quantum, so any tolerance loose enough to accommodate bf16 is also loose enough
to accommodate the bias being absent. **A gate must report its per-case margin,
and a margin below 1x is a failed gate rather than a passing one.**
`parity_rel_chunked_decode` on the same hardware printed 7.5x to 37.8x per case,
which is what the passing side of this rule looks like.

These figures are **journal-only**. The session committed
`journal/remote/local_sm120_s29/parity_rel_bias_coverage_sm120.json` as a
zero-byte file, so the 0.5x, the 13 of 13 and the `[504..511]` are transcribed
from [the session
write-up](../journal/regression-sm120-varlen-illegal-address.md) with nothing
machine-readable behind them. A rule about measurable margins whose own evidence
is a transcription is worth stating and worth fixing, and the fix is a laptop
re-run.

## Measured-or-null ledger

Every performance field is either measured with an artifact or written as `null`. Estimates, roofline arithmetic, vendor claims, and results from different hardware may guide investigation, but they do not fill result cells. This rule is visible in [LEDGER.md](../LEDGER.md), where end-to-end serving and kernel-gate fields remain unfilled until their exact gates run.

Each measurement record must identify:

- hardware and compute capability;
- exact workload shape and mode;
- baseline implementation;
- correctness state for the measured kernel;
- timing statistic and units;
- raw log, JSON, profiler output, or journal session;
- whether the result is local relative performance, a remote microbenchmark, or end-to-end serving.

Two strengths of record are allowed, and they are not the same thing. The strong
form is an artifact under `journal/remote/` that a reader can re-parse: JSON, a
raw log, a profiler export. The weak form is a number transcribed into a journal
session with no machine-readable record behind it. Both are permitted, because
refusing the weak form would have meant deleting true measurements. Mixing them
without saying which is which is not permitted, because it lets a reader assume
every number is checkable when some are not.

The weak-form numbers in this repository are the Nsight Compute
percentages, the 18.7% post-deploy A100 re-run (**withdrawn 2026-07-25**, see
[the Ampere record](../journal/regression-ampere-tile-sweep.md), and still listed
here because a withdrawn number is not a deleted one), the `sm_120` relative timings,
the 8x H100 memory recipe with its 0.77GB-per-0.01 sensitivity, the
`max 1.56e-2` per-op parity figure in the ledger, and every count from the local
`sm_120` session of 2026-07-25, whose three committed JSONs are zero-byte files.
The canonical list lives in
[journal/remote/README.md](../journal/remote/README.md); no count is repeated
across documents, because every document that carried one drifted. They are
listed there and labelled
at each point of use. Moving one of them into the strong form, by committing a
CSV export or a JSON, is preferred over restating it.

Evidence classes by hardware are also not interchangeable:

| Evidence class | Allowed claim |
|---|---|
| RTX 5090 Laptop, `sm_120` | Correctness development and same-machine relative comparisons only, on the architecture actually run. WDDM and the local power cap prevent serving claims, and so does the open multi-sequence varlen fault. A green gate here is not evidence about `sm_80`. [Hardware note](../journal/phase0.md#local-tier) |
| Single H100, `sm_90` | Architecture-local parity, kernel microbenchmarks, and profiler evidence. It is not an end-to-end serving result. [Session record](../journal/remote/h100-session1.md) |
| Multi-GPU full-model gate | Token-level and logprob agreement between two builds of the same checkpoint. It is a correctness result, not a performance result. [Artifact](../journal/remote/gate_logit_parity_8xh100.json) |
| Multi-GPU serving run | End-to-end throughput and latency requires matched stock and turbo checkpoint, quantization, hardware, workload, and SLO. No such run has produced retrievable results, so every serving field is `null`. [Ledger](../LEDGER.md) |

If a later investigation invalidates an interpretation, append the correction rather than silently rewriting history. The H100 journal's correction from misplaced bias to bias not consumed demonstrates this rule. [Correction](../journal/remote/h100-session1.md#post-session-correction-static-analysis-2026-07-18)

## Session-based GPU validation

Remote work is packaged as a bounded session with one stated payload. The launcher waits for an approved instance type, refuses a duplicate project instance, uploads the bootstrap and harnesses, captures stdout and JSON artifacts, and terminates the instance in a `finally` block by default.

The required session sequence is:

1. State the hypothesis, target architecture, test cases, expected artifacts, and spending limit.
2. Freeze the code and patch payload before launch.
3. Launch only an approved instance type and record its region and identifier in the private raw log as needed.
4. Bootstrap the pinned environment and verify that the intended source tree is imported at runtime.
5. Run parity first, then baseline and candidate microbenchmarks, then profiling if correctness is green.
6. Copy logs and machine-readable results into `journal/remote/`.
7. Terminate on success, failure, timeout, or interruption.
8. Add the duration, cost, result, and next payload to the ledger or session journal.

`scripts/grab_b200.py` implements the termination and artifact-capture boundary. `scripts/bootstrap_b200.sh` continues through individual harness failures so one paid session can capture a full diagnostic set. Therefore, the bootstrap exit code is not sufficient evidence. The parity lines and artifacts decide the gate.

## Per-architecture verification

CUDA architecture families are validated independently. An implementation that compiles or wins on one family is not presumed correct or fast on another.

### `sm_120`

- Use the generic attention path actually selected on RTX 50-series hardware.
- Apply and record any compatibility patches needed by the pinned CuTe DSL.
- Require per-op oracle parity for global, beyond-extent, and sliding-window modes.
- Report timings as relative-only.
- Do not infer Blackwell behavior from the `sm_120` architecture number. The instruction set and kernel path differ.
- Do not infer `sm_80` behavior from an `sm_120` result either, in the direction that helps. The two share `flash_fwd.py`, so a **defect** found on one is a reasonable expectation on the other and must be labelled an expectation until it is run there. A **green gate** on one is not evidence about the other at all, because different silicon is the whole reason this section exists.

The local hardware and instruction-set caveats are recorded in [phase 0](../journal/phase0.md#local-tier).

On 2026-07-25 this architecture carried the first execution of the generic
kernel's corrected shear shift on any silicon. Green: `parity_rel_chunked_decode`
7/7 with per-case signal 7.5x to 37.8x above tolerance, `parity_fa4_rel` 3/3,
`parity_rel_bias_coverage` 6/6, `parity_qkvr_prep` 5/5. Not green:
`parity_rel_varlen_batch` 1/12, an illegal memory access on the first
multi-sequence case, which is an **open defect** and the reason no `sm_120`
serving claim is available. All of those counts are **journal-only**: the three
JSONs the session committed under `journal/remote/local_sm120_s29/` are zero-byte
files and two of the five gates wrote no file at all, so they are transcribed
from [the session write-up](../journal/regression-sm120-varlen-illegal-address.md).

### `sm_90`

- Compile and run on H100 hardware. Local `sm_120` JIT success is not a Hopper compile test.
- Compare against vLLM's actual Hopper `score_mod` route, including its production split policy.
- Require parity before accepting a timing result, on the shape family being timed. Decode and chunked-prefill timings require a decode and chunked-prefill parity run, not a full-prefill one.
- Validate fragment coordinate mapping against a known-correct mask or oracle consumer rather than assuming layouts are shared across kernel families.
- Save profiler evidence only after the parity gate is green.

The Hopper port is complete and validated. Per-op parity is 3/3 on H100 and the kernel was reproduced on a second machine with a different torch and CUDA build. [Session record](../journal/u2-hopper-design.md#session-24-2026-07-20-sm_90-native-parity-33-green--race-won)

Getting there took seventeen debugging sessions. During that period a flight reached near-plain-attention timing while producing wrong output, and it was recorded as a failed kernel rather than a win. That record is kept in the journal rather than removed, because the rule it demonstrates is the point: a timing from a wrong-output path is not evidence.

### `sm_80`

- Run on A100 hardware. The generic kernel reaches Ampere through backward compatibility, which is not the same as being tested there.
- Require per-op oracle parity for all three modes before any tile-tuning result counts.
- Gate every tuning candidate on its own parity run, **on the shape family that candidate is going to be timed on**. A configuration that is fast and wrong is discarded, not reported.
- Gate every tuning candidate at its own tile size. A parity run at the default tile size says nothing about the others, and a sweep exists precisely to change that parameter.
- Make no speedup claim on this architecture. No day-0 path executes here, so there is no baseline to divide by. The claim is support, not speed.
- **Make no serving claim on this architecture either, and say why.** Every parity case behind the `sm_80` support claim passes one sequence. Serving batches sequences into one varlen call on every step, and `harness/parity_rel_varlen_batch.py` scores 1 of 12 on the generic kernel with an illegal memory access. That was observed on `sm_120`; `sm_80` runs the same file, so it is **expected** here and no A100 has been available to confirm it. The support claim is therefore single-sequence and full-prefill, and root-causing the fault gates any serving statement on Ampere. [The defect](../journal/regression-sm120-varlen-illegal-address.md).

The A100 record is in [session 26 and session 27](../journal/u2-hopper-design.md#session-26-2026-07-23-founder-8x-a100-node-sm_80-ours-runs-day-0-cannot).

The third and fourth bullets are there because both were violated, and the tile-tuning percentages measured on 2026-07-23 are **withdrawn** as a result. `harness/tune_sm80.py` times decode shapes, `T_q = 1` against `T_k = 65536`, while its `parity_ok()` built one `cu_seqlens` and passed it as both `cu_seqlens_q` and `cu_seqlens_k`, so the gate certified 512 query rows against 512 keys. The generic kernel carried the same `seqlen_q == seqlen_k` shear-shift specialisation that `sm_90` did, so the harness was green on the one family the kernel got right and released timings for the family it got wrong. This is the harness that CONTRIBUTING.md and the README both cite as the one place where rule 1 above is enforced in code rather than by habit, which is what makes it the sharpest example available: **an enforced gate aimed at the wrong shape family is more dangerous than no gate, because it gets cited.** The `sm_80` **support** claim does not depend on any of those percentages and is unaffected, though its parity evidence is `seqlen_q == seqlen_k` only, and single-sequence only. Full account, and what a re-measurement would take: [journal/regression-ampere-tile-sweep.md](../journal/regression-ampere-tile-sweep.md).

### `sm_100` and `sm_110`

- Run on the matching Blackwell hardware.
- Verify that the intended sheared-bias and block-scaled paths are selected at runtime.
- Re-run the parity suite, microbenchmarks, and profiler gate on that architecture.
- Do not substitute Hopper or `sm_120` measurements when capacity is unavailable.
- Keep performance fields `null` until the hardware-local artifacts exist.

No Blackwell hardware was available while this was built. The dispatch code exists and is untested. Every Blackwell performance field is `null` and stays that way until someone runs it on that silicon.

## Failure records

A failure is an artifact. It is written down at the same weight as a result, in the same file, at the time it happened, and it is not removed once a later session succeeds.

This rule is load-bearing rather than decorative. Three of the things this repository knows are only knowable because a failure was kept:

- The `pack_gqa` row contract is visible only through the sequence of bias addressing schemes that were correct on one architecture and wrong on another. A journal that recorded only the working `partition_C` approach would not explain why the obvious approaches do not work, which is the part a future contributor needs.
- The memory recipe for the full checkpoint is six rejected configurations and one that works. Publishing only the working line would present a narrow window as an arbitrary choice, and would lose the measured sensitivity that tells a reader what to change on different hardware.
- The full-model logprob gate failed as specified, and its same-build control failed too. The control failure is what turns a tolerance failure into information about the platform rather than information about the kernel. Waiving the control would have produced a pass that meant nothing.

Two further categories are kept for the same reason and are not quietly retried: capacity lost to defects in our own tooling, and results lost to an orchestration error. The end-to-end serving sweep was lost to a watchdog race caused by our own launcher, and that is recorded as the reason the serving fields are `null`, rather than the fields being filled from a partial run.

A published overstatement is corrected in place and the correction is left visible. A comparison against one of our own abandoned prototypes was published once as though it were a comparison against the shipped baseline. It was corrected publicly, and the abandoned prototypes stay in the measurement artifacts so the distinction is checkable rather than asserted.

The largest such correction is the decode speedup, withdrawn on 2026-07-25. It is instructive for a reason that has nothing to do with arithmetic: the numbers were measured, twice, on two machines, and both sides of the ratio were real timings of real kernels. What was wrong was that our kernel was doing a small fraction of the bias work the baseline was doing, so the ratio measured an omission. The like-for-like re-measurement then landed within a few percent of the withdrawn figures, which is the hardest case in which to hold the line: the temptation is to conclude the old numbers were fine after all. They were not fine, they were unfounded, and a number that lands near a sound one by luck has no more standing than one that does not. Both sets stay on the front page, the old one struck through, because a reader who saw the claim needs to be able to find out what happened to it.

## Claim checklist

Before adding a number to the README or ledger, verify all of the following:

- the value exists in a journal, raw log, JSON result, or profiler artifact;
- the hardware, architecture, workload, batch, context, and units are stated;
- the corresponding correctness gate is green;
- that gate ran on the same shape family as the number, `seqlen_q` against `seqlen_k` included;
- the gate is capable of failing, and you can say what result would have failed it;
- the baseline is the path used by the real integration, not a convenient substitute;
- the baseline is not one of our own superseded prototypes;
- local relative numbers are labeled relative-only;
- single-GPU kernel measurements are labeled microbenchmarks;
- end-to-end wording is reserved for matched serving runs;
- limitations and unmeasured fields remain visible;
- a reader can follow the link from the claim to its evidence.

For a capability claim, additionally verify:

- the parity run that establishes it ran on the architecture being claimed;
- the failure of every day-0 route on the same hardware is recorded, not assumed;
- no ratio is attached where no baseline executes.

For a defect claim, additionally verify:

- the pinned commit and the file and line are named in a tree a reader can check out;
- a reproduction exists that does not depend on this repository's kernels;
- observed behavior is separated from inferred behavior;
- the duplicate check against the upstream tracker has been run and its result recorded, including an empty result;
- the AI-assistance disclosure required by the upstream contribution policy is present.

When any item is missing, the result stays `null` or is described as an unresolved experiment.
