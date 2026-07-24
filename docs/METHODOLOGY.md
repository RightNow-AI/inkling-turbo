# Methodology

Inkling-turbo treats correctness, performance, and deployment scope as separate gates. A result can pass one and fail another. Public claims include the artifact that establishes each gate and the hardware context in which it was measured.

## Parity-oracle discipline

The parity oracle is an explicit PyTorch implementation of Inkling relative attention. It computes scaled QK scores, applies the learned relative-position value only when the causal distance is inside the configured extent, applies the global or sliding-window mask, then computes softmax and the value projection. The semantics were derived from the day-0 vLLM path and the sheared-bias writer, and are recorded in [the implementation study](../journal/day0-implementation.md#attention-path-nvidiaattentionpy-opsfa4_rel_attentionpy-u2u3u5-baseline).

Kernel and baseline backends consume inputs derived from the same seeded tensors. The current per-op suite covers three distinct semantic cases:

- a short global-attention case;
- a global case that crosses the relative-bias extent;
- a sliding-window case.

The local U2 gate is green only when all 3 cases pass the configured tolerance. The published `sm_120` result was 3/3 with maximum differences in the same range as the day-0 `score_mod` backend. [Recorded result](../journal/u2-hopper-design.md#v0-final-verdict-2026-07-19-parity-33-green-speed-failed-expected)

The full integration gate is stricter than the per-op gate. It requires the fixed prompt suite and batched-output versus batch-1 consistency before a kernel is considered complete at the model level. That gate ran for U2 on 8x H100 against the real checkpoint. The token half passed 32/32. The logprob half is recorded as a fail, because the a-priori tolerance turned out to be tighter than the platform's own batch reproducibility, and the same-build control failed too. The control failure is reported, not waived. Fields that no gate has filled stay `null` in the [kernel ledger](../LEDGER.md#kernel-gates-ncu-90-of-binding-roofline).

Rules:

1. Compare against the oracle before comparing speed.
2. Use the same tensors, masks, scales, and output dtype for every backend.
3. Record maximum and mean error, not only a pass label.
4. Treat a skipped backend, compile failure, non-finite result, or tolerance failure as not green.
5. Do not use timings from a wrong-output path as evidence for an optimization.
6. When the oracle catches a defect, preserve the failing case and root-cause record. The silent non-Blackwell `rel_bias` omission is the canonical example. [Issue draft](../journal/upstream/01-rel-bias-silently-ignored-non-blackwell.md)

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

Evidence classes are not interchangeable:

| Evidence class | Allowed claim |
|---|---|
| RTX 5090 Laptop, `sm_120` | Correctness development and same-machine relative comparisons only. WDDM and the local power cap prevent serving claims. [Hardware note](../journal/phase0.md#local-tier) |
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

The local hardware and instruction-set caveats are recorded in [phase 0](../journal/phase0.md#local-tier).

### `sm_90`

- Compile and run on H100 hardware. Local `sm_120` JIT success is not a Hopper compile test.
- Compare against vLLM's actual Hopper `score_mod` route, including its production split policy.
- Require parity before accepting a timing result.
- Validate fragment coordinate mapping against a known-correct mask or oracle consumer rather than assuming layouts are shared across kernel families.
- Save profiler evidence only after the parity gate is green.

The Hopper port is complete and validated. Per-op parity is 3/3 on H100 and the kernel was reproduced on a second machine with a different torch and CUDA build. [Session record](../journal/u2-hopper-design.md#session-24-2026-07-20-sm_90-native-parity-33-green--race-won)

Getting there took seventeen debugging sessions. During that period a flight reached near-plain-attention timing while producing wrong output, and it was recorded as a failed kernel rather than a win. That record is kept in the journal rather than removed, because the rule it demonstrates is the point: a timing from a wrong-output path is not evidence.

### `sm_80`

- Run on A100 hardware. The generic kernel reaches Ampere through backward compatibility, which is not the same as being tested there.
- Require per-op oracle parity for all three modes before any tile-tuning result counts.
- Gate every tuning candidate on its own parity run. A configuration that is fast and wrong is discarded, not reported.
- Make no speedup claim on this architecture. No day-0 path executes here, so there is no baseline to divide by. The claim is support, not speed.

The A100 record is in [session 26 and session 27](../journal/u2-hopper-design.md#session-26-2026-07-23-founder-8x-a100-node-sm_80-ours-runs-day-0-cannot).

### `sm_100` and `sm_110`

- Run on the matching Blackwell hardware.
- Verify that the intended sheared-bias and block-scaled paths are selected at runtime.
- Re-run the parity suite, microbenchmarks, and profiler gate on that architecture.
- Do not substitute Hopper or `sm_120` measurements when capacity is unavailable.
- Keep performance fields `null` until the hardware-local artifacts exist.

No Blackwell hardware was available while this was built. The dispatch code exists and is untested. Every Blackwell performance field is `null` and stays that way until someone runs it on that silicon.

## Claim checklist

Before adding a number to the README or ledger, verify all of the following:

- the value exists in a journal, raw log, JSON result, or profiler artifact;
- the hardware, architecture, workload, batch, context, and units are stated;
- the corresponding correctness gate is green;
- the baseline is the path used by the real integration, not a convenient substitute;
- local relative numbers are labeled relative-only;
- single-GPU kernel measurements are labeled microbenchmarks;
- end-to-end wording is reserved for matched serving runs;
- limitations and unmeasured fields remain visible;
- a reader can follow the link from the claim to its evidence.

When any item is missing, the result stays `null` or is described as an unresolved experiment.
