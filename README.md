# Inkling-turbo

Inkling-turbo is an open-source kernel project for serving TML's Inkling model on top of vLLM's day-0 implementation. It replaces selected GPU kernels and their integration code. It does not change the checkpoint, model architecture, or attention semantics.

The first target is Inkling's relative-attention path. The current work is building replacements for the Hopper and generic per-score bias callback using tile-level sheared bias. The public goal is a drop-in kernel path that preserves the model and makes each architecture prove its own correctness and performance.

This repository is still a kernel-development release, not a claim of end-to-end serving readiness. See [Methodology](docs/METHODOLOGY.md) for the evidence rules.

## Headline findings

| Finding | Result | Scope | Evidence |
|---|---|---|---|
| The replacement Hopper kernel beats every day-0 production variant with parity | Native `sm_90` kernel: parity 3/3, then 905.6 us versus 2375 to 6209 us production at batch-1 64K-KV global decode (2.6x to 6.9x) and 2.5x to 3.9x at 8K prefill; the relative-attention term costs 21% over plain attention | One H100 SXM5, `sm_90`, per-op microbenchmark, same machine and same harness for all paths. This is not end-to-end serving performance | [Session 24](journal/u2-hopper-design.md#session-24-2026-07-20-sm_90-native-parity-33-green--race-won), [JSON](journal/remote/microbench_attn_day0_native_sm90_session24.json) |
| Hopper long-context decode has large relative-attention overhead | vLLM's production `score_mod` path took 2375 us versus 743 us for plain attention, a 3.2x gap | One H100 SXM5, `sm_90`, batch 1, global decode, 64K KV, per-op microbenchmark. This is not end-to-end serving performance | [H100 session 4](journal/remote/h100-session1.md#h100-session-4-2026-07-18-12291237-utc-056--honest-baseline) |
| U2 tile-level sheared-bias attention is green on the local architecture | Parity passed 3/3, and the kernel beat the same-machine day-0 `score_mod` baseline on every reported case | RTX 5090 Laptop, `sm_120`. Timings are relative-only because the system is WDDM and power-capped | [U2 v1 results](journal/u2-hopper-design.md#v1-complete-parity-33--beats-score_mod-on-every-case-2026-07-19), [local hardware limits](journal/phase0.md#local-tier) |
| The day-0 stack contains correctness and compatibility defects | 8 findings are currently indexed, including `rel_bias` being silently ignored on non-Blackwell kernels and returning plausible wrong output | Draft upstream reports, not yet filed. The index is the source of truth for the count | [Upstream findings index](journal/upstream/00-INDEX.md) |

## The kernels

### U2: tile-level sheared-bias attention

Inkling attention has no RoPE. It adds learned relative-position terms to the pre-softmax scores. The day-0 Hopper path applies those terms through a per-score callback, while the Blackwell path uses a sheared layout. U2 loads a contiguous bias tile and applies it to the score fragment before softmax. The generic path is complete at its per-op gate and is also parity-proven on H100. The native Hopper kernel applies the sheared tile through the same tiled-copy machinery the kernel already uses for its P matrix, and is awaiting its architecture-local gate.

The replacement keeps Inkling's attention math intact:

- the checkpoint and model weights are untouched;
- relative distance outside the configured extent contributes zero;
- global and sliding-window attention retain their original masks;
- the kernel is accepted only when it agrees with the same PyTorch parity oracle used for the baseline.

| Architecture | Status | Current evidence |
|---|---|---|
| `sm_120` | Done for the current per-op gate | Parity 3/3 and faster than the day-0 `score_mod` path on all reported local cases. These are relative-only local measurements, not serving numbers. [Journal](journal/u2-hopper-design.md#v1-complete-parity-33--beats-score_mod-on-every-case-2026-07-19) |
| `sm_90` | Done for the current per-op gate | Native wgmma kernel: parity 3/3 on H100 (max error 1.6e-2) and faster than every day-0 production variant on the same machine: 905.6 us versus 2375 to 6209 us at batch-1 64K-KV global decode (2.6x to 6.9x), 3362 us versus 8483 to 13049 us at 8K global prefill (2.5x to 3.9x). The relative-attention term costs 21% over biasless plain attention. The root cause that blocked this path (GQA head packing changing the tile-row meaning) is documented, and `pack_gqa` is disabled with bias as the v0 tradeoff. These are per-op microbenchmarks, not serving numbers. [Session 24](journal/u2-hopper-design.md#session-24-2026-07-20-sm_90-native-parity-33-green--race-won), [microbench JSON](journal/remote/microbench_attn_day0_native_sm90_session24.json), [ncu analysis](journal/u2-hopper-design.md#session-24-ncu-kernel-gate-evidence-reports-in-journalncu) |
| `sm_100` / `sm_110` | Pending hardware capacity | The Blackwell variants have not received the required architecture-local validation. No Blackwell performance number is claimed. [Blockers](BLOCKERS.md#owner-decision-2026-07-19-lambda-only-lean-finish-plan) |

The implementation and reproducible patch sequence live under `kernels/tml_fa4_modified/` and `kernels/patches/`. Earlier correct-but-slow variants remain as evidence of rejected designs rather than being presented as wins.

## Reproduce the current kernel work

The scripts expect a Linux or WSL environment with a compatible vLLM checkout and CUDA toolchain. The fork base and pinned day-0 commits are recorded in [the implementation study](journal/day0-implementation.md).

### Apply the local toolchain fixes and U2 patch

Review each patch before applying it to your checkout.

```bash
bash scripts/apply_local_sm120_fixes.sh /path/to/vllm
python3 kernels/patches/u2_v0_generic_bias.py /path/to/vllm
python3 kernels/patches/u2_v1_smem_bias.py /path/to/vllm
```

The first script repairs known incompatibilities between the vendored attention code and the pinned CuTe DSL. The U2 scripts patch the generic `sm_120` path. The Hopper work is in `kernels/patches/u2_sm90_bias_port.py` and `kernels/patches/u2_sm90_direct_gmem.py`, and remains under its architecture-local parity gate.

### Run parity and microbenchmarks

Activate the vLLM environment, run from the vLLM checkout, and use absolute paths if this repository is elsewhere:

```bash
python /path/to/inkling-turbo/harness/parity_fa4_rel.py
python /path/to/inkling-turbo/harness/parity_shear_writer.py
python /path/to/inkling-turbo/harness/microbench_attn_scoremod.py
python /path/to/inkling-turbo/harness/microbench_attn_day0.py
```

The main relative-attention harness checks global, beyond-relative-extent, and sliding-window cases against one PyTorch oracle. A timing result is discarded when its corresponding parity result is not green. See [Methodology](docs/METHODOLOGY.md#parity-oracle-discipline).

### Run a remote validation session

`scripts/grab_b200.py` polls for an allowed instance type, uploads the bootstrap and harness payload, captures logs under `journal/remote/`, and terminates the instance in a `finally` block unless `--park` is explicitly supplied.

```powershell
py scripts/grab_b200.py --types gpu_1x_h100_sxm5 --max-hours 1
```

This command can incur external GPU charges and requires provider credentials. Set a budget before running it. The bootstrap is designed to collect all evidence even when an individual harness fails, so a zero bootstrap exit alone is not a correctness gate. Inspect the parity output and journal artifact.

## What is not claimed yet

- No end-to-end throughput, latency, TTFT, TPOT, or tokens-per-second improvement is claimed. Those fields remain `null` in the [measured-or-null ledger](LEDGER.md#e2e-serving-remote-vs-stock-day-0-build--same-checkpointquantgpus-slo).
- End-to-end serving curves are pending the planned 8-GPU integration and final validation sessions. They must compare the same checkpoint, quantization, GPU set, workload, and SLO. [Validation plan](BLOCKERS.md#owner-decision-2026-07-19-lambda-only-lean-finish-plan)
- The `sm_90` U2 path is not done until architecture-local parity passes. A fast parity failure is a failed kernel.
- Blackwell variants are pending hardware capacity and architecture-local verification. Hopper or RTX 5090 results are not projected onto Blackwell.
- RTX 5090 Laptop timings are relative-only. H100 figures in this README are per-op microbenchmarks, not end-to-end serving results. [Hardware ground truth](journal/phase0.md#hardware-ground-truth-measured-2026-07-17)

## Roadmap

1. `sm_90` performance pass: split-KV decode (batch-1 decode is parallelism-bound, measured DRAM 7% and occupancy 14%), re-enable `intra_wg_overlap` with bias, packed-GQA bias addressing, and shear-writer overlap. The current kernel ships on its measured 2.5x to 6.9x with the ceilings documented honestly in the ncu analysis.
2. Validate the `sm_100` and `sm_110` U2 variants when Blackwell capacity is available.
3. Validate U3 quantized paged KV per architecture. Its per-op parity is green locally (2/2 on `sm_120`); H100 and integration-level checks are pending.
4. Run the full prompt-level parity and batched-consistency integration gate.
5. Run stock-versus-turbo end-to-end serving sweeps on the same 8-GPU system and publish median, best, latency, throughput, and raw artifacts together.
6. Upstream the kernel and compatibility fixes after tracker duplicate checks.

The broader unit plan, including MoE, routing, QKVR, graphs, overlap, and batch-aware dispatch, is tracked in [the project rules](CLAUDE.md) and [the ledger](LEDGER.md).

## Upstream findings

The currently indexed issue drafts are under [`journal/upstream/`](journal/upstream/):

- [Silent `rel_bias` omission on non-Blackwell kernels](journal/upstream/01-rel-bias-silently-ignored-non-blackwell.md), a wrong-output correctness issue.
- [Four CuTe DSL compatibility breaks](journal/upstream/02-cutlass-4.6.0-api-drift-cluster.md) against the dependency version pinned by the day-0 stack.
- [Three generic-path defects](journal/upstream/03-vllm-flash-attn-generic-path-bugs.md) exposed on `sm_120`.

The drafts are evidence packages, not filed issue links. Duplicate-check the target trackers before filing, then update the [index](journal/upstream/00-INDEX.md) with the canonical upstream references.
