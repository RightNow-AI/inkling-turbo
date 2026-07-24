# Inkling-turbo

Open-source attention kernels for serving TML's Inkling on vLLM.

Inkling's attention is not standard. There is no RoPE. The model adds a learned relative-position term to every pre-softmax score, and it alternates global and sliding-window layers. Stock FlashAttention paths do not cover that shape, so vLLM's day-0 support implements the bias with a per-score callback, `score_mod`, on every architecture that is not Blackwell.

This repo replaces the callback with a tile-level sheared-bias kernel. The bias is built as a tile, laid out sheared so that a contiguous tile lines up with the scores it belongs to, and added to the MMA accumulator in one pass before softmax. The checkpoint is untouched. No quantization change, no retraining, no change to the attention math. Only the kernel and the code that dispatches to it.

## What is here that was not here before

Five things. The speed number is deliberately last, because attention is a slice of serving time and the first four are capabilities and defects rather than percentages.

**1. Inkling has an attention path on A100.** Day-0 has none. The model router sends everything that is not Blackwell to `score_mod`, and the cute backend hard-blocks `score_mod` on SM8x with a `NotImplementedError`. The two pieces of routing logic disagree and nothing checks between them, so the failure arrives at the first attention call, after the weights are resident and the KV cache is allocated. Our generic sheared-bias kernel runs there, parity 3 of 3 green against a float32 oracle. Because no day-0 path executes on that arch, this is a support result and not a speed result, and no speedup is claimed. [Reproducer and root cause](journal/upstream/05-no-sm8x-attention-path.md), [session record](journal/u2-hopper-design.md#session-26-2026-07-23-founder-8x-a100-node-sm_80-ours-runs-day-0-cannot).

**2. `rel_bias` is accepted and silently dropped on every non-Blackwell arch.** The library allocates the padded bias tensor, launches the `ShearingBias` pre-kernel, then constructs a forward kernel that has no bias argument, and returns plain bias-free attention as if it were the biased result. No error, no warning. A case-insensitive grep for `bias` returns zero matches in `flash_fwd_sm90.py`, `flash_fwd.py` and `flash_fwd_sm120.py`, against 236 matches in `flash_fwd_sm100.py`. This is silent numerical wrongness in a shipped stack, found by the parity oracle rather than by a crash. A correct bias implementation cannot be free: ours costs 15.8% over plain attention on the batch-1 64K decode case, 853 us against 736 us in the table below. The day-0 `rel_bias` path costs nothing there, because it does no bias work. [Full report with a runnable reproducer](journal/upstream/01-rel-bias-silently-ignored-non-blackwell.md).

**3. `pack_gqa` redefines what a score-tile row means, and nothing in the API says so.** It folds eight GQA query heads into the seqlen mode, so a 128-row score tile stops being 128 sequence positions and becomes 16 positions by 8 heads. Any row-indexed feature is silently wrong from that point on unless it also packs. It is not opt-in: a heuristic turns it on for every GQA model. The bias feature survives on Blackwell only because the sm_100 kernel and the shear writer each remembered to pack, independently and undocumented. It cost us 17 debugging sessions, and the diagnostic that ended it was a stride: the wgmma submode that steps 8 tile rows had stride 81920, which at the anchor shape is exactly one sequence row of the bias tensor. Anyone adding a row-indexed feature to these kernels will hit this. [Contract hazard report](journal/upstream/04-pack-gqa-row-semantics.md), [the debugging account](journal/u2-hopper-design.md#the-key-insight-2026-07-20-why-manual-coords-fail-on-wgmma).

**4. A measured recipe for fitting a 592GB checkpoint on 640GB of HBM.** Seven attempts on 8x H100. Six failed, each for a different measured reason: a missing worker-side dependency, an infeasible 16384 context, a CUDA graph capture 394MB short at utilization 0.95, a warmup activation 782MB short at the same setting, utilization 0.90 leaving no room for KV at all, and 0.93 still short with KV at 0.58GB against 1.27 needed. The seventh works. The window is narrow and the sensitivity is measured at roughly 0.77GB of KV headroom per 0.01 of utilization. The configuration is in [Serving the full model](#serving-the-full-model) and the failure sequence is in the [session record](journal/u2-hopper-design.md#session-28-2026-07-24-8x-h100-first-full-model-serving--logit-gate).

**5. Token-level equivalence on the real 975B model.** 32 of 32 prompts produced identical greedy tokens against the stock build, same checkpoint, TP8, 2369 tokens compared. That is a stronger statement than it looks, because the platform itself is not batch-deterministic at TP8 across 66 bf16 layers: the same-build control, batched against single, produced mismatched tokens for both stock and ours. We preserve the model's behavior exactly where the platform preserves it. The logprob half of that gate is recorded as a failure, honestly, and is explained in [What is not measured](#what-is-not-measured). [Artifact](journal/remote/gate_logit_parity_8xh100.json).

**And it is faster.** On an H100 the kernel runs 2.7x faster than the path vLLM actually serves with at 64K decode, and 1.45x faster at 8K global prefill. On sliding-window prefill it is currently 1.28x slower. All cases, including the one we lose, are in the table below.

## What is measured

Read [What is not measured](#what-is-not-measured) before quoting any number from this section.

![Attention kernel latency against the path vLLM ships](docs/figures/fig1_latency.png)

Attention kernel latency, microseconds per iteration, one H100 SXM5. Lower is better.

There is exactly one day-0 baseline on Hopper: `score_mod`, the per-score callback vLLM actually serves with. Everything is measured against that. Plain attention carries no bias at all and is the floor this feature can approach but never beat, not a baseline we are entitled to claim a win over.

| Case | Ours | day-0 `score_mod` | plain, no bias | Result |
|---|---|---|---|---|
| decode, batch 1, 64K KV | 853 | 2327 | 736 | **2.7x faster** |
| decode, batch 32, 64K KV | 855 | 2391 | 727 | **2.8x faster** |
| decode, batch 32, 8K KV | 124 | 304 | | **2.5x faster** |
| prefill 8K, global | 3309 | 4799 | | **1.45x faster** |
| prefill 8K, sliding window | 1223 | 957 | | **1.28x slower** |

Source: [`microbench_attn_day0_session25_h100.json`](journal/remote/microbench_attn_day0_session25_h100.json) and [`microbench_attn_scoremod_session25_h100.json`](journal/remote/microbench_attn_scoremod_session25_h100.json). Same box, same session, identical shapes in both harnesses.

Our prefill totals include the ShearingBias pre-kernel, which the `score_mod` path does not need: 827 us of the 3309, and 461 us of the 1223. That pre-kernel is why the sliding-window case loses, and removing it is the first item in [What comes next](#what-comes-next).

The `scoremod` JSON also contains two much slower paths, `relproj` at 7195 us and `relprojT` at 5155 us on the batch-1 decode case. **Those are ours, not vLLM's.** They are the register-resident designs we tried and abandoned, kept in `kernels/relproj_score_mod.py` and measured in the same runs so the dead ends stay on the record. Dividing our shipped kernel by our own failed prototype would produce a larger number and would not mean anything. That mistake was made once here and corrected publicly.

### Everything else that was gated

| Result | Numbers | Where it was run |
|---|---|---|
| Same tokens as stock on the real model | 32 of 32 prompts produced identical greedy tokens. 2369 tokens compared. | 8x H100, TP8, full NVFP4 checkpoint |
| The only working option on Ampere | Parity green on A100. Every day-0 path fails to run. | A100 SXM4 40GB |
| Tuned tile sizes for Ampere | 10% to 18.7% faster on decode shapes than the upstream default, which shipped with a "should tune" comment. | A100 SXM4 40GB |
| Reproduces on a second machine and a different software stack | Parity green again on torch 2.11/cu130 after the first run used cu129. The decode gap widened rather than shrank. | A second H100 SXM5 |
| Inkling fits and serves on 8x H100 | 592GB of weights on 640GB of HBM. The working configuration is in [Serving the full model](#serving-the-full-model). | 8x H100 |

Every timing has a passing parity run behind it. A fast kernel that returns the wrong answer is a failed kernel, and the harness discards its timing. During the 17 sessions on Hopper, one flight reached near-plain-attention timing while producing wrong output. It is recorded as a failed kernel, not a win.

![Full-model agreement with the stock build](docs/figures/fig2_correctness.png)

## What is not measured

Read this section before quoting any number above.

- **No end-to-end serving speedup is claimed.** The throughput rows in [LEDGER.md](LEDGER.md) are `null`. We ran the sweep on 8x H100 and lost the results when a safety watchdog killed the box mid-run. That was our own bug and it is [written up](journal/u2-hopper-design.md#session-28-postscript-e2e-curves-lost-to-a-watchdog-race-orchestrator-error) instead of quietly retried.
- **We lose on sliding-window prefill.** 1223 us against 957 us for the shipped path, measured on the same box in the same run. The ShearingBias pre-kernel costs 461 us there and the attention kernel alone does not make it back. 55 of Inkling's 66 layers are sliding-window, so this case matters. A fix exists and is correctness-validated but not yet speed-measured: see [Removing the pre-kernel](#removing-the-pre-kernel).
- Attention is only part of serving time. The MoE layers and the big GEMMs dominate. Do not assume a 2.7x kernel speedup becomes a 2.7x serving speedup. It will not.
- The decode numbers come from a microbenchmark that packs its query rows into one sequence. True multi-sequence decode was measured separately and is slower per sequence: 432 us per sequence at 32 sequences by 64K KV. Neither number is a serving result.
- **The logprob half of the full-model gate is recorded as a fail.** The a-priori tolerance turned out to be tighter than the platform's own batch reproducibility, and the same-build control failed too. Ours-against-stock mean delta was 0.048 where the same-build noise floor was 0.150. The control failure is reported, not waived. Only the token half passed.
- **Blackwell is untested.** The code dispatches to `sm_100`, but no B200 was available while this was built. No number here comes from Blackwell hardware.
- **Below the kernel roofline gate.** ncu on 8K prefill measured 45.6% SM SOL and 55.9% memory SOL at 490 GB/s, occupancy 14.0%. The project's own bar is 90% of the binding roofline. This does not meet it. The recoverable costs are named in [What comes next](#what-comes-next).
- The full-model gate compared tokens and logprobs between two builds. It is a correctness check, not a quality benchmark. We ran no downstream evals.
- RTX 5090 numbers are relative only. That machine is power-capped and on WDDM.
- U3 quantizes KV on write. Attention does not yet read the quantized cache directly.
- The upstream bug reports are written but not filed.

## Removing the pre-kernel

The sliding-window loss has one cause: our path runs a `ShearingBias` kernel that
`score_mod` does not need. It rewrites the relative-bias buffer into the sheared
layout the attention kernel reads. It costs 461 us of the 1223 us total.

`qkvr_prep` already computes and writes that buffer one step earlier, so it can
write it sheared in the first place and the pre-kernel disappears, along with the
two scheduler kernels that exist only to launch it.

That is implemented in `kernels/patches/u2_shear_fusion.py` and it is
**correctness-validated on GPU**: 16 of 16 gate cases in
`harness/parity_shear_fusion.py`, where the 14 writer cases are bit-exact against
stock `ShearingBias` output across global, sliding-window, varlen, batched,
prefill, chunked and decode.

**No speedup is claimed.** Removing a 461 us kernel should take sliding-window
prefill to roughly 762 us against 957 us, which would turn the loss into a win.
That number is arithmetic, not a measurement, and it stays that way until an H100
run produces a JSON artifact. The feature ships **off by default**; enable it with
`INKLING_TURBO_FUSED_SHEAR=1`.

Two constraints worth knowing. It requires `pack_gqa=False`, which is already
forced for Hopper with bias, and it must be applied **after**
`kernels/patches/u3_fp8_kv.py`, because it extends text that patch introduces.
Applying them out of order fails loudly rather than silently.

## Architecture support

![Validation status by unit and architecture](docs/figures/fig3_status.png)

| GPU | State | Detail |
|---|---|---|
| H100 (`sm_90`) | Working, per-op and full-model | Native wgmma kernel. Parity green, 2.7x faster than the shipped path at decode, slower on sliding-window prefill, token-identical to stock on the real model. |
| A100 (`sm_80`) | Working, and the only option | Parity green, tile sizes tuned. Day-0 cannot run here at all. |
| RTX 5090 (`sm_120`) | Working, per-op | Parity green, 2% to 10% faster than day-0 on that machine. The local headroom is structurally smaller than Hopper's. Timings relative only. |
| B200 (`sm_100`, `sm_110`) | Untested | Dispatch exists. No hardware was available. |

## How it works

The day-0 Hopper path calls a function for every score to add its bias term. We build a bias tile instead, lay it out sheared so that a contiguous tile lines up with the scores it belongs to, and add it to the accumulator in one pass before softmax.

The hard part was not the idea. It was that on `sm_90` the accumulator lives in wgmma fragments whose element-to-coordinate mapping you cannot compute by hand. Seventeen debugging sessions went into that. Linear indexing, `reshape_acc_to_mn` coordinates and `make_tiled_copy_C` were each correct or nearly correct on unpacked geometry and unfixable on packed geometry.

The fix is to stop computing coordinates. Partition the bias tile with the same partitioner that produced the accumulator, `thr_mma_qk.partition_C`, then pair them by flat index. They cannot disagree, because they came from the same object. That is the whole trick, and it is arch-portable in a way that a hand-derived mapping is not.

The bug underneath all seventeen sessions was `pack_gqa`, described above. Every coordinate-based scheme was wrong before it started. The full account, including the probes that finally exposed it, is in [journal/u2-hopper-design.md](journal/u2-hopper-design.md).

Behavior we preserve:

- weights and checkpoint untouched
- distances outside the configured extent contribute zero
- global and sliding-window masks unchanged
- a kernel ships only when it matches the same PyTorch oracle the baseline is checked against

## Reproducing this

You need Linux or WSL, a vLLM checkout at the pinned commit, and a CUDA toolchain. Pinned commits are in [journal/day0-implementation.md](journal/day0-implementation.md).

### Apply the patches

Read each patch before you run it.

```bash
bash scripts/apply_local_sm120_fixes.sh /path/to/vllm
python3 kernels/patches/u2_v0_generic_bias.py /path/to/vllm
python3 kernels/patches/u2_v1_smem_bias.py /path/to/vllm
python3 kernels/patches/u3_fp8_kv.py /path/to/vllm         # optional, FP8 KV writes
python3 kernels/patches/u2_serving_route.py /path/to/vllm  # send sm_90 and sm_120 serving here
```

The first script fixes incompatibilities between the vendored attention code and the pinned CuTe DSL. Full kernel sources are in `kernels/tml_fa4_modified/`. On `sm_90`, deploy those sources rather than patching: the native Hopper kernel is in `flash_fwd_sm90.py` there.

`kernels/patches/u2_sm90_bias_port.py` and `u2_sm90_direct_gmem.py` are not in the list above on purpose. They are the smem-staged and direct-gmem bias attempts, both superseded by the `partition_C` approach that actually works. They are kept because the journal refers to them, not because you should apply them.

### Run the gates

```bash
python harness/parity_fa4_rel.py           # main attention gate, global and SWA
python harness/parity_kv_fp8.py            # FP8 KV writes
python harness/parity_shear_writer.py      # shear layout contract
python harness/microbench_attn_day0.py     # our kernel, real shapes
python harness/microbench_attn_scoremod.py # the day-0 baseline, same shapes
python harness/tune_sm80.py                # tile sweep, parity-gated
```

Run these from inside the vLLM checkout with its environment active.

### Serving the full model

This configuration is measured, not guessed. It is the seventh of seven attempts, and the six that failed are listed in the [session record](journal/u2-hopper-design.md#session-28-2026-07-24-8x-h100-first-full-model-serving--logit-gate).

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
vllm serve /path/to/inkling --served-model-name inkling \
  --tensor-parallel-size 8 \
  --max-model-len 3072 \
  --gpu-memory-utilization 0.94 \
  --enforce-eager --seed 0
```

KV cache headroom moves by roughly 0.77GB per GPU for every 0.01 of `gpu-memory-utilization`. At 0.95 the warmup allocation fails. At 0.90 there is no room left for KV at all. 0.94 is the window. `--enforce-eager` is not a preference: CUDA graph capture came up 394MB short.

### Compare two builds on a real model

`scripts/gate_logit_parity.py` serves the model twice, once with the stock kernels and once with ours, sends the same 32 prompts to both, and compares tokens and logprobs. `scripts/grab_b200.py` provisions a cloud box, runs the harness, and terminates it in a `finally` block.

These scripts spend real money. Set a budget first.

## Upstream bugs found

Five reports covering ten distinct defects, in [journal/upstream/](journal/upstream/). We checked both trackers and found no existing coverage. They are written, reviewed, and not filed yet. A human files them under their own name after re-running the duplicate check, which is scripted in [`00-INDEX.md`](journal/upstream/00-INDEX.md).

1. [`rel_bias` is silently ignored on non-Blackwell kernels](journal/upstream/01-rel-bias-silently-ignored-non-blackwell.md). The kernel accepts the argument, drops it, and returns output that looks plausible and is wrong. This is the one that matters most.
2. [Four CuTe DSL breaks](journal/upstream/02-cutlass-4.6.0-api-drift-cluster.md) against the dependency version the day-0 stack pins. Nothing on this path runs against its own pin.
3. [Three generic-path defects](journal/upstream/03-vllm-flash-attn-generic-path-bugs.md) found on `sm_120`.
4. [`pack_gqa` changes what a score-tile row means](journal/upstream/04-pack-gqa-row-semantics.md), which breaks any row-indexed feature.
5. [No Inkling attention path exists on SM8x at all](journal/upstream/05-no-sm8x-attention-path.md).

## Repository layout

```
kernels/tml_fa4_modified/   modified kernel sources, the real implementation
kernels/patches/            idempotent patch scripts against a clean checkout
harness/                    parity oracles, microbenchmarks, the tile tuner
scripts/                    cloud provisioning, bootstrap, full-model gates
journal/                    the working record, including every dead end
journal/remote/             raw measurement artifacts as JSON, plus session logs
journal/upstream/           bug reports written against upstream
docs/METHODOLOGY.md         the evidence rules
docs/figures/               figures used in this README
LEDGER.md                   every number, measured or null
CONTRIBUTING.md             the gates a change has to pass
```

Start at [journal/README.md](journal/README.md). It says what each file is and which session produced which number. [journal/remote/README.md](journal/remote/README.md) maps every artifact to the claim it backs.

Nsight Compute reports are not in the repo. A single `.ncu-rep` is several megabytes and they were left out of git deliberately. The numbers read off them are quoted in [journal/u2-hopper-design.md](journal/u2-hopper-design.md), and the commands that regenerate them are in `scripts/bootstrap_8x.sh`.

## What comes next

1. Fold the ShearingBias pre-kernel into the attention kernel, or into `qkvr_prep` upstream of it. It is 25% to 38% of our prefill total and it is the whole reason the sliding-window prefill case loses.
2. Split-KV decode for `sm_90`. Batch-1 decode is parallelism-bound, not bandwidth-bound: 64 CTAs on 132 SMs, DRAM at 7%, occupancy at 14%. Splitting the KV range is the fix.
3. Re-enable `intra_wg_overlap` and `pack_gqa`, both forced off to get the bias path correct. Both cost prefill throughput today. Packed-bias addressing on `sm_90` is exactly the problem the `sm_100` path already solves.
4. Blackwell validation when hardware is available.
5. U3 read path, so attention consumes the FP8 cache directly.
6. Re-run the serving sweep, pulling artifacts after every config so a dead box costs one config instead of everything.
7. File the upstream reports.

Longer-term units (MoE grouped GEMM, router fusion, QKVR fusion, CUDA graphs, batch-aware dispatch) are tracked in [LEDGER.md](LEDGER.md).

## How we handle numbers

[LEDGER.md](LEDGER.md) contains no estimates. Every cell is a measurement or the word `null`. The rules are written down in [docs/METHODOLOGY.md](docs/METHODOLOGY.md) and they are the reason several fields on this page are empty rather than filled with something plausible.

Failures get the same space as wins. The GPU capacity lost to bugs in our own tooling, the serving results lost to a watchdog race, the seventeen sessions before the kernel was correct, and the one overstatement we published and then corrected are all in the journal. It is the working record, not a highlight reel. If you find a number here you cannot reproduce, open an issue.

## License

See [LICENSE](LICENSE).
