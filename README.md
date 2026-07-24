# Inkling-turbo

Open-source attention kernels for serving TML's Inkling on vLLM.

Inkling's attention is not standard. There is no RoPE. The model adds a learned relative-position term to every pre-softmax score, and it alternates global and sliding-window layers. Stock FlashAttention paths do not cover that shape, so vLLM's day-0 support implements the bias with a per-score callback, `score_mod`, on every architecture that is not Blackwell.

This repo replaces the callback with a tile-level sheared-bias kernel. The bias is built as a tile, laid out sheared so that a contiguous tile lines up with the scores it belongs to, and added to the MMA accumulator in one pass before softmax. The checkpoint is untouched. No quantization change, no retraining, no change to the attention math. Only the kernel and the code that dispatches to it.

## What is here that was not here before

Five things. The speed number is deliberately last, because attention is a slice of serving time and the first four are capabilities and defects rather than percentages.

**1. Inkling has an attention *kernel* on A100.** Day-0 has none. The model router sends everything that is not Blackwell to `score_mod`, and the cute backend hard-blocks `score_mod` on SM8x with a `NotImplementedError`. The two pieces of routing logic disagree and nothing checks between them, so the failure arrives at the first attention call, after the weights are resident and the KV cache is allocated. Our generic sheared-bias kernel runs there, parity 3 of 3 green against a float32 oracle. Because no day-0 path executes on that arch, this is a support result and not a speed result, and no speedup is claimed. Read the word *kernel* literally: this is a per-op parity result on one A100 SXM4 40GB. **We never served Inkling on A100 and cannot**, because 8x40GB is 320GB against a 592GB checkpoint. What is fixed is the attention gap; the memory gap is untouched. [Reproducer and root cause](journal/upstream/05-no-sm8x-attention-path.md), [session record](journal/u2-hopper-design.md#session-26-2026-07-23-founder-8x-a100-node-sm_80-ours-runs-day-0-cannot).

**2. `rel_bias` is accepted and silently dropped on every non-Blackwell arch.** The library allocates the padded bias tensor, launches the `ShearingBias` pre-kernel, then constructs a forward kernel that has no bias argument, and returns plain bias-free attention as if it were the biased result. No error, no warning. In **stock tml-fa4 at the pinned commit `13374f0c`**, `grep -ci bias` returns zero in `flash_attn/cute/flash_fwd_sm90.py`, `flash_fwd.py` and `flash_fwd_sm120.py`, against 236 in `flash_fwd_sm100.py`. Name the flag or the number does not reproduce: `-c` counts matching *lines*, so 236 is lines, and `grep -oi bias file | wc -l` counts raw occurrences and returns 346. The zeros are zero either way, which is the part that matters. Run that grep against a stock checkout, not against this repo: our `kernels/tml_fa4_modified/` copies of those files are the fix, and they match on `bias` by design. This is silent numerical wrongness in a shipped stack, found by the parity oracle rather than by a crash. A correct bias implementation cannot be free: ours costs 15.8% over plain attention on the batch-1 64K decode case, 853 us against 736 us in the table below. The day-0 `rel_bias` path costs nothing there, because it does no bias work. [Full report with a runnable reproducer](journal/upstream/01-rel-bias-silently-ignored-non-blackwell.md).

**3. `pack_gqa` redefines what a score-tile row means, and nothing in the API says so.** It folds eight GQA query heads into the seqlen mode, so a 128-row score tile stops being 128 sequence positions and becomes 16 positions by 8 heads. Any row-indexed feature is silently wrong from that point on unless it also packs. It is not opt-in: a heuristic turns it on for every GQA model. The bias feature survives on Blackwell only because the sm_100 kernel and the shear writer each remembered to pack, independently and undocumented. It cost us 17 debugging sessions, and the diagnostic that ended it was a stride: the wgmma submode that steps 8 tile rows had stride 81920, which at the anchor shape is exactly one sequence row of the bias tensor. Anyone adding a row-indexed feature to these kernels will hit this. [Contract hazard report](journal/upstream/04-pack-gqa-row-semantics.md), [the debugging account](journal/u2-hopper-design.md#the-key-insight-2026-07-20-why-manual-coords-fail-on-wgmma).

**4. A measured recipe for fitting a 592GB checkpoint on 640GB of HBM.** Seven attempts on 8x H100. Six failed, each for a different measured reason: a missing worker-side dependency, an infeasible 16384 context, a CUDA graph capture 394MB short at utilization 0.95, a warmup activation 782MB short at the same setting, utilization 0.90 leaving no room for KV at all, and 0.93 still short with KV at 0.58GB against 1.27 needed. The seventh works. The window is narrow and the sensitivity is measured at roughly 0.77GB of KV headroom per 0.01 of utilization. The configuration is in [Serving the full model](#serving-the-full-model) and the failure sequence is in the [session record](journal/u2-hopper-design.md#session-28-2026-07-24-8x-h100-first-full-model-serving--logit-gate).

**5. Token-level equivalence on the real 975B model.** 32 of 32 prompts produced identical greedy tokens against the stock build, same checkpoint, TP8, 2369 tokens compared. That is a stronger statement than it looks, because the platform itself is not batch-deterministic at TP8 across 66 bf16 layers: the same-build control, batched against single, produced mismatched tokens for both stock and ours. We preserve the model's behavior exactly where the platform preserves it. The logprob half of that gate is recorded as a failure, honestly, and is explained in [What is not measured](#what-is-not-measured). [Artifact](journal/remote/gate_logit_parity_8xh100.json).

**And it is faster.** On an H100 the attention kernel runs 2.7x to 2.8x faster than the path vLLM actually serves with at 64K decode, and about 1.45x faster at 8K global prefill, reproduced on two machines. On sliding-window prefill it is 1.28x to 1.41x **slower**, and 55 of Inkling's 66 layers are sliding-window. All cases, including the one we lose, are in the table below with both runs shown.

Read that paragraph as written. It is a kernel microbenchmark on one GPU, not a serving result. **No end-to-end serving speedup is claimed anywhere in this repository**, every throughput row in [LEDGER.md](LEDGER.md) is `null`, and attention is a slice of serving time that the MoE layers and the big GEMMs dominate. If you are about to quote "2.7x" without the word *attention* in the same sentence, the number does not support it.

## What is measured

Read [What is not measured](#what-is-not-measured) before quoting any number from this section.

![Attention kernel latency against the path vLLM ships](docs/figures/fig1_latency.png)

Attention kernel latency, microseconds per iteration, one H100 SXM5. Lower is better.

There is exactly one day-0 baseline on Hopper: `score_mod`, the per-score callback vLLM actually serves with. Everything is measured against that. Plain attention carries no bias at all and is the floor this feature can approach but never beat, not a baseline we are entitled to claim a win over.

| Case | Ours | day-0 `score_mod` | plain, no bias | Result |
|---|---|---|---|---|
| decode, batch 1, 64K KV | 853 / 860 | 2327 / 2412 | 736 | **2.7x to 2.8x faster** |
| decode, batch 32, 64K KV | 855 / 866 | 2391 / 2383 | 727 | **2.8x faster** |
| decode, batch 32, 8K KV | 124 / 124 | 304 / 304 | | **2.5x faster** |
| prefill 8K, global | 3309 / 3307 | 4799 / 4841 | | **1.45x to 1.46x faster** |
| prefill 8K, sliding window | 1223 / 1221 | 957 / 863 | | **1.28x to 1.41x slower** |

Two independent runs, on two different H100 SXM5 machines: session 25 and session 26b. Both figures are given wherever they differ, because a single number here would be a choice about which run to quote.

Sources: [session 25](journal/remote/microbench_attn_day0_session25_h100.json) with its [baseline](journal/remote/microbench_attn_scoremod_session25_h100.json), and [session 26b](journal/remote/validate_s26b_h100x1_route/), where ours and the baseline were timed in the **same container** minutes apart.

**Our kernel is the reproducible half of this comparison.** Across those two runs it moves by at most 1.9%, and on three of five cases by 0.1% or less. The `score_mod` baseline moves by up to 9.8%, which is the entire reason the sliding-window loss is quoted as a range rather than a figure: our number went 1223 to 1221, the baseline went 957 to 863. Treat any of these ratios as good to about one decimal place, not two, and treat the day-0 side as the noisy one.

Our prefill totals include the ShearingBias pre-kernel, which the `score_mod` path does not need: 827 us of the 3309, and 461 us of the 1223. That pre-kernel is why the sliding-window case loses. The obvious fix, folding it into the writer that produces the bias one step earlier, is implemented, and we measured it: **it makes things worse, not better.** See [Removing the pre-kernel](#removing-the-pre-kernel). Closing this case needs a cheaper way to produce the sheared layout, not the removal of the launch.

The `scoremod` JSON also contains two much slower paths, `relproj` at 7195 us and `relprojT` at 5155 us on the batch-1 decode case. **Those are ours, not vLLM's.** They are the register-resident designs we tried and abandoned, kept in `kernels/relproj_score_mod.py` and measured in the same runs so the dead ends stay on the record. Dividing our shipped kernel by our own failed prototype would produce a larger number and would not mean anything. That mistake was made once here and corrected publicly.

### Everything else that was gated

| Result | Numbers | Where it was run | Evidence |
|---|---|---|---|
| Same tokens as stock on the real model | 32 of 32 prompts produced identical greedy tokens. 2369 tokens compared. | 8x H100, TP8, full NVFP4 checkpoint | [JSON](journal/remote/gate_logit_parity_8xh100.json) |
| The only working attention kernel on Ampere | Parity green on A100. Every day-0 path fails to run. Per-op only; the checkpoint does not fit on A100. | A100 SXM4 40GB | journal session 26 |
| Tuned tile sizes for Ampere | 10.1% faster on batch-1 decode and 18.2% on the 32-sequence case, against the upstream default, which shipped with a "should tune" comment. A post-deploy re-run measured 18.7% on the 32-sequence case. | A100 SXM4 40GB | [JSON](journal/remote/tune_sm80_a100.json) for the sweep; the 18.7% re-run is journal session 27 only, no separate JSON |
| Reproduces on a second machine and a different software stack | Parity green again on torch 2.11/cu130 after the first run used cu129. The decode gap widened rather than shrank. | A second H100 SXM5 | session 24 and session 25 JSON |
| Inkling fits and serves on 8x H100 | 592GB of weights on 640GB of HBM. The working configuration is in [Serving the full model](#serving-the-full-model). | 8x H100 | journal session 28, prose |

A note on the last column, because the rule matters more than any row in the table. **Not every number on this page has a JSON artifact.** The attention latency table, the full-model gate and the Ampere tile sweep do. Some do not, and those are marked **journal-only** wherever they appear: the Nsight Compute percentages, the 18.7% post-deploy re-run, the `sm_120` relative timings, and the memory recipe with its 0.77GB-per-0.01 sensitivity. Journal-only means one of us read a number off a tool and wrote it down, with no machine-readable record you can re-parse. Treat those as weaker evidence than the rest, because they are. The label at the point of use is the authority and [journal/remote/README.md](journal/remote/README.md) holds the canonical list; this page deliberately does not restate a count, because every document that carried one drifted out of step with the others.

Every timing quoted here has a passing parity run behind it, as a rule we follow, and in one place as a rule the code enforces: `harness/tune_sm80.py` refuses to report a configuration's timing unless that configuration's own parity run was green. The other harnesses do not enforce it and cannot. `harness/microbench_attn_day0.py` will happily time a stock build that silently drops the bias, which is exactly upstream finding 01, and its own docstring says so. During the 17 sessions on Hopper, one flight reached near-plain-attention timing while producing wrong output. It is recorded as a failed kernel, not a win.

![Full-model agreement with the stock build](docs/figures/fig2_correctness.png)

Full-model agreement, 8x H100 at TP8. The token half of this gate passed 32 of 32. **The logprob half is recorded as a failure**, and the figure shows why the failure is not informative about our kernel: the a-priori tolerance sits below the platform's own batch reproducibility, and both same-build controls failed too. The bars are mean absolute per-token logprob difference. Lower is better, and the two grey bars are the floor.

## What is not measured

Read this section before quoting any number above.

- **No end-to-end serving speedup is claimed.** The throughput rows in [LEDGER.md](LEDGER.md) are `null`. We ran the sweep on 8x H100 and lost the results when a safety watchdog killed the box mid-run. That was our own bug and it is [written up](journal/u2-hopper-design.md#session-28-postscript-e2e-curves-lost-to-a-watchdog-race-orchestrator-error) instead of quietly retried.
- **We lose on sliding-window prefill.** 1223 us against 957 us for the shipped path, measured on the same box in the same run. The ShearingBias pre-kernel costs 461 us there and the attention kernel alone does not make it back. 55 of Inkling's 66 layers are sliding-window, so this case matters. A fix exists and is correctness-validated but not yet speed-measured: see [Removing the pre-kernel](#removing-the-pre-kernel).
- Attention is only part of serving time. The MoE layers and the big GEMMs dominate. Do not assume a 2.7x kernel speedup becomes a 2.7x serving speedup. It will not.
- The decode numbers come from a microbenchmark that packs its query rows into one sequence. True multi-sequence decode was measured separately and is slower per sequence: 432 us per sequence at 32 sequences by 64K KV. Neither number is a serving result.
- **The logprob half of the full-model gate is recorded as a fail.** The a-priori tolerance turned out to be tighter than the platform's own batch reproducibility, and the same-build control failed too. Ours-against-stock mean delta was 0.048 where the same-build noise floor was 0.150. The control failure is reported, not waived. Only the token half passed.
- **Blackwell is untested.** The code dispatches to `sm_100`, but no B200 was available while this was built. No number here comes from Blackwell hardware.
- **Below the kernel roofline gate.** ncu on 8K prefill measured 45.6% SM SOL and 55.9% memory SOL at 490 GB/s, occupancy 14.0%. The project's own bar is 90% of the binding roofline. This does not meet it. The recoverable costs are named in [What comes next](#what-comes-next).
- **The ncu numbers in the line above are the weakest evidence on this page, and you cannot check them.** The three `.ncu-rep` files exist, but they are several megabytes each and are gitignored, so what ships is our transcription of four metrics and nothing else. There is no CSV export in the repo either, and the profiling run was done by hand rather than from a script, so no committed script regenerates them. Exporting the section summaries to a committed CSV is on the list below. Until that happens, treat 45.6 / 55.9 / 490 / 14.0 as reported, not as verifiable.
- **Inkling was never served on A100, and the checkpoint does not fit there.** The Ampere result is a per-op attention kernel that passes parity. 8x A100 40GB is 320GB against a 592GB checkpoint. Closing the attention gap does not close the memory gap.
- The full-model gate compared tokens and logprobs between two builds. It is a correctness check, not a quality benchmark. We ran no downstream evals.
- RTX 5090 numbers are relative only. That machine is power-capped and on WDDM, and those timings live in the journal with no JSON artifact.
- U3 quantizes KV on write. Attention does not yet read the quantized cache directly.
- **The shear fusion is measured and it loses on prefill.** It costs 1019 us on global prefill and 561 us on sliding-window prefill, and saves 5 us on batch-32 decode. It also cannot run at all on `sm_90` on the read side: the pre-sheared `bias=` path hits an unbound `n_block` in `flash_fwd_sm90.py`, so the Hopper gate is 14/16. Ships off. See [Removing the pre-kernel](#removing-the-pre-kernel).
- **Split-KV decode is still unvalidated on any hardware.** Its first execution, on Hopper in session 26, hit the same `n_block` defect.
- The upstream bug reports are written but not filed. Report 03 targets a tracker whose duplicate check has not been run; see [Upstream bugs found](#upstream-bugs-found).

## Removing the pre-kernel

The sliding-window loss has one cause: our path runs a `ShearingBias` kernel that
`score_mod` does not need. It rewrites the relative-bias buffer into the sheared
layout the attention kernel reads. It costs 461 us of the 1223 us total.

`qkvr_prep` already computes and writes that buffer one step earlier, so it can
write it sheared in the first place and the pre-kernel disappears, along with the
two scheduler kernels that exist only to launch it.

That is implemented in `kernels/patches/u2_shear_fusion.py`. The writer is
correct: 14 of 14 writer cases in `harness/parity_shear_fusion.py` are bit-exact
against stock `ShearingBias` output across global, sliding-window, varlen,
batched, prefill, chunked and decode, and that now holds on **both** `sm_120`
(RTX 5090) and `sm_90` (H100, session 26).

### It does not pay, and we measured it

The idea was that removing a 461 us kernel would take sliding-window prefill from
1223 us to roughly 762 us and turn our one loss into a win. That was arithmetic,
it was labelled as arithmetic, and an H100 has now measured it. **It is wrong.**

The fused writer has to emit `rel_extent + 256` columns into a
`(T + 128, H, ext + 256)` buffer instead of `rel_extent` columns into
`(T, H, ext)`. That costs far more than the `ShearingBias` launch it removes.
Both paths timed in one process on the same inputs, per kernel, us/iter:

| shape | natural writer + ShearingBias | fused writer | net |
|---|---|---|---|
| prefill, 8K, global | 1312.1 | 2336.1 | **+1019.4, loss** |
| prefill, 8K, sliding window | 685.9 | 1251.6 | **+561.1, loss** |
| decode, batch 32, 64K KV | 10.8 | 10.3 | **-4.7, win** |

Attention consumes an identical buffer either way, so that delta is the entire
effect of the fusion, not half of it. The writer is 5.7x slower on sliding-window
prefill than the one it replaces. The pre-kernel is not the thing to remove.

Evidence: [journal/remote/validate_s26_h100x1/](journal/remote/validate_s26_h100x1/).
The `net` column excludes a `torch.full(NaN)` the harness runs each iteration so
the parity gate can catch unwritten columns; production allocates with
`torch.empty` (`kernels/tml_fa4_modified/interface.py:725,735`). Including it
gives +606.0 / +314.9 / -13.3, which flatters the fusion and is still a loss on
both prefill shapes.

The feature ships **off by default** and should stay off for prefill. Enable with
`INKLING_TURBO_FUSED_SHEAR=1`.

### And on Hopper the read path does not run at all

The same session-26 run found that the pre-sheared `bias=` path cannot execute on
`sm_90`. Both `attention_consumes_*` gate cases, all four `presheared_*`
microbenchmarks and both `splitkv_*` cases fail with

```
NameError: cannot access local variable 'n_block' where it is not associated with a value
  --> flash_fwd_sm90.py
```

so the gate scores **14/16** on Hopper rather than 16/16. The writer half is
correct there; the consumer half is not reachable. That is an open defect, not a
tolerance question, and it is tracked in
[journal/remote/validate_s26_h100x1/README.md](journal/remote/validate_s26_h100x1/README.md).

The `sm_120` 16/16 remains journal-only: that gate wrote no file, and its only
record is commit `7375849`. The harness now writes
`harness/parity_shear_fusion_sm<cc>.json` with device, torch version, capability
and every case's pass state, which is how the `sm_90` result above has an
artifact.

Two constraints worth knowing. It requires `pack_gqa=False`, which is already
forced for Hopper with bias. And **if u3 is also being applied, u3 goes first**,
because both patches edit the same two places in `qkvr_prep.py`: the
`fused_qkvr_prep` signature tail and the `_run_fused_small(...)` call tail.
`u3_fp8_kv.py` anchors on the stock form of that text, and this patch rewrites
it, so u3 has to land while the stock form is still there. That is an ordering
constraint between the two, not a dependency: this patch anchors on nothing u3
introduces and applies cleanly to a tree that has never seen u3. Applying them
in the wrong order fails loudly rather than corrupting the tree, because
`u3_fp8_kv.py` aborts on the anchor it can no longer find.

## Architecture support

![Validation status by unit and architecture](docs/figures/fig3_status.png)

| GPU | State | Detail |
|---|---|---|
| H100 (`sm_90`) | Working, per-op and full-model | Native wgmma kernel. Parity green, 2.7x faster than the shipped path at decode, slower on sliding-window prefill, token-identical to stock on the real model. |
| A100 (`sm_80`) | Working per-op, and the only attention kernel that runs | Parity green, tile sizes tuned. Day-0 cannot run here at all. The checkpoint does not fit on A100, so this is a kernel result, not a deployment. |
| RTX 5090 (`sm_120`) | Working, per-op | Parity green, 2% to 10% faster than day-0 on that machine. The local headroom is structurally smaller than Hopper's. Timings relative only, journal-recorded, no JSON artifact. |
| B200 (`sm_100`, `sm_110`) | Untested | Dispatch exists. No hardware was available. Nothing on this row has ever run. |

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

The order below is the order to run them in, and it is not arbitrary. Read each patch before you run it.

```bash
bash scripts/apply_local_sm120_fixes.sh /path/to/vllm
python3 kernels/patches/u2_v0_generic_bias.py /path/to/vllm
python3 kernels/patches/u2_v1_smem_bias.py /path/to/vllm
python3 kernels/patches/u2_serving_route.py /path/to/vllm  # send sm_90 and sm_120 serving here
python3 kernels/patches/u3_fp8_kv.py /path/to/vllm         # FP8 KV writes
python3 kernels/patches/u2_shear_fusion.py /path/to/vllm   # off unless INKLING_TURBO_FUSED_SHEAR=1
```

The first script fixes incompatibilities between the vendored attention code and the pinned CuTe DSL. Full kernel sources are in `kernels/tml_fa4_modified/`. On `sm_90`, deploy those sources rather than patching: the native Hopper kernel is in `flash_fwd_sm90.py` there. `u2_serving_route.py` is order-independent, since it only rewrites the body of `_use_sheared_bias`.

The last two lines are each optional, and independently so. You may stop after `u2_serving_route.py` and have the shipping kernel. `u3_fp8_kv.py` and `u2_shear_fusion.py` can each be applied without the other; `u2_shear_fusion.py` on a tree with no u3 applies all 28 edits (15/3/6/4), compiles, and leaves no u3 symbols behind. **If you want both, u3 goes first**, for the anchor reason given in [Removing the pre-kernel](#removing-the-pre-kernel).

Three of the seven patch scripts apply to the tree this repo ships, and all three are idempotent. Measured on copies of the tree, second run:

| patch | second run | exit |
|---|---|---|
| `u2_serving_route.py` | `already applied` | 0 |
| `u3_fp8_kv.py` | `qkvr_prep.py: 0 edits applied` | 0 |
| `u2_shear_fusion.py` | `already applied, nothing to do` | 0 |

The wording differs between them; only `u3_fp8_kv.py` literally reports `0 applied`.

The other four (`u2_v0_generic_bias.py`, `u2_v1_smem_bias.py`, `u2_sm90_bias_port.py`, `u2_sm90_direct_gmem.py`) anchor on **stock** tml-fa4 text. They are the historical steps that produced `kernels/tml_fa4_modified/`, so against a tree that already has those sources they abort on a missing anchor on the *first* run, by design. Idempotence is not a meaningful property of them and is not claimed.

On anchor strictness the seven are not uniform. `u2_shear_fusion.py` requires every anchor to match **exactly once** (`count = source.count(old); assert count == 1`) and aborts otherwise. The rest assert only that the anchor is *present* and then replace its first occurrence, so they stop on a missing anchor, which is what a wrong order produces, but a duplicated anchor would be patched at the first hit rather than refused.

`kernels/patches/u2_sm90_bias_port.py` and `u2_sm90_direct_gmem.py` are not in the list on purpose. They are the smem-staged and direct-gmem bias attempts, both superseded by the `partition_C` approach that actually works. They are kept because the journal refers to them, not because you should apply them.

### Run the gates

```bash
python harness/parity_fa4_rel.py           # main attention gate, global and SWA
python harness/parity_kv_fp8.py            # FP8 KV writes, needs u3_fp8_kv.py
python harness/parity_shear_writer.py      # shear layout contract
python harness/parity_shear_fusion.py      # shear fusion, needs u2_shear_fusion.py
python harness/microbench_attn_day0.py     # our kernel, real shapes
python harness/microbench_attn_scoremod.py # the day-0 baseline, same shapes
python harness/tune_sm80.py                # tile sweep, parity-gated, A100 only
```

Run these from inside the vLLM checkout with its environment active. `parity_fa4_rel.py` is the one that decides whether a kernel ships: three semantic cases, green means 3 of 3 inside tolerance, and a skipped backend counts as red. Run the two microbenchmarks on the same box in the same session or the comparison is not controlled.

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

Five reports covering ten distinct defects, in [journal/upstream/](journal/upstream/). They are written, reviewed, and not filed yet. A human files them under their own name after re-running the duplicate check, which is scripted in [`00-INDEX.md`](journal/upstream/00-INDEX.md).

These target three trackers, and our duplicate check covered two of them. The sweep on 2026-07-21 searched `vllm-project/vllm` and `vllm-project/tml-fa4` and found nothing. It did not search `vllm-project/flash-attention`, which is where two of report 03's three defects belong. That gap is recorded in `00-INDEX.md` and has to close before anything is filed.

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

Nsight Compute reports are not in the repo. A single `.ncu-rep` is several megabytes and they were left out of git deliberately. The four metrics read off them are quoted in the session 24 ncu entry of [journal/u2-hopper-design.md](journal/u2-hopper-design.md), and that transcription is the only form the evidence ships in. **No script in this repo regenerates them.** The profiling was done by hand on the session-24 box against `harness/microbench_attn_day0.py`, and the invocation was never committed. Committing a CSV export of the section summaries, which is small enough for git and re-parseable, is item 8 in [What comes next](#what-comes-next).

## What comes next

1. Kill the ShearingBias cost, which is 25% to 38% of our prefill total and the whole reason the sliding-window case loses. Folding it upstream into `qkvr_prep` is **done and refuted**: the sheared writer costs more than the launch it removes, measured, see [Removing the pre-kernel](#removing-the-pre-kernel). What is left is to build the sheared tile inside the attention kernel, in shared memory, per tile, and never materialize the padded buffer at all. That is a real kernel change rather than a re-plumbing, and it is the honest next attempt.
2. Split-KV decode for `sm_90`. Batch-1 decode is parallelism-bound, not bandwidth-bound: 64 CTAs on 132 SMs, DRAM at 7%, occupancy at 14%. Splitting the KV range is the fix.
3. Re-enable `intra_wg_overlap` and `pack_gqa`, both forced off to get the bias path correct. Both cost prefill throughput today. Packed-bias addressing on `sm_90` is exactly the problem the `sm_100` path already solves.
4. Blackwell validation when hardware is available.
5. U3 read path, so attention consumes the FP8 cache directly.
6. Re-run the serving sweep, pulling artifacts after every config so a dead box costs one config instead of everything.
7. File the upstream reports, after running the duplicate check against `vllm-project/flash-attention` that has not been run.
8. Commit a CSV export of the ncu section summaries, so the roofline numbers are checkable rather than transcribed.
9. Re-run the shear-fusion gate on `sm_90`. It scored 14/16 there because both attention cases hit the `n_block` defect that is now fixed; the writer half is already 14/14 bit-exact on Hopper. The speed question is answered and the answer is no.

Longer-term units (MoE grouped GEMM, router fusion, QKVR fusion, CUDA graphs, batch-aware dispatch) are tracked in [LEDGER.md](LEDGER.md).

## How we handle numbers

[LEDGER.md](LEDGER.md) contains no estimates. Every cell is a measurement or the word `null`. The rules are written down in [docs/METHODOLOGY.md](docs/METHODOLOGY.md) and they are the reason several fields on this page are empty rather than filled with something plausible.

Failures get the same space as wins. The GPU capacity lost to bugs in our own tooling, the serving results lost to a watchdog race, the seventeen sessions before the kernel was correct, and the one overstatement we published and then corrected are all in the journal. It is the working record, not a highlight reel.

## What to do next

Pick the one that matches why you are here.

- **You want to know whether to believe the 2.7x.** Open the two session-25 JSONs linked under [What is measured](#what-is-measured) and divide the numbers yourself. Then read [What is not measured](#what-is-not-measured), which is where the load-bearing caveats are. Then read [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the rules the numbers were produced under.
- **You maintain vLLM, tml-fa4 or flash-attention.** Read [`01-rel-bias-silently-ignored-non-blackwell.md`](journal/upstream/01-rel-bias-silently-ignored-non-blackwell.md) first. It is a silent wrong-output path in shipped code, it has a fifteen-line reproducer that does not depend on this repository, and the fix can be a `NotImplementedError`. Report [04](journal/upstream/04-pack-gqa-row-semantics.md) is the one to read before anyone adds another row-indexed feature to those kernels.
- **You want to run it.** [Reproducing this](#reproducing-this), in the order given. `harness/parity_fa4_rel.py` is the gate that decides whether a kernel is real.
- **You want to work on it.** The open items are in [What comes next](#what-comes-next); item 2, split-KV decode for `sm_90`, is the largest measured headroom and is written up in the journal. The bar for a patch is in [CONTRIBUTING.md](CONTRIBUTING.md): a timing without a passing parity run is not accepted.
- **You found a number here you cannot reproduce.** Open an issue. Say which number and what you got. That is the most useful thing anyone can send us.

## License

See [LICENSE](LICENSE).
