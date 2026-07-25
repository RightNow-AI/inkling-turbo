# Inkling-turbo

Open-source attention kernels for serving TML's Inkling on vLLM.

Inkling's attention is not standard. There is no RoPE. The model adds a learned relative-position term to every pre-softmax score, and it alternates global and sliding-window layers. Stock FlashAttention paths do not cover that shape, so vLLM's day-0 support implements the bias with a per-score callback, `score_mod`, on every architecture that is not Blackwell.

This repo replaces the callback with a tile-level sheared-bias kernel. The bias is built as a tile, laid out sheared so that a contiguous tile lines up with the scores it belongs to, and added to the MMA accumulator in one pass before softmax. The checkpoint is untouched. No quantization change, no retraining, no change to the attention math. Only the kernel and the code that dispatches to it.

## What is here that was not here before

Five things. The speed number is deliberately last, because attention is a slice of serving time and the first four are capabilities and defects rather than percentages.

**1. Inkling has an attention *kernel* on A100.** Day-0 has none. The model router sends everything that is not Blackwell to `score_mod`, and the cute backend hard-blocks `score_mod` on SM8x with a `NotImplementedError`. The two pieces of routing logic disagree and nothing checks between them, so the failure arrives at the first attention call, after the weights are resident and the KV cache is allocated. Our generic sheared-bias kernel runs there, parity 3 of 3 green against a float32 oracle, **on three cases that all pass `seqlen_q == seqlen_k`**, so what is certified is the full-prefill family and nothing else. Because no day-0 path executes on that arch, this is a support result and not a speed result, and no speedup is claimed. Read the word *kernel* literally: this is a per-op parity result on one A100 SXM4 40GB. The tile-tuning percentages that used to sit beside this claim are **withdrawn**, for reasons that do not touch the support claim: [the record](journal/regression-ampere-tile-sweep.md). **We never served Inkling on A100 and cannot**, because 8x40GB is 320GB against a 592GB checkpoint. What is fixed is the attention gap; the memory gap is untouched. [Reproducer and root cause](journal/upstream/05-no-sm8x-attention-path.md), [session record](journal/u2-hopper-design.md#session-26-2026-07-23-founder-8x-a100-node-sm_80-ours-runs-day-0-cannot).

**2. `rel_bias` is accepted and silently dropped on every non-Blackwell arch.** The library allocates the padded bias tensor, launches the `ShearingBias` pre-kernel, then constructs a forward kernel that has no bias argument, and returns plain bias-free attention as if it were the biased result. No error, no warning. In **stock tml-fa4 at the pinned commit `13374f0c`**, `grep -ci bias` returns zero in `flash_attn/cute/flash_fwd_sm90.py`, `flash_fwd.py` and `flash_fwd_sm120.py`, against 236 in `flash_fwd_sm100.py`. Name the flag or the number does not reproduce: `-c` counts matching *lines*, so 236 is lines, and `grep -oi bias file | wc -l` counts raw occurrences and returns 346. The zeros are zero either way, which is the part that matters. Run that grep against a stock checkout, not against this repo: our `kernels/tml_fa4_modified/` copies of those files are the fix, and they match on `bias` by design. This is silent numerical wrongness in a shipped stack, found by the parity oracle rather than by a crash. A correct bias implementation cannot be free: ours costs **21.6%** over biasless attention on the batch-1 64K decode case, 895 us against 736 us, both timed in the same container. This paragraph used to put that cost at ~~15.8%, 853 us against 736 us~~, and **that figure is withdrawn**: the 853 us was our own kernel skipping almost all of its bias gather at decode, so it understated the cost rather than measuring it. The day-0 `rel_bias` path still costs nothing there, because it does no bias work at all. [Full report with a runnable reproducer](journal/upstream/01-rel-bias-silently-ignored-non-blackwell.md), [why the old figure is withdrawn](journal/regression-sm90-bias-shift.md).

**3. `pack_gqa` redefines what a score-tile row means, and nothing in the API says so.** It folds eight GQA query heads into the seqlen mode, so a 128-row score tile stops being 128 sequence positions and becomes 16 positions by 8 heads. Any row-indexed feature is silently wrong from that point on unless it also packs. It is not opt-in: a heuristic turns it on for every GQA model. The bias feature survives on Blackwell only because the sm_100 kernel and the shear writer each remembered to pack, independently and undocumented. It cost us 17 debugging sessions, and the diagnostic that ended it was a stride: the wgmma submode that steps 8 tile rows had stride 81920, which at the anchor shape is exactly one sequence row of the bias tensor. Anyone adding a row-indexed feature to these kernels will hit this. [Contract hazard report](journal/upstream/04-pack-gqa-row-semantics.md), [the debugging account](journal/u2-hopper-design.md#the-key-insight-2026-07-20-why-manual-coords-fail-on-wgmma).

**4. A measured recipe for fitting a 592GB checkpoint on 640GB of HBM.** Seven attempts on 8x H100. Six failed, each for a different measured reason: a missing worker-side dependency, an infeasible 16384 context, a CUDA graph capture 394MB short at utilization 0.95, a warmup activation 782MB short at the same setting, utilization 0.90 leaving no room for KV at all, and 0.93 still short with KV at 0.58GB against 1.27 needed. The seventh works. The window is narrow and the sensitivity is measured at roughly 0.77GB of KV headroom per 0.01 of utilization. The configuration is in [Serving the full model](#serving-the-full-model) and the failure sequence is in the [session record](journal/u2-hopper-design.md#session-28-2026-07-24-8x-h100-first-full-model-serving--logit-gate).

**5. Agreement with the stock build on the real 975B model, on prompt positions only.** 32 of 32 prompts, same checkpoint, TP8, 2369 positions compared, every token identical and a mean logprob delta of 0.048 where the platform's own same-build batch noise floor is 0.150. **Read that qualifier, because we did not write it here originally and it is load bearing.** The stock-against-ours comparison ran `max_tokens=0` with `echo=True`, so all 2369 of those positions are echoed prompt tokens and not one of them was generated. The token half of it is close to a tautology: the script's docstring at the time said a token mismatch in an echoed prompt is impossible with one tokenizer. So this is prefill agreement, the logprob delta is the informative half, and it says nothing at all about decode. The same script does contain one sub-gate that generates tokens, 32 each on 4 prompts, but it compares a build against *itself* at two batch sizes, so it is the noise floor and not a stock-against-ours result. It is also one of the three gates that failed to catch [the shear-shift defect](journal/regression-sm90-bias-shift.md). The logprob half is recorded as a formal failure against its a-priori tolerance, honestly, and is explained in [What is not measured](#what-is-not-measured). [Artifact](journal/remote/gate_logit_parity_8xh100.json).

**And it is faster.** On an H100 the attention kernel runs 2.66x faster than the path vLLM actually serves with at batch-1 64K decode, 2.75x at batch-32 64K, 2.10x at batch-32 8K, and 1.44x at 8K global prefill. On sliding-window prefill it is 1.27x **slower**, and 55 of Inkling's 66 layers are sliding-window. All five cases are in the table below.

> **Those decode numbers are not the ones this page carried until 2026-07-25.** It said 2.7x to 2.8x and 2.5x, and **those figures are withdrawn**, because the kernel that produced them was applying Inkling's relative-position bias to one KV block instead of ten while the baseline it was divided by gathered every score correctly. The defect is fixed, the fix passes a new decode-shape parity gate 7 of 7 on an H100, and the numbers above are the like-for-like re-measurement. They are lower. The withdrawn rows are struck through rather than deleted, and being close to the sound numbers does not make them sound. [Full account](journal/regression-sm90-bias-shift.md).

Read the claim as written. It is a kernel microbenchmark on one GPU, not a serving result. **No end-to-end serving speedup is claimed anywhere in this repository**, every throughput row in [LEDGER.md](LEDGER.md) is `null`, and attention is a slice of serving time that the MoE layers and the big GEMMs dominate. If you are about to quote "2.66x" without the word *attention* in the same sentence, the number does not support it. If you are about to quote "2.7x" or "2.8x" at all, that number is withdrawn and the figures and git history that still show it are superseded.

## What is measured

Read [What is not measured](#what-is-not-measured) before quoting any number from this section.

There is exactly one day-0 baseline on Hopper: `score_mod`, the per-score callback vLLM actually serves with. Everything is measured against that. Plain attention carries no bias at all and is the floor this feature can approach but never beat, not a baseline we are entitled to claim a win over.

This table is the **post-fix** measurement. Every row comes from one container in which our kernel and the baseline were timed minutes apart, after the sheared-bias shift defect described below was fixed. Microseconds per iteration, one H100 SXM5, lower is better.

| Case | Ours | day-0 `score_mod` | plain, no bias | Result |
|---|---|---|---|---|
| prefill 8K, global | 3354 | 4821 | | **1.44x faster** |
| prefill 8K, sliding window | 1225 | 965 | | **1.27x slower** |
| decode, batch 1, 64K KV | 895 | 2376 | 736 | **2.66x faster** |
| decode, batch 32, 64K KV | 868 | 2389 | 736 | **2.75x faster** |
| decode, batch 32, 8K KV | 146 | 308 | | **2.10x faster** |

Source: [journal/remote/validate_s27_decodefix/](journal/remote/validate_s27_decodefix/). The same run has `parity_rel_chunked_decode` at 7 of 7 and `parity_fa4_rel` at 3 of 3, so every timing above has a passing parity run behind it **on its own shape family**, which is a sentence this page could not honestly have written before 2026-07-25.

> ### Withdrawn: the decode figures published before 2026-07-25
>
> | Case | Ours | day-0 `score_mod` | as published |
> |---|---|---|---|
> | decode, batch 1, 64K KV | ~~853 / 860~~ | ~~2327 / 2412~~ | ~~2.7x to 2.8x faster~~ |
> | decode, batch 32, 64K KV | ~~855 / 866~~ | ~~2391 / 2383~~ | ~~2.8x faster~~ |
> | decode, batch 32, 8K KV | ~~124 / 124~~ | ~~304 / 304~~ | ~~2.5x faster~~ |
> | session 24, decode batch 1, 64K KV | ~~906~~ | ~~2375~~ | ~~2.6x faster~~ |
>
> One provenance note on that last row, found while verifying this page. The 906 is session 24's own measurement (`microbench_attn_day0_native_sm90_session24.json`, 905.6). The 2375 is **not**: session 24's own `score_mod` reading is 2433.0, and 2375 is the session-1 H100 figure (2374.9, `journal/remote/h100-session1.md`). So that row was a cross-session ratio as well as a not-like-for-like one. Both defects are moot now that it is withdrawn, but it is recorded rather than tidied.
>
> Our `sm_90` kernel derived its sheared-bias shift from `128 * (m_block + 1)`, which is the `seqlen_q == seqlen_k` special case of the layout contract. At batch-1 decode with 64K of KV the shift came out +9 where the contract requires -502, and because the consumer skips any tile outside the buffer, the kernel applied the learned relative-position bias to exactly one KV block of 512, the oldest one, where ten should have had it. The `score_mod` baseline gathered every score correctly. The two columns were not doing the same work, so the ratio measured the omission rather than the design. **These rows are withdrawn, not corrected.** The re-measured numbers in the table above are close to them, and that is luck rather than vindication: a number that was nearly right by accident was still unfounded. [Full account](journal/regression-sm90-bias-shift.md).

The prefill rows were never affected, because at `seqlen_q == seqlen_k` the specialisation is an identity. They now have three independent runs behind them on two machines and two software stacks: ours 3309, 3307, 3354 against `score_mod` 4799, 4841, 4821 on global, and ours 1223, 1221, 1225 against 957, 863, 965 on sliding window. That is 1.44x to 1.46x faster and 1.27x to 1.41x slower respectively, and the 1.44x is the fixed kernel, which pays 0.7% at prefill for computing the general form of the shift. Earlier sources: [session 25](journal/remote/microbench_attn_day0_session25_h100.json) with its [baseline](journal/remote/microbench_attn_scoremod_session25_h100.json), and [session 26b](journal/remote/validate_s26b_h100x1_route/).

**What the correct bias costs.** A control run put the defective shift back so the cost of the fix could be measured: batch-1 64K decode 852 to 895, so +5.0%; batch-32 64K 852 to 868, +1.9%; batch-32 8K 122 to 146, +19.7%; global prefill 3330 to 3354, +0.7%; sliding-window prefill 1211 to 1225, +1.1%. The two prefill figures are the cost of the extra index computation alone, since the shift is unchanged there. The batch-32 8K figure is why that case went from 2.5x to 2.10x. Those two runs are different containers of the same box class, so read them as percentages good to about a point. Source: [journal/remote/validate_s27_brokencontrol/](journal/remote/validate_s27_brokencontrol/).

**Our kernel is the reproducible half of this comparison.** Across the pre-fix runs it moved by at most 1.9%, and on the two prefill cases by 0.13% and 0.06%. The `score_mod` baseline moves by up to 10.6% across those three runs, which is the entire reason the sliding-window loss is quoted as a range rather than a figure: our number went 1223 to 1221 to 1225, the baseline went 957 to 863 to 965. Treat these ratios as good to about one decimal place, not two, and treat the day-0 side as the noisy one. Note also what reproducibility does not buy: a kernel can be reproducible to 0.1% while doing the wrong amount of work, which is exactly what the withdrawn rows were.

**The decode rows above rest on a single post-fix container.** The prefill rows have three runs; decode has one. Reproducing the decode measurement on a second machine is item 1 in [What comes next](#what-comes-next).

![Attention kernel latency against the path vLLM ships](docs/figures/fig1_latency.png)

The same five cases as the table, on a log scale, one H100 SXM5. The open grey markers are biasless attention, a floor rather than a baseline. This figure carried the withdrawn pre-fix decode ratios until 2026-07-25, which is a problem a caption cannot fix, because a figure travels without its caption. It is now generated from the artifacts by [`scripts/make_fig_latency.py`](scripts/make_fig_latency.py), which reads the two `validate_s27_decodefix` JSONs and types in no value at all, and it prints its source path into its own subtitle. Re-run `py scripts/make_fig_latency.py` and the bytes should not change. A published chart that cannot be regenerated from the artifacts is a claim that cannot be corrected, and that is why the previous one had to be pulled rather than relabelled.

Our prefill totals include the ShearingBias pre-kernel, which the `score_mod` path does not need: in the post-fix run, 827.6 us of the 3353.7 and 459.5 us of the 1224.5, with the attention kernel itself at 2523.5 and 762.3. That pre-kernel is why the sliding-window case loses. It costs 3.5 us on batch-1 decode, so it is a prefill problem only. The obvious fix, folding it into the writer that produces the bias one step earlier, is implemented, and we measured it: **it makes things worse, not better.** See [Removing the pre-kernel](#removing-the-pre-kernel). Closing this case needs a cheaper way to produce the sheared layout, not the removal of the launch.

The `scoremod` JSON also contains two much slower paths, `relproj` at 7319 us and `relprojT` at 5185 us on the batch-1 decode case in the post-fix run, and 7195 and 5155 in session 25. **Those are ours, not vLLM's.** They are the register-resident designs we tried and abandoned, kept in `kernels/relproj_score_mod.py` and measured in the same runs so the dead ends stay on the record. Dividing our shipped kernel by our own failed prototype would produce a larger number and would not mean anything. That mistake was made once here and corrected publicly.

### Everything else that was gated

| Result | Numbers | Where it was run | Evidence |
|---|---|---|---|
| Same tokens as stock on the real model, **on prompt positions only** | 32 of 32 prompts, 2369 positions, all identical, mean logprob delta 0.048 against a same-build noise floor of 0.150. This comparison ran `max_tokens=0` with `echo=True`, so it generated no tokens and exercised no decode call. The token half is near-tautological by the script's own docstring. The 0.150 floor comes from the one sub-gate that does generate tokens, and that sub-gate compares a build against itself. | 8x H100, TP8, full NVFP4 checkpoint | [JSON](journal/remote/gate_logit_parity_8xh100.json), [what it does not cover](journal/regression-sm90-bias-shift.md) |
| The only working attention kernel on Ampere | Parity 3 of 3 green on A100, **all three cases `seqlen_q == seqlen_k`**, so the certified family is full prefill and no decode-shape correctness result exists on `sm_80` on any hardware. Every day-0 path fails to run. Per-op only; the checkpoint does not fit on A100. This is a capability claim and it survives the withdrawal in the row below. | A100 SXM4 40GB | journal session 26, [the caveat](journal/regression-ampere-tile-sweep.md#the-support-claim-survives-with-one-caveat-it-did-not-carry-before) |
| ~~Tuned tile sizes for Ampere~~ **WITHDRAWN 2026-07-25** | ~~10.1% faster on batch-1 decode~~, ~~18.2% on the 32-sequence case~~, ~~18.7% on a post-deploy re-run~~. **Do not quote any of these.** `harness/tune_sm80.py` times decode shapes, `T_q=1` against `T_k=65536`, while its `parity_ok()` built one `cu_seqlens` and passed it as both `cu_seqlens_q` and `cu_seqlens_k`, so it verified `seqlen_q == seqlen_k`. The generic kernel was wrong on exactly the family its own gate never exercised, so the winning tile size was selected under a reader that was addressing the bias out of its own tile domain at the timed shapes. This is the harness this page cites as the one place a code rule enforces parity before a timing is reported, which makes it the sharpest available example of the failure. The two **prefill** rows of the same sweep are `seqlen_q == seqlen_k` and survive, as does the `tile_n=128` collapse. | A100 SXM4 40GB | [the withdrawal record](journal/regression-ampere-tile-sweep.md), [the JSON is kept](journal/remote/tune_sm80_a100.json), [the shift defect](journal/regression-sm90-bias-shift.md) |
| Reproduces on a second machine and a different software stack | Parity green again on torch 2.11/cu130 after the first run used cu129. ~~The decode gap widened rather than shrank.~~ The decode half of that statement is withdrawn with the decode ratios; what reproduced is the parity result and the prefill timings. | A second H100 SXM5 | session 24 and session 25 JSON |
| Inkling fits and serves on 8x H100 | 592GB of weights on 640GB of HBM. The working configuration is in [Serving the full model](#serving-the-full-model). | 8x H100 | journal session 28, prose |

A note on the last column, because the rule matters more than any row in the table. **Not every number on this page has a JSON artifact.** The attention latency table, the full-model gate and the Ampere tile sweep do, though the Ampere sweep's decode percentages are withdrawn and its JSON is now kept as the record of a withdrawal rather than as the backing for a live claim. Some do not, and those are marked **journal-only** wherever they appear: the Nsight Compute percentages, the 18.7% post-deploy re-run (**also withdrawn**, see the Ampere row above), the `sm_120` relative timings, and the memory recipe with its 0.77GB-per-0.01 sensitivity. Journal-only means one of us read a number off a tool and wrote it down, with no machine-readable record you can re-parse. Treat those as weaker evidence than the rest, because they are. The label at the point of use is the authority and [journal/remote/README.md](journal/remote/README.md) holds the canonical list; this page deliberately does not restate a count, because every document that carried one drifted out of step with the others.

Every timing quoted here has a passing parity run behind it, as a rule we follow, and in one place as a rule the code enforces: `harness/tune_sm80.py` refuses to report a configuration's timing unless that configuration's own parity run was green. The other harnesses do not enforce it and cannot. `harness/microbench_attn_day0.py` will happily time a stock build that silently drops the bias, which is exactly upstream finding 01, and its own docstring says so. During the 17 sessions on Hopper, one flight reached near-plain-attention timing while producing wrong output. It is recorded as a failed kernel, not a win.

**That rule was weaker than it reads, and the withdrawn decode rows are the proof.** The parity run behind them was green, on three cases that all had `cu_seqlens_q == cu_seqlens_k`, while the decode timings were taken at `seqlen_q != seqlen_k`. A passing parity run is evidence only for the shapes it ran. The rule that follows is now written down in [docs/METHODOLOGY.md](docs/METHODOLOGY.md#parity-oracle-discipline): a parity suite has to cover every shape family the kernel dispatches to. `harness/parity_rel_chunked_decode.py` covers the family that was missing, it passes 7 of 7 on Hopper, and it was also run against a deliberately re-broken kernel so that it has been observed failing on the defect it exists for. A gate only ever seen passing is not known to be a gate. That control run immediately earned its cost: under the original tolerance one defective case passed, so `TOL_MEAN` was tightened from 5e-3 to 5e-4, where the worst legitimate case is 6.96e-05 and the best defective one is 3.28e-03.

![Full-model agreement with the stock build](docs/figures/fig2_correctness.png)

Full-model agreement, 8x H100 at TP8. The bars are mean absolute per-token logprob difference, lower is better, and the two grey bars are the floor. Read the top bar and the two grey bars as measuring different things, which the figure does not say and which we got wrong in an earlier caption: the **top** bar is ours against stock and ran `max_tokens=0` with `echo=True`, so all 2369 of its positions are echoed prompt tokens and none was generated; the two **grey** control bars ran `max_tokens=32`, so 128 of their 348 positions per build are generated tokens. That asymmetry is a flaw in the comparison, not a subtlety: the cross-build number that matters is measured on prefill positions only, and the floor it is being read against is measured partly on decode positions. **The logprob half is recorded as a failure**, and the figure shows why the failure is not informative about our kernel: the a-priori tolerance sits below the platform's own batch reproducibility, and both same-build controls failed too. The figure also renders "32 of 32 prompts produced identical greedy tokens" without the prompt-positions qualifier, and unlike [fig1](#what-is-measured) it has no generator, so it cannot be corrected without being redrawn. That is item 11 in [What comes next](#what-comes-next).

## What is not measured

Read this section before quoting any number above.

- **The decode ratios published before 2026-07-25 are withdrawn, and the ones that replace them have one run behind them.** The old figures timed a kernel that was applying the relative-position bias to one KV block instead of ten. The new figures come from a single post-fix container, where the prefill figures have three. Decode has not been reproduced on a second machine. [Write-up](journal/regression-sm90-bias-shift.md).
- **The generic kernel had the same specialisation, the fix is in the tree, and no GPU has run it.** `kernels/tml_fa4_modified/flash_fwd.py:917-919` and `:1359` carried the identical expression, and that is the path used on A100 and RTX 5090, so those two architectures had the defect Hopper had. Both sites were changed in `9b63979`. **That change has never executed on an A100 or a 5090, and reading it a second time found it wrong in a new way**: it multiplies a block count that `BlockInfo` returns in units of `tile_n` by the fixed 128 the shear writer actually uses, which is correct only at `tile_n == 128`, and the generic path selects 32 or 64. The full account and the arithmetic are in [the Ampere record](journal/regression-ampere-tile-sweep.md#a-blocking-finding-the-ported-fix-does-not-match-the-writers-contract), including a second candidate form now in the tree that no gate in this repository can distinguish from a correct one, because every gated shape has `seqlen_k` a multiple of 128 and real serving does not. Read the consequence conservatively: **`sm_80` and `sm_120` now have no correctness result for the code in the tree, on any shape family.** The published 3/3 was measured on the pre-fix file at `seqlen_q == seqlen_k`. `harness/parity_rel_chunked_decode.py` has run on Hopper only, and `parity_rel_bias_coverage.py`, `parity_rel_paged.py` and `parity_rel_varlen_batch.py` have not run anywhere. One A100 session closes all of it.
- **The Ampere tile-sweep percentages are withdrawn.** 10.1%, 18.2% and 18.7% selected a tile size while the generic reader was addressing the bias outside its own tile domain at the timed shapes, under a parity gate that checked a different shape family. [Record](journal/regression-ampere-tile-sweep.md). The `sm_80` support claim does not depend on them and survives.
- **No end-to-end serving speedup is claimed.** The throughput rows in [LEDGER.md](LEDGER.md) are `null`. We ran the sweep on 8x H100 and lost the results when a safety watchdog killed the box mid-run. That was our own bug and it is [written up](journal/u2-hopper-design.md#session-28-postscript-e2e-curves-lost-to-a-watchdog-race-orchestrator-error) instead of quietly retried.
- **We lose on sliding-window prefill.** In the post-fix run that the table above quotes, 1224.5 us against 965.4 us for the shipped path, both timed in one container, and 1223 against 957 in session 25 and 1221 against 863 in session 26b. The ShearingBias pre-kernel costs 459.5 us of our total there and the attention kernel alone does not make it back. 55 of Inkling's 66 layers are sliding-window, so this case matters. A fix exists and is correctness-validated but not yet speed-measured: see [Removing the pre-kernel](#removing-the-pre-kernel).
- Attention is only part of serving time. The MoE layers and the big GEMMs dominate. Do not assume a 2.66x kernel speedup becomes a 2.66x serving speedup. It will not.
- The decode microbenchmark packs its query rows into one sequence. True multi-sequence decode is measured separately and is slower per sequence: 459 us per sequence at 32 sequences by 64K KV on the fixed kernel, 14698 us for the batch. The pre-fix figure for the same case was 432 us per sequence and is withdrawn with the other pre-fix decode numbers. Neither is a serving result.
- **The full-model gate's stock-against-ours comparison never generated a token, and its logprob half is recorded as a fail.** That comparison ran `max_tokens=0` with `echo=True`, so both its halves compare echoed prompt positions and neither says anything about decode. Be precise about the scope, because a first pass at this caveat overstated it: the *script* did generate tokens, 32 each on 4 prompts, but only inside the same-build batch-consistency controls, which compare a build against itself and therefore cannot separate ours from stock. So no generated token was ever compared **across builds**. On the logprob half the a-priori tolerance turned out to be tighter than the platform's own batch reproducibility, and the same-build control failed too: ours-against-stock mean delta was 0.048 where the same-build noise floor was 0.150. Those two are not measured on the same position mix, which weakens the comparison further. The control failure is reported, not waived. Only the token half passed, and on echoed positions that half is close to a tautology.
- **A cross-build decode comparison is written but has not run.** `scripts/gate_logit_parity.py` in the working tree now requests a second comparison with `echo=False` and `max_tokens=32`, which is the only way this gate can reach `seqlen_k > seqlen_q`. It has never been executed, no artifact exists for it, and running it needs 8x H100 again. Until then the recorded gate is prefill-only, and note that the new script writes `parity_prompt_echo` and `parity_decode` where [the artifact](journal/remote/gate_logit_parity_8xh100.json) has a single `parity` key, so the two are not the same schema.
- **Blackwell is untested.** The code dispatches to `sm_100`, but no B200 was available while this was built. No number here comes from Blackwell hardware.
- **Below the kernel roofline gate.** ncu on 8K prefill measured 45.6% SM SOL and 55.9% memory SOL at 490 GB/s, occupancy 14.0%. The project's own bar is 90% of the binding roofline. This does not meet it. The recoverable costs are named in [What comes next](#what-comes-next).
- **The ncu numbers in the line above are the weakest evidence on this page, and you cannot check them.** The three `.ncu-rep` files exist, but they are several megabytes each and are gitignored, so what ships is our transcription of four metrics and nothing else. There is no CSV export in the repo either, and the profiling run was done by hand rather than from a script, so no committed script regenerates them. Exporting the section summaries to a committed CSV is on the list below. Until that happens, treat 45.6 / 55.9 / 490 / 14.0 as reported, not as verifiable.
- **Inkling was never served on A100, and the checkpoint does not fit there.** The Ampere result is a per-op attention kernel that passes parity. 8x A100 40GB is 320GB against a 592GB checkpoint. Closing the attention gap does not close the memory gap.
- The full-model gate compared tokens and logprobs between two builds. It is a correctness check, not a quality benchmark. We ran no downstream evals.
- RTX 5090 numbers are relative only. That machine is power-capped and on WDDM, and those timings live in the journal with no JSON artifact.
- U3 quantizes KV on write. Attention does not yet read the quantized cache directly.
- **The shear fusion is measured and it loses on prefill.** It costs 1019 us on global prefill and 561 us on sliding-window prefill, and saves 5 us on batch-32 decode. In session 26 it also could not run at all on `sm_90` on the read side, because the pre-sheared `bias=` path hit an unbound `n_block` in `flash_fwd_sm90.py`, which is why the recorded Hopper gate is 14/16. That defect is since fixed and the gate has not been re-run, so 14/16 is the last measured state rather than the current one. Ships off. See [Removing the pre-kernel](#removing-the-pre-kernel).
- **Split-KV decode is still unvalidated on any hardware.** Its first execution, on Hopper in session 26, hit the same `n_block` defect.
- The upstream bug reports are written but not filed. Report 03 targets a tracker whose duplicate check has not been run; see [Upstream bugs found](#upstream-bugs-found).

## Removing the pre-kernel

The sliding-window loss has one cause: our path runs a `ShearingBias` kernel that
`score_mod` does not need. It rewrites the relative-bias buffer into the sheared
layout the attention kernel reads. It costs 459.5 us of the 1224.5 us total in the post-fix run, and 461 of 1223 in session 25, which is the run the projection below was built on.

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
| H100 (`sm_90`) | Working, per-op and full-model, and the only arch with a decode-shape correctness result | Native wgmma kernel. Parity 3/3 at `seqlen_q == seqlen_k` and 7/7 on chunked prefill and decode shapes. 2.66x faster than the shipped path at batch-1 64K decode, 1.44x at global prefill, slower on sliding-window prefill, token-identical to stock on prompt positions of the real model. Decode carried [a bias-shift defect](journal/regression-sm90-bias-shift.md) until 2026-07-25, which is why the decode figures moved. |
| A100 (`sm_80`) | Working per-op at `seqlen_q == seqlen_k`, and the only attention kernel that runs | Parity 3/3 green, all three cases `seqlen_q == seqlen_k`. Day-0 cannot run here at all. The checkpoint does not fit on A100, so this is a kernel result, not a deployment. **The tile-tuning percentages are withdrawn**, see the row in [Everything else that was gated](#everything-else-that-was-gated). The generic kernel carried the same shift specialisation `sm_90` did; it is fixed in the tree at `b5f0f7e` but no A100 has been available since, so **the fix is unrun and decode correctness here is untested on any hardware**. |
| RTX 5090 (`sm_120`) | Working, per-op | Parity green at `seqlen_q == seqlen_k`, 2% to 10% faster than day-0 on that machine. The local headroom is structurally smaller than Hopper's. Timings relative only, journal-recorded, no JSON artifact. Same generic kernel as A100, so the same unrun shift fix and the same untested decode shapes. |
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
python harness/parity_rel_chunked_decode.py # chunked prefill and decode, seqlen_q != seqlen_k
python harness/parity_rel_bias_coverage.py  # does every KV block get its bias tile, at depth
python harness/parity_rel_paged.py          # paged KV, the only shape vLLM calls with
python harness/parity_rel_varlen_batch.py   # more than one sequence in the batch
python harness/parity_kv_fp8.py            # FP8 KV writes, needs u3_fp8_kv.py
python harness/parity_shear_writer.py      # shear layout contract
python harness/parity_shear_fusion.py      # shear fusion, needs u2_shear_fusion.py
python harness/microbench_attn_day0.py     # our kernel, real shapes
python harness/microbench_attn_scoremod.py # the day-0 baseline, same shapes
python harness/tune_sm80.py                # tile sweep, A100 only; its numbers are WITHDRAWN
```

**Three of those have never run on a GPU.** `parity_rel_bias_coverage.py`, `parity_rel_paged.py` and `parity_rel_varlen_batch.py` were written after the shift defect, for shape families nothing in this repository had ever reached, and they are unexercised. Do not read their presence as coverage. `tune_sm80.py` is the harness whose parity gate checked a different shape family from the one it timed, which is why [the Ampere percentages are withdrawn](journal/regression-ampere-tile-sweep.md); its `parity_ok()` now covers the timed family, at a shorter context than the timed one.

Run these from inside the vLLM checkout with its environment active. `parity_fa4_rel.py` decides whether a kernel ships on the full-prefill family: three semantic cases, green means 3 of 3 inside tolerance, and a skipped backend counts as red. **It is not sufficient on its own**, which is the lesson of [the shear-shift defect](journal/regression-sm90-bias-shift.md): all three of its cases pass `cu_seqlens_q == cu_seqlens_k`, so `parity_rel_chunked_decode.py` has to pass too before a decode or chunked-prefill number means anything. Run the two microbenchmarks on the same box in the same session or the comparison is not controlled.

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

1. Reproduce the post-fix decode measurement on a second machine, because it currently rests on one container while every prefill figure has three. **Then one A100, to un-strand the Ampere half of this project.** The shift fix is ported to the generic kernel in `9b63979`, reading it again found that port dimensionally wrong for every `tile_n` that path selects, and `b5f0f7e` corrected all three sites. **None of that has ever executed on an `sm_80` or `sm_120` GPU**, so it is three commits of unrun code and should be treated as such. Then run `parity_fa4_rel.py` at each swept tile size, `parity_rel_chunked_decode.py`, and the three gates that have never run anywhere, and only then re-run `harness/tune_sm80.py` to replace [the withdrawn tile percentages](journal/regression-ampere-tile-sweep.md#what-re-measurement-requires). That session is unpriced: no A100 rate is committed in this repo.
2. Kill the ShearingBias cost, which is 25% to 38% of our prefill total and the whole reason the sliding-window case loses. Folding it upstream into `qkvr_prep` is **done and refuted**: the sheared writer costs more than the launch it removes, measured, see [Removing the pre-kernel](#removing-the-pre-kernel). What is left is to build the sheared tile inside the attention kernel, in shared memory, per tile, and never materialize the padded buffer at all. That is a real kernel change rather than a re-plumbing, and it is the honest next attempt.
3. Split-KV decode for `sm_90`. Batch-1 decode is parallelism-bound, not bandwidth-bound: 64 CTAs on 132 SMs, and DRAM at 7% with occupancy at 14%. The CTA count is structural; the two percentages were profiled on the kernel that was skipping its decode bias gather, so re-profile them alongside item 1. Splitting the KV range is the fix either way.
4. Re-enable `intra_wg_overlap` and `pack_gqa`, both forced off to get the bias path correct. Both cost prefill throughput today. Packed-bias addressing on `sm_90` is exactly the problem the `sm_100` path already solves.
5. Blackwell validation when hardware is available.
6. U3 read path, so attention consumes the FP8 cache directly.
7. Re-run the serving sweep, pulling artifacts after every config so a dead box costs one config instead of everything.
8. File the upstream reports, after running the duplicate check against `vllm-project/flash-attention` that has not been run.
9. Commit a CSV export of the ncu section summaries, so the roofline numbers are checkable rather than transcribed.
10. Re-run the shear-fusion gate on `sm_90`. It scored 14/16 there because both attention cases hit the `n_block` defect that is now fixed; the writer half is already 14/14 bit-exact on Hopper. The speed question is answered and the answer is no.
11. Give `fig2_correctness.png` and `fig3_status.png` generators, the way `fig1_latency.png` now has [`scripts/make_fig_latency.py`](scripts/make_fig_latency.py). Both are binaries with no source. `fig2` currently renders "32 of 32 prompts produced identical greedy tokens" without the prompt-positions qualifier that the prose around it now carries, and it cannot be corrected without being redrawn by hand. That is the same trap that made the pre-fix latency figure uncorrectable, and it is a documented lesson rather than a hypothetical.

Longer-term units (MoE grouped GEMM, router fusion, QKVR fusion, CUDA graphs, batch-aware dispatch) are tracked in [LEDGER.md](LEDGER.md).

## How we handle numbers

[LEDGER.md](LEDGER.md) contains no estimates. Every cell is a measurement or the word `null`. The rules are written down in [docs/METHODOLOGY.md](docs/METHODOLOGY.md) and they are the reason several fields on this page are empty rather than filled with something plausible.

Failures get the same space as wins. The GPU capacity lost to bugs in our own tooling, the serving results lost to a watchdog race, the seventeen sessions before the kernel was correct, and the overstatements we published and then corrected are all in the journal. It is the working record, not a highlight reel.

The most recent correction is the largest. The decode speedups were the headline of this page, they were measured twice on two machines, and they were unfounded because our kernel was doing less work than the baseline it was divided by. The re-measurement then landed within a few percent of them, which is the most tempting situation to say nothing about. They are struck through above rather than deleted, and the mechanism, the three gates that missed it and the lessons are in [journal/regression-sm90-bias-shift.md](journal/regression-sm90-bias-shift.md). A repository that only published its wins would have quietly swapped the numbers and moved on.

## What to do next

Pick the one that matches why you are here.

- **You want to know whether to believe the numbers, or you came here for the 2.7x.** The 2.7x is withdrawn and the current figure is 2.66x from a different run; start at [journal/regression-sm90-bias-shift.md](journal/regression-sm90-bias-shift.md) and decide for yourself whether the withdrawal is honest. Then open the JSONs linked under [What is measured](#what-is-measured) and divide the numbers yourself. Then read [What is not measured](#what-is-not-measured), which is where the load-bearing caveats are, and [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the rules the numbers were produced under.
- **You maintain vLLM, tml-fa4 or flash-attention.** Read [`01-rel-bias-silently-ignored-non-blackwell.md`](journal/upstream/01-rel-bias-silently-ignored-non-blackwell.md) first. It is a silent wrong-output path in shipped code, it has a fifteen-line reproducer that does not depend on this repository, and the fix can be a `NotImplementedError`. Report [04](journal/upstream/04-pack-gqa-row-semantics.md) is the one to read before anyone adds another row-indexed feature to those kernels.
- **You want to run it.** [Reproducing this](#reproducing-this), in the order given. `harness/parity_fa4_rel.py` and `harness/parity_rel_chunked_decode.py` are the two gates that decide whether a kernel is real, and both have to be green on the architecture you are claiming.
- **You want to work on it.** The open items are in [What comes next](#what-comes-next). Item 1 has two halves and they unblock different things: a second H100 is what puts a second run behind every decode number on this page, and one A100 is what un-strands the whole Ampere half, where three commits of shift-fix code have never executed and the tile percentages are withdrawn until they do. Item 3, split-KV decode for `sm_90`, is the largest measured headroom and is written up in the journal. The bar for a patch is in [CONTRIBUTING.md](CONTRIBUTING.md): a timing without a passing parity run is not accepted, and after 2026-07-25 that means a parity run on the shape family being timed.
- **You found a number here you cannot reproduce.** Open an issue. Say which number and what you got. That is the most useful thing anyone can send us.

## License

See [LICENSE](LICENSE).
