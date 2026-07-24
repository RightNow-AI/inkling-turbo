# LEDGER, Inkling-turbo

Every number here is measured-or-null. `null` = not yet measured. No estimates in this file, ever.
Every non-null number cites the artifact under `journal/remote/` that produced it. See [journal/remote/README.md](journal/remote/README.md) for what each artifact is.
Baseline reference (external claim, NOT ours): vLLM day-0 blog reports 380 tok/s/user (MTP8) / 140 tok/s/user (no MTP) on 4x GB200 TP8, 8K in / 1K out.

## E2E serving (remote, vs stock day-0 build, same checkpoint/quant/GPUs/SLO)

| Metric | Stock day-0 | Inkling-turbo | Delta | Evidence |
|---|---|---|---|---|
| Decode tok/s/user @ bs1 | null | null | null | null |
| Throughput tok/s @ bs8 | null | null | null | null |
| Throughput tok/s @ bs32 | null | null | null | null |
| Throughput tok/s @ bs128 | null | null | null | null |
| TTFT p50 prefill-heavy | null | null | null | null |
| TPOT p50 decode-heavy | null | null | null | null |

Why every cell above is `null`: the sweep was launched on 8x H100 and the box
self-terminated mid-run because of a watchdog race in our own orchestration. The
stock side had completed roughly 26 to 30 of 30 runs and none of it was
retrieved. Nothing partial is reported from it. Written up in
[journal/u2-hopper-design.md](journal/u2-hopper-design.md#session-28-postscript-e2e-curves-lost-to-a-watchdog-race-orchestrator-error).

## Kernel gates (ncu, >=90% of binding roofline)

| Unit | Kernel | Binding roofline | % achieved | Parity 32/32 | batched==bs1 | Profile |
|---|---|---|---|---|---|---|
| U1 | fused NVFP4 MoE grouped GEMM | null | null | null | null | null |
| U2 | rel-attn prefill | ncu ran, 90% gate NOT met (SM SOL 45.6%, MemSOL 55.9%, occupancy 14.0% at prefill 8k; decode b1 is parallelism-bound at 64 CTAs on 132 SMs, DRAM 7.2%) | below gate, ceiling documented in journal session 24 ncu | per-op 3/3 on sm_90+sm_80+sm_120; FULL-MODEL 32/32 greedy-token match vs stock on 8x H100 (2026-07-24); logprob-tolerance gate fail-as-specified with control (noise floor) also failing - see journal session 28 | batched==bs1 FAILS ON STOCK ITSELF (platform nondeterminism, documented) | session 24: decode b1 kv64k 905.6us vs day-0 score_mod 2375 (2.6x); prefill 8k 3362us vs day-0 score_mod 5368 (1.6x). Also measured on the same box: relprojT 4163 / relproj 6209 at decode and 8483 / 13049 at prefill, but those are OUR abandoned V1/V1.5 prototypes, NOT day-0 paths, and no speedup is claimed against them. journal/remote/microbench_attn_day0_native_sm90_session24.json + microbench_attn_scoremod_session24.json |
| U2 | rel-attn, session-25 reproduction (different box, torch 2.11/cu130) | see row above | see row above | per-op 3/3 native sm_90, max 1.56e-2 | not re-run | vs the only day-0 path (score_mod): decode b1 kv64k 852.6 vs 2326.6 (2.7x); decode b32 854.8 vs 2391.2 (2.8x); prefill_global_8k 3308.8 vs 4799.4 (1.45x); **prefill_swa_8k 1223.0 vs 956.5 = 1.28x SLOWER, we lose this case**. Reference points, not baselines: plain no-bias 736.0 (decode b1); our abandoned relprojT 5154.7 / relproj 7194.5. True 32-seq decode kv64k 13821.4 (431.9/seq). journal/remote/microbench_attn_day0_session25_h100.json + microbench_attn_scoremod_session25_h100.json |
| U2 | rel-attn on sm_80 (A100) | null | null | per-op 3/3 (2026-07-23) | null | no speedup ratio exists: every day-0 path raises NotImplementedError on SM8x, so there is no baseline. Support claim only. Tile sweep (parity-gated, journal/remote/tune_sm80_a100.json): tile_n=32 b1 decode 5350.1us vs untuned tile_n=64 5953.7 = 10.1% faster; 32-seq 60801.4 vs 74356.6 = 18.2% faster. Post-deploy re-validation on A100, 32-seq 60977.5 vs 75013.4 = 18.7% faster, journal session 27 (no separate JSON) |
| U2 | rel-attn split-KV decode | null | null | null | null | null |
| U3 | quantized paged KV | null | null | per-op 2/2 on sm_120 + sm_90/H100 + sm_80/A100 (2026-07-20/23); 32/32 pending | null | null |
| U4 | router+dispatch fusion | null | null | null | null | null |
| U5 | QKVR fused GEMM | null | null | null | null | null |
| U6 | graphs+overlap | n/a (timeline) | null | null | null | null |
| U7 | batch-aware dispatch | null | null | null | null | null |

## last_error

null
