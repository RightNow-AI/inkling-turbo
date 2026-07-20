# LEDGER — Inkling-turbo

Every number here is measured-or-null. `null` = not yet measured. No estimates in this file, ever.
Baseline reference (external claim, NOT ours): vLLM day-0 blog reports 380 tok/s/user (MTP8) / 140 tok/s/user (no MTP) on 4x GB200 TP8, 8K in / 1K out.

## E2E serving (remote, vs stock day-0 build — same checkpoint/quant/GPUs/SLO)

| Metric | Stock day-0 | Inkling-turbo | Delta | Evidence |
|---|---|---|---|---|
| Decode tok/s/user @ bs1 | null | null | null | — |
| Throughput tok/s @ bs8 | null | null | null | — |
| Throughput tok/s @ bs32 | null | null | null | — |
| Throughput tok/s @ bs128 | null | null | null | — |
| TTFT p50 prefill-heavy | null | null | null | — |
| TPOT p50 decode-heavy | null | null | null | — |

## Kernel gates (ncu, >=90% of binding roofline)

| Unit | Kernel | Binding roofline | % achieved | Parity 32/32 | batched==bs1 | Profile |
|---|---|---|---|---|---|---|
| U1 | fused NVFP4 MoE grouped GEMM | null | null | null | null | — |
| U2 | rel-attn prefill | null (ncu pending) | null (ncu pending) | per-op 3/3 sm_90 native H100 + 3/3 sm_120 (2026-07-20); 32/32 pending | null | decode b1 kv64k 905.6us vs 2375-6209 prod (2.6-6.9x), prefill 8k 3362us vs 8483-13049 (2.5-3.9x) — journal/remote/microbench_attn_day0_native_sm90_session24.json |
| U2 | rel-attn split-KV decode | null | null | null | null | — |
| U3 | quantized paged KV | null | null | per-op 2/2 local sm_120 (2026-07-20, parity_kv_fp8.py); arch-local + 32/32 pending | null | — |
| U4 | router+dispatch fusion | null | null | null | null | — |
| U5 | QKVR fused GEMM | null | null | null | null | — |
| U6 | graphs+overlap | n/a (timeline) | null | null | null | — |
| U7 | batch-aware dispatch | null | null | null | null | — |

## Spend

| Date | Resource | Duration | Cost | Purpose |
|---|---|---|---|---|
| 2026-07-18 | Lambda 1x H100 SXM5 (us-south-2) | 0.13 h | $0.55 | First contact: parity 3/3 green on sm_90 score_mod path; gate kernel 4.3us@T1 / 22.2us@T4096; found make_fragment cutlass-4.6.0 drift (3rd upstream break). Auto-terminated. |
| 2026-07-18 | Lambda 1x H100 SXM5 (us-south-2) | 0.14 h | $0.61 | Session 2: score_mod parity re-green post-rename; found 4th drift (fmax/atomicrmw nvvm branch keyed to CUDA 12.9 instead of DSL version) — tml-fa4 direct sm_90 path still blocked; gate timings reproduced (4.3/22.2us, stable across sessions). Auto-terminated. |
| 2026-07-18 | Lambda 1x H100 SXM5 (us-south-2) | 0.14 h | $0.61 | Session 3: all drift fixes green; sm_90 rel_bias path runs but is NUMERICALLY WRONG (max_diff 0.9-1.6, harness catch #1); indicative attn timings captured (decode kv64k b32~=b1: KV-bound, supports U3). Auto-terminated. |
| 2026-07-18 | Lambda 1x H100 SXM5 (us-south-2) | 0.13 h | $0.56 | Session 4 HONEST BASELINE: sm_90 production score_mod path = 2375us @ b1/kv64k decode vs 743us plain attention (3.2x overhead measured); num_splits=1 confirmed production on sm_90; gate stable 4th session. Auto-terminated. |
| 2026-07-20 | Lambda 1x H100 x3 (sessions 5-7) | 0.39 h | $1.65 | sm_90 U2 kernel first flights: s5 CRLF abort; s6 SPEED TARGET HIT (745.7us vs plain 736.9 vs prod 2411 = 3.24x available) but parity FAIL (misplaced bias, max ~2.3); s7 warpgroup-lockstep fix — no change (race ruled out). Debug-dump payload next. All auto-terminated. |
| 2026-07-20 | Lambda 1x H100 x14 (sessions 8-23) | ~1.9 h | ~$13.4 | sm_90 debug campaign -> resolution: sentinel probe found unthreaded mBias (11 dead flights); probe ladder proved scale/row; sm_100 tiled-copy insight; SESSION 23: generic-routed parity 3/3 GREEN ON H100 (sm_90 correctness ACHIEVED), routed speed 31x slow = reference-only. All auto-terminated. |
| | | | **~$17.4 total** | |

## last_error

null
