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
| U2 | rel-attn prefill | null | null | null | null | — |
| U2 | rel-attn split-KV decode | null | null | null | null | — |
| U3 | quantized paged KV | null | null | null | null | — |
| U4 | router+dispatch fusion | null | null | null | null | — |
| U5 | QKVR fused GEMM | null | null | null | null | — |
| U6 | graphs+overlap | n/a (timeline) | null | null | null | — |
| U7 | batch-aware dispatch | null | null | null | null | — |

## Spend

| Date | Resource | Duration | Cost | Purpose |
|---|---|---|---|---|
| — | — | — | $0.00 total | — |

## last_error

null
