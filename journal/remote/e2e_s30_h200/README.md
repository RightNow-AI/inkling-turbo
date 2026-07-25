# Session 30: the first end-to-end serving A/B

8x NVIDIA H200 143771 MiB, TP8, the real 592GB NVFP4 checkpoint. Two containers,
one per build, $59.22 and $19.18. Ledger $141.51 of the $200 cap.

Both builds: `--tensor-parallel-size 8 --max-model-len 3072
--gpu-memory-utilization 0.94 --enforce-eager --seed 0`, with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Client is
`vllm bench serve`, `--percentile-metrics ttft,tpot,itl,e2el`, `--ignore-eos`,
seed 0. Stock is day-0: the unpatched router sends `sm_90` to the `score_mod`
gather. Ours is `kernels/tml_fa4_modified/` plus `u2_serving_route.py`, with u3
and the shear fusion asserted absent.

**Every E2E row in LEDGER.md was `null` before this run.**

## The result

Prefill-heavy, 2048 in / 128 out, concurrency 8, **3 runs against 3**, median:

| metric | stock day-0 | Inkling-turbo | change |
|---|---|---|---|
| output throughput | 63.709 tok/s | **70.453** | **1.106x** |
| request throughput | 0.498 req/s | **0.550** | **1.106x** |
| median TTFT | 377.961 ms | **340.199** | **1.111x better** |
| median TPOT | 117.863 ms | **106.566** | **1.106x** |
| p99 TPOT | 128.235 ms | **115.677** | **1.109x** |
| p99 end to end | 17492.6 ms | **15975.5** | **1.095x** |

Decode-heavy, 512 in / 1024 out, concurrency 8, **3 runs against 1**, so
indicative rather than a matched median: output throughput 68.008 to 75.266,
median TPOT 117.350 to 106.048, both 1.107x. The `ours` container died after its
first decode run, so that row is one sample and is labelled as one sample.

**Read the size of this number, because it is the honest one.** The attention
kernel is 2.66x faster at batch-1 64K decode in isolation. End to end it buys
about 10%. Attention is a slice of serving time and the MoE layers and the big
GEMMs dominate, which is exactly what this repository has said all along while
having no number to prove it. Now it has one, and it is 1.10x rather than 2.66x.
Anyone quoting the kernel ratio as a serving ratio is off by 2.4x.

The consistency is worth a second look: six metrics, two mixes, all landing
between 1.095x and 1.111x. A uniform per-token improvement is what you would
expect if attention time fell by a fixed fraction of every decode step, which is
what the microbenchmarks measured.

## Why a cross-container comparison is defensible here

The two builds ran in separate containers, because the first one used its whole
allowance on stock. That is a real weakness and the reason the KV check exists:

    kv_pool  stock 188160 tokens, max concurrency 61.25x
             ours  188160 tokens, max concurrency 61.25x
             drift 0.0000%, identical

Both servers profiled memory independently at util 0.94 and arrived at the same
KV budget to the token. Since the pool is what would otherwise confound a
throughput comparison, and it is identical, the comparison holds. Had it drifted
more than 2% the harness would have refused to publish it.

## Generated-token equivalence, for the first time

Four fixed prompts per build, temperature 0, `max_tokens=32`, and `--ignore-eos`
deliberately NOT set, so these are real generated tokens rather than echoed
prompt positions:

**4 of 4 completions byte-identical between stock and ours.**

This matters more than it looks. The full-model gate in
`gate_logit_parity_8xh100.json` ran `max_tokens=0` with `echo=True`, so its
"2369 tokens matched" is prompt echo, which that script's own docstring says
cannot differ under one tokenizer. **This is the first time in the project that a
generated token has been compared between the two builds**, and they agree on all
four prompts including a multiple-choice answer and two arithmetic continuations.

The stored `kv_pool_match` and `greedy_match` flags read `None` because the
`ours` container exited before its final validity write. The underlying per-build
records are present and the equality above was computed from them directly.

## What is missing from this run, stated plainly

- Decode-heavy is 3 runs against 1. Not a matched median.
- Concurrency 1 has stock only, 3 runs of each mix, no `ours` counterpart. The
  `conc1` decode config costs about 32 minutes per run, because 16 prompts at
  concurrency 1 is 16384 strictly sequential decode steps, and it was dropped to
  buy the matched `conc8` comparison instead.
- No `bs8 / bs32 / bs128` row is filled, and none can be from this data.
  `--max-concurrency 8` is the offered client concurrency. The server's own
  reported ceiling is 61.25x here, so 8 is genuinely batched rather than queued,
  but it is still not the same quantity as a server batch size and is not
  labelled as one.
- `--enforce-eager`, so no CUDA graphs on either side.
- H200, not H100. Same architecture, sm_90, so the kernel under test is the one
  the microbenchmarks measured. What differs is memory headroom: 8x143GB leaves
  room for 61 concurrent maximum-length requests where 8x80GB leaves room for
  1.43, which is why the H100 attempt could not produce a batched comparison at
  all.
