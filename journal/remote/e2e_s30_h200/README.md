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

### Completed 2026-07-25: decode-heavy is now 3 against 3, and conc 1 exists

A second `ours` container finished the two decode runs and the three
concurrency-1 prefill runs that the first one's deadline refused. All 14
pre-existing run files are **byte-identical** between the two fetches, so this is
a superset and not a re-measurement.

Decode-heavy, 512 in / 1024 out, concurrency 8, **3 runs against 3**:

| metric | stock day-0 | Inkling-turbo | change |
|---|---|---|---|
| output throughput | 68.008 tok/s | **75.266** | **1.107x** |
| request throughput | 0.066 req/s | **0.074** | **1.107x** |
| median TPOT | 117.350 ms | **106.048** | **1.107x** |
| p99 end to end | 120655.0 ms | **109170.6** | **1.105x** |

The medians did not move when the two missing runs landed. What changed is the
evidence class: this was labelled indicative on one sample and is now a matched
median.

Prefill-heavy at concurrency 1, **3 runs against 3**, which had no `ours` side at
all before:

| metric | stock day-0 | Inkling-turbo | change |
|---|---|---|---|
| output throughput | 8.641 tok/s | **9.588** | **1.110x** |
| request throughput | 0.068 req/s | **0.075** | **1.110x** |
| median TPOT | 115.429 ms | **103.671** | **1.113x** |
| p99 end to end | 14894.3 ms | **13630.8** | **1.093x** |

**The intervals do not touch.** In all three matched comparisons, on output
throughput and on median TPOT, the slowest `ours` run beats the fastest `stock`
run:

| comparison | metric | stock range | ours range |
|---|---|---|---|
| prefill, conc 1 | out tok/s | [8.630, 8.651] | [9.557, 9.655] |
| prefill, conc 8 | out tok/s | [61.684, 66.784] | [67.772, 73.790] |
| decode, conc 8 | out tok/s | [67.712, 68.013] | [74.477, 75.289] |
| prefill, conc 1 | TPOT p50 ms | [115.414, 115.584] | [103.269, 103.931] |
| prefill, conc 8 | TPOT p50 ms | [117.656, 119.377] | [106.277, 108.067] |
| decode, conc 8 | TPOT p50 ms | [117.342, 117.562] | [105.805, 106.164] |

Three independent ratios at 1.106x, 1.107x and 1.110x across two mixes and two
concurrencies is why this is quoted as "about 10%".

**TTFT is not claimed, in any comparison.** Its medians favour ours everywhere,
by 1.065x to 1.111x, but one cold-start run per build makes the ranges overlap in
every case. The worst is TTFT p99 for `ours` at conc 1, which spans 130.073 ms to
1103.033 ms across three runs against a stock range of [143.702, 148.714]. A
median that flatters us inside a range that overlaps is not a result, and the
table above omits TTFT for that reason rather than reporting the flattering half.

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

The stored `kv_pool_match` and `greedy_match` flags read `None` in the first
fetch, because the `ours` container exited before its final validity write, and
the equality above was computed from the per-build records directly. **The second
container wrote them: both are now `true` in `validity.json`**, so the flags and
the hand computation agree.

One run is recorded as a failure rather than dropped. `ours/decode/conc1/run1`
was killed at 914 s by the per-run timeout instead of being allowed to burn to
the container cap, so it is `null` in the summary and decode at concurrency 1
has no `ours` side. `failed_runs: 1` and `aborted_early: true` in
`summary.json` are that run, not a kernel fault.

## What is missing from this run, stated plainly

- ~~Decode-heavy is 3 runs against 1.~~ **Closed**, it is 3 against 3.
- ~~Concurrency 1 has stock only.~~ **Partly closed**: prefill at concurrency 1
  is 3 against 3. Decode at concurrency 1 still has one stock run and no `ours`
  counterpart, so there is no comparison there and none is claimed. That config
  costs about 32 minutes per run, because 16 prompts at concurrency 1 is 16384
  strictly sequential decode steps, and the deadline refused it twice.
- **TTFT, at every concurrency and in both mixes.** Ranges overlap; see above.
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
