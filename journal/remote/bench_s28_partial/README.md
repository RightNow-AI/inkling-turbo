# Session 28: 8x H100 serving, stopped on purpose, and one durable measurement

`RUNS=3 CONCURRENCIES="1 4" modal run --detach scripts/modal_e2e_bench.py --step bench`
1.30 hours of 8x H100, $44.85. Stopped by hand at 9 of 12 stock runs.

## Why it was stopped

The container mounted `kernels/tml_fa4_modified/` at launch, which was **before**
the `bias_tile_shift` fix landed. So its `ours` build would have served a kernel
whose decode path applied the relative-position bias to one KV block out of 512,
and that block the oldest. Its decode throughput numbers would have been
unpublishable, and the defect was already proven four independent ways: the
arithmetic, the consumer code, a block-coverage table, and a deliberately broken
control run on an H100.

Stock had finished 9 of its 12 runs and those are valid and kept. Continuing
would have spent roughly $45 more on `ours` numbers that would be thrown away.
Relaunched instead with the fixed kernel; the resume logic skips the 9 completed
stock runs.

Kept: `prefill/1`, `prefill/4`, `decode/4`, three runs each, for `stock`.
Missing: `stock/decode/1`, which the cost model orders last because it is the
most expensive config.

## The measurement worth keeping

vLLM reports its own KV budget at startup, and nothing in this project had ever
read it:

```
kv_cache_tokens      4379
max_model_len_tokens 3072
max_concurrency_x    1.43
```

**On 8x H100 at the working memory recipe, a 592GB Inkling checkpoint leaves room
for 1.43 concurrent maximum-length requests.**

That single number settles a question this repo had been getting wrong in
planning. A throughput-latency curve across a batch sweep is not available for
this model on this hardware, at any concurrency setting, because the server
cannot hold the batch. Asking for `--max-concurrency 8` or `32` measures client
side queueing, not a server batch of 8 or 32.

Consequences, stated plainly:

- The `Throughput tok/s @ bs8 / bs32 / bs128` rows in LEDGER.md **cannot be
  honestly filled on 8x H100**. bs128 has no configuration that could produce it
  and bs8 and bs32 would be offered concurrency mislabelled as batch size.
- Any serving comparison from this hardware must be labelled `--max-concurrency`,
  the value handed to the client, and never `bs`.
- This is a property of a 592GB checkpoint on 640GB of HBM, not a property of our
  kernel. It applies equally to the stock build.

An earlier audit predicted about 2 concurrent sequences from arithmetic over the
memory recipe. The measured 1.43 confirms the prediction and is tighter than it.

## Greedy probe, stock build

Four prompts, temperature 0, 32 tokens, `--ignore-eos` deliberately NOT set. The
benchmark itself runs with `--ignore-eos`, which makes numerically garbage output
indistinguishable from good output in every metric it collects, so this probe is
the only thing in the run that looks at a token.

Stock produced coherent text on all four, including the correct answer to a
primary-colours multiple choice. Full completions in `validity.json`.

The build-versus-build comparison of these completions needs both builds, so it
belongs to the relaunched run. This file records the stock side.
