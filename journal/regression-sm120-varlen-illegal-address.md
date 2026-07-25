# Open defect: the generic kernel faults on some multi-sequence varlen batches

Found 2026-07-25 on a local RTX 5090 Laptop GPU (`sm_120`, capability 12.0),
torch 2.11.0+cu130, by running `harness/parity_rel_varlen_batch.py` for the first
time. Cost nothing. **Not fixed.**

This is the production call shape. vLLM serving batches multiple sequences into
one varlen call on every step. It is also the shape no gate in this repository
had ever constructed: `parity_fa4_rel.py`, `parity_rel_chunked_decode.py` and
`parity_shear_fusion.py`'s attention cases all pass a single `cu_seqlens` pair
describing **one** sequence.

## Symptom

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
cutlass.cutlass_dsl.tvm_ffi_provider.CUDADialectError:
  cudaErrorIllegalAddress (error code: 700)
```

Reproduced with `CUDA_LAUNCH_BLOCKING=1`, so the fault is at the launch that
caused it and not a deferred report.

`harness/parity_rel_varlen_batch.py` scores **1 of 12** on `sm_120`. Only the
single-sequence control passes. The first multi-sequence case faults, and every
case after it fails as a cascade, because a 700 poisons the CUDA context for the
rest of the process.

## What actually triggers it, isolated one variable at a time

Each probe in its own subprocess, because of the context poisoning. `rel_bias`
present unless stated.

| sequences | total_q | result |
|---|---|---|
| `[512]` | 512 | OK |
| `[256]` | 256 | OK |
| `[137]` | 137 | OK |
| `[256, 137, 64]` | 457 | **OK** |
| `[256, 256]` | 512 | **cudaErrorIllegalAddress** |
| `[128, 128]` | 256 | **cudaErrorIllegalAddress** |

So it is **not** simply "more than one sequence": three sequences of 256, 137 and
64 pass. The two failing cases are the two whose `total_q` is an exact multiple
of 128, and the passing multi-sequence case is the one whose `total_q` is not.
That is a correlation over six data points and a hypothesis, not a root cause.
The bias buffer is allocated `(total_q + tile_m, num_head, rel_extent_padded)`,
so a block count derived from `total_q` rather than from the padded row extent
would come out one tile short exactly when `total_q % 128 == 0`. That is the
first thing to check and it has not been checked.

## What is NOT the cause

**Not the harness.** The harness passes `rel_bias` in the natural
`(total_q, H, rel_extent)` layout with cumulative `cu_seqlens`, which is the
documented varlen contract. The library allocates and shears the padded buffer
itself inside `_flash_attn_fwd`, so the caller cannot get its shape wrong, and
the interface's own shape assertion on that buffer passes. An `AssertionError`
would have fired before any kernel launch; instead a kernel launched and read out
of bounds.

**Not today's shear-shift work.** The failing shapes have zero cached context, so
`seqlen_q == seqlen_k` per sequence, which is the family every form of the shift
expression agrees on and the family that was already green.

**Not `sm_90`.** `microbench_attn_day0.py`'s `batched_decode_case` builds 32 true
sequences with `cu_seqlens_q = arange(B+1)` and has produced timings on H100
repeatedly, most recently 14697.5 us/iter in session 27. Multi-sequence works
there. `sm_90` runs `flash_fwd_sm90.py`; `sm_80` and `sm_120` run `flash_fwd.py`.
This is a `flash_fwd.py` defect, so it is expected to affect **A100 as well**, and
that has not been confirmed because no A100 has been available.

## What it costs the claims in this repository

The `sm_80` **support** claim, that our kernel is the only thing that runs on
Ampere because every day-0 path raises `NotImplementedError` there, is now
narrower than it reads. It is established for single-sequence calls. Serving
batches. Until this is root-caused, the honest form of that claim is: our kernel
is the only one that runs on Ampere, and on the generic path some multi-sequence
batches fault, so it is not yet a serving claim on that architecture.

Nothing about the `sm_90` numbers changes.

## A second finding, from the control arm

The no-bias probes all fail with a **different** error:

```
NameError in `__call__`: cannot access local variable 'mDynamicCausal'
```

That is upstream's own defect, not ours, and it is the exact pair of bugs already
reported in **vllm-project/flash-attention PR #156** ("[Bugfix] Fix two
SM80/SM120 forward kernel bugs: missing is_split_kv default, mDynamicCausal
NameError", open since 2026-06-30). So this run independently confirms that PR's
second bug on real `sm_120` hardware, and it is the reason the no-bias arm cannot
serve as a control here: the generic path cannot currently run **without**
`rel_bias` on this architecture at all.

That also settles the duplicate check for upstream report 03: it is a duplicate
of #156 and must not be filed. Note that `vllm-project/flash-attention` has
issues disabled entirely, so a PR is the only route there in any case.

## What the gates that DID pass on sm_120 establish

Run in the same session, same deployment, all green:

| gate | result |
|---|---|
| `parity_rel_chunked_decode` | **7/7**, and every case carries a signal 7.5x to 37.8x above tolerance, so each one could have failed |
| `parity_fa4_rel` | **3/3** for `tml_fa4_rel_bias` |
| `parity_rel_bias_coverage` | **6/6** |
| `parity_qkvr_prep` | **5/5** |

That is the first execution of the generic kernel's corrected shear shift on any
hardware, and it passes on single-sequence decode and chunked-prefill shapes.
`parity_rel_bias_coverage` also reported something worth keeping: at production
decode geometry its oracle comparison is **blind** to a dropped bias, signal 0.5x
of tolerance, and the probe that walks individual distances is the check that
actually discriminates. It confirmed 13 of 13 distances move the output and named
the KV tiles touched, `[504..511]` at 64K, which are the newest blocks and are
what the corrected shift is supposed to select.

## Next step

Root-cause the fault before claiming anything about `sm_80` or `sm_120` serving.
The 128-multiple correlation gives a concrete first hypothesis and the repro is
six lines of shapes, runs on a laptop, and costs nothing.

The reproducer is committed as
`harness/repro_sm120_varlen_illegal_address.py`. It isolates sequence count
against `rel_bias` presence for the `[256, 137, 64]` case and runs each probe in
its own subprocess, because a 700 poisons the CUDA context and every later probe
in one process would fault as a cascade regardless of its own inputs. Start
there rather than from the full gate.

## What this narrowed, and where that is written down

This defect cost no published number. Every parity case and every timing in this
repository was taken with one sequence in the batch, so single-sequence results
are unaffected and the `sm_90` numbers are untouched. What it cost is the
**scope** of the `sm_80` support claim, which is now stated as single-sequence
and explicitly not a serving claim, in:

- [README.md](../README.md) claim 1, the "only working attention kernel on
  Ampere" row of "Everything else that was gated", the "What is not measured"
  list, and both generic-path rows of the architecture table;
- [LEDGER.md](../LEDGER.md), the `sm_80` kernel-gate row, the Ampere tile-sweep
  incident row, and a Reproducibility-incidents row of its own;
- [docs/METHODOLOGY.md](../docs/METHODOLOGY.md), the capability row of the
  claim-class table, the paragraph under it on call-shape scope, and the `sm_80`
  and `sm_120` per-architecture sections;
- [regression-ampere-tile-sweep.md](regression-ampere-tile-sweep.md), which now
  carries this as its second caveat on the support claim;
- `docs/figures/fig3_status.png`, which gained a multi-sequence varlen row and a
  *ran, and faults* status distinct from *not done*.

`journal/upstream/05-no-sm8x-attention-path.md` states the same support claim and
has **not** been updated, because another lane owns that directory. Its evidence
list at `:129-134` is the place the single-sequence qualifier belongs.
