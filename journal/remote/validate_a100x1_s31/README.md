# Session 31: A100-SXM4-40GB, the first correctness result on Ampere

2026-07-25. Modal, `A100-40GB:1`, 0.216 h, **$0.99** at $2.0988/GPU-hr.
Tag `modal_a10040gbx1_route`, patch set `route`.

This is the first time any `sm_80` GPU has run this project's relative-bias
kernel on a shape family other than single-sequence full prefill, and the first
A100 session with a recorded duration and cost. It closes the "Cost: unpriced"
gap that [../../regression-ampere-tile-sweep.md](../../regression-ampere-tile-sweep.md)
recorded against `docs/METHODOLOGY.md` step 8.

## Provenance

| | |
|---|---|
| device | NVIDIA A100-SXM4-40GB, 108 SMs, 42.4 GB, driver 580.95.05 |
| capability | (8, 0), **`arch_matches_request: true`** |
| torch | 2.11.0+cu130, CUDA 13.0 |
| vllm | 0.23.1rc1.dev1237+g850295881.precompiled |
| cutlass dsl | 4.6.0 |
| deployed | `flash_fwd.py:739fb92a`, `flash_fwd_sm90.py:5490c64d`, `interface.py:f7e72e54` |
| absent, asserted | `u2_shear_relshearspec`, `u3_quantize_kv` |

The capability assertion matters: the previous attempt at this session aborted
because `VALIDATE_GPU_KIND` was read at import time and Modal does not propagate
env into containers, so a correctly provisioned A100 was rejected as "not
sm_90". That cost $0.00 and is why `gpu_kind` is a function parameter now.

## Results

| step | verdict | what it actually says |
|---|---|---|
| `parity_fa4_rel` | FAIL rc=1 | **Our path is 3/3.** See below. |
| `parity_rel_chunked_decode` | FAIL rc=0 | **7/7 green.** Harness naming bug. See below. |
| `parity_rel_varlen_batch` | FAIL rc=1 | **11/12.** One real defect. See below. |
| `parity_rel_bias_coverage` | PASS | 6/6 |
| `parity_qkvr_prep` | PASS | 5/5 |
| `tune_sm80` | PASS | parity green on 3 of 3 configs, and see the verdict below |
| `microbench_attn_scoremod` | PASS | 13 of 13 day-0 arms cannot run on SM8x |
| `microbench_ours_attn_shearfusion_OFF` | PASS | timings only, no baseline to divide by |

**Read the three FAIL rows before quoting them.** Two are not failures of this
project's kernel.

### `parity_fa4_rel`, FAIL rc=1, and our arm is 3/3

```
[global_short]          tml_fa4_rel_bias: max_diff=7.8125e-03 mean=1.9599e-04 OK
[global_beyond_extent]  tml_fa4_rel_bias: max_diff=7.8125e-03 mean=6.7980e-05 OK
[swa_512]               tml_fa4_rel_bias: max_diff=7.8125e-03 mean=7.9319e-05 OK
```

The six failures are the other two arms, day-0's `score_mod` and our abandoned
`relproj_v1` prototype, and every one of them raises the same thing:

```
NotImplementedError: Custom user-provided score_mod is not supported on SM8x architectures.
```

The step exits non-zero because it counts all nine case-by-backend checks. On
Ampere the baseline arms cannot run at all, which is
[../../upstream/05-no-sm8x-attention-path.md](../../upstream/05-no-sm8x-attention-path.md).
So this step failing is the support claim's own evidence wearing a red label.

### `parity_rel_chunked_decode`, FAIL rc=0, and it is 7/7

```
[control_full_prefill]        OK max=7.8125e-03 mean=6.8168e-05 signal=37.7x TOL_MEAN
[chunked_global_128_on_1408]  OK max=4.8828e-04 mean=3.7134e-05 signal=17.9x
[chunked_global_256_on_768]   OK max=9.7656e-04 mean=4.6922e-05 signal=21.3x
[decode_global_ctx2047]       OK max=2.4414e-04 mean=2.8564e-05 signal=12.4x
[decode_global_ctx4095]       OK max=2.4414e-04 mean=2.0370e-05 signal=6.7x
[decode_swa_ctx4095]          OK max=4.8828e-04 mean=5.7560e-05 signal=29.9x
[chunked_swa_128_on_1408]     OK max=9.7656e-04 mean=6.3586e-05 signal=1.795e-02 35.9x
```

`rc=0`. The FAIL verdict comes from the runner's artifact check, which looks for
`parity_rel_chunked_decode_sm90.json` on every architecture. That is a bug in
`scripts/modal_e2e_bench.py`, not a result. **Fixed after this session**: the
runner now matches any capability suffix, so session 32 reports this step as PASS.

**Every one of the seven had power to fail**, 6.7x to 37.7x above `TOL_MEAN`, so
these are certifications and not vacuous passes. This is the single most
important row in the session: chunked prefill and decode on `sm_80` had **no
correctness result on any hardware** before it, and
[../../regression-ampere-tile-sweep.md](../../regression-ampere-tile-sweep.md)
listed closing that as step 3 of its re-measurement recipe.

### `parity_rel_varlen_batch`, 11/12, and the one failure is real

`single_m_tail_chunked`, max 0.092529, mean 1.9666e-03 against `TOL_MEAN` 5e-4.
The 5090 reported 1.9674e-03 on the same case, so this is the same defect on two
architectures and not a local artifact.

Root-caused and fixed the same day:
[../../regression-pack-gqa-shear-granularity.md](../../regression-pack-gqa-shear-granularity.md).
The shear writer was packing GQA heads while the generic reader was not. **This
run predates the fix.** [Session 32](../validate_a100x1_s32_packgqa/) ran it on
this same architecture and scored **12/12**, with this case at max 4.8828e-04 and
mean 4.1555e-05.

Also of note: the varlen multi-sequence crash
([../../regression-sm120-varlen-illegal-address.md](../../regression-sm120-varlen-illegal-address.md))
does not reproduce here, which is the first execution of that fix on Ampere.

### `tune_sm80`: what this session concluded, and what session 32 corrected

Parity went green on all three configs. The timings:

| shape | (128,32) | (128,64) upstream default | (128,128) |
|---|---|---|---|
| `prefill_global_8k` | **9943.2** | 10404.4 | 359001.7 |
| `prefill_swa_8k` | 10664.8 | **10468.4** | 357130.5 |
| `decode_b1_global_kv64k` | 4669.5 | **4356.3** | 129286.4 |
| `decode_32seqs_global_kv64k` | **51393.7** | 53833.3 | 2816014.3 |

Against the shipped default, `tile_n=32` gains 4.4% and 4.5% on two shapes and
loses 1.9% and 7.2% on the other two. But the decisive comparison is not between
configs, it is between runs. Against the earlier
[../tune_sm80_a100.json](../tune_sm80_a100.json):

| config | prefill_global | prefill_swa | decode_b1 | decode_32seqs |
|---|---|---|---|---|
| (128,32), old to new | 0.93x | 1.01x | 0.87x | 0.85x |
| (128,64), old to new | 0.94x | 1.14x | 0.73x | 0.72x |

**Run-to-run drift on an unchanged config reaches 27.6%, while the largest
difference between configs is 7.2%.** The noise is roughly four times the signal.
The sweep cannot rank tile sizes, and the withdrawn 10.1% / 18.2% / 18.7% were
single samples read off exactly this noise. They stay withdrawn permanently. This
is now a refutation and not a suspicion.

> **CORRECTED a few hours later by [session 32](../validate_a100x1_s32_packgqa/),
> and the paragraph above is left in place because that is this directory's rule.**
> Two things in it are wrong. First, the configurations were **not** unchanged: the
> kernel differed between every pair of those runs, since `tune_sm80_a100.json`
> predates the shear-shift fix and this session predates the `pack_gqa` guard, and
> that guard changes `ShearingBias`'s grid factorisation while `ShearingBias` runs
> inside the timed region. A cross-session delta there conflates a code change
> with noise. Second, the sweep **can** rank tile sizes: given five interleaved
> rounds it decides three of the four shapes with disjoint intervals, and seven of
> eight cells repeat to within 2.2%.
>
> The withdrawal stands and is stronger than this paragraph made it. On batch-1
> decode at 64K, the shape the 10.1% claim was about, the repeat measurement puts
> `tile_n=64` ahead of `tile_n=32` by **9.7%**. The published figure had the wrong
> sign, not merely wide error bars.

Caveat on the comparison: the old artifact records no device, capability or torch
version, because it predates the fields being written. It is named for an A100
and cannot be proven to be one. That is a confound, and it cuts the same way,
since the withdrawn claim was made from that file.

`tile_n=128` collapsing by a factor of 34 to 50 reproduces in both runs and is
far outside the noise. That conclusion stands.

`harness/tune_sm80.py` was rewritten after this run to repeat in interleaved
rounds and to refuse a winner unless the sample intervals are disjoint. Fed the
two real samples per config, it reports NO WINNER on `decode_b1`, the case that
produced "10.1% faster".

### An upstream defect this session found by accident

Two arms of `microbench_attn_scoremod` fail differently from the other eleven:

```
decode_b1_plain_kv64k, decode_b32_plain_kv64k:
  /opt/vllm/vllm/vllm_flash_attn/cute/flash_fwd.py:840
  psc = mDynamicCausal[batch_size] if const_expr(mDynamicCausal is not None) else None
  NameError: name 'mDynamicCausal' is not defined
```

That path is `vllm_flash_attn`, vLLM's bundled fork, **not** our
`third_party/tml_fa4` tree, and the shape carries no `rel_bias`. So plain
attention at 64K KV is broken on SM8x in stock vLLM independent of anything here.
It is the same defect class as the `n_block` scoping trap in
[../../regression-sm90-bias-shift.md](../../regression-sm90-bias-shift.md): a
name bound in one traced branch and read in another. Candidate addition to
[../../upstream/03-vllm-flash-attn-generic-path-bugs.md](../../upstream/03-vllm-flash-attn-generic-path-bugs.md).
Not reduced to a minimal reproducer, so not filed.

## What this session does not establish

- **No Ampere speedup number exists and none can.** All 13 day-0 arms raise
  `NotImplementedError` on SM8x, so there is no baseline to divide by. The
  `microbench_ours_attn_shearfusion_OFF` timings in this directory are absolute
  numbers with nothing to compare them to. The Ampere claim is a capability
  claim.
- **The `pack_gqa` fix has not run on `sm_80`.** This session observed the
  defect there; the fix was verified on `sm_120` only.
- **Depth.** `tune_sm80`'s parity probes run at `ctx` 4095 and shallower while
  the timed cases run at 65535, because `TOL_MEAN` is absolute and the bias moves
  a 64K global decode by less than the tolerance. A green oracle certifies the
  timed family and head geometry, not the timed depth.
  `parity_rel_bias_coverage` is the instrument for 64K, and it is 6/6 here.
