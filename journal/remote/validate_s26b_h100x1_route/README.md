# Session 26b: the regression fix, verified on one H100

`VALIDATE_PATCHES=route modal run --detach scripts/modal_e2e_bench.py --step validate`
2.0 minutes of GPU, $0.34. Ledger total $1.03 of the $200 cap.

Device: NVIDIA H100 80GB HBM3, capability (9, 0). torch 2.11 / cu130.

> **Withdrawal notice added 2026-07-25, after the fact, because this file had
> none.** Everything here predates the sheared-bias shift fix. The `n_block`
> hoist that this session verified is a different defect from the shift, so the
> `n_block` answer stands, and so does every **prefill** row. But the three
> **decode** ratios below, 2.81x, 2.75x and 2.45x, were measured on a kernel that
> was applying Inkling's relative-position bias to one KV block instead of ten
> while the `score_mod` baseline it is divided by gathered every score. **They are
> withdrawn**, together with every other pre-fix decode ratio in this repository.
> They are struck through below rather than deleted. The decode *timings* stand as
> timings; what does not stand is dividing them by anything. The like-for-like
> re-measurement is 2.66x / 2.75x / 2.10x in `../validate_s27_decodefix/`, and the
> proximity is luck rather than vindication.
> [Full account](../../regression-sm90-bias-shift.md).

This run exists to answer two questions that a static reading could not settle:

1. Does the `n_block` hoist actually fix the sm_90 regression on silicon?
2. Do the published speedups still reproduce from the fixed code?

Yes to the first. **Read the second one narrowly, and it is the reason for the
notice above**: what reproduced was that the same code produces the same
timings, to within 2%. Reproducibility was never the problem with those decode
numbers. A kernel can be reproducible to a tenth of a percent while doing the
wrong amount of work, and that is what this run measured three times.

## Configuration

`patches=route`, which deploys `kernels/tml_fa4_modified/` plus
`u2_serving_route.py` and **nothing else**. That is byte-for-byte what the 8x
end-to-end bench deploys for its `ours` build, which is why this run is the
go/no-go gate for the expensive one. The manifest records
`patches_confirmed_absent: [u2_shear_relshearspec, u3_quantize_kv]`, so the
absence of u3 and the shear fusion is asserted rather than assumed. The four
gates that need those patches were dropped rather than run and reported red.

## Result: all green

| gate | before the fix (session 26) | after |
|---|---|---|
| `parity_fa4_rel` | FAIL, `tml_fa4_rel_bias` failed all 3 cases | **PASS**, 3/3 |
| `parity_qkvr_prep` | PASS | PASS |
| `microbench_attn_scoremod` | PASS | PASS |
| `microbench_ours_attn_*` | PASS with **all 7 attention cases null** | **PASS with all 7 producing numbers** |

`max_diff=7.8125e-03` on all three parity cases, identical to `score_mod` on
the same inputs, which is the bf16 quantum at these magnitudes.

## The published timings reproduce

This heading read "The published numbers reproduce" until 2026-07-25. That was
too strong in exactly the way that mattered: the timings reproduced and the
comparison they fed was unsound. Against session 25, on a different machine and a
different container:

| workload | session 25 | session 26b | delta |
|---|---|---|---|
| prefill_global_8k | 3308.8 | 3306.9 | -0.1% |
| prefill_swa_8k | 1223.0 | 1221.4 | -0.1% |
| decode_b32_global_kv8k | 124.1 | 124.1 | 0.0% |
| decode_b32_global_kv64k | 854.8 | 866.3 | +1.3% |
| decode_b1_global_kv64k | 852.6 | 859.6 | +0.8% |
| decode_32seqs_global_kv8k | 1799.1 | 1810.5 | +0.6% |
| decode_32seqs_global_kv64k | 13821.4 | 14079.6 | +1.9% |

## Speedups, both builds measured in the same container

This is a stronger comparison than session 25's, because ours and the day-0
baseline were timed in one process on one machine minutes apart, with no
cross-session or cross-toolchain gap to argue about.

| workload | ours | day-0 `score_mod` | ratio |
|---|---|---|---|
| decode, batch 1, 64K KV | 859.6 | 2412.1 | ~~**2.81x faster**~~ **WITHDRAWN** |
| decode, batch 32, 64K KV | 866.3 | 2382.5 | ~~**2.75x faster**~~ **WITHDRAWN** |
| decode, batch 32, 8K KV | 124.1 | 303.6 | ~~**2.45x faster**~~ **WITHDRAWN** |
| prefill, 8K, global | 3306.9 | 4841.2 | **1.46x faster**, stands |
| prefill, 8K, sliding window | 1221.4 | 863.2 | **0.71x, we lose**, stands |

The two prefill rows are unaffected because at `seqlen_q == seqlen_k` the
defective shift is an identity, so the fixed kernel computes the same value. The
three decode rows are not, for the reason in the notice at the top of this file.

Reference points from the same file, which are **not** baselines: biasless
attention 736.2 / 736.6 us (the floor this feature can approach and never
beat), and our own abandoned prototypes `relproj` 7330.9 / 15358.9 and
`relprojT` 5176.1 / 10571.3.

## One thing worth reading carefully

The sliding-window loss is **1.41x** here and **1.28x** in session 25. Our
kernel did not move: 1223.0 to 1221.4, a tenth of a percent. The `score_mod`
baseline moved 956.5 to 863.2, which is 9.8%. So the honest statement of that
loss is a range, 1.28x to 1.41x slower, and the uncertainty is in the baseline
rather than in us. Anyone quoting a single figure for this case should say which
run it came from.

The same asymmetry appears in the global prefill baseline, 4799.4 to 4841.2, and
in decode, 2326.6 to 2412.1. Our numbers are reproducible to about 2%; the
day-0 path is reproducible to about 10%. That is worth knowing before reading
any single ratio to two decimal places.
