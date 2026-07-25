# Session 32: the pack_gqa fix on Ampere, and the tile sweep done properly

2026-07-25, a few hours after [session 31](../validate_a100x1_s31/). Modal
`A100-40GB:1`, capability (8, 0) asserted, **$0.61**. Tag
`modal_a10040gbx1_route_s32packgqa`, patch set `route`.

Two jobs, and both came back with a clear answer. The `pack_gqa`
shear-granularity guard closes Ampere at 12 of 12, and the repeat-enabled tile
sweep says something sharper than session 31 could: the withdrawn Ampere
percentage was not merely unreliable, it had **the wrong sign** on the shape it
claimed most confidently.

## Provenance

| | |
|---|---|
| device | NVIDIA A100-SXM4-40GB, capability (8, 0), `arch_matches_request: true` |
| deployed | `flash_fwd.py:739fb92a`, `flash_fwd_sm90.py:5490c64d`, **`interface.py:d5e88990`** |
| absent, asserted | `u3_quantize_kv`, `u2_shear_relshearspec` |
| cost | 0.14 h, $0.61 at $2.0988/GPU-hr |

`interface.py` hashes `d5e88990` here against `f7e72e54` in session 31. That is
the only code difference between the two sessions and it is the `pack_gqa` guard.

## The fix closes Ampere

| gate | session 31 | session 32 |
|---|---|---|
| `parity_rel_varlen_batch` | 11/12 | **12/12** |
| `parity_rel_chunked_decode` | 7/7 | 7/7 |
| `parity_rel_bias_coverage` | 6/6 | 6/6 |
| `parity_fa4_rel`, our arm | 3/3 | 3/3 |
| `parity_qkvr_prep` | 5/5 | 5/5 |

`single_m_tail_chunked`, the case that failed in session 31, moves from max
9.2529e-02 / mean 1.9666e-03 to **max 4.8828e-04 / mean 4.1555e-05**. The
`sm_120` post-fix value for the same case is 4.1567e-05, so the two
architectures land in the same place. **11 of the 12 cases could have failed if
the bias were dropped entirely**; the twelfth, `multi_prefill_offset_only`,
prints `<-- NO POWER` about itself at 1.0x `TOL_MEAN` and is counted separately
rather than banked.

`parity_rel_bias_coverage`'s probe still names KV tiles [504..511] as touched at
64K, the newest eight blocks, which is what the corrected shift must select. Its
oracle at that depth reports itself **BLIND** at 0.5x tolerance. Both halves are
printed.

## The tile sweep, five interleaved rounds, disjoint-interval rule

`harness/tune_sm80.py` was rewritten after session 31 to repeat in interleaved
rounds and to refuse a winner unless the winner's worst sample beats the
runner-up's best sample.

| shape | `tile_n=32` median [lo, hi] | `tile_n=64` median [lo, hi] | verdict |
|---|---|---|---|
| `prefill_global_8k` | 9037.8 [8926.4, 10889.9] | 9495.0 [9426.1, 9542.7] | **NO WINNER**, ranges overlap |
| `prefill_swa_8k` | 8532.0 [8520.4, 8541.7] | 8420.4 [8397.8, 8435.7] | `tile_n=64` by 1.3%, disjoint |
| `decode_b1_global_kv64k` | 3693.4 [3692.5, 3695.1] | 3366.5 [3365.8, 3366.8] | **`tile_n=64` by 9.7%**, disjoint |
| `decode_32seqs_global_kv64k` | 53326.1 [52377.0, 53487.2] | 55064.4 [54345.2, 55209.3] | `tile_n=32` by 3.3%, disjoint |

`tile_n=128` was dropped after one round on all four shapes by the
dominated-config rule, at 37x to 52x the best median. That is logged per shape
rather than silently skipped, and the collapse now reproduces across three
independent runs.

### The withdrawn percentage had the wrong sign

This is the finding. The withdrawn claim was **"`tile_n=32` is 10.1% faster than
`tile_n=64` on batch-1 decode at 64K KV"**, from a single sample per cell.
Measured with five interleaved rounds, on that exact shape, **`tile_n=64` is 9.7%
faster**, and it is the cleanest cell in the table: the two intervals span 0.03%
and 0.07% of their medians and are nowhere near each other.

The second withdrawn figure, 18.2% for `tile_n=32` on the 32-sequence decode,
points the right way but is **5.5x too large**: the measured separation is 3.3%.

So the shipped upstream default, `FwdConfig(128, 64)` with its literal "should
tune" comment, wins the two shapes it was most suspected on and loses one by
3.3%. Nothing here justifies changing it, and now that is a measurement rather
than an absence of one.

### Repeatability is excellent within a session, and one cell is not

Within-session spread across the five rounds, as a percentage of the low value:

| config | prefill_global | prefill_swa | decode_b1 | decode_32seqs |
|---|---|---|---|---|
| `tile_n=32` | **22.00%** | 0.25% | 0.07% | 2.12% |
| `tile_n=64` | 1.24% | 0.45% | 0.03% | 1.59% |

Seven of the eight cells repeat to within 2.2%, and three of them to within
0.5%. The exception is `tile_n=32` on global 8K prefill, which swings 22% and is
the single reason that shape has no winner. It is one erratic cell, not a noisy
harness, and the interleaved rounds are what makes the difference visible rather
than averaging it away.

## Correction to what session 31 concluded

Session 31's write-up compared its single-sample timings against the older
`tune_sm80_a100.json` and reported "run-to-run drift on an unchanged config
reaches 27.6%, so the sweep cannot rank tile sizes". **That framing was wrong on
two counts and is corrected here.**

1. **The configs were not unchanged.** The kernel differed between every pair of
   those runs. The old artifact predates the shear-shift fix; session 31 has it
   but not the `pack_gqa` guard; session 32 has both. The guard changes
   `ShearingBias`'s grid factorisation, and `ShearingBias` runs inside the timed
   region, so it can move these timings without any correctness difference. A
   cross-session delta there conflates a code change with measurement noise and
   cannot be attributed to either.
2. **The sweep can rank tile sizes.** With repeats it ranks three of the four
   shapes with disjoint intervals. "Too noisy to rank" was an artifact of one
   sample per cell, which is the same methodological hole in a different place.

What survives, and is stronger than the withdrawn-because-noisy version: the
percentages were selected from single samples, and a proper measurement of the
headline shape gives the **opposite** result. The withdrawal stands on evidence
now instead of on doubt.

## Still not established

- **No Ampere speedup number, and none is possible.** All 13 day-0 arms of
  `microbench_attn_scoremod` raise `NotImplementedError` on SM8x. There is no
  baseline to divide by, so the Ampere claim is a capability claim. The
  `microbench_ours_attn_shearfusion_OFF` timings here are absolute numbers with
  nothing to compare them to.
- **These timings do not transfer off this machine.** They are repeatable to
  within 2.2% on one A100 in one container. Nothing here says what they are on a
  different A100, and the cross-session comparison above is exactly why.
- **Depth.** The sweep's parity probes now match the timed head geometry, `Hq=64`
  over `Hkv=8` and `Hq=64` over `Hkv=16`, and include a `ctx = 1000` shape at 104
  mod 128. They still run shallower than the timed 65535, because `TOL_MEAN` is
  absolute and the bias moves a 64K global decode by less than it.
  `parity_rel_bias_coverage` is the instrument at that depth.
- `parity_fa4_rel` shows FAIL rc=1 here as in session 31. Our arm is 3/3; the six
  failures are day-0 `score_mod` and our abandoned `relproj_v1`, both of which
  cannot run on SM8x at all.
