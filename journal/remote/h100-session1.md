# H100 session 1, first contact (2026-07-18)

Instance: gpu_1x_h100_sxm5, us-south-2.
11:48:12 -> 11:55:55 UTC = 0.13 h = $0.55. Auto-terminated (finally block). Log:
b200_first_contact_20260718_1155.log.

## Results

- Pipeline proven end-to-end: launch -> boot (3.5 min) -> bootstrap (precompiled
  install on Lambda Stack, torch 2.11.0+cu129) -> harness -> evidence -> kill.
- Parity (sm_90, score_mod = vLLM's actual Hopper path): 3/3 OK, max_diff
  <= 1.6e-2, matches sm_120 results exactly.
- Gate-select kernel measured on H100: 4.3 us @ T=1, 22.2 us @ T=4096.
  (SGLang claims 7.72 us @ T=4096 on B200 for their fused router, different
  hardware, NOT comparable; measure both on the same box before any claim.)
- Attention microbench cases failed: tml-fa4 pinned commit uses
  cute.make_fragment, REMOVED in nvidia-cutlass-dsl 4.6.0 (vLLM's own pin), renamed make_rmem_tensor. Call sites: utils.py(8), softmax.py(5)
  flash_fwd_sm100.py(7!), flash_fwd.py(2), pack_gqa.py(1). sm_100 having call
  sites means the DAY-0 B200 SHEARED PATH cannot run against the pinned
  toolchain as installed here, third API-drift break (with ThrMma/TiledMma).
  Upstream issue material; verify how vLLM CI pins cutlass before filing.
- Fixes applied to bootstrap + local trees (rename verified: local sm_120
  parity 3/3 green post-rename, no regression).

## Next session payload

Patched bootstrap re-run on H100: full attention microbench via tml-fa4
(sm_90 path now importable end-to-end), same auto-terminate discipline.

# H100 session 2 (2026-07-18, 12:00-12:09 UTC, $0.61)

Patched payload (ThrMma/make_fragment fixes in). Results:
- score_mod parity: 3/3 OK again (stable across sessions/archs).
- gate-select: 4.3 us @ T=1, 22.2 us @ T=4096, identical to session 1;
  numbers are stable, kernel is launch-bound at T=1.
- FOURTH cutlass-4.6.0 drift: tml-fa4 utils.py fmax/atomic_add_i32 select
  old-vs-new nvvm API by "CUDA_VERSION == 12.9" (utils.py:352,496), but the
  binding signature tracks the DSL version (4.6.0 = new API, 2 positional).
  Lambda torch = cu129 -> wrong branch -> TypeError in every tml-fa4 sm_90
  attention call. Local cu130 never hits it (why sessions were needed).
  Fix: dead-branch the check (DSL is pinned 4.6.0 by vLLM requirements).
  Applied to WSL tree (parity 3/3 post-fix), bootstrap, fixes script.
- Pattern for upstream report: tml-fa4 @13374f0c is incompatible with
  vLLM's own nvidia-cutlass-dsl==4.6.0 pin in FOUR distinct ways; the
  sheared/direct FA4 path cannot have been CI-tested against this pin.

Session 3 payload: attention microbench through tml-fa4 sm_90 end-to-end.

# H100 session 3 (2026-07-18, 12:14-12:23 UTC, $0.61)

All four drift fixes green, tml-fa4 direct path now RUNS on sm_90. Findings:

## Parity: sm_90 rel_bias path is NUMERICALLY WRONG (harness catch #1)

- tml_fa4 flash_attn_varlen_func(rel_bias=...) on sm_90: max_diff 0.90-1.63
  vs reference on all 3 cases (mean 0.02-0.06 -> scattered wrong positions
  not a global offset). score_mod path on the same inputs: 7.8e-3. NOT a vLLM
  production path (vLLM uses score_mod on Hopper), but it accepts the input
  and returns wrong attention silently -> upstream-report material.
  Hypothesis to verify before filing: ShearingBias 128-wide block layout vs
  sm_90 kernel tile/consumption mismatch. DO NOT use this path for anything.

## First attention timings on H100 (INDICATIVE ONLY, from the wrong-output
## rel_bias path; same work shape, but no optimization claims on these)

| case | us/iter |
|---|---|
| prefill_global_8k (64q/8kv, ext1024) | 2609.6 |
| prefill_swa_8k (64q/16kv, ext512, win511) | 853.8 |
| decode_b32_global_kv8k | 109.5 |
| decode_b32_global_kv64k | 747.1 |
| decode_b1_global_kv64k | 739.6 |

- decode kv64k: b32 ~= b1 (747 vs 740us), KV-read bound, batch nearly free.
  Directly supports U3 (quantized KV) as top-leverage for long-ctx decode.
- global prefill = 3.1x SWA prefill at 8K.
- gate-select 4.3/22.2us, identical across 3 sessions; measurement noise
  floor is excellent on these boxes.

## Program status after 3 sessions ($1.77 total)

Remote pipeline hardened (4 toolchain breaks fixed in bootstrap), parity
oracle proven as a bug-catcher, first indicative timings in. Score_mod
timings at same shapes = next session payload (the honest sm_90 baseline).

# H100 session 4 (2026-07-18, 12:29-12:37 UTC, $0.56), HONEST BASELINE

score_mod = what vLLM actually serves on Hopper (sheared is cap 10/11 only
fa4_rel_attention.py:20-23). Crucially num_splits=1 IS production on sm_90
(fa4_rel_attention.py:71-72 forces 1 for capability major 9), so these
numbers are the faithful Hopper serving baseline, not a strawman.

| case | score_mod (prod) | plain attn (no bias) | rel_bias path (s3, WRONG output) |
|---|---|---|---|
| prefill_global_8k | 5372.0 us |, | 2609.6 us |
| prefill_swa_8k | 948.9 us |, | 853.8 us |
| decode_b32_kv8k | 324.7 us |, | 109.5 us |
| decode_b32_kv64k | 2435.7 us | 743.2 us | 747.1 us |
| decode_b1_kv64k | 2374.9 us | 742.7 us | 739.6 us |

## Findings (measured, sm_90)

1. **The day-0 Hopper path leaves ~3.2x on the table at long-ctx global
   decode**: score_mod 2375us vs plain 743us at kv64k. The per-element
   score-mod gather costs 220% on top of the attention itself.
2. Prefill global 8K: score_mod is 2.1x the rel_bias-path time.
3. The (incorrect) sheared-style path runs at ~plain-attention speed
   (747 vs 743us) while mostly applying bias (mean err 0.02-0.06, max 0.9-1.6
   at a subset of positions) -> HYPOTHESIS: a correct sheared-style bias on
   sm_90 costs ~0 over plain attention. Verify after fixing correctness, speed of a wrong kernel is not evidence of anything by itself.
4. Plain b1 kv64k = 743us = 268MB KV / 743us ~= 361 GB/s ~= 11% of H100 HBM, with num_splits=1, b1 decode runs ~8 CTAs (one per KV head): the GPU is
   mostly IDLE. Split-KV (banned on sm_90 by the day-0 heuristic) is itself
   a large lever before any quantization.
5. Gate timings 4.2/22.0us, 4th consecutive session within noise.

## Hopper-tier unit reordering (evidence-based)

U2-Hopper (bias mechanism + split-KV on sm_90) is now measured at ~3.2x
potential on long-ctx decode and ~2x on global prefill, LARGER than U3's
2x byte-halving, and U3 multiplies on top of a fixed kernel. New order:
**U2-Hopper -> U3 -> rest**. (Blackwell ranking unchanged, pending B200.)

# Post-session correction (static analysis, 2026-07-18)

Session-3 catch root-caused: sm_90 kernel HAS no bias path (ctor takes no
bias arg; flash_fwd_sm90.py bias-free). rel_bias on sm_90 = plain attention
+ wasted shear launch, returned silently. Journal entries above claiming
"sheared-style path mostly applies bias" are WRONG, the observed
mean-error pattern was bias-absence, not bias-misplacement. Detail:
journal/u2-hopper-design.md CORRECTION.
