# Local tier bring-up (WSL2 + RTX 5090 Laptop, sm_120)

Date: 2026-07-17.

## Environment (verified)

- WSL2 Ubuntu, torch 2.11.0+cu130, CUDA passthrough OK, device capability (12, 0).
- Fork installed editable with VLLM_USE_PRECOMPILED=1 at base `850295881`;
  `vllm import OK: 0.23.1rc1.dev1237+g850295881`; Inkling in model registry.
- WSL checkout: `~/inkling-turbo/vllm` (cloned from the Windows-side fork clone).

## Three sm_120 bugs found & fixed (scripts/apply_local_sm120_fixes.sh)

Stock day-0 Inkling attention is BROKEN on every sm_120 GPU (RTX 50-series
Blackwell GeForce/workstation). Chain of three independent failures:

1. **tml-fa4 @13374f0c vs nvidia-cutlass-dsl 4.6.0 (vLLM's own pin):**
   `cute.core.ThrMma` / `cute.core.TiledMma` no longer exist, moved to
   `cutlass.cute.ThrMma` / `.TiledMma` (4.6.0 defines ThrMma in
   `cutlass/cute/atom.py`). 14 references fail at import. Upstream tml-fa4 main
   (09d2482) does NOT fix it.
2. **`vllm_flash_attn/cute/flash_fwd.py`:** kernel body line ~840 reads
   `mDynamicCausal`, but the tensor is only a parameter of the launcher, never
   threaded through the `@cute.kernel` signature nor passed at `.launch()`.
   DSL staging error. Only the GENERIC forward is affected; sm90/sm100 have
   their own kernel files → CI on H100/B200 never exercises this line.
3. **`vllm_flash_attn/cute/flash_fwd_sm120.py`:** `FlashAttentionForwardSm120`
   (subclass of the SM80-family class) never sets `self.is_split_kv`, which the
   shared `epilogue` in flash_fwd.py reads. AttributeError at first call.

All three are upstream-reportable with minimal repros. TODO: file issues
(vllm-project/tml-fa4 for 1, vllm-project/flash-attention or vllm for 2-3)
after checking their trackers for duplicates (AGENTS.md duplicate-work rule).

## Parity gate 1, GREEN (harness/parity_fa4_rel.py)

Backend actually used by vLLM on sm_120: `score_mod` gather path
(`_use_sheared_bias()` is major-10/11 only; sheared path asserts tile_n==128
and sm_120 tiles differ → correctly unsupported).

| case | shape | max_diff | mean_diff | verdict |
|---|---|---|---|---|
| global_short | T=128, 8q/1kv, hd128, ext1024 | 1.5625e-02 | 2.02e-04 | OK (tol 2e-2, bf16) |
| global_beyond_extent | T=1536, ext1024 (zero-bias region live) | 7.8125e-03 | 6.81e-05 | OK |
| swa_512 | T=1536, 8q/2kv, win 511 | 7.8125e-03 | 7.99e-05 | OK |

Reference implementation (PyTorch, fp32 accum) matches the production kernel →
the reference is now trusted as the parity oracle for U2/U3 kernel work.

## Implications

- Local U2/U3 kernel dev on sm_120 must target/extend the sm_120 path (SM80-style
  mma.sync pipeline, 99KB smem), deployment kernels (sm_100 tcgen05) validated
  remotely on B200. Two-arch reality documented; per-arch instruction verification
  mandatory (BLOCKERS B3).
- The sheared-bias tml-fa4 kernel only accepts tile_n==128, an sm_120-compatible
  variant is prerequisite for LOCAL U2 iteration (or U2 iterates score_mod-side
  locally + sheared-side remotely).
