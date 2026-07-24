# [vllm-flash-attn cute] Generic (SM80-family/sm_120) forward path: three latent bugs, never CI-exercised

**Repo:** vllm-project/flash-attention (vllm_flash_attn/cute, as shipped in
vLLM wheel for fork-base `850295881`); items 2-3 also present in
vllm-project/tml-fa4 @ `13374f0c`.

sm_90/sm_100 have dedicated kernel files, so CI on H100/B200 never compiles
the generic path, every RTX 50-series (sm_120) user hits these on first
call. All three verified on RTX 5090; stock vLLM Inkling attention is
unusable on sm_120 without the fixes.

## 1. `mDynamicCausal` used but never threaded through the kernel

`flash_fwd.py` kernel body (~:840) reads `mDynamicCausal`, but the tensor is
only a parameter of the launcher, absent from the `@cute.kernel` signature
and the `.launch()` call. DSL staging error on any generic-path call:
`NameError ... name 'mDynamicCausal' is not defined`.
Fix: add the defaulted parameter + pass it at the call site (2 lines).

## 2. `FlashAttentionForwardSm120` missing `is_split_kv`

The sm_120 shim (subclass of the SM80-family class) never sets
`self.is_split_kv`, which the shared epilogue reads (`flash_fwd.py:365`):
`AttributeError` at first call. Fix: default it False in the shim ctor.

## 3. (tml-fa4 generic path) pack_gqa declared but tensors never packed +
##    TMA-O wrongly enabled by the sm_120 shim

- `pack_gqa=True` reaches the generic ctor but the generic `__call__` never
  calls `pack_gqa_layout` (the vllm_flash_attn copy does, :726-729), rank
  mismatch crash in the epilogue (`'cute.slice' ... weakly congruent`).
- The sm_120 shim leaves `self.arch = sm_120`, so
  `use_tma_O = arch >= sm_90` enables the TMA output path that the
  SM80-family kernel never configures (`tma_atom_O` None / ragged-branch
  2D-view slice errors).
Fixes: force `pack_gqa=False` for the generic family until packing is
implemented; hard-set `use_tma_O = False` in the SM80-family `__call__`.

## Environment

RTX 5090 Laptop (sm_120), WSL2 Ubuntu, torch 2.11.0+cu130
nvidia-cutlass-dsl 4.6.0. Minimal repro for each available; combined patch
script exists and restores the full varlen path (parity vs fp32 reference:
max diff 7.8e-3 across causal/beyond-extent/SWA cases).

Found during Inkling-turbo kernel work; fixes available for upstreaming.
