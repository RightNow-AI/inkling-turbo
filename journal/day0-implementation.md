# Day-0 vLLM Inkling implementation study

Date: 2026-07-17. Source: fork base `850295881` (day-0 series merged:
`6570c9800` [1/N] model, `251f7e478` [2/N] PW CUDA graphs, `fb5ec0dc9` [3/N] MTP=1,
`f61163e6c` Hopper FA4 rel-attn, `f3e9497e9` [4/N] LoRA, `166a8e954` [5/N] multi-depth MTP).
Code: `vllm/vllm/models/inkling/` (`common/`, `nvidia/`, `configs.py`, `nvfp4.py`).

## NVFP4 layout — B2 RESOLVED (`nvfp4.py`)

- ModelOpt checkpoint format: weights FP4 **E2M1** (`num_bits (2,1)`), block scales FP8
  **E4M3** (`scale_bits (4,3)`), **group_size 16** (asserted; only 16 supported).
- Per-tensor second-level scale `*_scale_2` + `input_amax` → input_scale =
  amax / (448.0 * 6.0). Constants: FLOAT8_E4M3_MAX=448, FLOAT4_E2M1_MAX=6.
- Only routed experts quantized; `exclude_modules` keeps attention/dense-MLP/sink
  experts BF16, decided per layer via `experts_quantized(layer_id)`.
- Checkpoint w13 rows interleaved `[g0,u0,g1,u1,...]`, de-interleaved on device at load.

## MoE path (`nvidia/moe.py`) — U1/U4/U7 baseline

- Gate logits GEMM: `ll_bf16` CUTE-DSL low-latency GEMM for **<=64 tokens** (sm_90+),
  else `torch.mm` with fp32 out. (U7 note: a crossover already exists here, at 64.)
- Selection: ONE Triton kernel `_inkling_gate_select_kernel` per layer: sigmoid +
  selection bias, iterative top-6 argmax, sink ids appended (R+j), log-sigmoid renorm
  over the 8 active logits, × route_scale × global_scale. Grid [T].
- Routed experts: vLLM standard `FusedMoE` + `ModelOptNvFp4Config` → backend picked by
  vLLM's fused-MoE stack (TRTLLM kernels referenced re: expert-slab padding). U1 must
  first identify which backend actually runs on sm_100 (profile, not guess).
- Sink experts (2, always BF16): stacked into TWO dense GEMMs with fused
  `sink_silu_mul_epilogue`; run on **aux stream** overlapped with routed MoE when
  tokens <= VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD; replicated across EP,
  TP-sharded on intermediate dim.
- `skip_final_all_reduce = True`: decoder does reduce-scatter → sharded sconv →
  all-gather instead of all-reduce (sconv-aware TP from the blog).

## Attention path (`nvidia/attention.py`, `ops/fa4_rel_attention.py`) — U2/U3/U5 baseline

- **U5 largely exists day-0**: `qkvr` is ONE MergedColumnParallelLinear over
  [Q 64·128 | K kv·128 | V kv·128 | R 64·16] (single weight read). KV heads padded up
  to tp_size when tp > kv_heads (GQA replication at TP8 on 8-KV-head global layers!).
- `fused_qkvr_prep` (918-line op): single launch doing K/V sconv (cache insert + conv +
  residual), Q/K per-head RMSNorm, paged KV-cache write, R→rel_logits projection,
  log-scaling (global layers only). Only q and rel_logits are materialized.
- Scaling: q/k unit-normed → softmax_scale = **1/head_dim** (not 1/sqrt).
  Log scaling tau = 1 + alpha·log(clamp((pos+1)/n_floor, min=1)), n_floor=128000.
- **FA4 = CuTe-DSL Python flash-attention**, external repo
  `github.com/vllm-project/tml-fa4` @ `13374f0c` (installed as
  `vllm.third_party.tml_fa4`, string-rewritten from `flash_attn.cute`).
  - Blackwell (cap major 10/11): native `rel_bias=` kwarg — "sheared relative-bias
    layout" kernel (TML+Colfax). Hopper (cap 9): generic vllm_flash_attn CuTe
    `score_mod` gather per element, num_splits forced 1 — second-class.
  - Bias semantics (from score_mod): rel_dist = (q_pos - kv_pos); bias =
    rel_logits[q, h, dist] iff 0 <= dist < rel_extent, else **0.0** (NOT clamped).
  - rel_logits is **materialized** as (T, H_local, rel_extent) bf16 and re-read by the
    attention kernel. Global: 64·1024·2B = 128KB/token/layer; local: 64·512·2B = 64KB.
    Prefill 8K tokens ≈ 1.0GB (global) / 0.5GB (local) written+read per layer →
    tens of GB of extra HBM traffic per prefill pass. **This is the U2 target: fold
    the (T,H,16)×(16,extent) projection into the kernel, registers-only.**
    (Caveat for Phase 0b: verify the sheared layout doesn't already tile-cache it.)
- Split-KV: hand-tuned `inkling_fa4_num_splits` (batch-1 decode caps by kv_heads/TP,
  measured to 1M KV); local layers never split (window 512).
- Window: local = (511, 0); global = full causal. Local layers `SlidingWindowSpec`,
  global `FullAttentionSpec`, FlashAttentionBackend metadata reused.
- **U3 confirmed open**: `k_scale`/`v_scale` buffers exist but fixed 1.0;
  kv_cache_dtype follows config ("auto" → BF16); blog states FP8 global-attn not done.
  FA4 CuTe kernel needs an FP8-KV load path + per-block scales.

## Collectives / overlap (`ops/lamport.py`, commit [2/N]) — U6 baseline

- Lamport-protocol low-latency RS/AG kernels exist (blog: 40µs → 8µs at bs1).
- Piecewise CUDA graph support added in [2/N] (`eager_break_during_capture` on
  `_attention` — FA4 python call breaks graph capture; PW graphs stitch around it).
- Remaining U6 surface: expert-weight staging overlap for layer N+1, verify comm truly
  overlapped under TP4/TP8 decode in the Phase 0b timeline.

## Cross-reference

- SGLang day-0 (S4, lmsys blog) — TODO next session.
- tml-fa4 internals (sheared-bias layout, tile scheduling) — TODO: clone + read before
  any U2 design. DO NOT design U2 until the sheared layout is understood.

## Revised unit leverage (pre-profile hypothesis, Phase 0b decides)

1. U3 (KV cache FP8/NVFP4 + per-block scales) — explicitly absent today, long-ctx
   global layers KV-bound, blog itself names it.
2. U2 (rel-bias in registers; drop rel_logits materialization) — measurable HBM
   traffic, clean win if sheared layout doesn't already avoid it.
3. U1 (MoE grouped GEMM) — depends on which FusedMoE backend actually runs; skewed
   256-expert loads at batch are the stated weakness of slab-based kernels.
4. U6 (staging/comm overlap gaps found in nsys timeline).
5. U4 (gate is already 1 kernel; permute/scatter inside FusedMoE — profile first).
6. U7 (crossovers exist at 64 tokens for gate; measure, extend to MoE dispatch).
7. U5 (merged QKVR exists; only prep-kernel micro-wins remain).
