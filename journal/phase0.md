# Phase 0a, Verified Inkling Configuration

Date: 2026-07-17. All values pulled from primary sources, zero guessed shapes.

## Sources

- S1: `huggingface.co/thinkingmachines/Inkling/resolve/main/config.json` (fetched 2026-07-17)
- S2: vLLM day-0 blog: https://vllm.ai/blog/2026-07-15-inkling (fetched 2026-07-17)
- S3: vLLM day-0 PR: https://github.com/vllm-project/vllm/pull/48768 (WoosukKwon)
- S4: SGLang day-0 blog: https://www.lmsys.org/blog/2026-07-15-inkling-day0-support (not yet read, TODO)

## Model config (S1, verbatim from text_config)

| Field | Value |
|---|---|
| architectures | InklingForConditionalGeneration, model_type `inkling_mm_model` |
| hidden_size | 6144 |
| num_hidden_layers | 66 |
| vocab_size | 201024 (unpadded 200058) |
| model_max_length | 1,048,576 |
| torch_dtype | bfloat16 (BF16 repo) |
| **Global attention** | 64 Q heads, **8 KV heads**, head_dim 128 |
| **SWA attention** | 64 Q heads, **16 KV heads**, head_dim 128, sliding_window_size **512** |
| **R projection** | `d_rel: 16`, `rel_extent: 1024` |
| log scaling | `log_scaling_n_floor: 128000`, `log_scaling_alpha: 0.1` |
| q_bias / o_bias | false / false |
| rms_norm_eps | 1e-6, `use_embed_norm: true` |
| **sconv** | `use_sconv: true`, `sconv_kernel_size: 4` |
| **MoE** | n_routed_experts **256**, top-k **6**, n_shared_experts **2**, `shared_expert_sink: true` |
| expert intermediate | 3072 (routed), dense_intermediate_size 24576, `dense_mlp_idx: 2` |
| router | route_scale 8.0, gate_activation **sigmoid**, use_gate_bias true, norm_after_topk true, use_global_scale true |
| logits | mup_width_multiplier 24.0, final_logit_softcapping null |
| MTP | num_nextn_predict_layers **8**, chained; MTP local_layer_ids [0,2,4,5,6,7] |
| eos_token_id | 200006 |

### Layer pattern (S1 `local_layer_ids`, 66 layers)

Local (SWA) layers: 55 → all except `{5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65}`.
**Global layers = every 6th layer starting at 5** (11 total). Pattern: 5×SWA → 1×global, repeated 11 times.

### Multimodal (S1)

- audio: dmel mode, 80 mel bins, mel_vocab 16, decoder_dmodel 6144
- vision: hmlp encoder, patch 40, temporal_patch 2, 4 layers
- Day-0 note (S2): encoders run **eager mode** (no CUDA graphs yet), out of kernel scope for now.

## Quantization (S2, NVFP4 checkpoint)

> "only the routed experts are quantized to NVFP4; all other parameters, including the shared experts and the qkvr linears, remain in BF16."

- **OPEN ITEM**: `Inkling-NVFP4/config.json` fetched, contains NO `quantization_config` block (looks identical to BF16 config). NVFP4 block-scale layout metadata must live elsewhere (likely `hf_quant_config.json` / modelopt-style, or encoded in vLLM loader). Resolve from vLLM PR code once clone completes. **Do not assume group size / scale dtype until verified.**
- Consequence for units: U1 (MoE grouped GEMM) is the FP4 surface. U5 (QKVR fused GEMM) is BF16, it is a pure memory/fusion win, not a quant kernel.

## Day-0 baseline implementation facts (S2)

- Attention: **FA4 kernel with "sheared-bias technique"** (TML + Colfax Research collab). Relative attention = learned relative-position terms added to pre-softmax logits; **no RoPE**. num_splits selected per (batch, TP, KV-len).
- sconv: cached via KV-cache manager as a "virtual sliding-window attention layer" (window = kernel size). TP sharding via reduce-scatter/all-gather across channels.
- Collectives: Lamport-protocol low-latency RS/AG; bs=1 kernel time 40µs → 8µs claimed.
- Reference perf (S2): **380 tok/s/user with MTP8** (mean acceptance 4.5), **140 tok/s/user without MTP**, on 4× GB200, `--tensor-parallel-size 8`, 8K in / 1K out.
- Day-0 self-declared gaps (S2): FP8 for global-attention layers **not done (BF16 today)** → supports U3; encoders eager; AMD unsupported.

## Hardware ground truth (measured 2026-07-17)

### Local tier
- RTX 5090 **Laptop** GPU: 24,463 MiB VRAM, 115W power cap, WDDM mode, driver 581.57, CUDA 13.0.
- Consequences: (a) sm_120, the tcgen05 block-scaled-FP4-MMA trap applies, verify every PTX instruction per-arch; (b) 24GB VRAM bounds microbench shapes, single-layer at real dims fits (6144 hidden, 256×3072 experts ≈ 2.4GB BF16 per expert-stack slice, use per-unit slices); (c) WDDM + 115W cap → local timings are RELATIVE only, never quoted as serving numbers.
- WSL2 Ubuntu present (stopped), all builds happen there; Windows-native vLLM build is not supported.

### Remote tier, BLOCKED on capacity
Lambda account instance-type enumeration (API, 2026-07-17):
- **No B300 instance type exists on Lambda.** (Spec's TP4 W4A4 target impossible there today.)
- **No H200 instance type exists on Lambda.** (Earlier positive was a substring bug matching GH200.)
- B200: 1x/2x/8x listed, **zero capacity in all regions right now**. 8x B200 = $53.52/hr.
- Available today: 1x/2x H100 SXM5, 8x A100 (40/80GB), 1x A100/A10, 8x V100.
- Feasibility: NVFP4 ~592GB weights → needs 8× B200 (1.44TB, FP4 MMA native). 8× H100 (640GB) cannot hold weights+KV+activations comfortably and Hopper has no FP4 MMA (would be W4A16 dequant path, spec calls this second-class). A100/V100: no.

## Decode roofline sanity (from spec, to re-verify in Phase 0b)

41B active × NVFP4 (~0.5 byte incl. scales) ≈ 21GB weight reads/token/forward. B200 HBM3e ≈ 8TB/s → theoretical ~380 dense decode steps/s/GPU-set; MoE routing skew and KV traffic land on top. Global layers at long ctx are KV-bound → U3.

## Phase 0b status

NOT STARTED, blocked on remote capacity. No optimization work may claim victory before the profile exists. Local-tier prep (build, parity fixtures, day-0 code study) proceeds in parallel.
