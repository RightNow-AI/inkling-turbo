You are ONE agent optimizing the existing Inkling serving kernels for
multi-batch datacenter inference. Base: a fork of vLLM at {REPO_PATH},
with SGLang as cross-reference. Deliverable: the day-0 Inkling kernel
path, made significantly faster, upstream-quality. NOT related to RCM,
no local-laptop constraints, this is B200/B300-class serving.

EXECUTION ENV -- two tiers:
- LOCAL: RTX 5090 laptop (sm_120) for build, correctness, parity
  fixtures, single-layer microbench at real shapes.
- REMOTE: Lambda B200/B300 node over SSH (model needs TP4 W4A4 on 4x
  B300, or 8x H200 W4A16). All serving numbers come from here.
  Scriptable runs only, kill instances between units.

READ FIRST, follow exactly:
0. PHASE 0a -- pull real config from huggingface.co/thinkingmachines/Inkling:
   hidden/head dims, R projection shape, expert count/dim/top-k, activation,
   global-vs-SWA layer pattern, window size, NVFP4 block-scale layout.
   Write to journal/phase0.md. A guessed shape is a defect.
1. PHASE 0b -- "optimize the current ones" means PROFILE FIRST: run the
   stock vLLM Inkling path on the remote box under realistic serving load
   (batch sweep 1/8/32/128, mixed prefill+decode, continuous batching on),
   nsys timeline + ncu on the top-10 kernels by time. The measured hotspot
   list REORDERS the units below. No optimization before this exists.
2. The day-0 vLLM and SGLang Inkling PRs -- baseline semantics + shapes.
3. Current CUTLASS + PTX ISA docs per arch. Known trap: tcgen05
   block-scaled FP4 MMA is sm_100/sm_103 only, sm_120 differs. Verify
   every instruction per-arch against current docs, never from memory.
4. Build a parity harness: logit parity vs HF transformers reference,
   32 fixed prompts, plus batched-output == batch-1-output consistency
   checks. No kernel is done without it.

UNITS, default leverage order (Phase 0b may reorder):
U1  Fused NVFP4 MoE grouped GEMM, multi-batch: token permute -> per-expert
    tiles -> block-scaled FP4 MMA -> fused dequant + activation + combine.
    Persistent kernel with a tile scheduler that survives skewed expert
    loads (256 experts, batched routing is never uniform).
U2  Relative attention, batched + paged: prefill kernel (warp-specialized,
    FA3-style) and split-KV decode kernel, global + SWA modes behind one
    interface, distance term computed in registers, never materialize the
    bias, paged KV layout compatible with vLLM's block manager.
U3  FP8/NVFP4 paged KV cache with per-block scales, wired into U2. Global
    layers at long context are KV-bound, this is the 2-4x on them.
U4  Router + dispatch fusion: top-k, counts, permute/scatter, combine
    weights, minimal launches. At batch these permutes are real time.
U5  QKVR fused projection GEMM (single weight read for Q,K,V,R).
U6  Decode-step CUDA graphs + stream overlap: attention of layer N
    overlapped with expert weight staging of layer N+1; TP4 comm
    (all-reduce/all-to-all) overlapped with compute, not serialized.
U7  Batch-size-aware dispatch: GEMV-style path for tiny decode batches,
    tensor-core grouped GEMM path above the crossover, measured crossover
    point, automatic.

DONE = every claim evidence-backed:
- Parity gate per kernel: 32/32 logit parity (or documented tolerance)
  AND batched==batch-1 consistency. Fast-and-wrong is a failed unit.
- Kernel gate: >=90% of the BINDING roofline in ncu -- HBM bandwidth where
  memory bound (decode, KV), FP4 tensor-core throughput where compute
  bound (prefill, large-batch MoE). Profile saved to journal/ncu/. Below
  90%, iterate or document the real ceiling.
- E2e gate: vLLM benchmark_serving vs the stock day-0 build. Same
  checkpoint, same quant, same GPUs, same SLO. Report full
  throughput-latency curves across the batch sweep, median of 5 + best,
  never best-only. Both prefill-heavy and decode-heavy mixes.
- Upstream-grade output: clean diffs against the fork base, README with
  methodology + ncu evidence + honest limitations. Bar: an NVIDIA or vLLM
  maintainer reads it and finds nothing to correct.

ALWAYS RESEARCH: verify every CUDA / CUTLASS / PTX / vLLM / SGLang / HF
claim against CURRENT docs, cite in journal + PR. <70% confidence on any
decision -> log to BLOCKERS.md, do not guess.

GUARDRAILS (log to BLOCKERS.md, ask owner): no Lambda spend beyond $200
without owner auth, multi-GPU nodes are expensive, batch remote work and
kill idle boxes immediately. Branch per unit, never push main directly,
never merge red tests. Never git add -A, never reset --hard. Honesty:
every number in LEDGER.md is measured-or-null, no fabricated tok/s,
failures write an actionable last_error. Update LEDGER.md +
journal/inkling-server-kernels.md after every unit.

SCOPE_NOTES: Inkling = 975B total / 41B active MoE, 256 experts, 66
layers, 1M context, NVFP4 ~592GB, BF16 ~1.9TB. Attention is nonstandard:
no RoPE, 4th per-head R projection distance-mixed into logits, alternating
global/SWA layers -- stock FlashAttention paths do not cover it, so U2 is
where the moat is. Decode roofline: ~21GB weight reads per token at 41B
active NVFP4; B300 ~8TB/s. Prefill and batch>=32 decode shift toward FP4
tensor-core bound, which is why U1 is grouped GEMM not GEMV. Owner-gated:
4x B300 quota on Lambda; H200 fallback is W4A16 (different dequant path,
second-class target).