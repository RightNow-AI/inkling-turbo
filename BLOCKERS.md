# BLOCKERS

## B1 — Remote hardware: spec'd GPUs do not exist on this Lambda account [OPEN, owner decision needed]

Logged: 2026-07-17. Confidence in facts: high (API-enumerated, not guessed).

- Spec asks for 4× B300 (TP4 W4A4) or 8× H200 (W4A16 fallback).
- Lambda API instance-type list contains **no B300 and no H200 type at all**.
- Closest real option: `gpu_8x_b200_sxm6` @ $53.52/hr — currently **no capacity in any region** (fluctuates; will poll).
- $200 guardrail ≈ 3.7 hours on 8× B200. One full Phase 0b profiling session (launch + install + 592GB weight pull + batch sweep + nsys/ncu) is realistically 2.5–4 h ≈ $135–215 — AT or ABOVE the cap.

Owner options:
1. Authorize up to ~$250 for ONE batched Phase 0b session on 8× B200 when capacity appears (recommended; single session, everything scripted, instance killed after).
2. Provide access to B200/B300/GB200/H200 on another provider.
3. Local-only until capacity: proceed U-unit kernel work on sm_120 with per-op parity + microbench, accept that e2e serving numbers wait.

Until answered: proceeding with option 3 work that is common to all outcomes.

## B2 — NVFP4 block-scale layout [RESOLVED 2026-07-17]

Resolved from `vllm/models/inkling/nvfp4.py` + `moe.py` (fork base 850295881): ModelOpt
format, FP4 E2M1 weights, FP8 E4M3 block scales, group_size 16 (only 16 supported),
per-tensor `*_scale_2` + input_scale = amax/(448·6). Routed experts only; attention,
dense MLP, sink experts BF16 via exclude_modules. Details: journal/day0-implementation.md.

## B3 — sm_120 vs sm_100 FP4 MMA instruction set [OPEN, research]

Known trap per spec: tcgen05 block-scaled FP4 MMA is sm_100/sm_103 only. Local RTX 5090 Laptop is sm_120. Every FP4 MMA instruction used in U1/U3 must be verified against the CURRENT PTX ISA per-arch table before use; local kernels may need an sm_120 variant that differs from the B200 (sm_100) deployment path. No instruction written from memory.
