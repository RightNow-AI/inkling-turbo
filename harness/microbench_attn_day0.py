#!/usr/bin/env python3
"""Day-0 FA4 rel-attention microbench at real Inkling shapes (single GPU).

Measures per-CUDA-kernel time via torch.profiler so the ShearingBias /
forward / combine split is visible, the U2 evidence (bias share of
attention time). Sections are independent; one failure doesn't kill the rest.

Output: printed tables + JSON next to this script (microbench_attn_day0.json).
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch

RESULTS: dict = {}


def profile_case(name: str, fn, iters: int = 20, warmup: int = 5) -> None:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    agg: dict[str, float] = {}
    for ev in prof.key_averages():
        if ev.device_type == torch.profiler.DeviceType.CUDA or ev.self_device_time_total > 0:
            t = ev.self_device_time_total  # us, summed over iters
            if t > 0:
                agg[ev.key] = agg.get(ev.key, 0.0) + t / iters
    top = dict(sorted(agg.items(), key=lambda kv: -kv[1])[:8])
    total = sum(agg.values())
    RESULTS[name] = {"total_us_per_iter": total, "kernels_us": top}
    print(f"\n[{name}] total {total:.1f} us/iter")
    for k, v in top.items():
        print(f"    {v:9.1f} us  {v / total * 100:5.1f}%  {k[:90]}")


def attn_case(T_q: int, T_k: int, Hq: int, Hkv: int, ext: int,
              window_left: int | None):
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func

    D = 128
    dev = "cuda"
    q = torch.randn(T_q, Hq, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=dev)
    rel = torch.randn(T_q, Hq, ext, dtype=torch.bfloat16, device=dev)
    cu_q = torch.tensor([0, T_q], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, T_k], dtype=torch.int32, device=dev)
    window = (None, None) if window_left is None else (window_left, 0)

    def fn():
        flash_attn_varlen_func(
            q=q, k=k, v=v, rel_bias=rel,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=T_q, max_seqlen_k=T_k,
            softmax_scale=1.0 / D, causal=True, window_size=window)
    return fn


def batched_decode_case(B: int, L: int, Hq: int, Hkv: int, ext: int,
                        window_left: int | None):
    """TRUE multi-sequence decode: B sequences, 1 query token each, each with
    its own KV of length L. The plain attn_case "b32" proxy packs 32 rows into
    ONE sequence, which keeps the grid at heads x 1 CTAs and hides batch
    parallelism (session 24 ncu: identical profile to b1)."""
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func

    D = 128
    dev = "cuda"
    q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B * L, Hkv, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B * L, Hkv, D, dtype=torch.bfloat16, device=dev)
    rel = torch.randn(B, Hq, ext, dtype=torch.bfloat16, device=dev)
    cu_q = torch.arange(B + 1, dtype=torch.int32, device=dev)
    cu_k = torch.arange(B + 1, dtype=torch.int32, device=dev) * L
    window = (None, None) if window_left is None else (window_left, 0)

    def fn():
        flash_attn_varlen_func(
            q=q, k=k, v=v, rel_bias=rel,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=1, max_seqlen_k=L,
            softmax_scale=1.0 / D, causal=True, window_size=window)
    return fn


def gate_case(T: int):
    from vllm.models.inkling.nvidia.moe import inkling_gate_select

    dev = "cuda"
    logits = torch.randn(T, 264, dtype=torch.float32, device=dev)
    bias = torch.randn(256, dtype=torch.float32, device=dev)
    gscale = torch.ones(1, dtype=torch.float32, device=dev)

    def fn():
        inkling_gate_select(logits, 258, 256, 6, 2, bias, 8.0, gscale)
    return fn


def main() -> None:
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"capability {torch.cuda.get_device_capability(0)}")

    # Full-model head geometry (no TP on 1 GPU): global 64q/8kv ext1024,
    # SWA 64q/16kv ext512 win511.
    cases = [
        ("prefill_global_8k", lambda: attn_case(8192, 8192, 64, 8, 1024, None)),
        ("prefill_swa_8k", lambda: attn_case(8192, 8192, 64, 16, 512, 511)),
        ("decode_b32_global_kv8k", lambda: attn_case(32, 8192, 64, 8, 1024, None)),
        ("decode_b32_global_kv64k", lambda: attn_case(32, 65536, 64, 8, 1024, None)),
        ("decode_b1_global_kv64k", lambda: attn_case(1, 65536, 64, 8, 1024, None)),
        ("decode_32seqs_global_kv8k", lambda: batched_decode_case(32, 8192, 64, 8, 1024, None)),
        ("decode_32seqs_global_kv64k", lambda: batched_decode_case(32, 65536, 64, 8, 1024, None)),
        ("gate_select_T1", lambda: gate_case(1)),
        ("gate_select_T4096", lambda: gate_case(4096)),
    ]
    for name, make in cases:
        try:
            profile_case(name, make())
        except Exception:  # noqa: BLE001
            print(f"\n[{name}] FAILED:")
            traceback.print_exc()
            RESULTS[name] = {"error": traceback.format_exc(limit=3)}

    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
