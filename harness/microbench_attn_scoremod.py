#!/usr/bin/env python3
"""Honest sm_90 attention baseline: score_mod path (vLLM's Hopper route) at
real Inkling shapes, per-kernel breakdown via torch.profiler.

Unlike microbench_attn_day0.py (rel_bias path, numerically wrong on sm_90,
journal/remote/h100-session1.md session 3), this measures the path vLLM
actually serves with on Hopper, parity-proven to 7.8e-3.

Output: printed tables + microbench_attn_scoremod.json next to this script.
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
        t = ev.self_device_time_total
        if t > 0:
            agg[ev.key] = agg.get(ev.key, 0.0) + t / iters
    top = dict(sorted(agg.items(), key=lambda kv: -kv[1])[:8])
    total = sum(agg.values())
    RESULTS[name] = {"total_us_per_iter": total, "kernels_us": top}
    print(f"\n[{name}] total {total:.1f} us/iter")
    for k, v in top.items():
        print(f"    {v:9.1f} us  {v / total * 100:5.1f}%  {k[:90]}")


def attn_case(T_q: int, T_k: int, Hq: int, Hkv: int, ext: int,
              window_left: int | None, mode: str = "score_mod"):
    from vllm.models.inkling.nvidia.ops.fa4_rel_attention import _get_score_mod
    from vllm.vllm_flash_attn.cute import flash_attn_varlen_func

    D = 128
    dev = "cuda"
    q = torch.randn(T_q, Hq, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=dev)
    cu_q = torch.tensor([0, T_q], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, T_k], dtype=torch.int32, device=dev)
    window = (None, None) if window_left is None else (window_left, 0)

    if mode in ("relproj", "relproj_v15"):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from kernels.relproj_score_mod import (
            get_relproj_score_mod,
            get_relproj_score_mod_v15)

        r = torch.randn(T_q, Hq, 16, dtype=torch.bfloat16, device=dev)
        proj = torch.randn(16, ext, dtype=torch.bfloat16, device=dev)
        if mode == "relproj_v15":
            score_mod = get_relproj_score_mod_v15(ext)
            aux = [r.contiguous(), proj.T.contiguous()]
        else:
            score_mod = get_relproj_score_mod(ext)
            aux = [r.contiguous(), proj.contiguous()]
    else:
        rel = torch.randn(T_q, Hq, ext, dtype=torch.bfloat16, device=dev)
        score_mod = _get_score_mod(ext)
        aux = [rel.contiguous()]

    def fn():
        flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=T_q, max_seqlen_k=T_k,
            softmax_scale=1.0 / D, causal=True, window_size=window,
            score_mod=score_mod, aux_tensors=aux)
    return fn


def main() -> None:
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"capability {torch.cuda.get_device_capability(0)}")

    cases = [
        ("prefill_global_8k", lambda: attn_case(8192, 8192, 64, 8, 1024, None)),
        ("prefill_swa_8k", lambda: attn_case(8192, 8192, 64, 16, 512, 511)),
        ("decode_b32_global_kv8k", lambda: attn_case(32, 8192, 64, 8, 1024, None)),
        ("decode_b32_global_kv64k", lambda: attn_case(32, 65536, 64, 8, 1024, None)),
        ("decode_b1_global_kv64k", lambda: attn_case(1, 65536, 64, 8, 1024, None)),
        # bf16 KV-read roofline anchors for U3: same decode, no bias at all
        ("decode_b1_plain_kv64k", lambda: attn_case_plain(1, 65536, 64, 8)),
        ("decode_b32_plain_kv64k", lambda: attn_case_plain(32, 65536, 64, 8)),
        # U2-Hopper Design B V1: register-resident r-projection bias
        ("relproj_prefill_global_8k",
         lambda: attn_case(8192, 8192, 64, 8, 1024, None, mode="relproj")),
        ("relproj_decode_b32_kv64k",
         lambda: attn_case(32, 65536, 64, 8, 1024, None, mode="relproj")),
        ("relproj_decode_b1_kv64k",
         lambda: attn_case(1, 65536, 64, 8, 1024, None, mode="relproj")),
        ("relprojT_decode_b1_kv64k",
         lambda: attn_case(1, 65536, 64, 8, 1024, None, mode="relproj_v15")),
        ("relprojT_decode_b32_kv64k",
         lambda: attn_case(32, 65536, 64, 8, 1024, None, mode="relproj_v15")),
        ("relprojT_prefill_global_8k",
         lambda: attn_case(8192, 8192, 64, 8, 1024, None, mode="relproj_v15")),
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


def attn_case_plain(T_q: int, T_k: int, Hq: int, Hkv: int):
    from vllm.vllm_flash_attn.cute import flash_attn_varlen_func

    D = 128
    dev = "cuda"
    q = torch.randn(T_q, Hq, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=dev)
    cu_q = torch.tensor([0, T_q], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, T_k], dtype=torch.int32, device=dev)

    def fn():
        flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=T_q, max_seqlen_k=T_k,
            softmax_scale=1.0 / D, causal=True)
    return fn


if __name__ == "__main__":
    main()
