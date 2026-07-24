#!/usr/bin/env python3
"""Parity: Inkling SwiGLU kernels vs PyTorch references.

Covers (from vllm/models/inkling/nvidia/ops/silu_and_mul.py):
  silu_and_mul_triton:     out[:, j] = silu(in[:, 2j]) * in[:, 2j+1]
  sink_silu_mul_epilogue:  out[t, s*F+j] =
      silu(raw[t, s*2F+2j] * alpha_s) * (raw[t, s*2F+2j+1] * alpha_s)
      * (gamma[t, s] * ratio_s)
Includes the strided-input cases the production path exercises (raw as a
column slice of a padded GEMM output; gammas as a column slice of the
gate-select weights).

Run (WSL): cd ~/inkling-turbo/vllm && source .venv/bin/activate && \
  python $REPO/harness/parity_swiglu_sink.py
"""

from __future__ import annotations

import torch


def ref_silu_mul(gateup: torch.Tensor) -> torch.Tensor:
    g = gateup[:, 0::2].float()
    u = gateup[:, 1::2].float()
    return (torch.nn.functional.silu(g) * u).to(gateup.dtype)


def ref_sink_epilogue(raw, alphas, gammas, ratios, S, out_dtype):
    T = raw.shape[0]
    F = raw.shape[1] // (2 * S)
    r = raw.float().view(T, S, F, 2)
    g = r[..., 0] * alphas.view(1, S, 1)
    u = r[..., 1] * alphas.view(1, S, 1)
    w = (gammas.float() * ratios.view(1, S)).view(T, S, 1)
    return (torch.nn.functional.silu(g) * u * w).reshape(T, S * F).to(out_dtype)


def main() -> None:
    from vllm.models.inkling.nvidia.ops.silu_and_mul import (
        silu_and_mul_triton,
        sink_silu_mul_epilogue,
    )

    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}")
    failures = 0

    # Dense MLP SwiGLU: real dense_intermediate 24576 (2N=49152) and TP shards.
    for M, N in [(1, 24576), (17, 12288), (256, 3072), (4096, 3072)]:
        torch.manual_seed(M * 100000 + N)
        x = torch.randn(M, 2 * N, dtype=torch.bfloat16, device=dev)
        d = (silu_and_mul_triton(x).float() - ref_silu_mul(x).float()).abs().max()
        ok = d.item() <= 2e-2
        failures += 0 if ok else 1
        print(f"[silu_and_mul M={M} N={N}] max_diff={d.item():.3e} "
              f"{'OK' if ok else 'FAIL'}")

    # Sink epilogue: S=2, F=3072 (no TP) and 1536 (TP2), contiguous + strided.
    for T, F, strided in [(1, 3072, False), (33, 3072, True), (256, 1536, True),
                          (4096, 3072, False)]:
        S = 2
        torch.manual_seed(T * 7 + F)
        if strided:
            padded = torch.randn(T, S * 2 * F + 128, dtype=torch.bfloat16,
                                 device=dev)
            raw = padded[:, : S * 2 * F]
            wsel = torch.rand(T, 8, dtype=torch.float32, device=dev)
            gammas = wsel[:, 6:]
        else:
            raw = torch.randn(T, S * 2 * F, dtype=torch.bfloat16, device=dev)
            gammas = torch.rand(T, S, dtype=torch.float32, device=dev)
        alphas = torch.rand(S, dtype=torch.float32, device=dev) + 0.5
        ratios = torch.rand(S, dtype=torch.float32, device=dev) + 0.5

        out = sink_silu_mul_epilogue(raw, alphas, gammas, ratios, S,
                                     torch.bfloat16)
        ref = ref_sink_epilogue(raw, alphas, gammas, ratios, S, torch.bfloat16)
        d = (out.float() - ref.float()).abs().max().item()
        ok = d <= 2e-2
        failures += 0 if ok else 1
        print(f"[sink T={T} F={F} strided={int(strided)}] max_diff={d:.3e} "
              f"{'OK' if ok else 'FAIL'}")

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
