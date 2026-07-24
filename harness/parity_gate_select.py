#!/usr/bin/env python3
"""Parity: Inkling gate-select Triton kernel vs PyTorch reference.

Semantics (from vllm/models/inkling/nvidia/moe.py `_inkling_gate_select_kernel`,
journal/day0-implementation.md):
  sel      = sigmoid(routed_logits) (+ selection bias)      [selection only]
  top-k    = iterative argmax over sel (K=6), stable ties
  active   = RAW logits of the k selected + the S shared logits (ids R..R+S-1)
  weights  = softmax(logsigmoid(active)) * route_scale * global_scale
Outputs: ids [T, K+S] int32, weights [T, K+S] fp32.

Run (WSL): cd ~/inkling-turbo/vllm && source .venv/bin/activate && \
  python $REPO/harness/parity_gate_select.py
"""

from __future__ import annotations

import torch


def reference_gate_select(
    logits: torch.Tensor,  # [T, >=R+S] fp32 (may carry GEMM padding)
    n_routed: int,
    topk: int,
    n_shared: int,
    bias: torch.Tensor | None,
    route_scale: float,
    global_scale: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    lr = logits[:, :n_routed].float()
    sel = torch.sigmoid(lr)
    if bias is not None:
        sel = sel + bias.float()
    top_ids = torch.topk(sel, topk, dim=1).indices.to(torch.int32)
    active_routed = lr.gather(1, top_ids.long())
    shared = logits[:, n_routed : n_routed + n_shared].float()
    active = torch.cat([active_routed, shared], dim=1)
    lp = torch.nn.functional.logsigmoid(active)
    w = torch.softmax(lp, dim=1) * route_scale
    if global_scale is not None:
        w = w * global_scale.float()
    shared_ids = (
        torch.arange(n_routed, n_routed + n_shared, dtype=torch.int32,
                     device=logits.device).expand(logits.shape[0], -1)
    )
    return w, torch.cat([top_ids, shared_ids], dim=1)


def run_case(T: int, use_bias: bool, use_gscale: bool, seed: int) -> list[str]:
    from vllm.models.inkling.nvidia.moe import inkling_gate_select

    R, K, S = 256, 6, 2
    G = R + S
    pad = G + (-G) % 8
    torch.manual_seed(seed)
    dev = "cuda"
    logits = torch.randn(T, pad, dtype=torch.float32, device=dev) * 2
    bias = torch.randn(R, dtype=torch.float32, device=dev) * 0.1 if use_bias else None
    gscale = (torch.rand(1, dtype=torch.float32, device=dev) + 0.5
              if use_gscale else None)

    w_k, ids_k = inkling_gate_select(logits, G, R, K, S, bias, 8.0, gscale)
    w_r, ids_r = reference_gate_select(logits, R, K, S, bias, 8.0, gscale)

    errs = []
    # Routed ids compare as SETS per token (kernel: iterative argmax order,
    # reference: topk order, both descending-sel so order should match too;
    # check order strictly and fall back to set equality for diagnostics).
    if not torch.equal(ids_k, ids_r):
        same_set = (ids_k[:, :K].sort(dim=1).values
                    == ids_r[:, :K].sort(dim=1).values).all()
        errs.append(f"ids mismatch (same set: {bool(same_set)})")
    dw = (w_k - w_r).abs().max().item()
    if dw > 1e-5:
        errs.append(f"weights max_diff={dw:.3e}")
    return errs


def main() -> None:
    print(f"device: {torch.cuda.get_device_name(0)}")
    failures = 0
    for T in (1, 7, 64, 333, 4096):
        for use_bias in (True, False):
            for use_gscale in (True, False):
                name = f"T{T}_bias{int(use_bias)}_gs{int(use_gscale)}"
                errs = run_case(T, use_bias, use_gscale,
                                seed=hash(name) % (2**31))
                if errs:
                    failures += 1
                    print(f"[{name}] FAIL: {'; '.join(errs)}")
                else:
                    print(f"[{name}] OK")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
