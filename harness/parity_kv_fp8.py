#!/usr/bin/env python3
"""Parity for U3 per-block FP8 attention-cache writes.

This extends ``parity_qkvr_prep.py`` by reusing its exact BF16 qkvr oracle
and exercising both the fused-small and tiled launch paths with pre-existing
quantized page contents.  Each FP32 scale covers one physical cache block and
one KV head across every page token and all 128 head dimensions.

Tolerance derivation: E4M3 has three stored mantissa bits, hence four bits of
normal significand precision.  Round-to-nearest therefore contributes at most
``2**-4 = 1/16`` relative error for normal values.  Its minimum subnormal is
``2**-9`` in scaled units, so subnormal rounding contributes at most
``scale * 2**-10`` absolute error.  The original BF16 kernel/reference gate
allows ``2e-2`` before cache quantization.  For reference value ``x`` the
elementwise acceptance bound is thus
``2e-2 * (1 + 1/16) + abs(x)/16 + scale/1024 + 1e-6``.  The scale itself is
checked against ``amax(reference_block_head) / 448`` with the BF16 allowance
divided by 448.

Run after applying ``kernels/patches/u3_fp8_kv.py`` to a vLLM checkout.
"""

from __future__ import annotations

import torch

from parity_qkvr_prep import (
    CONV_BS,
    D,
    D_REL,
    HKV,
    HQ,
    KVW,
    OFF_K,
    OFF_V,
    PAGE,
    QW,
    W,
    reference,
)

FP8_MAX = 448.0
BF16_KERNEL_ATOL = 2e-2
NORMAL_REL_ERROR = 1.0 / 16.0
SUBNORMAL_ABS_UNITS = 1.0 / 1024.0


def encode_fp8_pages(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode page values with one amax/448 scale per block and head."""
    values = values.float()
    scale = values.abs().amax(dim=(1, 3)) / FP8_MAX
    divisor = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = values / divisor[:, None, :, None]
    normalized = normalized.clamp(-FP8_MAX, FP8_MAX)
    return normalized.to(torch.float8_e4m3fn), scale


def cache_error(
    name: str,
    cache: torch.Tensor,
    scale: torch.Tensor,
    expected: torch.Tensor,
    active_blocks: torch.Tensor,
) -> str | None:
    got_scale = scale.index_select(0, active_blocks).float().cpu()
    got_cache = cache.index_select(0, active_blocks).float().cpu()
    got = got_cache * got_scale[:, None, :, None]
    ref_blocks = active_blocks.cpu()
    ref = expected.index_select(0, ref_blocks).float()
    bound = (
        BF16_KERNEL_ATOL * (1.0 + NORMAL_REL_ERROR)
        + ref.abs() * NORMAL_REL_ERROR
        + got_scale[:, None, :, None] * SUBNORMAL_ABS_UNITS
        + 1e-6
    )
    excess = (got - ref).abs() - bound
    max_excess = excess.max().item()
    if max_excess > 0:
        max_diff = (got - ref).abs().max().item()
        return (
            f"{name} dequant max_diff={max_diff:.3e} "
            f"max_excess={max_excess:.3e}"
        )
    return None


def scale_error(
    name: str,
    scale: torch.Tensor,
    expected: torch.Tensor,
    active_blocks: torch.Tensor,
) -> str | None:
    got = scale.index_select(0, active_blocks).float().cpu()
    ref_blocks = active_blocks.cpu()
    ref_values = expected.index_select(0, ref_blocks).float()
    ref = ref_values.abs().amax(dim=(1, 3)) / FP8_MAX
    diff = (got - ref).abs().max().item()
    tolerance = BF16_KERNEL_ATOL / FP8_MAX + 2e-6
    if diff > tolerance:
        return f"{name} scale max_diff={diff:.3e} tol={tolerance:.3e}"
    return None


def run_case(
    name: str,
    seq_lens: list[int],
    ext: int,
    use_log: bool,
    seed: int,
) -> list[str]:
    from vllm.models.inkling.nvidia.ops.qkvr_prep import fused_qkvr_prep

    torch.manual_seed(seed)
    device = "cuda"
    tokens = sum(seq_lens)
    num_requests = len(seq_lens)
    eps = 1e-6
    width = QW + 2 * KVW + HQ * D_REL

    qkvr = torch.randn(
        tokens,
        width,
        dtype=torch.bfloat16,
        device=device,
    )
    k_weight = (
        torch.randn(HKV * D, W, dtype=torch.bfloat16, device=device) * 0.3
    )
    v_weight = (
        torch.randn(HKV * D, W, dtype=torch.bfloat16, device=device) * 0.3
    )
    q_norm_weight = (
        torch.rand(D, dtype=torch.bfloat16, device=device) + 0.5
    )
    k_norm_weight = (
        torch.rand(D, dtype=torch.bfloat16, device=device) + 0.5
    )
    rel_proj = (
        torch.randn(D_REL, ext, dtype=torch.bfloat16, device=device) * 0.5
    )

    positions_list: list[int] = []
    seq_idx_list: list[int] = []
    query_start_list: list[int] = []
    row = 0
    for request, length in enumerate(seq_lens):
        positions_list.extend(range(length))
        seq_idx_list.extend([request] * length)
        query_start_list.extend([row] * length)
        row += length
    positions = torch.tensor(
        positions_list,
        dtype=torch.int64,
        device=device,
    )
    seq_idx = torch.tensor(
        seq_idx_list,
        dtype=torch.int32,
        device=device,
    )
    query_start = torch.tensor(
        query_start_list,
        dtype=torch.int32,
        device=device,
    )

    max_length = max(seq_lens)
    max_blocks = (max_length + CONV_BS - 1) // CONV_BS + 1
    num_conv_blocks = num_requests * max_blocks + 3
    permutation = torch.randperm(num_conv_blocks - 1, device=device) + 1
    conv_table = permutation[: num_requests * max_blocks].reshape(
        num_requests,
        max_blocks,
    )
    conv_table = conv_table.to(torch.int32)
    conv_cache = torch.zeros(
        num_conv_blocks,
        HKV,
        CONV_BS,
        256,
        dtype=torch.bfloat16,
        device=device,
    )
    conv_slots = torch.tensor(
        [
            int(conv_table[int(seq_idx[i]), int(positions[i]) // CONV_BS])
            * CONV_BS
            + int(positions[i]) % CONV_BS
            for i in range(tokens)
        ],
        dtype=torch.int32,
        device=device,
    )

    num_pages = (tokens + PAGE - 1) // PAGE + 2
    attention_slots = torch.randperm(
        num_pages * PAGE,
        device=device,
    )[:tokens].to(torch.int32)
    initial_key = (
        torch.randn(
            num_pages,
            PAGE,
            HKV,
            D,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.05
    )
    initial_value = (
        torch.randn(
            num_pages,
            PAGE,
            HKV,
            D,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.05
    )
    key_cache, k_cache_scale = encode_fp8_pages(initial_key)
    value_cache, v_cache_scale = encode_fp8_pages(initial_value)
    expected_key = (
        key_cache.float() * k_cache_scale[:, None, :, None]
    ).cpu()
    expected_value = (
        value_cache.float() * v_cache_scale[:, None, :, None]
    ).cpu()

    log_scaling = None
    if use_log:
        log_scaling = (
            torch.rand(tokens, dtype=torch.float32, device=device) * 0.3
            + 1.0
        )

    conv_cache_before = conv_cache.clone()
    q_out, rel_out = fused_qkvr_prep(
        qkvr,
        k_weight,
        v_weight,
        q_norm_weight,
        k_norm_weight,
        rel_proj,
        eps,
        HQ,
        HKV,
        D,
        D_REL,
        conv_cache,
        key_cache,
        value_cache,
        positions,
        conv_table,
        seq_idx,
        conv_slots,
        query_start,
        attention_slots,
        OFF_K,
        OFF_V,
        CONV_BS,
        log_scaling,
        k_cache_scale=k_cache_scale,
        v_cache_scale=v_cache_scale,
        quantize_kv=True,
    )
    torch.cuda.synchronize()

    ref_q, ref_rel, key_writes, value_writes, ref_conv = reference(
        qkvr.cpu(),
        k_weight.cpu(),
        v_weight.cpu(),
        q_norm_weight.cpu(),
        k_norm_weight.cpu(),
        rel_proj.cpu(),
        eps,
        conv_cache_before.cpu(),
        conv_table.cpu(),
        positions.cpu(),
        seq_idx.cpu(),
        query_start.cpu(),
        conv_slots.cpu(),
        attention_slots.cpu(),
        log_scaling.cpu() if log_scaling is not None else None,
    )
    for (slot, head), values in key_writes.items():
        expected_key[slot // PAGE, slot % PAGE, head] = values
    for (slot, head), values in value_writes.items():
        expected_value[slot // PAGE, slot % PAGE, head] = values

    errors: list[str] = []
    q_diff = (
        q_out.view(tokens, HQ, D).float().cpu() - ref_q
    ).abs().max().item()
    if q_diff > BF16_KERNEL_ATOL:
        errors.append(f"q max_diff={q_diff:.3e}")
    rel_diff = (rel_out.float().cpu() - ref_rel).abs().max().item()
    if rel_diff > BF16_KERNEL_ATOL:
        errors.append(f"rel max_diff={rel_diff:.3e}")
    conv_diff = (
        conv_cache.float().cpu() - ref_conv
    ).abs().max().item()
    if conv_diff > BF16_KERNEL_ATOL:
        errors.append(f"conv cache max_diff={conv_diff:.3e}")

    active_blocks = torch.unique(
        (attention_slots.to(torch.long) // PAGE),
        sorted=True,
    )
    checks = [
        scale_error(
            "key",
            k_cache_scale,
            expected_key,
            active_blocks,
        ),
        scale_error(
            "value",
            v_cache_scale,
            expected_value,
            active_blocks,
        ),
        cache_error(
            "key",
            key_cache,
            k_cache_scale,
            expected_key,
            active_blocks,
        ),
        cache_error(
            "value",
            value_cache,
            v_cache_scale,
            expected_value,
            active_blocks,
        ),
    ]
    errors.extend(error for error in checks if error is not None)
    return errors


def main() -> None:
    print(f"device: {torch.cuda.get_device_name(0)}")
    cases = [
        ("fp8_small_global", [5, 9, 3], 1024, True, 3103),
        ("fp8_tiled_local", [80, 80], 512, False, 3104),
    ]
    failures = 0
    for name, seq_lens, extent, use_log, seed in cases:
        try:
            errors = run_case(name, seq_lens, extent, use_log, seed)
        except Exception as exc:  # noqa: BLE001
            errors = [f"EXCEPTION {type(exc).__name__}: {exc}"]
        if errors:
            failures += 1
            print(f"[{name}] FAIL: {'; '.join(errors)}")
        else:
            print(f"[{name}] OK")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
