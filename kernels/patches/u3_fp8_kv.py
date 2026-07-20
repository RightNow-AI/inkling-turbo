#!/usr/bin/env python3
"""U3: add optional per-cache-block FP8 K/V writes to qkvr_prep.py.

The existing Triton kernels still produce the exact BF16 K/V values.  When
``quantize_kv`` is true, their attention-cache writes are redirected into a
compact BF16 staging cache whose block ids cover only the touched physical
pages.  The host wrapper then merges each touched page with the dequantized
old page, computes one FP32 scale per ``(cache_block, kv_head)`` as
``amax / 448``, and requantizes the full page to ``float8_e4m3fn``.  Rewriting
the whole touched page keeps earlier entries valid when an incremental write
increases the shared block scale.

Scale tensors have shape ``(num_cache_blocks, num_kv_heads)``.  This changes
only the attention KV-cache storage format; qkvr weights, convolution weights,
the BF16 convolution cache, Q output, and relative-bias output are untouched.
With ``quantize_kv=False`` (the default), the original path is unchanged.

Usage: python u3_fp8_kv.py /path/to/vllm
"""

import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
TARGET = ROOT / "vllm/models/inkling/nvidia/ops/qkvr_prep.py"

HELPERS = '''def _prepare_fp8_kv_staging(
    qkvr: torch.Tensor,
    key_cache: torch.Tensor,
    attention_slot_mapping: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build compact BF16 pages and remap physical attention slots."""
    page_size = key_cache.shape[1]
    valid = attention_slot_mapping >= 0
    safe_slots = attention_slot_mapping.clamp_min(0).to(torch.long)
    blocks = safe_slots // page_size
    offsets = safe_slots % page_size
    active_blocks = torch.unique(blocks[valid], sorted=True)
    compact_blocks = torch.searchsorted(active_blocks, blocks)
    compact_slots = compact_blocks * page_size + offsets
    compact_slots = torch.where(valid, compact_slots, -1)
    compact_slots = compact_slots.to(attention_slot_mapping.dtype)

    staging_blocks = max(active_blocks.shape[0], 1)
    cache_shape = (staging_blocks, page_size, num_kv_heads, head_dim)
    key_staging = qkvr.new_zeros(cache_shape)
    value_staging = qkvr.new_zeros(cache_shape)
    written = torch.zeros(
        (active_blocks.shape[0], page_size),
        dtype=torch.bool,
        device=qkvr.device,
    )
    written[compact_blocks[valid], offsets[valid]] = True
    return (
        key_staging,
        value_staging,
        compact_slots,
        active_blocks,
        written,
    )


def _commit_fp8_cache_blocks(
    cache: torch.Tensor,
    scale: torch.Tensor,
    staging: torch.Tensor,
    active_blocks: torch.Tensor,
    written: torch.Tensor,
) -> None:
    """Merge and requantize touched pages with one scale per block/head."""
    if active_blocks.numel() == 0:
        return
    num_blocks = active_blocks.shape[0]
    old_scale = scale.index_select(0, active_blocks)
    values = cache.index_select(0, active_blocks).float()
    values *= old_scale[:, None, :, None]
    fresh = staging[:num_blocks].float()
    values = torch.where(written[:, :, None, None], fresh, values)

    block_scale = values.abs().amax(dim=(1, 3)) / FP8_E4M3_MAX
    divisor = torch.where(
        block_scale > 0,
        block_scale,
        torch.ones_like(block_scale),
    )
    quantized = values / divisor[:, None, :, None]
    quantized = quantized.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    # index_copy_cuda is not implemented for float8_e4m3fn; scatter through a
    # same-itemsize uint8 view instead.
    cache.view(torch.uint8).index_copy_(
        0,
        active_blocks,
        quantized.to(cache.dtype).view(torch.uint8),
    )
    scale.index_copy_(0, active_blocks, block_scale)


'''

EDITS = [
    (
        "KV_NUM_WARPS = 2\n",
        "KV_NUM_WARPS = 2\nFP8_E4M3_MAX = 448.0\n",
    ),
    (
        "def fused_qkvr_prep(\n",
        HELPERS + "def fused_qkvr_prep(\n",
    ),
    (
        "    conv_block_size: int,\n"
        "    log_scaling: torch.Tensor | None = None,\n"
        ") -> tuple[torch.Tensor, torch.Tensor]:",
        "    conv_block_size: int,\n"
        "    log_scaling: torch.Tensor | None = None,\n"
        "    k_cache_scale: torch.Tensor | None = None,\n"
        "    v_cache_scale: torch.Tensor | None = None,\n"
        "    quantize_kv: bool = False,\n"
        ") -> tuple[torch.Tensor, torch.Tensor]:",
    ),
    (
        "    assert conv_cache.stride(3) == 1\n"
        "    assert key_cache.stride(3) == 1 and value_cache.stride(3) == 1\n"
        "    tokens = qkvr.shape[0]",
        "    assert conv_cache.stride(3) == 1\n"
        "    assert key_cache.stride(3) == 1 and value_cache.stride(3) == 1\n"
        "    if quantize_kv:\n"
        "        assert key_cache.dtype == torch.float8_e4m3fn\n"
        "        assert value_cache.dtype == torch.float8_e4m3fn\n"
        "        assert key_cache.shape == value_cache.shape\n"
        "        assert k_cache_scale is not None\n"
        "        assert v_cache_scale is not None\n"
        "        scale_shape = (key_cache.shape[0], num_kv_heads)\n"
        "        assert k_cache_scale.shape == scale_shape\n"
        "        assert v_cache_scale.shape == scale_shape\n"
        "        assert k_cache_scale.dtype == torch.float32\n"
        "        assert v_cache_scale.dtype == torch.float32\n"
        "        assert k_cache_scale.device == key_cache.device\n"
        "        assert v_cache_scale.device == value_cache.device\n"
        "        assert k_cache_scale.is_contiguous()\n"
        "        assert v_cache_scale.is_contiguous()\n"
        "    tokens = qkvr.shape[0]",
    ),
    (
        "    if tokens == 0:\n"
        "        return q_out, rel_out\n\n"
        "    if tokens < SMALL_TOKEN_THRESHOLD:",
        "    if tokens == 0:\n"
        "        return q_out, rel_out\n\n"
        "    write_key_cache = key_cache\n"
        "    write_value_cache = value_cache\n"
        "    write_attention_slots = attention_slot_mapping\n"
        "    if quantize_kv:\n"
        "        (\n"
        "            write_key_cache,\n"
        "            write_value_cache,\n"
        "            write_attention_slots,\n"
        "            active_blocks,\n"
        "            written,\n"
        "        ) = _prepare_fp8_kv_staging(\n"
        "            qkvr,\n"
        "            key_cache,\n"
        "            attention_slot_mapping,\n"
        "            num_kv_heads,\n"
        "            head_dim,\n"
        "        )\n\n"
        "    if tokens < SMALL_TOKEN_THRESHOLD:",
    ),
    (
        "        _run_fused_small(\n"
        "            qkvr,\n"
        "            q_norm_weight,\n"
        "            q_out,\n"
        "            rel_proj,\n"
        "            rel_out,\n"
        "            k_weight,\n"
        "            v_weight,\n"
        "            k_norm_weight,\n"
        "            conv_cache,\n"
        "            key_cache,\n"
        "            value_cache,\n"
        "            positions,\n"
        "            seq_idx,\n"
        "            conv_slot_mapping,\n"
        "            conv_block_table,\n"
        "            query_start,\n"
        "            attention_slot_mapping,",
        "        _run_fused_small(\n"
        "            qkvr,\n"
        "            q_norm_weight,\n"
        "            q_out,\n"
        "            rel_proj,\n"
        "            rel_out,\n"
        "            k_weight,\n"
        "            v_weight,\n"
        "            k_norm_weight,\n"
        "            conv_cache,\n"
        "            write_key_cache,\n"
        "            write_value_cache,\n"
        "            positions,\n"
        "            seq_idx,\n"
        "            conv_slot_mapping,\n"
        "            conv_block_table,\n"
        "            query_start,\n"
        "            write_attention_slots,",
    ),
    (
        "            log_scaling=log_scaling,\n"
        "        )\n"
        "        return q_out, rel_out",
        "            log_scaling=log_scaling,\n"
        "        )\n"
        "        if quantize_kv:\n"
        "            _commit_fp8_cache_blocks(\n"
        "                key_cache,\n"
        "                k_cache_scale,\n"
        "                write_key_cache,\n"
        "                active_blocks,\n"
        "                written,\n"
        "            )\n"
        "            _commit_fp8_cache_blocks(\n"
        "                value_cache,\n"
        "                v_cache_scale,\n"
        "                write_value_cache,\n"
        "                active_blocks,\n"
        "                written,\n"
        "            )\n"
        "        return q_out, rel_out",
    ),
    (
        "    with torch.cuda.stream(kv_stream):\n"
        "        _run_tiled_kv(\n"
        "            qkvr,\n"
        "            k_weight,\n"
        "            v_weight,\n"
        "            k_norm_weight,\n"
        "            conv_cache,\n"
        "            key_cache,\n"
        "            value_cache,\n"
        "            positions,\n"
        "            seq_idx,\n"
        "            conv_slot_mapping,\n"
        "            conv_block_table,\n"
        "            query_start,\n"
        "            attention_slot_mapping,",
        "    with torch.cuda.stream(kv_stream):\n"
        "        _run_tiled_kv(\n"
        "            qkvr,\n"
        "            k_weight,\n"
        "            v_weight,\n"
        "            k_norm_weight,\n"
        "            conv_cache,\n"
        "            write_key_cache,\n"
        "            write_value_cache,\n"
        "            positions,\n"
        "            seq_idx,\n"
        "            conv_slot_mapping,\n"
        "            conv_block_table,\n"
        "            query_start,\n"
        "            write_attention_slots,",
    ),
    (
        "    current_stream.wait_stream(kv_stream)\n"
        "    return q_out, rel_out",
        "    current_stream.wait_stream(kv_stream)\n"
        "    if quantize_kv:\n"
        "        _commit_fp8_cache_blocks(\n"
        "            key_cache,\n"
        "            k_cache_scale,\n"
        "            write_key_cache,\n"
        "            active_blocks,\n"
        "            written,\n"
        "        )\n"
        "        _commit_fp8_cache_blocks(\n"
        "            value_cache,\n"
        "            v_cache_scale,\n"
        "            write_value_cache,\n"
        "            active_blocks,\n"
        "            written,\n"
        "        )\n"
        "    return q_out, rel_out",
    ),
]


def apply(path: Path) -> int:
    source = path.read_text()
    edits = 0
    for old, new in EDITS:
        if new in source:
            continue
        assert old in source, f"anchor not found in {path.name}: {old[:80]!r}"
        source = source.replace(old, new, 1)
        edits += 1
    path.write_text(source)
    return edits


if __name__ == "__main__":
    print(f"qkvr_prep.py: {apply(TARGET)} edits applied")
