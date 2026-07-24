#!/usr/bin/env python3
"""U2 shear fusion: make qkvr_prep write the ShearingBias layout directly.

Today the relative-bias path is three kernels:

    qkvr_prep      -> rel_logits (T, H, rel_extent), natural layout
    ShearingBias   -> reads it, writes a sheared copy (T+128, H, rel_extent+256)
    attention      -> reads 128x128 sheared tiles

ShearingBias is pure data movement. It costs 460.9us of our 1223.0us SWA-8K
prefill and 827.2us of our 3308.8us global-8K prefill (session 25, H100,
journal/remote/microbench_attn_day0_session25_h100.json). qkvr_prep already
writes the rel output, so it can write it sheared and the middle kernel
disappears.

LAYOUT CONTRACT (derived from tml-fa4 shearing_bias.py:357-476, generalised
from the machine-verified contract in journal/u2-hopper-design.md:244-258)

  journal, "Layout contract COMPLETE (machine-verified, 2026-07-19)":
    "col(i, k) = k + padded - 128 * (m_block(i) + 1),  m_block(i) = i // 128"
  with padded = rel_extent + 256, verified 20,100/20,100 positions at T=200.

That form assumes seqlen_q == seqlen_k (pure prefill).  The writer's actual
per-row placement is

    col(i, d) = bias_idx_right(i) - 1 - d        d = q_pos - kv_pos = rel index
    bias_idx_right(i) = n_idx_right(i) + rel_extent_padded
                        - 128 * n_block_max(m_block(i))

so with the reversed index c = rel_extent - 1 - d the destination column is
contiguous and increasing:

    col(i, c) = base(i) + c
    base(i)   = n_idx_right(i) + 256 - 128 * n_block_max(m_block(i))

    n_idx_right(i) = i_local + 1 + (seqlen_k - seqlen_q) + window_right
    n_block_max(m) = min(ceil(seqlen_k / 128),
                         ceil((128 * (m + 1) + seqlen_k - seqlen_q
                               + window_right) / 128))

base(i) is provably in [1, 256] and bias_idx_right in [rel_extent+1, padded],
so the 256-column pad is exactly the room the shear needs.  Both closed forms
were re-derived from the writer arithmetic and checked over 6906 row
configurations (rel_extent 128/512/1024, prefill / decode / chunked, causal
and sliding-window) and against the journal contract, before this patch was
written.  They are STILL UNVALIDATED ON SILICON: no GPU was available.

VALUE / PAD REGIONS, per row (this is the whole output row, nothing else
writes it once ShearingBias is gone):

    [0, base)                      left_pad
    [base, base + rel_extent)      value if c >= rel_extent - num_bias_vals
                                   else left_pad
    [base + rel_extent, padded)    -inf   (right pad, always)

    num_bias_vals = n_idx_right - max(n_idx_left, n_idx_right - rel_extent)
    left_pad      = -inf if window_size_left is not None else 0.0
                    (shearing_bias.py:88-89)

The three regions are disjoint and cover [0, padded), so every column is
written exactly once and no zero-fill pass is needed.

CONSTRAINTS
  - pack_gqa must be False.  Packing redefines the 128-row block over packed
    rows, so base() would become head-dependent.  arch 9 + bias already forces
    pack_gqa=False (interface.py, session-24 root cause); this patch asserts it.
  - varlen only for the fused writer.  A batched problem is the same thing with
    uniform cu_seqlens, and harness/parity_shear_fusion.py checks the fused
    output against the BATCHED ShearingBias reference to prove it.
  - causal (window None) or local with window_size_right set, matching the
    interface's existing rel-bias assertion.

Enabled per call: attention.py builds a RelShearSpec only when
INKLING_TURBO_FUSED_SHEAR=1.  Default off, because none of this has run on a
GPU yet.

Targets (all under the vLLM tree; interface.py is the deployed copy of
kernels/tml_fa4_modified/interface.py, re-apply after every bootstrap):
  vllm/models/inkling/nvidia/ops/qkvr_prep.py
  vllm/models/inkling/nvidia/ops/fa4_rel_attention.py
  vllm/models/inkling/nvidia/attention.py
  vllm/third_party/tml_fa4/interface.py

ORDER MATTERS.  Apply LAST:
    deploy kernels/tml_fa4_modified -> u2_serving_route.py -> u3_fp8_kv.py
    -> u2_shear_fusion.py
u3_fp8_kv.py anchors on the `fused_qkvr_prep` signature tail and on the
`_run_fused_small(...)` call tail, both of which this patch extends, so u3
must go first.  u2_serving_route.py is order-independent (it rewrites only the
body of `_use_sheared_bias`).  Verified by dry run on copies of the tree.

Usage: python u2_shear_fusion.py /path/to/vllm
"""

import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

QKVR = ROOT / "vllm/models/inkling/nvidia/ops/qkvr_prep.py"
FA4 = ROOT / "vllm/models/inkling/nvidia/ops/fa4_rel_attention.py"
ATTN = ROOT / "vllm/models/inkling/nvidia/attention.py"
IFACE = ROOT / "vllm/third_party/tml_fa4/interface.py"


# --------------------------------------------------------------------------
# qkvr_prep.py
# --------------------------------------------------------------------------

QKVR_HEADER_OLD = """import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import aux_stream
"""

QKVR_HEADER_NEW = '''from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import aux_stream

# ShearingBias layout constants (tml-fa4 shearing_bias.py, interface.py:691/711).
# tl.constexpr, not a plain int: Triton refuses to read a plain Python
# global from inside a @triton.jit function.
SHEAR_PAD = tl.constexpr(256)  # rel_extent_padded = rel_extent + SHEAR_PAD
SHEAR_ROW_PAD = 128  # bias buffer carries tile_m extra rows, left unwritten


@dataclass(frozen=True)
class RelShearSpec:
    """Everything the fused writer needs to place a row in sheared columns.

    The fields mirror exactly what the attention call passes to FA4, so the
    fused writer and the kernel it feeds cannot disagree about geometry:
    ``cu_seqlens_q`` is FA4's ``cu_seqlens_q``, ``seq_lens`` is its
    ``seqused_k``.  ``seq_idx`` maps token -> request.  ``num_tokens`` is
    ``num_actual_tokens``; rows past it are not written, matching the rows
    ShearingBias leaves untouched today.
    """

    cu_seqlens_q: torch.Tensor  # (batch + 1,) int32
    seq_lens: torch.Tensor  # (batch,) int32, == seqused_k
    seq_idx: torch.Tensor  # (tokens,) int32
    num_tokens: int
    window_left: int | None = None
    window_right: int = 0

    @property
    def is_local(self) -> bool:
        return self.window_left is not None

    @property
    def left_pad(self) -> float:
        # shearing_bias.py:88-89
        return float("-inf") if self.window_left is not None else 0.0


def _validate_rel_shear(spec: RelShearSpec, tokens: int, rel_extent: int) -> None:
    assert rel_extent % 128 == 0, "shear contract requires rel_extent % 128 == 0"
    assert 0 <= spec.num_tokens <= tokens
    assert spec.cu_seqlens_q.dtype == torch.int32
    assert spec.seq_lens.dtype == torch.int32
    assert spec.seq_idx.dtype == torch.int32
    assert spec.cu_seqlens_q.is_contiguous()
    assert spec.seq_lens.is_contiguous()
    assert spec.seq_idx.is_contiguous()
    assert spec.cu_seqlens_q.shape[0] >= spec.seq_lens.shape[0] + 1
    assert spec.seq_idx.shape[0] >= spec.num_tokens
    if spec.window_left is not None:
        assert spec.window_right is not None, (
            "local rel-bias requires window_size_right (interface.py:695-699)"
        )
        assert spec.window_left + spec.window_right + 1 == rel_extent, (
            "local window length must equal rel_extent (interface.py:695-699)"
        )
    else:
        assert spec.window_right == 0


@triton.jit
def _shear_row_geometry(
    token,
    valid,
    seq_idx_ptr,
    cu_seqlens_q_ptr,
    seq_lens_ptr,
    REL_EXTENT: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
):
    """Return (base, num_bias_vals) for each token.

    ``base`` is the destination column of reversed index 0; the value whose
    relative distance is ``d`` lands at ``base + REL_EXTENT - 1 - d``.  See the
    module docstring of kernels/patches/u2_shear_fusion.py for the derivation
    from shearing_bias.py:357-476.
    """
    req = tl.load(seq_idx_ptr + token, mask=valid, other=0)
    q_start = tl.load(cu_seqlens_q_ptr + req, mask=valid, other=0)
    q_end = tl.load(cu_seqlens_q_ptr + req + 1, mask=valid, other=1)
    seqlen_k = tl.load(seq_lens_ptr + req, mask=valid, other=1)
    i_local = token - q_start
    ctx = seqlen_k - (q_end - q_start)  # seqlen_k - seqlen_q
    n_idx_right = i_local + 1 + ctx + WINDOW_RIGHT
    if IS_LOCAL:
        n_idx_left = tl.maximum(i_local + ctx - WINDOW_LEFT, 0)
    else:
        n_idx_left = i_local * 0
    m_block = i_local // 128
    n_block_max = tl.minimum(
        (seqlen_k + 127) // 128,
        ((m_block + 1) * 128 + ctx + WINDOW_RIGHT + 127) // 128,
    )
    base = n_idx_right + 256 - n_block_max * 128
    num_bias_vals = n_idx_right - tl.maximum(n_idx_left, n_idx_right - REL_EXTENT)
    return base, num_bias_vals


@triton.jit
def _shear_store_pads(
    rel_out_ptr,
    row,
    row_mask,
    base,
    pad_pid,
    REL_EXTENT: tl.constexpr,
    OUT_ROW_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    PAD_BLOCK: tl.constexpr,
    LEFT_PAD: tl.constexpr,
    RIGHT_PAD: tl.constexpr,
):
    """Write the two pad wings: [0, base) and [base + REL_EXTENT, padded).

    Both wings are at most 256 wide (base is in [1, 256]), so they always sit
    inside columns [0, 256) and [REL_EXTENT, REL_EXTENT + 256).  The masks are
    disjoint from each other and from the value window, so every column of the
    row is written exactly once across the launch.
    """
    p = pad_pid * PAD_BLOCK + tl.arange(0, PAD_BLOCK)
    is_left = p < SHEAR_PAD
    pad_col = tl.where(is_left, p, p - SHEAR_PAD + REL_EXTENT)
    pad_val = tl.where(is_left, LEFT_PAD, RIGHT_PAD)
    keep = tl.where(
        is_left[None, :],
        pad_col[None, :] < base[:, None],
        pad_col[None, :] >= base[:, None] + REL_EXTENT,
    )
    keep = keep & row_mask[:, None] & (p < 2 * SHEAR_PAD)[None, :]
    vals = tl.broadcast_to(pad_val[None, :], (BLOCK_M, PAD_BLOCK))
    tl.store(
        rel_out_ptr + row[:, None] * OUT_ROW_STRIDE + pad_col[None, :],
        vals.to(rel_out_ptr.dtype.element_ty),
        mask=keep,
    )
'''


QKVR_LOWLAT_OLD = '''@triton.jit(do_not_specialize=["rows"])
def _rel_proj_low_latency_kernel(
    qkvr_ptr,
    rel_proj_ptr,
    rel_out_ptr,
    log_scaling_ptr,
    rows,
    stride_x_t,
    R_OFFSET: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    APPLY_LOG_SCALING: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, 16)
    token = row // NUM_Q_HEADS
    head = row % NUM_Q_HEADS
    relative = tl.load(
        qkvr_ptr
        + token[:, None] * stride_x_t
        + R_OFFSET
        + head[:, None] * 16
        + inner[None, :],
        mask=row[:, None] < rows,
        other=0.0,
    )
    projection = tl.load(
        rel_proj_ptr + inner[:, None] * REL_EXTENT + col[None, :],
        mask=col[None, :] < REL_EXTENT,
        other=0.0,
    )
    values = tl.dot(relative, projection, out_dtype=tl.float32).to(
        rel_out_ptr.dtype.element_ty
    )
    values = values.to(tl.float32)
    if APPLY_LOG_SCALING:
        values *= tl.load(log_scaling_ptr + token, mask=row < rows, other=1.0)[:, None]
    tl.store(
        rel_out_ptr + row[:, None] * REL_EXTENT + col[None, :],
        values.to(rel_out_ptr.dtype.element_ty),
        mask=(row[:, None] < rows) & (col[None, :] < REL_EXTENT),
    )
'''

QKVR_LOWLAT_NEW = '''@triton.jit(do_not_specialize=["rows"])
def _rel_proj_low_latency_kernel(
    qkvr_ptr,
    rel_proj_ptr,
    rel_out_ptr,
    log_scaling_ptr,
    seq_idx_ptr,
    cu_seqlens_q_ptr,
    seq_lens_ptr,
    rows,
    stride_x_t,
    R_OFFSET: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    APPLY_LOG_SCALING: tl.constexpr,
    SHEAR: tl.constexpr,
    OUT_ROW_STRIDE: tl.constexpr,
    PAD_BLOCK: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    LEFT_PAD: tl.constexpr,
    RIGHT_PAD: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, 16)
    token = row // NUM_Q_HEADS
    head = row % NUM_Q_HEADS
    relative = tl.load(
        qkvr_ptr
        + token[:, None] * stride_x_t
        + R_OFFSET
        + head[:, None] * 16
        + inner[None, :],
        mask=row[:, None] < rows,
        other=0.0,
    )
    if SHEAR:
        # reversed column order, so the sheared destination is contiguous
        proj_col = tl.maximum(REL_EXTENT - 1 - col, 0)
    else:
        proj_col = col
    projection = tl.load(
        rel_proj_ptr + inner[:, None] * REL_EXTENT + proj_col[None, :],
        mask=col[None, :] < REL_EXTENT,
        other=0.0,
    )
    values = tl.dot(relative, projection, out_dtype=tl.float32).to(
        rel_out_ptr.dtype.element_ty
    )
    values = values.to(tl.float32)
    if APPLY_LOG_SCALING:
        values *= tl.load(log_scaling_ptr + token, mask=row < rows, other=1.0)[:, None]
    if SHEAR:
        base, num_bias_vals = _shear_row_geometry(
            token,
            row < rows,
            seq_idx_ptr,
            cu_seqlens_q_ptr,
            seq_lens_ptr,
            REL_EXTENT,
            IS_LOCAL,
            WINDOW_LEFT,
            WINDOW_RIGHT,
        )
        keep = col[None, :] >= (REL_EXTENT - num_bias_vals)[:, None]
        tl.store(
            rel_out_ptr
            + row[:, None] * OUT_ROW_STRIDE
            + base[:, None]
            + col[None, :],
            tl.where(keep, values, LEFT_PAD).to(rel_out_ptr.dtype.element_ty),
            mask=(row[:, None] < rows) & (col[None, :] < REL_EXTENT),
        )
        _shear_store_pads(
            rel_out_ptr,
            row,
            row < rows,
            base,
            tl.program_id(1),
            REL_EXTENT,
            OUT_ROW_STRIDE,
            BLOCK_M,
            PAD_BLOCK,
            LEFT_PAD,
            RIGHT_PAD,
        )
    else:
        tl.store(
            rel_out_ptr + row[:, None] * OUT_ROW_STRIDE + col[None, :],
            values.to(rel_out_ptr.dtype.element_ty),
            mask=(row[:, None] < rows) & (col[None, :] < REL_EXTENT),
        )
'''


QKVR_THRU_OLD = '''@triton.jit(do_not_specialize=["rows"])
def _rel_proj_throughput_kernel(
    qkvr_ptr,
    rel_proj_ptr,
    rel_out_ptr,
    log_scaling_ptr,
    rows,
    stride_x_t,
    R_OFFSET: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    APPLY_LOG_SCALING: tl.constexpr,
):
    row_group = tl.program_id(0)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, 16)
    projection = tl.load(
        rel_proj_ptr + inner[:, None] * REL_EXTENT + col[None, :],
        mask=col[None, :] < REL_EXTENT,
        other=0.0,
    )
    row_offsets = tl.arange(0, BLOCK_M)
    for group_offset in tl.static_range(GROUP_M):
        row = (row_group * GROUP_M + group_offset) * BLOCK_M + row_offsets
        token = row // NUM_Q_HEADS
        head = row % NUM_Q_HEADS
        relative = tl.load(
            qkvr_ptr
            + token[:, None] * stride_x_t
            + R_OFFSET
            + head[:, None] * 16
            + inner[None, :],
            mask=row[:, None] < rows,
            other=0.0,
        )
        values = tl.dot(relative, projection, out_dtype=tl.float32).to(
            rel_out_ptr.dtype.element_ty
        )
        values = values.to(tl.float32)
        if APPLY_LOG_SCALING:
            values *= tl.load(log_scaling_ptr + token, mask=row < rows, other=1.0)[
                :, None
            ]
        tl.store(
            rel_out_ptr + row[:, None] * REL_EXTENT + col[None, :],
            values.to(rel_out_ptr.dtype.element_ty),
            mask=(row[:, None] < rows) & (col[None, :] < REL_EXTENT),
        )
'''

QKVR_THRU_NEW = '''@triton.jit(do_not_specialize=["rows"])
def _rel_proj_throughput_kernel(
    qkvr_ptr,
    rel_proj_ptr,
    rel_out_ptr,
    log_scaling_ptr,
    seq_idx_ptr,
    cu_seqlens_q_ptr,
    seq_lens_ptr,
    rows,
    stride_x_t,
    R_OFFSET: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    APPLY_LOG_SCALING: tl.constexpr,
    SHEAR: tl.constexpr,
    OUT_ROW_STRIDE: tl.constexpr,
    PAD_BLOCK: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    LEFT_PAD: tl.constexpr,
    RIGHT_PAD: tl.constexpr,
):
    row_group = tl.program_id(0)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, 16)
    if SHEAR:
        proj_col = tl.maximum(REL_EXTENT - 1 - col, 0)
    else:
        proj_col = col
    projection = tl.load(
        rel_proj_ptr + inner[:, None] * REL_EXTENT + proj_col[None, :],
        mask=col[None, :] < REL_EXTENT,
        other=0.0,
    )
    row_offsets = tl.arange(0, BLOCK_M)
    for group_offset in tl.static_range(GROUP_M):
        row = (row_group * GROUP_M + group_offset) * BLOCK_M + row_offsets
        token = row // NUM_Q_HEADS
        head = row % NUM_Q_HEADS
        relative = tl.load(
            qkvr_ptr
            + token[:, None] * stride_x_t
            + R_OFFSET
            + head[:, None] * 16
            + inner[None, :],
            mask=row[:, None] < rows,
            other=0.0,
        )
        values = tl.dot(relative, projection, out_dtype=tl.float32).to(
            rel_out_ptr.dtype.element_ty
        )
        values = values.to(tl.float32)
        if APPLY_LOG_SCALING:
            values *= tl.load(log_scaling_ptr + token, mask=row < rows, other=1.0)[
                :, None
            ]
        if SHEAR:
            base, num_bias_vals = _shear_row_geometry(
                token,
                row < rows,
                seq_idx_ptr,
                cu_seqlens_q_ptr,
                seq_lens_ptr,
                REL_EXTENT,
                IS_LOCAL,
                WINDOW_LEFT,
                WINDOW_RIGHT,
            )
            keep = col[None, :] >= (REL_EXTENT - num_bias_vals)[:, None]
            tl.store(
                rel_out_ptr
                + row[:, None] * OUT_ROW_STRIDE
                + base[:, None]
                + col[None, :],
                tl.where(keep, values, LEFT_PAD).to(rel_out_ptr.dtype.element_ty),
                mask=(row[:, None] < rows) & (col[None, :] < REL_EXTENT),
            )
            _shear_store_pads(
                rel_out_ptr,
                row,
                row < rows,
                base,
                tl.program_id(1),
                REL_EXTENT,
                OUT_ROW_STRIDE,
                BLOCK_M,
                PAD_BLOCK,
                LEFT_PAD,
                RIGHT_PAD,
            )
        else:
            tl.store(
                rel_out_ptr + row[:, None] * OUT_ROW_STRIDE + col[None, :],
                values.to(rel_out_ptr.dtype.element_ty),
                mask=(row[:, None] < rows) & (col[None, :] < REL_EXTENT),
            )
'''


QKVR_DRIVER_OLD = '''def qkvr_rel_proj(
    qkvr: torch.Tensor,
    rel_proj: torch.Tensor,
    rel_out: torch.Tensor,
    log_scaling: torch.Tensor | None,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    d_rel: int,
) -> None:
    rows = qkvr.shape[0] * num_q_heads
    rel_extent = rel_proj.shape[1]
    assert d_rel == 16 and rel_proj.shape[0] == 16
    r_offset = num_q_heads * head_dim + 2 * num_kv_heads * head_dim
    log_scaling_ptr = log_scaling if log_scaling is not None else qkvr
    common = dict(
        R_OFFSET=r_offset,
        NUM_Q_HEADS=num_q_heads,
        REL_EXTENT=rel_extent,
        APPLY_LOG_SCALING=log_scaling is not None,
    )

    if use_rel_proj_throughput(rows, rel_extent):
        grid = (
            triton.cdiv(rows, THROUGHPUT_BLOCK_M * THROUGHPUT_GROUP_M),
            triton.cdiv(rel_extent, THROUGHPUT_BLOCK_N),
        )
        _rel_proj_throughput_kernel[grid](
            qkvr,
            rel_proj,
            rel_out,
            log_scaling_ptr,
            rows,
            qkvr.stride(0),
            BLOCK_M=THROUGHPUT_BLOCK_M,
            BLOCK_N=THROUGHPUT_BLOCK_N,
            GROUP_M=THROUGHPUT_GROUP_M,
            num_warps=THROUGHPUT_NUM_WARPS,
            **common,
        )
        return

    grid = (
        triton.cdiv(rows, LOW_BLOCK_M),
        triton.cdiv(rel_extent, LOW_BLOCK_N),
    )
    _rel_proj_low_latency_kernel[grid](
        qkvr,
        rel_proj,
        rel_out,
        log_scaling_ptr,
        rows,
        qkvr.stride(0),
        BLOCK_M=LOW_BLOCK_M,
        BLOCK_N=LOW_BLOCK_N,
        num_warps=LOW_NUM_WARPS,
        **common,
    )
'''

QKVR_DRIVER_NEW = '''def _shear_kernel_args(
    shear: RelShearSpec | None,
    rel_extent: int,
    block_n: int,
) -> dict:
    """Constexpr block shared by both rel-projection kernels.

    Each column program also owns one slice of the 2*SHEAR_PAD pad columns, so
    the pad write is spread evenly instead of landing on program 0, and the
    launch grid is unchanged.
    """
    if shear is None:
        return dict(
            SHEAR=False,
            OUT_ROW_STRIDE=rel_extent,
            PAD_BLOCK=1,
            IS_LOCAL=False,
            WINDOW_LEFT=0,
            WINDOW_RIGHT=0,
            LEFT_PAD=0.0,
            RIGHT_PAD=0.0,
        )
    col_progs = triton.cdiv(rel_extent, block_n)
    return dict(
        SHEAR=True,
        OUT_ROW_STRIDE=rel_extent + SHEAR_PAD,
        PAD_BLOCK=triton.next_power_of_2(triton.cdiv(2 * SHEAR_PAD, col_progs)),
        IS_LOCAL=shear.is_local,
        WINDOW_LEFT=shear.window_left if shear.window_left is not None else 0,
        WINDOW_RIGHT=shear.window_right,
        LEFT_PAD=shear.left_pad,
        RIGHT_PAD=float("-inf"),
    )


def qkvr_rel_proj(
    qkvr: torch.Tensor,
    rel_proj: torch.Tensor,
    rel_out: torch.Tensor,
    log_scaling: torch.Tensor | None,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    d_rel: int,
    shear: RelShearSpec | None = None,
) -> None:
    rel_extent = rel_proj.shape[1]
    if shear is None:
        rows = qkvr.shape[0] * num_q_heads
        seq_idx = qkvr
        cu_seqlens_q = qkvr
        seq_lens = qkvr
    else:
        _validate_rel_shear(shear, qkvr.shape[0], rel_extent)
        rows = shear.num_tokens * num_q_heads
        seq_idx = shear.seq_idx
        cu_seqlens_q = shear.cu_seqlens_q
        seq_lens = shear.seq_lens
    assert d_rel == 16 and rel_proj.shape[0] == 16
    r_offset = num_q_heads * head_dim + 2 * num_kv_heads * head_dim
    log_scaling_ptr = log_scaling if log_scaling is not None else qkvr
    common = dict(
        R_OFFSET=r_offset,
        NUM_Q_HEADS=num_q_heads,
        REL_EXTENT=rel_extent,
        APPLY_LOG_SCALING=log_scaling is not None,
    )
    if rows == 0:
        return

    if use_rel_proj_throughput(rows, rel_extent):
        grid = (
            triton.cdiv(rows, THROUGHPUT_BLOCK_M * THROUGHPUT_GROUP_M),
            triton.cdiv(rel_extent, THROUGHPUT_BLOCK_N),
        )
        _rel_proj_throughput_kernel[grid](
            qkvr,
            rel_proj,
            rel_out,
            log_scaling_ptr,
            seq_idx,
            cu_seqlens_q,
            seq_lens,
            rows,
            qkvr.stride(0),
            BLOCK_M=THROUGHPUT_BLOCK_M,
            BLOCK_N=THROUGHPUT_BLOCK_N,
            GROUP_M=THROUGHPUT_GROUP_M,
            num_warps=THROUGHPUT_NUM_WARPS,
            **common,
            **_shear_kernel_args(shear, rel_extent, THROUGHPUT_BLOCK_N),
        )
        return

    grid = (
        triton.cdiv(rows, LOW_BLOCK_M),
        triton.cdiv(rel_extent, LOW_BLOCK_N),
    )
    _rel_proj_low_latency_kernel[grid](
        qkvr,
        rel_proj,
        rel_out,
        log_scaling_ptr,
        seq_idx,
        cu_seqlens_q,
        seq_lens,
        rows,
        qkvr.stride(0),
        BLOCK_M=LOW_BLOCK_M,
        BLOCK_N=LOW_BLOCK_N,
        num_warps=LOW_NUM_WARPS,
        **common,
        **_shear_kernel_args(shear, rel_extent, LOW_BLOCK_N),
    )
'''


QKVR_SMALL_SIG_OLD = '''@triton.jit(do_not_specialize=["tokens", "stride_block_table_req", "max_blocks"])
def _qkvr_qkv_kernel(
    qkvr_ptr,
    q_norm_weight_ptr,
    q_out_ptr,
    rel_proj_ptr,
    rel_out_ptr,
    k_weight_ptr,
'''

QKVR_SMALL_SIG_NEW = '''@triton.jit(
    do_not_specialize=["tokens", "rel_rows", "stride_block_table_req", "max_blocks"]
)
def _qkvr_qkv_kernel(
    qkvr_ptr,
    q_norm_weight_ptr,
    q_out_ptr,
    rel_proj_ptr,
    rel_out_ptr,
    cu_seqlens_q_ptr,
    seq_lens_ptr,
    k_weight_ptr,
'''

QKVR_SMALL_ARGS_OLD = '''    log_scaling_ptr,
    tokens,
    eps,
'''

QKVR_SMALL_ARGS_NEW = '''    log_scaling_ptr,
    tokens,
    rel_rows,
    eps,
'''

QKVR_SMALL_CONSTEXPR_OLD = '''    D_REL: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    REL_EXTENT_PADDED: tl.constexpr,
):
'''

QKVR_SMALL_CONSTEXPR_NEW = '''    D_REL: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    REL_EXTENT_PADDED: tl.constexpr,
    SHEAR: tl.constexpr,
    OUT_ROW_STRIDE: tl.constexpr,
    IS_LOCAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    LEFT_PAD: tl.constexpr,
    RIGHT_PAD: tl.constexpr,
):
'''

QKVR_SMALL_BODY_OLD = '''        rel_cols = tl.arange(0, REL_EXTENT_PADDED)
        rel_mask = rel_cols < REL_EXTENT
        projected = tl.zeros([REL_EXTENT_PADDED], dtype=tl.float32)
        rel_offset = Q_WIDTH + 2 * KV_WIDTH + head * D_REL
        for rel_dim in tl.static_range(D_REL):
            rel_value = tl.load(
                qkvr_ptr + token * stride_x_t + rel_offset + rel_dim
            ).to(tl.float32)
            proj = tl.load(
                rel_proj_ptr + rel_dim * REL_EXTENT + rel_cols,
                mask=rel_mask,
                other=0.0,
            ).to(tl.float32)
            projected += rel_value * proj
        projected = projected.to(rel_out_ptr.dtype.element_ty).to(tl.float32)
        if APPLY_LOG_SCALING:
            projected *= tl.load(log_scaling_ptr + token)
        tl.store(
            rel_out_ptr + row * REL_EXTENT + rel_cols,
            projected.to(rel_out_ptr.dtype.element_ty),
            mask=rel_mask,
        )
'''

QKVR_SMALL_BODY_NEW = '''        rel_cols = tl.arange(0, REL_EXTENT_PADDED)
        rel_mask = rel_cols < REL_EXTENT
        if SHEAR:
            proj_cols = tl.maximum(REL_EXTENT - 1 - rel_cols, 0)
        else:
            proj_cols = rel_cols
        projected = tl.zeros([REL_EXTENT_PADDED], dtype=tl.float32)
        rel_offset = Q_WIDTH + 2 * KV_WIDTH + head * D_REL
        for rel_dim in tl.static_range(D_REL):
            rel_value = tl.load(
                qkvr_ptr + token * stride_x_t + rel_offset + rel_dim
            ).to(tl.float32)
            proj = tl.load(
                rel_proj_ptr + rel_dim * REL_EXTENT + proj_cols,
                mask=rel_mask,
                other=0.0,
            ).to(tl.float32)
            projected += rel_value * proj
        projected = projected.to(rel_out_ptr.dtype.element_ty).to(tl.float32)
        if APPLY_LOG_SCALING:
            projected *= tl.load(log_scaling_ptr + token)
        if SHEAR:
            if row < rel_rows:
                base, num_bias_vals = _shear_row_geometry(
                    token,
                    token >= 0,
                    seq_idx_ptr,
                    cu_seqlens_q_ptr,
                    seq_lens_ptr,
                    REL_EXTENT,
                    IS_LOCAL,
                    WINDOW_LEFT,
                    WINDOW_RIGHT,
                )
                keep = rel_cols >= REL_EXTENT - num_bias_vals
                tl.store(
                    rel_out_ptr + row * OUT_ROW_STRIDE + base + rel_cols,
                    tl.where(keep, projected, LEFT_PAD).to(
                        rel_out_ptr.dtype.element_ty
                    ),
                    mask=rel_mask,
                )
                pad = tl.arange(0, 2 * SHEAR_PAD)
                is_left = pad < SHEAR_PAD
                pad_col = tl.where(is_left, pad, pad - SHEAR_PAD + REL_EXTENT)
                pad_val = tl.where(is_left, LEFT_PAD, RIGHT_PAD)
                tl.store(
                    rel_out_ptr + row * OUT_ROW_STRIDE + pad_col,
                    pad_val.to(rel_out_ptr.dtype.element_ty),
                    mask=tl.where(
                        is_left, pad_col < base, pad_col >= base + REL_EXTENT
                    ),
                )
        else:
            tl.store(
                rel_out_ptr + row * OUT_ROW_STRIDE + rel_cols,
                projected.to(rel_out_ptr.dtype.element_ty),
                mask=rel_mask,
            )
'''


QKVR_RUNSMALL_SIG_OLD = '''    conv_block_size: int,
    log_scaling: torch.Tensor | None,
) -> None:
    tokens = qkvr.shape[0]

    num_q_rows = tokens * num_q_heads
    grid = (num_q_rows + tokens * num_kv_heads,)
    _qkvr_qkv_kernel[grid](
        qkvr,
        q_norm_weight,
        q_out,
        rel_proj,
        rel_out,
        k_weight,
'''

QKVR_RUNSMALL_SIG_NEW = '''    conv_block_size: int,
    log_scaling: torch.Tensor | None,
    shear: RelShearSpec | None = None,
) -> None:
    tokens = qkvr.shape[0]
    rel_extent = rel_proj.shape[1]
    if shear is None:
        rel_rows = tokens * num_q_heads
        cu_seqlens_q = qkvr
        seq_lens_shear = qkvr
    else:
        _validate_rel_shear(shear, tokens, rel_extent)
        rel_rows = shear.num_tokens * num_q_heads
        cu_seqlens_q = shear.cu_seqlens_q
        seq_lens_shear = shear.seq_lens

    num_q_rows = tokens * num_q_heads
    grid = (num_q_rows + tokens * num_kv_heads,)
    _qkvr_qkv_kernel[grid](
        qkvr,
        q_norm_weight,
        q_out,
        rel_proj,
        rel_out,
        cu_seqlens_q,
        seq_lens_shear,
        k_weight,
'''

QKVR_RUNSMALL_CALL_OLD = '''        log_scaling if log_scaling is not None else positions,
        tokens,
        eps,
'''

QKVR_RUNSMALL_CALL_NEW = '''        log_scaling if log_scaling is not None else positions,
        tokens,
        rel_rows,
        eps,
'''

QKVR_RUNSMALL_CONSTEXPR_OLD = '''        D_REL=16,
        REL_EXTENT=rel_proj.shape[1],
        REL_EXTENT_PADDED=triton.next_power_of_2(rel_proj.shape[1]),
        num_warps=SMALL_NUM_WARPS,
    )
'''

QKVR_RUNSMALL_CONSTEXPR_NEW = '''        D_REL=16,
        REL_EXTENT=rel_proj.shape[1],
        REL_EXTENT_PADDED=triton.next_power_of_2(rel_proj.shape[1]),
        SHEAR=shear is not None,
        OUT_ROW_STRIDE=(
            rel_extent if shear is None else rel_extent + SHEAR_PAD
        ),
        IS_LOCAL=False if shear is None else shear.is_local,
        WINDOW_LEFT=(
            0 if shear is None or shear.window_left is None else shear.window_left
        ),
        WINDOW_RIGHT=0 if shear is None else shear.window_right,
        LEFT_PAD=0.0 if shear is None else shear.left_pad,
        RIGHT_PAD=float("-inf"),
        num_warps=SMALL_NUM_WARPS,
    )
'''


QKVR_FUSED_SIG_OLD = '''    log_scaling: torch.Tensor | None = None,
'''

QKVR_FUSED_SIG_NEW = '''    log_scaling: torch.Tensor | None = None,
    rel_shear: RelShearSpec | None = None,
'''

QKVR_FUSED_ALLOC_OLD = '''    rel_out = torch.empty(
        (tokens, num_q_heads, rel_proj.shape[1]),
        dtype=qkvr.dtype,
        device=qkvr.device,
    )
    if tokens == 0:
        return q_out, rel_out
'''

QKVR_FUSED_ALLOC_NEW = '''    if rel_shear is None:
        rel_out = torch.empty(
            (tokens, num_q_heads, rel_proj.shape[1]),
            dtype=qkvr.dtype,
            device=qkvr.device,
        )
    else:
        # Same shape the FA4 interface allocates for the sheared bias
        # (interface.py:711-718): total_q + tile_m rows, rel_extent + 256 cols.
        # The trailing SHEAR_ROW_PAD rows stay uninitialised, exactly as
        # ShearingBias leaves them today.
        _validate_rel_shear(rel_shear, tokens, rel_proj.shape[1])
        rel_out = torch.empty(
            (
                rel_shear.num_tokens + SHEAR_ROW_PAD,
                num_q_heads,
                rel_proj.shape[1] + SHEAR_PAD,
            ),
            dtype=qkvr.dtype,
            device=qkvr.device,
        )
    if tokens == 0:
        return q_out, rel_out
'''

QKVR_FUSED_SMALLCALL_OLD = '''            conv_block_size=conv_block_size,
            log_scaling=log_scaling,
'''

QKVR_FUSED_SMALLCALL_NEW = '''            conv_block_size=conv_block_size,
            log_scaling=log_scaling,
            shear=rel_shear,
'''

QKVR_FUSED_PROJCALL_OLD = '''        d_rel=d_rel,
    )
'''

QKVR_FUSED_PROJCALL_NEW = '''        d_rel=d_rel,
        shear=rel_shear,
    )
'''


# --------------------------------------------------------------------------
# fa4_rel_attention.py
# --------------------------------------------------------------------------

FA4_HELPER_OLD = '''@cache
def _get_score_mod(rel_extent: int) -> Callable:
'''

FA4_HELPER_NEW = '''@cache
def use_fused_shear() -> bool:
    """Inkling-turbo U2: qkvr_prep writes the ShearingBias layout directly.

    Opt-in until it has run on a GPU.  Requires the sheared attention path
    (Blackwell natively, Hopper/sm_120 with kernels/tml_fa4_modified deployed).
    """
    import os

    if not _use_sheared_bias():
        return False
    return os.environ.get("INKLING_TURBO_FUSED_SHEAR", "0") == "1"


@cache
def _get_score_mod(rel_extent: int) -> Callable:
'''

FA4_SIG_OLD = '''    num_splits: int = 32,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
'''

FA4_SIG_NEW = '''    num_splits: int = 32,
    sheared: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
'''

FA4_DISPATCH_OLD = '''    rel_logits = rel_logits.contiguous()
    if _use_sheared_bias():
        from vllm.third_party.tml_fa4 import flash_attn_varlen_func

        bias_kwargs: dict[str, Any] = {"rel_bias": rel_logits}
    else:
'''

FA4_DISPATCH_NEW = '''    rel_logits = rel_logits.contiguous()
    if sheared:
        # Already in ShearingBias layout: hand it straight to the kernel as
        # `bias` so the interface skips the pre-kernel (u2_shear_fusion).
        from vllm.third_party.tml_fa4 import flash_attn_varlen_func

        bias_kwargs: dict[str, Any] = {"bias": rel_logits}
    elif _use_sheared_bias():
        from vllm.third_party.tml_fa4 import flash_attn_varlen_func

        bias_kwargs = {"rel_bias": rel_logits}
    else:
'''


# --------------------------------------------------------------------------
# attention.py
# --------------------------------------------------------------------------

ATTN_IMPORT_OLD = "from .ops.qkvr_prep import fused_qkvr_prep\n"
ATTN_IMPORT_NEW = "from .ops.qkvr_prep import RelShearSpec, fused_qkvr_prep\n"

ATTN_HELPER_OLD = '''    def forward(
        self,
        positions: torch.Tensor,
'''

ATTN_HELPER_NEW = '''    def _rel_shear_spec(self, fa_md, conv_meta) -> "RelShearSpec | None":
        """Geometry for the fused sheared rel writer, or None to keep the
        ShearingBias pre-kernel.

        The tensors are the same ones handed to FA4 below (query_start_loc ->
        cu_seqlens_q, seq_lens -> seqused_k), so the writer and the reader
        cannot disagree about seqlen_q / seqlen_k.
        """
        from .ops.fa4_rel_attention import use_fused_shear

        if not use_fused_shear():
            return None
        window_left, window_right = self.window_size
        local = window_left >= 0
        return RelShearSpec(
            cu_seqlens_q=fa_md.query_start_loc,
            seq_lens=fa_md.seq_lens,
            seq_idx=conv_meta.seq_idx,
            num_tokens=int(fa_md.num_actual_tokens),
            window_left=window_left if local else None,
            window_right=window_right if local else 0,
        )

    def forward(
        self,
        positions: torch.Tensor,
'''

ATTN_CALL_OLD = '''            q, rel_logits = fused_qkvr_prep(
'''

ATTN_CALL_NEW = '''            rel_shear = self._rel_shear_spec(fa_md, conv_meta)
            q, rel_logits = fused_qkvr_prep(
'''

ATTN_CALL_TAIL_OLD = '''                log_scaling if not self.is_local else None,
            )
            q = q.view(num_tokens, self.num_heads, self.head_dim)
            self._attention(q, rel_logits, attn_output)
'''

ATTN_CALL_TAIL_NEW = '''                log_scaling if not self.is_local else None,
                rel_shear=rel_shear,
            )
            q = q.view(num_tokens, self.num_heads, self.head_dim)
            self._attention(
                q, rel_logits, attn_output, sheared=rel_shear is not None
            )
'''

ATTN_ATTN_SIG_OLD = '''    def _attention(
        self,
        q: torch.Tensor,
        rel_logits: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
'''

ATTN_ATTN_SIG_NEW = '''    def _attention(
        self,
        q: torch.Tensor,
        rel_logits: torch.Tensor,
        output: torch.Tensor,
        sheared: bool = False,
    ) -> None:
'''

ATTN_ATTN_CALL_OLD = '''            rel_logits=rel_logits[:nt],
'''

ATTN_ATTN_CALL_NEW = '''            rel_logits=rel_logits if sheared else rel_logits[:nt],
            sheared=sheared,
'''


# --------------------------------------------------------------------------
# third_party/tml_fa4/interface.py
# --------------------------------------------------------------------------

IFACE_NORMALIZE_OLD = '''    dtype = torch2cute_dtype_map[q.dtype]
    use_block_sparsity = block_sparse_tensors is not None
'''

IFACE_NORMALIZE_NEW = '''    # Inkling-turbo U2 shear fusion: a caller that already produced the sheared
    # (..., rel_extent + 256) buffer passes it as `bias` with rel_bias=None.
    # Alias it onto rel_bias so every downstream has_bias gate stays true, and
    # remember to skip the ShearingBias pre-kernel below.
    rel_bias_presheared = rel_bias is None and bias is not None
    if rel_bias_presheared:
        rel_bias = bias

    dtype = torch2cute_dtype_map[q.dtype]
    use_block_sparsity = block_sparse_tensors is not None
'''

IFACE_BLOCK_OLD = '''    if rel_bias is not None:
        rel_extent = rel_bias.shape[-1]
        rel_extent_padded = rel_extent + 256
'''

IFACE_BLOCK_NEW = '''    if rel_bias_presheared:
        # qkvr_prep already emitted the ShearingBias layout; the pre-kernel,
        # its compile cache and its scheduler tensors are all skipped.
        bias = rel_bias
        rel_extent_padded = bias.shape[-1]
        rel_extent = rel_extent_padded - 256
        assert rel_extent > 0 and rel_extent % 128 == 0
        assert tile_m == 128
        assert tile_n == 128 or arch // 10 in [8, 9, 12]
        assert not pack_gqa, (
            "pre-sheared bias assumes unpacked 128-row blocks; pack_gqa "
            "redefines what a tile row is (journal/upstream/04)"
        )
        assert (
            causal
            or window_size_left is None
            or (window_size_right is not None and window_size_left + window_size_right + 1 == rel_extent)
        ), "for relative bias, require causal (with possibly shifted diagonal) or window length == rel_extent"
        if cu_seqlens_q is None:
            assert bias.shape[0] == batch_size
            assert bias.shape[2] == num_head and bias.shape[3] == rel_extent_padded
            assert bias.shape[1] >= (seqlen_q + tile_m - 1) // tile_m * tile_m
        else:
            assert bias.shape == (total_q + tile_m, num_head, rel_extent_padded)
    elif rel_bias is not None:
        rel_extent = rel_bias.shape[-1]
        rel_extent_padded = rel_extent + 256
'''

IFACE_VARLEN_SIG_OLD = '''def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_bias: Optional[torch.Tensor] = None,
'''

IFACE_VARLEN_SIG_NEW = '''def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel_bias: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
'''

IFACE_VARLEN_CALL_OLD = '''        rel_bias=rel_bias,
        scheduler_metadata=scheduler_metadata,
'''

IFACE_VARLEN_CALL_NEW = '''        rel_bias=rel_bias,
        bias=bias,
        scheduler_metadata=scheduler_metadata,
'''


EDITS = {
    QKVR: [
        (QKVR_HEADER_OLD, QKVR_HEADER_NEW),
        (QKVR_LOWLAT_OLD, QKVR_LOWLAT_NEW),
        (QKVR_THRU_OLD, QKVR_THRU_NEW),
        (QKVR_DRIVER_OLD, QKVR_DRIVER_NEW),
        (QKVR_SMALL_SIG_OLD, QKVR_SMALL_SIG_NEW),
        (QKVR_SMALL_ARGS_OLD, QKVR_SMALL_ARGS_NEW),
        (QKVR_SMALL_CONSTEXPR_OLD, QKVR_SMALL_CONSTEXPR_NEW),
        (QKVR_SMALL_BODY_OLD, QKVR_SMALL_BODY_NEW),
        (QKVR_RUNSMALL_SIG_OLD, QKVR_RUNSMALL_SIG_NEW),
        (QKVR_RUNSMALL_CALL_OLD, QKVR_RUNSMALL_CALL_NEW),
        (QKVR_RUNSMALL_CONSTEXPR_OLD, QKVR_RUNSMALL_CONSTEXPR_NEW),
        (QKVR_FUSED_SIG_OLD, QKVR_FUSED_SIG_NEW),
        (QKVR_FUSED_ALLOC_OLD, QKVR_FUSED_ALLOC_NEW),
        (QKVR_FUSED_SMALLCALL_OLD, QKVR_FUSED_SMALLCALL_NEW),
        (QKVR_FUSED_PROJCALL_OLD, QKVR_FUSED_PROJCALL_NEW),
    ],
    FA4: [
        (FA4_HELPER_OLD, FA4_HELPER_NEW),
        (FA4_SIG_OLD, FA4_SIG_NEW),
        (FA4_DISPATCH_OLD, FA4_DISPATCH_NEW),
    ],
    ATTN: [
        (ATTN_IMPORT_OLD, ATTN_IMPORT_NEW),
        (ATTN_HELPER_OLD, ATTN_HELPER_NEW),
        (ATTN_CALL_OLD, ATTN_CALL_NEW),
        (ATTN_CALL_TAIL_OLD, ATTN_CALL_TAIL_NEW),
        (ATTN_ATTN_SIG_OLD, ATTN_ATTN_SIG_NEW),
        (ATTN_ATTN_CALL_OLD, ATTN_ATTN_CALL_NEW),
    ],
    IFACE: [
        (IFACE_NORMALIZE_OLD, IFACE_NORMALIZE_NEW),
        (IFACE_BLOCK_OLD, IFACE_BLOCK_NEW),
        (IFACE_VARLEN_SIG_OLD, IFACE_VARLEN_SIG_NEW),
        (IFACE_VARLEN_CALL_OLD, IFACE_VARLEN_CALL_NEW),
    ],
}


def apply(path: Path, edits: list[tuple[str, str]]) -> tuple[int, int]:
    """Anchor-replace, idempotent.  Every anchor must be found exactly once."""
    assert path.exists(), f"missing target: {path}"
    source = path.read_text()
    applied = skipped = 0
    for old, new in edits:
        if new in source:
            skipped += 1
            continue
        count = source.count(old)
        assert count == 1, (
            f"{path.name}: anchor found {count} times (want 1): {old[:70]!r}"
        )
        source = source.replace(old, new, 1)
        applied += 1
    path.write_text(source)
    return applied, skipped


def main() -> None:
    total = 0
    for path, edits in EDITS.items():
        applied, skipped = apply(path, edits)
        total += applied
        print(f"{path.name}: {applied} applied, {skipped} already present")
    if total == 0:
        print("u2 shear fusion: already applied, nothing to do")
    else:
        print(
            "u2 shear fusion applied. Enable per run with "
            "INKLING_TURBO_FUSED_SHEAR=1 (default off: UNVALIDATED on GPU)."
        )


if __name__ == "__main__":
    main()
