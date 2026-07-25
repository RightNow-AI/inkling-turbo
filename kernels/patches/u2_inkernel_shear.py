#!/usr/bin/env python3
"""U2 in-kernel shear: make the sm_90 attention kernel read the NATURAL
relative-bias buffer, so the ShearingBias pass stops existing.

Today the sm_90 relative-bias path is three kernels:

    qkvr_prep      -> rel_logits (T, H, ext), natural layout
    ShearingBias   -> reads it, writes a sheared copy (T + 128, H, ext + 256)
                      plus CuSeqlensToBlocksKernel + CuBlocksToBatchKernel,
                      which exist only to schedule it
    attention      -> reads 128x128 tiles of the sheared copy

ShearingBias is pure data movement running at 88 percent of HBM roofline
(2.94 / 2.95 TB/s measured on both prefill shapes, journal/remote/
microbench_attn_day0_session25_h100.json plus the arithmetic in
kernels/patches/u2_inkernel_shear_notes.md). It cannot be tuned; it has to
stop running. This patch makes the attention kernel do the shear itself, in
its own bias address math, reading the natural buffer.

Gated behind INKLING_TURBO_INKERNEL_SHEAR, DEFAULT OFF. Unset, every byte of
behaviour is what it is today.

    INKLING_TURBO_INKERNEL_SHEAR=1        band-view addressing (the real path)
    INKLING_TURBO_INKERNEL_SHEAR=coord    identical math, per-element
                                          coordinate addressing. Slower by
                                          construction; it exists so that a
                                          CuTe DSL failure in the band layout
                                          can be isolated without a second
                                          GPU session. See the .md.


WHAT CHANGES, AND WHAT DELIBERATELY DOES NOT
--------------------------------------------

The design note asks for the shear to happen "while staging into shared
memory". There is no shared-memory staging of bias in this kernel to change:
flash_fwd_sm90.py reads the bias tile gmem->rmem through
`thr_mma_qk.partition_C(gBias_tile)` inside `apply_rel_bias_sm90`, and
`_get_shared_storage_cls` allocates sBias with size 0 ("bias is read
gmem->rmem through a tiled_copy_C partition; no smem stage, so no
shared-memory cost"). So the shear is applied where the address is formed,
which is strictly less machinery than the note assumed. Consequences:

  * the note's cost 1 ("TMA is lost for the bias operand") does not apply.
    Bias never used TMA on this kernel.
  * the note's cost 3 ("bank conflicts on the reversed smem write") does not
    apply. There is no smem write.
  * the note's cost 2 (redundant reads across n_blocks) applies unchanged and
    is still the main risk to the estimate.
  * no extra shared memory is consumed, so num_stages does not have to move.

The apply side keeps its hard-won property verbatim: the bias tile is
partitioned with `thr_mma_qk.partition_C`, the SAME partitioner that produced
acc_S, and paired with acc_S by flat index. What changes is only WHICH tensor
is handed to partition_C.


THE SHEAR AS A LAYOUT
---------------------

ShearingBias places natural column d of row `row` at sheared column

    col(row, d) = bias_idx_right(row) - 1 - d
    bias_idx_right(row) = n_idx_right(row) + rel_extent_padded
                          - 128 * n_block_max(m_block(row))

(shearing_bias.py:357-476, re-derived in
kernels/patches/u2_shear_fusion_notes.md and machine-verified there). So a
reader that wants kv index `kv` reads sheared column
`kv + padded - 128 * n_block_max`, and the natural column it actually lands on
is

    d = n_idx_right(row) - 1 - kv

with, from BlockInfo.get_n_idx_left_right for the causal / local-with-right
form that ShearingBias requires,

    n_idx_right(row) = row + 1 + (seqlen_k - seqlen_q) + window_right

hence, for score-tile element (i, j) of tile (m_block, n_block),

    d(i, j) = i - j + shear_k
    shear_k = m_block * tile_m - n_block * tile_n
              + (seqlen_k - seqlen_q) + window_right

`d` is the plain relative distance the model means, which is the same thing
harness/parity_fa4_rel.py's reference calls `dist = i - j`.

The natural address of element (i, j) is therefore

    (row0 + i) * s_r + d(i, j) * s_c
      = [row0 * s_r + shear_k * s_c] + i * (s_r + s_c) - j * s_c

so a cute layout of shape (tile_m, tile_n) and stride (s_r + s_c, -s_c) over
the natural buffer IS the shear, and the bracket is one domain_offset. That is
the whole mechanism: a diagonal band instead of a box. No coordinate
arithmetic enters the address.


VALIDITY, REDUCED TO A CONSTANT BOUND
-------------------------------------

ShearingBias writes a real value only where `d < num_bias_vals`, and pads
elsewhere: right_pad = -inf for d < 0, left_pad for d >= num_bias_vals, with
left_pad = -inf when window_size_left is set and 0.0 otherwise
(shearing_bias.py:88-89, plus the two edge fixups at 463-466 which pin the
per-element boundary to exactly `0 <= d < num_bias_vals`).

    num_bias_vals(row) = n_idx_right - max(n_idx_left, n_idx_right - rel_extent)

`d < num_bias_vals` is equivalent to `d < rel_extent` AND `kv >= n_idx_left`,
and the second reduces to a bound on d alone:

  * n_idx_left == 0 (not local, or no window_size_left): vacuous.
  * n_idx_left == max(row + ctx - wl, 0): when the max does not clamp,
    n_idx_right - n_idx_left == wl + wr + 1 exactly, so the condition is
    d < wl + wr + 1. When it does clamp (row + ctx < wl), the condition is
    d < n_idx_right, which is vacuous, and d <= row + ctx + wr < wl + wr there
    anyway, so applying the unclamped bound rejects nothing.

So, for every configuration ShearingBias accepts,

    valid(i, j)  <=>  0 <= d < d_max
    d_max = rel_extent                              if no window_size_left
            min(rel_extent, wl + wr + 1)            otherwise

which is loop-invariant. The per-element work is two comparisons against a
per-tile scalar, and for tiles entirely inside or entirely outside the band it
is nothing at all.


BOTH PATHS NOW USE THE GENERAL FORM
-----------------------------------

flash_fwd_sm90.py's `bias_tile_shift` USED to be

    padded // tile_n - (128 * (m_block + 1)) // tile_n

which hardcodes `n_block_max(m_block) == m_block + 1`, the
seqlen_q == seqlen_k specialisation of the layout contract. That was fixed on
this branch: commit eb1e487 for this kernel, 9b63979 for the sm_80 / sm_120
generic one. The shift now comes from n_block_max fetched with absolute=True.
Write-up in journal/regression-sm90-bias-shift.md, gate in
harness/parity_rel_chunked_decode.py.

This patch carries the FIXED block through verbatim on the gate-off branch, so
with INKLING_TURBO_INKERNEL_SHEAR unset the pre-sheared reader is byte-for-byte
what kernels/tml_fa4_modified/flash_fwd_sm90.py has today, n_block_max
included. That is the whole point of the gate-off passthrough and it is checked
mechanically: the patched file must contain
"- (128 * n_block_max_bias) // self.tile_n" and zero occurrences of
"- (128 * (m_block + 1)) // self.tile_n".

Consequence for validation, and this CHANGED from the first draft of this
patch: gate-on and gate-off should now agree on EVERY shape the constraints
below allow, decode and chunked prefill included, to within bf16 rounding of
the same values. A disagreement is a defect in one of the two paths, not an
expected difference.

See kernels/patches/u2_inkernel_shear.md, "The specialisation bug this
uncovered, and its fix".


CONSTRAINTS ASSERTED
--------------------

  * pack_gqa must be False. Packed tile rows interleave qhead_per_kvhead
    q-heads per sequence position, so a 128-row block is not a 128-position
    block and the per-row shear offset becomes head-dependent
    (journal/upstream/04-pack-gqa-row-semantics.md, the session-24 root
    cause). interface.py already forces pack_gqa=False for arch 9 with
    rel_bias; both the interface gate and the kernel constructor now assert
    it instead of trusting it.
  * causal, or local with window_size_right set. That is the only form for
    which BlockInfo produces n_idx_right = row + 1 + ctx + w_right, and it is
    also the only form ShearingBias itself accepts (it asserts
    `is_causal or is_local`).
  * arch 9 only. sm_100 and sm_120 keep the pre-sheared path.

Targets (both under the DEPLOYED tml_fa4 namespace, which is the copy of
kernels/tml_fa4_modified/ made by the deploy step; re-apply after every
bootstrap):
  vllm/third_party/tml_fa4/flash_fwd_sm90.py
  vllm/third_party/tml_fa4/interface.py

Order: independent of u2_serving_route.py and u3_fp8_kv.py. It does NOT
compose with u2_shear_fusion.py: that patch makes qkvr_prep emit the sheared
layout, which is the thing this patch removes the need for. Apply one or the
other, never both.

Usage: python u2_inkernel_shear.py /path/to/vllm
"""

import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

SM90 = ROOT / "vllm/third_party/tml_fa4/flash_fwd_sm90.py"
IFACE = ROOT / "vllm/third_party/tml_fa4/interface.py"

# Present in a patched tree, absent in a clean one.
SENTINEL = "INKLING_TURBO_INKERNEL_SHEAR"


# ==========================================================================
# flash_fwd_sm90.py
# ==========================================================================

SM90_CTOR_OLD = """        *args,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        paged_kv_non_tma: bool = False,
        is_split_kv: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.intra_wg_overlap = intra_wg_overlap
        self.mma_pv_is_rs = mma_pv_is_rs
        self.is_split_kv = is_split_kv
"""

SM90_CTOR_NEW = '''        *args,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        paged_kv_non_tma: bool = False,
        is_split_kv: bool = False,
        inkernel_shear: bool = False,
        inkernel_shear_coord: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.intra_wg_overlap = intra_wg_overlap
        self.mma_pv_is_rs = mma_pv_is_rs
        self.is_split_kv = is_split_kv
        # U2 in-kernel shear (INKLING_TURBO_INKERNEL_SHEAR, default off).
        # When on, mBias is the NATURAL (T, H, rel_extent) relative-bias
        # buffer rather than the (T + 128, H, rel_extent + 256) sheared one,
        # ShearingBias never ran, and the shear happens in this kernel's bias
        # address math (apply_rel_bias_inkernel_shear_sm90).
        # Derivation and validation status: kernels/patches/
        # u2_inkernel_shear.md.
        self.inkernel_shear = inkernel_shear
        self.inkernel_shear_coord = inkernel_shear_coord
        if inkernel_shear:
            assert self.has_bias, "inkernel_shear requires has_bias=True"
            assert not self.pack_gqa, (
                "in-kernel shear requires pack_gqa=False: packed tile rows "
                "interleave qhead_per_kvhead q-heads per sequence position, "
                "so a 128-row block is not a 128-position block and the "
                "per-row shear offset becomes head-dependent "
                "(journal/upstream/04-pack-gqa-row-semantics.md)"
            )
            assert self.is_causal or self.is_local, (
                "in-kernel shear reproduces ShearingBias, which asserts "
                "is_causal or is_local"
            )
        else:
            assert not inkernel_shear_coord, (
                "inkernel_shear_coord is a diagnostic mode of inkernel_shear"
            )
'''

SM90_ALIGN_OLD = """        if const_expr(mBias is not None):
            mBias = layout_utils.select(assume_tensor_aligned(mBias), QO_layout_transpose)
"""

SM90_ALIGN_NEW = """        if const_expr(mBias is not None):
            if const_expr(self.inkernel_shear):
                # The in-kernel shear reads the natural buffer along a
                # DIAGONAL BAND: the effective row stride is s_r + s_c (odd
                # whenever s_c is 1) and the tile's first column is dynamic.
                # A 16-byte alignment claim on that pointer would let the
                # vectorizer merge 2-byte loads that are not 4-byte aligned,
                # so claim only element alignment. Every bias load on this
                # path is a single element anyway.
                mBias = layout_utils.select(
                    assume_tensor_aligned(mBias, align=mBias.element_type.width // 8),
                    QO_layout_transpose,
                )
            else:
                mBias = layout_utils.select(assume_tensor_aligned(mBias), QO_layout_transpose)
"""

SM90_MMA_OLD = """            elif const_expr(self.has_bias and mBias is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mBias_cur = mBias[None, None, head_idx, batch_idx]
                else:
                    mBias_cur = cute.domain_offset(
                        (seqlen.offset_q, 0), mBias[None, None, head_idx]
                    )
                padded_bias = mBias_cur.shape[1]
                gBias_tiles = cute.local_tile(
                    mBias_cur, (self.tile_m, self.tile_n), (m_block, None)
                )
                # The shear shift is set by n_block_max, NOT by m_block + 1.
                #
                # This used to read `128 * (m_block + 1)`, which is the
                # seqlen_q == seqlen_k specialisation of the layout contract.
                # It is right for full prefill, where n_block_max really does
                # equal m_block + 1, and wrong for every shape where the two
                # lengths differ. The general contract is
                #   base(i) = n_idx_right(i) + 256 - 128 * n_block_max
                # and at tile granularity that is exactly what is below.
                #
                # How wrong it was: at batch-1 decode with 64K of KV,
                # n_block_max is 512 while m_block + 1 is 1, so the shift came
                # out +9 instead of -502. Since apply_rel_bias_sm90 guards on
                # `0 <= tile_idx < bias_num_tiles`, only n_block 0 passed, so
                # the kernel added bias to the OLDEST KV block and none at all
                # to the other 511, including the recent ones the relative
                # position term is actually about. Chunked prefill had the same
                # defect. Nothing caught it: all three parity_fa4_rel cases pass
                # cu_seqlens_q == cu_seqlens_k, the decode microbenchmarks check
                # no output, and the full-model gate ran with max_tokens=0 so it
                # never decoded.
                #
                # tile_n | 128 and tile_n | padded make the shift a whole number
                # of tiles, so every tile is fully in-shear or fully out.
                # absolute=True on purpose. ShearingBias runs once per sequence
                # and knows nothing about splits, so the layout it wrote is
                # defined by the FULL block count. Passing the split's own
                # n_block_max here would shift the bias by a per-split amount
                # against a buffer that was never sheared that way.
                _, n_block_max_bias = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, batch_idx, absolute=True
                )
                bias_tile_shift = (
                    padded_bias // self.tile_n
                    - (128 * n_block_max_bias) // self.tile_n
                )
                bias_num_tiles = padded_bias // self.tile_n
                score_mod_fn = partial(
                    self.apply_rel_bias_sm90,
                    thr_mma_qk,
                    gBias_tiles,
                    bias_tile_shift,
                    bias_num_tiles,
                    softmax_scale,
                )
"""

SM90_MMA_NEW = '''            elif const_expr(self.has_bias and mBias is not None):
                if const_expr(self.inkernel_shear):
                    # mBias is the NATURAL (rows, rel_extent) buffer for this
                    # head. No ShearingBias pass ran; the shear is done by
                    # apply_rel_bias_inkernel_shear_sm90.
                    assert self.is_causal or (
                        self.is_local and block_info.window_size_right is not None
                    ), (
                        "in-kernel shear needs BlockInfo's causal / "
                        "local-with-right form of n_idx_right"
                    )
                    if const_expr(not seqlen.has_cu_seqlens_q):
                        mBiasNat = mBias[None, None, head_idx, batch_idx]
                        # rows of this slice are already batch-local
                        bias_row_abs0 = m_block * self.tile_m
                    else:
                        mBiasNat = mBias[None, None, head_idx]
                        # rows of this slice are absolute; offset_q is folded
                        # into the address base, NOT into the row limit below
                        bias_row_abs0 = seqlen.offset_q + m_block * self.tile_m
                    # d(row, kv) = n_idx_right(row) - 1 - kv
                    #            = (row - kv) + (seqlen_k - seqlen_q) + w_right
                    # so inside the tile d(i, j) = i - j + bias_shear_m
                    #                                       - n_block * tile_n
                    bias_shear_m = (
                        m_block * self.tile_m + seqlen.seqlen_k - seqlen.seqlen_q
                    )
                    if const_expr(block_info.window_size_right is not None):
                        bias_shear_m = bias_shear_m + block_info.window_size_right
                    # ShearingBias writes a value iff 0 <= d < num_bias_vals,
                    # which reduces to 0 <= d < d_max with d_max loop
                    # invariant (kernels/patches/u2_inkernel_shear.md).
                    bias_d_max = Int32(mBiasNat.shape[1])
                    if const_expr(block_info.window_size_left is not None):
                        bias_w_right = (
                            block_info.window_size_right
                            if const_expr(block_info.window_size_right is not None)
                            else Int32(0)
                        )
                        bias_d_max = cutlass.min(
                            bias_d_max,
                            block_info.window_size_left + bias_w_right + 1,
                        )
                    # ShearingBias pad values (shearing_bias.py:88-89)
                    bias_left_pad = (
                        -Float32.inf
                        if const_expr(block_info.window_size_left is not None)
                        else Float32(0.0)
                    )
                    score_mod_fn = partial(
                        self.apply_rel_bias_inkernel_shear_sm90,
                        thr_mma_qk,
                        mBiasNat,
                        bias_row_abs0,
                        seqlen.seqlen_q - m_block * self.tile_m,
                        bias_shear_m,
                        bias_d_max,
                        bias_left_pad,
                        softmax_scale,
                    )
                else:
                    if const_expr(not seqlen.has_cu_seqlens_q):
                        mBias_cur = mBias[None, None, head_idx, batch_idx]
                    else:
                        mBias_cur = cute.domain_offset(
                            (seqlen.offset_q, 0), mBias[None, None, head_idx]
                        )
                    padded_bias = mBias_cur.shape[1]
                    gBias_tiles = cute.local_tile(
                        mBias_cur, (self.tile_m, self.tile_n), (m_block, None)
                    )
                    # The shear shift is set by n_block_max, NOT by m_block + 1.
                    #
                    # This used to read `128 * (m_block + 1)`, which is the
                    # seqlen_q == seqlen_k specialisation of the layout contract.
                    # It is right for full prefill, where n_block_max really does
                    # equal m_block + 1, and wrong for every shape where the two
                    # lengths differ. The general contract is
                    #   base(i) = n_idx_right(i) + 256 - 128 * n_block_max
                    # and at tile granularity that is exactly what is below.
                    #
                    # How wrong it was: at batch-1 decode with 64K of KV,
                    # n_block_max is 512 while m_block + 1 is 1, so the shift came
                    # out +9 instead of -502. Since apply_rel_bias_sm90 guards on
                    # `0 <= tile_idx < bias_num_tiles`, only n_block 0 passed, so
                    # the kernel added bias to the OLDEST KV block and none at all
                    # to the other 511, including the recent ones the relative
                    # position term is actually about. Chunked prefill had the same
                    # defect. Nothing caught it: all three parity_fa4_rel cases pass
                    # cu_seqlens_q == cu_seqlens_k, the decode microbenchmarks check
                    # no output, and the full-model gate ran with max_tokens=0 so it
                    # never decoded.
                    #
                    # tile_n | 128 and tile_n | padded make the shift a whole number
                    # of tiles, so every tile is fully in-shear or fully out.
                    # absolute=True on purpose. ShearingBias runs once per sequence
                    # and knows nothing about splits, so the layout it wrote is
                    # defined by the FULL block count. Passing the split's own
                    # n_block_max here would shift the bias by a per-split amount
                    # against a buffer that was never sheared that way.
                    _, n_block_max_bias = block_info.get_n_block_min_max(
                        seqlen, m_block, split_idx, batch_idx, absolute=True
                    )
                    bias_tile_shift = (
                        padded_bias // self.tile_n
                        - (128 * n_block_max_bias) // self.tile_n
                    )
                    bias_num_tiles = padded_bias // self.tile_n
                    score_mod_fn = partial(
                        self.apply_rel_bias_sm90,
                        thr_mma_qk,
                        gBias_tiles,
                        bias_tile_shift,
                        bias_num_tiles,
                        softmax_scale,
                    )
'''

SM90_APPLY_ANCHOR = """    @cute.jit
    def apply_score_mod(
"""

SM90_APPLY_NEW = '''    @cute.jit
    def apply_rel_bias_inkernel_shear_sm90(
        self,
        thr_mma_qk,
        mBiasNat,
        row_abs0,
        row_limit,
        shear_m,
        d_max,
        left_pad,
        softmax_scale,
        acc_S,
        n_block=None,
        seqlen=None,
    ):
        """In-kernel shear: add relative bias straight out of the NATURAL
        (rows, rel_extent) buffer, so ShearingBias never has to run.

        Element (i, j) of score tile (m_block, n_block) needs natural column

            d(i, j) = n_idx_right(row) - 1 - kv = i - j + shear_k
            shear_k = shear_m - n_block * tile_n

        and is a real bias value iff 0 <= d < d_max; otherwise it takes
        ShearingBias's pad: -inf for d < 0 (keys the row cannot see) and
        left_pad for d >= d_max (history beyond the projection's extent, 0.0
        globally and -inf under a left window). Rows past seqlen_q take
        left_pad; the pre-sheared path read uninitialised buffer rows there
        and let the mask deal with it, this one does not read at all.

        The gmem address carries the shear as a LAYOUT, not as arithmetic:
        element (i, j) sits at

            (row_abs0 + i) * s_r + d(i, j) * s_c
              = [row_abs0 * s_r + shear_k * s_c] + i * (s_r + s_c) - j * s_c

        so shape (tile_m, tile_n) with stride (s_r + s_c, -s_c) is exactly the
        diagonal band this tile needs, and the bracket is one domain_offset
        with coord (row_abs0, row_abs0 - shear_k) (that coord solves
        i0 * (s_r + s_c) - j0 * s_c == the bracket for any strides).

        That keeps the property that took seventeen sessions to get right: the
        bias tile is handed to thr_mma_qk.partition_C, the SAME partitioner
        that produced acc_S, and paired with acc_S by flat index. Coordinates
        are used only for the range predicate, the way mask.apply_mask already
        uses them, never to derive an address.
        """
        shear_k = shear_m - n_block * self.tile_n
        # d over the tile spans [shear_k - (tile_n - 1), shear_k + (tile_m - 1)]
        d_lo = shear_k - (self.tile_n - 1)
        d_hi = shear_k + (self.tile_m - 1)
        right_pad = -Float32.inf

        # Coordinates of acc_S, same partitioner, so tScS[i] describes acc_S[i].
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS = thr_mma_qk.partition_C(cS)

        # Diagonal-band view of the natural buffer. Bound at function scope:
        # a non-const_expr `if` body is its own traced scope
        # (journal/regression-sm90-n-block.md).
        tCgBias = None
        if const_expr(not self.inkernel_shear_coord):
            nat_stride_row = mBiasNat.stride[0]
            nat_stride_col = mBiasNat.stride[1]
            gBias_band = cute.domain_offset(
                (row_abs0, row_abs0 - shear_k),
                cute.make_tensor(
                    mBiasNat.iterator,
                    cute.make_layout(
                        (self.tile_m, self.tile_n),
                        stride=(nat_stride_row + nat_stride_col, -nat_stride_col),
                    ),
                ),
            )
            tCgBias = thr_mma_qk.partition_C(gBias_band)

        if d_hi < 0:
            # Whole tile sits right of the band: nothing this row may attend.
            for i in cutlass.range(cute.size(acc_S), unroll_full=True):
                acc_S[i] = acc_S[i] * softmax_scale + right_pad
        elif d_lo >= d_max:
            # Whole tile sits past the end of the band. This is the tile the
            # pre-sheared path skipped via `tile_idx < 0`; left_pad is 0.0
            # exactly when that skip was equivalent to adding nothing.
            for i in cutlass.range(cute.size(acc_S), unroll_full=True):
                acc_S[i] = acc_S[i] * softmax_scale + left_pad
        elif d_lo >= 0 and d_hi < d_max and self.tile_m <= row_limit:
            # Every element is a real value: no predicate, no coordinates,
            # the same inner loop shape as the pre-sheared path.
            for i in cutlass.range(cute.size(acc_S), unroll_full=True):
                if const_expr(self.inkernel_shear_coord):
                    acc_S[i] = acc_S[i] * softmax_scale + Float32(
                        mBiasNat[
                            row_abs0 + tScS[i][0],
                            tScS[i][0] - tScS[i][1] + shear_k,
                        ]
                    )
                else:
                    acc_S[i] = acc_S[i] * softmax_scale + Float32(tCgBias[i])
        else:
            # Band edge: predicate per element. `d` is bound in the loop body
            # scope and only READ inside the branches, never assigned there.
            for i in cutlass.range(cute.size(acc_S), unroll_full=True):
                d = tScS[i][0] - tScS[i][1] + shear_k
                if d >= 0 and d < d_max and tScS[i][0] < row_limit:
                    if const_expr(self.inkernel_shear_coord):
                        acc_S[i] = acc_S[i] * softmax_scale + Float32(
                            mBiasNat[row_abs0 + tScS[i][0], d]
                        )
                    else:
                        acc_S[i] = acc_S[i] * softmax_scale + Float32(tCgBias[i])
                elif d < 0:
                    acc_S[i] = acc_S[i] * softmax_scale + right_pad
                else:
                    acc_S[i] = acc_S[i] * softmax_scale + left_pad

    @cute.jit
    def apply_score_mod(
'''

SM90_EDITS = [
    ("ctor: inkernel_shear flags + pack_gqa assert", SM90_CTOR_OLD, SM90_CTOR_NEW),
    ("__call__: element-only alignment for the natural buffer",
     SM90_ALIGN_OLD, SM90_ALIGN_NEW),
    ("mma: build the in-kernel-shear score_mod_fn", SM90_MMA_OLD, SM90_MMA_NEW),
    ("apply_rel_bias_inkernel_shear_sm90", SM90_APPLY_ANCHOR, SM90_APPLY_NEW),
]


# ==========================================================================
# interface.py
# ==========================================================================

IFACE_ENV_OLD = """_U2_SM90_GENERIC = os.environ.get("U2_SM90_GENERIC") == "1"
"""

IFACE_ENV_NEW = '''_U2_SM90_GENERIC = os.environ.get("U2_SM90_GENERIC") == "1"
# U2 in-kernel shear. "1" makes the sm_90 attention kernel read the NATURAL
# (T, H, rel_extent) relative-bias buffer and perform the shear in its own
# address math, so ShearingBias and the two Cu*Kernel launches that only exist
# to schedule it never run and the (T + 128, H, rel_extent + 256) buffer is
# never allocated. "coord" is the same arithmetic with per-element coordinate
# addressing instead of the diagonal-band layout: slower, and there only to
# isolate a CuTe DSL failure in the band layout. DEFAULT OFF; no performance
# number has been measured for either mode.
# kernels/patches/u2_inkernel_shear.md
_INKLING_INKERNEL_SHEAR_MODE = os.environ.get("INKLING_TURBO_INKERNEL_SHEAR", "")
_INKLING_INKERNEL_SHEAR_ON = _INKLING_INKERNEL_SHEAR_MODE in ("1", "coord")
_INKLING_INKERNEL_SHEAR_COORD = _INKLING_INKERNEL_SHEAR_MODE == "coord"
'''

IFACE_GATE_OLD = """    # rel_bias -> bias
    cu_total_m_blocks_bias = None
    blocks_to_batch_idx = None
"""

IFACE_GATE_NEW = '''    # U2 in-kernel shear: pass rel_bias STRAIGHT to the attention kernel and
    # skip ShearingBias entirely. sm_90 only; every other arch keeps the
    # pre-sheared buffer. DEFAULT OFF.
    # U2_SM90_GENERIC routes sm_90 + bias through FlashAttentionForwardSm80,
    # which only knows the pre-sheared layout, so it wins and ShearingBias
    # still runs.
    inkernel_shear = (
        _INKLING_INKERNEL_SHEAR_ON
        and rel_bias is not None
        and arch // 10 == 9
        and not _U2_SM90_GENERIC
    )
    if inkernel_shear:
        assert not pack_gqa, (
            "in-kernel shear requires pack_gqa=False; packed rows make the "
            "per-row shear offset head-dependent"
        )
        assert causal or (local and window_size_right is not None), (
            "in-kernel shear needs n_idx_right = q_pos + 1 + "
            "(seqlen_k - seqlen_q) + window_right, which BlockInfo only "
            "produces for causal or local-with-window_right"
        )
        assert bias is None, (
            "in-kernel shear consumes rel_bias; a pre-sheared bias= as well "
            "is ambiguous"
        )

    # rel_bias -> bias
    cu_total_m_blocks_bias = None
    blocks_to_batch_idx = None
'''

IFACE_ALLOC_OLD = """        if cu_seqlens_q is None:
            bias_seqlen_q_rounded = (seqlen_q + tile_m - 1) // tile_m * tile_m
            assert rel_bias.shape == (batch_size, seqlen_q, num_head, rel_extent)
            bias = torch.empty(
                batch_size,
                bias_seqlen_q_rounded,
                num_head,
                rel_extent_padded,
                dtype=rel_bias.dtype,
                device=device)
        else:
            bias_total_q_padded = total_q + tile_m
            assert rel_bias.shape == (total_q, num_head, rel_extent)
            bias = torch.empty(
                bias_total_q_padded,
                num_head,
                rel_extent_padded,
                dtype=rel_bias.dtype,
                device=device)
"""

IFACE_ALLOC_NEW = """        if cu_seqlens_q is None:
            bias_seqlen_q_rounded = (seqlen_q + tile_m - 1) // tile_m * tile_m
            assert rel_bias.shape == (batch_size, seqlen_q, num_head, rel_extent)
            # in-kernel shear: the natural buffer IS the operand
            bias = rel_bias if inkernel_shear else torch.empty(
                batch_size,
                bias_seqlen_q_rounded,
                num_head,
                rel_extent_padded,
                dtype=rel_bias.dtype,
                device=device)
        else:
            bias_total_q_padded = total_q + tile_m
            assert rel_bias.shape == (total_q, num_head, rel_extent)
            bias = rel_bias if inkernel_shear else torch.empty(
                bias_total_q_padded,
                num_head,
                rel_extent_padded,
                dtype=rel_bias.dtype,
                device=device)
"""

IFACE_PREPARE_OLD = """        use_prepare_bias_kernel = (
            cu_seqlens_q is not None
"""

IFACE_PREPARE_NEW = """        use_prepare_bias_kernel = (
            not inkernel_shear
            and cu_seqlens_q is not None
"""

IFACE_SHEAR_COMPILE_OLD = """        if compile_key not in _flash_attn_fwd.compile_cache_shear_bias:
"""

IFACE_SHEAR_COMPILE_NEW = """        if not inkernel_shear and (
            compile_key not in _flash_attn_fwd.compile_cache_shear_bias
        ):
"""

IFACE_SHEAR_LAUNCH_OLD = """        if not is_fake_mode():
            _flash_attn_fwd.compile_cache_shear_bias[compile_key](
                rel_bias,
                bias,
"""

IFACE_SHEAR_LAUNCH_NEW = """        if not inkernel_shear and not is_fake_mode():
            _flash_attn_fwd.compile_cache_shear_bias[compile_key](
                rel_bias,
                bias,
"""

IFACE_KEY_OLD = """        bias is not None,
        tile_bias,
        rel_extent,
"""

IFACE_KEY_NEW = """        bias is not None,
        inkernel_shear,
        _INKLING_INKERNEL_SHEAR_COORD,
        tile_bias,
        rel_extent,
"""

IFACE_CTOR_OLD = """                is_split_kv=is_split_kv,
                has_bias=bias is not None)
"""

IFACE_CTOR_NEW = """                is_split_kv=is_split_kv,
                has_bias=bias is not None,
                inkernel_shear=inkernel_shear,
                inkernel_shear_coord=_INKLING_INKERNEL_SHEAR_COORD)
"""

IFACE_TENSOR_OLD = """                to_cute_tensor(bias, assumed_align=16) if bias is not None else None,
"""

IFACE_TENSOR_NEW = """                (
                    # the sheared operand is read in 16-byte-aligned tiles;
                    # the natural one is read element-wise along a diagonal
                    # band, so it must not claim more than element alignment
                    to_cute_tensor(bias, assumed_align=2 if inkernel_shear else 16)
                    if bias is not None
                    else None
                ),
"""

IFACE_EDITS = [
    ("env gate", IFACE_ENV_OLD, IFACE_ENV_NEW),
    ("resolve + validate the gate", IFACE_GATE_OLD, IFACE_GATE_NEW),
    ("skip the padded-buffer allocation", IFACE_ALLOC_OLD, IFACE_ALLOC_NEW),
    ("skip CuSeqlensToBlocks / CuBlocksToBatch", IFACE_PREPARE_OLD, IFACE_PREPARE_NEW),
    ("skip the ShearingBias compile", IFACE_SHEAR_COMPILE_OLD, IFACE_SHEAR_COMPILE_NEW),
    ("skip the ShearingBias launch", IFACE_SHEAR_LAUNCH_OLD, IFACE_SHEAR_LAUNCH_NEW),
    ("compile key", IFACE_KEY_OLD, IFACE_KEY_NEW),
    ("sm_90 constructor", IFACE_CTOR_OLD, IFACE_CTOR_NEW),
    ("bias operand alignment", IFACE_TENSOR_OLD, IFACE_TENSOR_NEW),
]


# ==========================================================================
# driver
# ==========================================================================

def apply(path: Path, edits) -> int:
    if not path.is_file():
        raise SystemExit(f"missing target: {path}")
    src = path.read_text()
    applied = 0
    for name, old, new in edits:
        count = src.count(old)
        assert count == 1, (
            f"{path.name}: anchor {name!r} matched {count} times, expected 1"
        )
        src = src.replace(old, new, 1)
        applied += 1
    path.write_text(src)
    return applied


def main() -> None:
    targets = [(SM90, SM90_EDITS), (IFACE, IFACE_EDITS)]
    present = [p for p, _ in targets if p.is_file() and SENTINEL in p.read_text()]
    if present:
        missing = [p for p, _ in targets if p not in present]
        if missing:
            raise SystemExit(
                "partially applied tree: "
                + ", ".join(p.name for p in present)
                + " patched but "
                + ", ".join(p.name for p in missing)
                + " not. Restore from kernels/tml_fa4_modified/ and re-run."
            )
        print("already applied, nothing to do")
        return

    counts = [(path, apply(path, edits)) for path, edits in targets]
    for path, n in counts:
        print(f"{path.name}: {n} edits applied")
    print(f"total: {sum(n for _, n in counts)} edits over {len(counts)} files")
    print("gate is OFF by default; set INKLING_TURBO_INKERNEL_SHEAR=1 to use it")


if __name__ == "__main__":
    main()
