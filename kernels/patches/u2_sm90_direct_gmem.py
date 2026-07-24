#!/usr/bin/env python3
"""U2 sm_90 v2: replace buggy smem-staged bias with direct-gmem per-fragment
read.

SUPERSEDED, DO NOT APPLY. This fixed the smem corruption but still failed
parity, because it computed fragment coordinates by hand and wgmma accumulator
fragments do not index that way. The real cause turned out to be pack_gqa
(session 24). Kept because the journal walks through it. The shipping sm_90
kernel is in kernels/tml_fa4_modified/flash_fwd_sm90.py. Root cause of sessions 6-10 parity FAIL: the bias cp.async tiled-copy
was built (in Base._setup_attributes) for num_producer_threads (32) but
issued from the consumer MMA threads (128-256) sliced by tidx%num_mma_threads
-> partition mismatch corrupted most of sBias (fingerprint: row 0 exact,
127/128 rows wrong, invariant across apply-index fixes).

This removes all smem staging: each MMA thread reads exactly the bias for its
own wgmma fragment elements, using reshape_acc_to_mn coords (the proven
AttentionMask pattern). Correctness first; smem staging done right is a later
perf pass. Idempotent.

Usage: python3 u2_sm90_direct_gmem.py /path/to/vllm
"""

import sys
from pathlib import Path

SM90 = Path(sys.argv[1]) / "vllm/third_party/tml_fa4/flash_fwd_sm90.py"

BRANCH_OLD = '''            elif const_expr(self.has_bias and mBias is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mBias_sliced = mBias[None, None, head_idx, batch_idx]
                else:
                    mBias_sliced = cute.domain_offset(
                        (seqlen.offset_q, 0), mBias[None, None, head_idx]
                    )
                _bias_ptr = cute.make_ptr(
                    mBias.element_type,
                    mBias_sliced.iterator.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                mBias_cur = assume_tensor_aligned(cute.make_tensor(
                    _bias_ptr,
                    cute.make_layout(
                        mBias_sliced.shape, stride=mBias_sliced.layout.stride
                    ),
                ))
                padded_bias = mBias_cur.shape[1]
                gBias_tiles = cute.local_tile(
                    mBias_cur, (self.tile_m, self.tile_n), (m_block, None)
                )
                gmem_thr_copy_Bias = gmem_tiled_copy_Bias.get_slice(
                    tidx % self.num_mma_threads
                )
                tBgBias = gmem_thr_copy_Bias.partition_S(gBias_tiles)
                tBsBias = gmem_thr_copy_Bias.partition_D(sBias)
                bias_tile_shift = (
                    padded_bias // self.tile_n
                    - (128 * (m_block + 1)) // self.tile_n
                )
                score_mod_fn = partial(
                    self.apply_rel_bias_sm90,
                    thr_mma_qk,
                    gmem_tiled_copy_Bias,
                    tBgBias,
                    tBsBias,
                    sBias,
                    bias_tile_shift,
                    softmax_scale,
                )'''

BRANCH_NEW = '''            elif const_expr(self.has_bias and mBias is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mBias_cur = mBias[None, None, head_idx, batch_idx]
                else:
                    mBias_cur = cute.domain_offset(
                        (seqlen.offset_q, 0), mBias[None, None, head_idx]
                    )
                score_mod_fn = partial(
                    self.apply_rel_bias_sm90,
                    thr_mma_qk,
                    mBias_cur,
                    m_block,
                    softmax_scale,
                )'''

SIG_OLD = '''    @cute.jit
    def apply_rel_bias_sm90(
        self,
        thr_mma_qk,
        gmem_tiled_copy_Bias,
        tBgBias,
        tBsBias,
        sBias,
        bias_tile_shift,
        softmax_scale,
        acc_S,
        n_block=None,
        seqlen=None,
    ):'''

SIG_NEW = '''    @cute.jit
    def apply_rel_bias_sm90(
        self,
        thr_mma_qk,
        mBias_cur,
        m_block,
        softmax_scale,
        acc_S,
        n_block=None,
        seqlen=None,
    ):'''

BODY_OLD = '''        \"\"\"Consumer-side sheared-bias tile: cp.async load -> barrier 7
        (MMA warpgroups) -> fragment-coord smem apply. Tiles are never
        partial (tile_mn forced (128,128) with bias; k_min % tile_n == 0).
        acc = acc*scale + bias, or scale-only for out-of-range tiles.\"\"\"
        tile_valid = (n_block + bias_tile_shift) >= 0
        if tile_valid:
            cute.copy(
                gmem_tiled_copy_Bias,
                tBgBias[None, None, None, n_block + bias_tile_shift],
                tBsBias,
            )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier(
            barrier_id=7, number_of_threads=self.num_mma_threads
        )
        # Mirror AttentionMask.apply_mask (the proven-correct sm_90 per-element
        # coordinate consumer): reshape the wgmma accumulator to a clean 2D
        # (m, n) view; the raw fragment layout is ((2,2,N/8),MMA_M,MMA_N) and
        # does NOT index linearly as (row, col) (session 8-9: linear indexing
        # left 127/128 rows misplaced). sBias is tile-local (row, col).
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
        n_rows = cutlass.const_expr(cute.size(tScS_mn.shape[0]))
        n_cols = cutlass.const_expr(cute.size(tScS_mn.shape[1]))
        if tile_valid:
            for r in cutlass.range(n_rows, unroll_full=True):
                for c in cutlass.range(n_cols, unroll_full=True):
                    row = tScS_mn[r, c][0]
                    col = tScS_mn[r, c][1]
                    acc_S_mn[r, c] = acc_S_mn[r, c] * softmax_scale + Float32(
                        sBias[(row, col)]
                    )
        else:
            for r in cutlass.range(n_rows, unroll_full=True):
                for c in cutlass.range(n_cols, unroll_full=True):
                    acc_S_mn[r, c] = acc_S_mn[r, c] * softmax_scale'''

BODY_NEW = '''        \"\"\"Direct-gmem sheared bias at wgmma fragment coords. Each mma thread
        reads exactly the bias for its own fragment elements -- no cross-thread
        smem partition to mis-tile. reshape_acc_to_mn gives a clean 2D (m,n)
        view (the wgmma frag is ((2,2,N/8),MMA_M,MMA_N), NOT linearly
        (row,col)-indexable). For global row i=m_block*tile_m+r, kv
        j=n_block*tile_n+c, the sheared column is j+padded-128*(m_block+1);
        out-of-range cols contribute 0 (the -inf right-pad handles causal).\"\"\"
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
        n_rows = cutlass.const_expr(cute.size(tScS_mn.shape[0]))
        n_cols = cutlass.const_expr(cute.size(tScS_mn.shape[1]))
        padded_bias = mBias_cur.shape[1]
        shift = padded_bias - 128 * (m_block + 1)
        for r in cutlass.range(n_rows, unroll_full=True):
            for c in cutlass.range(n_cols, unroll_full=True):
                row_g = m_block * self.tile_m + tScS_mn[r, c][0]
                sheared_col = n_block * self.tile_n + tScS_mn[r, c][1] + shift
                val = Float32(0.0)
                if (sheared_col >= 0 and sheared_col < padded_bias
                        and row_g < seqlen.seqlen_q):
                    val = Float32(mBias_cur[row_g, sheared_col])
                acc_S_mn[r, c] = acc_S_mn[r, c] * softmax_scale + val'''


def main() -> None:
    s = SM90.read_text()
    if "Direct-gmem sheared bias" in s:
        print("already applied")
        return
    for old, new, name in [(BRANCH_OLD, BRANCH_NEW, "branch"),
                           (SIG_OLD, SIG_NEW, "signature"),
                           (BODY_OLD, BODY_NEW, "body")]:
        assert old in s, f"anchor missing: {name}"
        s = s.replace(old, new, 1)
    SM90.write_text(s)
    print("direct-gmem applied: 3 regions")


if __name__ == "__main__":
    main()
