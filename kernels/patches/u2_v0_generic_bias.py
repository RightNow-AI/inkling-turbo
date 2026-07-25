#!/usr/bin/env python3
"""U2 v0: tile-level sheared-bias add in tml_fa4's generic (SM80/sm_120) kernel.

Applies to the WSL editable install. v0 = direct-gmem fragment add (no smem
staging yet): per acc element, one gmem load at fragment coords + FMA, no
divmods, no callback SSA machinery (the measured 3.2x overhead of score_mod).
v1 (smem staging + cp.async pipeline) follows only if v0 misses target.

Layout contract (machine-verified, journal/u2-hopper-design.md):
  bias col(i_global, k) = k + padded - 128 * (i_global // 128 + 1)
  -> per tile (m_block, n_block), col = n_block*tile_n + c + padded - 128*(m_block+1)
  -inf right-pad gives in-tile causal masking for free; out-of-range cols
  (col < 0 or >= padded) contribute 0 (global beyond-extent) and window
  masking handles local-mode exclusion.

Scale semantics: mimics score_mod contract, acc = acc*scale + bias applied
to every tile element; softmax_scale_log2 switches to log2e-only mode.

Usage: python3 u2_v0_generic_bias.py /path/to/vllm
"""
#
# PREDATES THE BIAS-SHIFT FIX. This script installs the shear shift in its
# `128 * (m_block + 1)` form, which is the seqlen_q == seqlen_k specialisation
# of the layout contract and is WRONG for every chunked-prefill and decode
# shape. See journal/regression-sm90-bias-shift.md. The shipped sources in
# kernels/tml_fa4_modified/ carry the corrected `n_block_max` form and are
# authoritative for every architecture. Apply this only to reproduce the
# historical state the journal describes, never to build something to serve
# with.


import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
FWD = ROOT / "vllm/third_party/tml_fa4/flash_fwd.py"
IFACE = ROOT / "vllm/third_party/tml_fa4/interface.py"

EDITS_FWD = [
    # 1. Base ctor: has_bias flag
    (
        "        has_aux_tensors: bool = False,\n"
        "        q_subtile_factor: int | None = None,\n"
        "    ):",
        "        has_aux_tensors: bool = False,\n"
        "        q_subtile_factor: int | None = None,\n"
        "        has_bias: bool = False,\n"
        "    ):"),
    (
        "        self.score_mod = score_mod\n"
        "        self.mask_mod = mask_mod\n"
        "        self.qk_acc_dtype = Float32",
        "        self.score_mod = score_mod\n"
        "        self.mask_mod = mask_mod\n"
        "        self.has_bias = has_bias\n"
        "        self.qk_acc_dtype = Float32"),
    # 2. SM80 __call__: accept mBias, transpose, thread into kernel
    (
        "        aux_tensors=None,\n"
        "        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,\n"
        "        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).\n"
        "        stream: cuda.CUstream = None,\n"
        "    ):",
        "        aux_tensors=None,\n"
        "        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,\n"
        "        mBias: Optional[cute.Tensor] = None,\n"
        "        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).\n"
        "        stream: cuda.CUstream = None,\n"
        "    ):"),
    (
        "        if const_expr(mLSE is not None):\n"
        "            LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]\n"
        "            mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))",
        "        if const_expr(mLSE is not None):\n"
        "            LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]\n"
        "            mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))\n"
        "        if const_expr(mBias is not None):\n"
        "            # (b, s, h, padded) -> (s, padded, h, b); varlen (total_q, h, padded) -> (total_q, padded, h)\n"
        "            Bias_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]\n"
        "            mBias = cute.make_tensor(\n"
        "                assume_tensor_aligned(mBias).iterator,\n"
        "                cute.select(mBias.layout, mode=Bias_layout_transpose),\n"
        "            )"),
    (
        "            tile_sched_params,\n"
        "            TileScheduler,\n"
        "            aux_tensors,\n"
        "            fastdiv_mods,\n"
        "        ).launch(",
        "            tile_sched_params,\n"
        "            TileScheduler,\n"
        "            aux_tensors,\n"
        "            fastdiv_mods,\n"
        "            mBias,\n"
        "        ).launch("),
    # 3. kernel signature (SM80 kernel: aux_tensors=None, fastdiv_mods=None tail)
    (
        "        aux_tensors=None,\n"
        "        fastdiv_mods=None,\n"
        "    ):\n"
        "        # Thread index, block index",
        "        aux_tensors=None,\n"
        "        fastdiv_mods=None,\n"
        "        mBias: Optional[cute.Tensor] = None,\n"
        "    ):\n"
        "        # Thread index, block index"),
    # 4. kernel body: slice mBias per (head, varlen offset) alongside mQ_cur
    (
        "        gQ = cute.local_tile(mQ_cur, blkQ_shape, (m_block, 0))",
        "        if const_expr(mBias is not None):\n"
        "            if const_expr(not seqlen.has_cu_seqlens_q):\n"
        "                mBias_cur = mBias[None, None, num_head, batch_size]\n"
        "            else:\n"
        "                mBias_cur = cute.domain_offset(\n"
        "                    (seqlen.offset_q, 0), mBias[None, None, num_head]\n"
        "                )\n"
        "        else:\n"
        "            mBias_cur = None\n"
        "        gQ = cute.local_tile(mQ_cur, blkQ_shape, (m_block, 0))"),
    # 5. thread mBias_cur through compute_one_n_block partial
    (
        "        compute_one_n_block = partial(\n"
        "            self.compute_one_n_block,\n"
        "            mma_params=mma_params,\n"
        "            smem_copy_params=smem_copy_params,\n"
        "            softmax=softmax,\n"
        "            load_K=load_K,",
        "        compute_one_n_block = partial(\n"
        "            self.compute_one_n_block,\n"
        "            mma_params=mma_params,\n"
        "            smem_copy_params=smem_copy_params,\n"
        "            softmax=softmax,\n"
        "            mBias_cur=mBias_cur,\n"
        "            load_K=load_K,"),
    # 6. compute_one_n_block: new param + bias application after gemm
    (
        "        mask_fn: Optional[Callable] = None,\n"
        "        is_first_n_block: cutlass.Constexpr = False,\n"
        "        check_inf: cutlass.Constexpr = True,\n"
        "    ):",
        "        mask_fn: Optional[Callable] = None,\n"
        "        is_first_n_block: cutlass.Constexpr = False,\n"
        "        check_inf: cutlass.Constexpr = True,\n"
        "        mBias_cur=None,\n"
        "    ):"),
    (
        "        if const_expr(score_mod is not None):\n"
        "            self.apply_score_mod(\n"
        "                mma_params.thr_mma_qk,\n"
        "                batch_idx,\n"
        "                head_idx,\n"
        "                m_block,\n"
        "                acc_S,\n"
        "                n_block,\n"
        "                softmax_scale=softmax.softmax_scale,\n"
        "                seqlen=seqlen,\n"
        "                aux_tensors=aux_tensors,\n"
        "                fastdiv_mods=fastdiv_mods,\n"
        "            )",
        "        if const_expr(score_mod is not None):\n"
        "            self.apply_score_mod(\n"
        "                mma_params.thr_mma_qk,\n"
        "                batch_idx,\n"
        "                head_idx,\n"
        "                m_block,\n"
        "                acc_S,\n"
        "                n_block,\n"
        "                softmax_scale=softmax.softmax_scale,\n"
        "                seqlen=seqlen,\n"
        "                aux_tensors=aux_tensors,\n"
        "                fastdiv_mods=fastdiv_mods,\n"
        "            )\n"
        "        if const_expr(self.has_bias and mBias_cur is not None):\n"
        "            self.apply_rel_bias(\n"
        "                mma_params.thr_mma_qk,\n"
        "                m_block,\n"
        "                n_block,\n"
        "                acc_S,\n"
        "                softmax.softmax_scale,\n"
        "                mBias_cur,\n"
        "                seqlen,\n"
        "            )"),
    # 7. apply_rel_bias method (before apply_score_mod def)
    (
        "    @cute.jit\n"
        "    def apply_score_mod(\n"
        "        self,\n"
        "        thr_mma_qk,\n"
        "        batch_idx,\n"
        "        head_idx,\n"
        "        m_block,",
        "    @cute.jit\n"
        "    def apply_rel_bias(\n"
        "        self,\n"
        "        thr_mma_qk,\n"
        "        m_block,\n"
        "        n_block,\n"
        "        acc_S,\n"
        "        softmax_scale,\n"
        "        mBias_cur,\n"
        "        seqlen,\n"
        "    ):\n"
        "        \"\"\"Tile-level sheared-bias add: acc = acc*scale + bias[frag coords].\n"
        "\n"
        "        Sheared layout contract (verified): for global row i, kv k, the bias\n"
        "        lives at column k + padded - 128*(i//128 + 1). tile_m == 128 is\n"
        "        asserted at construction, so i//128 == m_block for every tile row.\n"
        "        Out-of-range columns contribute 0 (beyond-extent semantics); the\n"
        "        tensor's -inf right-pad masks future positions inside the tile.\n"
        "        \"\"\"\n"
        "        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))\n"
        "        tScS = thr_mma_qk.partition_C(cS)\n"
        "        padded = mBias_cur.shape[1]\n"
        "        shift = n_block * self.tile_n + padded - 128 * (m_block + 1)\n"
        "        n_vals = cutlass.const_expr(cute.size(acc_S.shape))\n"
        "        for i in cutlass.range(0, n_vals, 1, unroll_full=True):\n"
        "            r_local = tScS[i][0]\n"
        "            col = shift + tScS[i][1]\n"
        "            row_g = m_block * self.tile_m + r_local\n"
        "            val = Float32(0.0)\n"
        "            if col >= 0 and col < padded and row_g < seqlen.seqlen_q:\n"
        "                val = Float32(mBias_cur[row_g, col])\n"
        "            acc_S[i] = acc_S[i] * softmax_scale + val\n"
        "\n"
        "    @cute.jit\n"
        "    def apply_score_mod(\n"
        "        self,\n"
        "        thr_mma_qk,\n"
        "        batch_idx,\n"
        "        head_idx,\n"
        "        m_block,"),
]

# Interface: route rel_bias to the generic family (sm_120/sm_80) with the
# scale-mode switch; relax tile_n==128 (contract needs tile_m==128 only).
EDITS_IFACE = [
    (
        "        assert tile_m == 128\n"
        "        assert tile_n == 128",
        "        assert tile_m == 128\n"
        "        assert tile_n == 128 or arch // 10 in [8, 12]"),
]


def apply(path: Path, edits) -> int:
    s = path.read_text()
    n = 0
    for old, new in edits:
        if new in s:
            continue  # idempotent
        assert old in s, f"anchor not found in {path.name}: {old[:80]!r}"
        s = s.replace(old, new, 1)
        n += 1
    path.write_text(s)
    return n


if __name__ == "__main__":
    print(f"flash_fwd.py: {apply(FWD, EDITS_FWD)} edits applied")
    print(f"interface.py: {apply(IFACE, EDITS_IFACE)} edits applied")
    print("NOTE: scale-mode switch (softmax_scale_log2 when has_bias) and the "
          "interface ctor/call plumbing for the generic family are applied in "
          "the follow-up edit set, run the parity harness to drive debugging.")
