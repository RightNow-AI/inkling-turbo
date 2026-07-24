#!/usr/bin/env python3
"""U2 sm_90 port: sheared-bias smem tile machinery in flash_fwd_sm90.py.

SUPERSEDED, DO NOT APPLY. This is the sessions 6-10 attempt. It failed parity:
the bias tiled-copy was built for the 32 producer threads and issued from the
consumer MMA threads, which corrupted most of sBias. Kept because the journal
walks through it. The shipping sm_90 kernel is in
kernels/tml_fa4_modified/flash_fwd_sm90.py and uses partition_C instead.

Rides the score_mod_fn slot (invoked identically at all 3 mainloop sites:
score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)). Consumer-side
cp.async load -> named barrier 7 (free slot; enum uses 1-6) across MMA
warpgroups -> fragment-coord smem apply. tile_mn forced (128,128) when
rel_bias on sm_90 so bias tiles are never partial (k_min % tile_n == 0).

Usage: python3 u2_sm90_bias_port.py /path/to/vllm
"""

import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
SM90 = ROOT / "vllm/third_party/tml_fa4/flash_fwd_sm90.py"
IFACE = ROOT / "vllm/third_party/tml_fa4/interface.py"

E_SM90 = [
    # 1. storage: sBias in both variants
    (
        "        @cute.struct\n"
        "        class SharedStorageQKV:\n"
        "            mbar_ptr_Q: mbar_ptr_Q_struct\n"
        "            mbar_ptr_K: mbar_ptr_K_struct\n"
        "            mbar_ptr_V: mbar_ptr_V_struct\n"
        "            sV: sV_struct\n"
        "            sQ: sQ_struct\n"
        "            sK: sK_struct\n"
        "            sP: sP_struct\n",
        "        sBias_struct = cute.struct.Align[\n"
        "            cute.struct.MemRange[\n"
        "                self.dtype,\n"
        "                cute.cosize(self.sBias_layout) if self.has_bias else 0,\n"
        "            ],\n"
        "            1024,\n"
        "        ]\n"
        "\n"
        "        @cute.struct\n"
        "        class SharedStorageQKV:\n"
        "            mbar_ptr_Q: mbar_ptr_Q_struct\n"
        "            mbar_ptr_K: mbar_ptr_K_struct\n"
        "            mbar_ptr_V: mbar_ptr_V_struct\n"
        "            sV: sV_struct\n"
        "            sQ: sQ_struct\n"
        "            sK: sK_struct\n"
        "            sP: sP_struct\n"
        "            sBias: sBias_struct\n",
    ),
    (
        "        @cute.struct\n"
        "        class SharedStorageSharedQV:\n"
        "            mbar_ptr_Q: mbar_ptr_Q_struct\n"
        "            mbar_ptr_K: mbar_ptr_K_struct\n"
        "            mbar_ptr_V: mbar_ptr_V_struct\n"
        "            sQ: sQV_struct\n"
        "            sK: sK_struct\n"
        "            sP: sP_struct\n",
        "        @cute.struct\n"
        "        class SharedStorageSharedQV:\n"
        "            mbar_ptr_Q: mbar_ptr_Q_struct\n"
        "            mbar_ptr_K: mbar_ptr_K_struct\n"
        "            mbar_ptr_V: mbar_ptr_V_struct\n"
        "            sQ: sQV_struct\n"
        "            sK: sK_struct\n"
        "            sP: sP_struct\n"
        "            sBias: sBias_struct\n",
    ),
    # 2. __call__: mBias param + transpose + scale sentinel
    (
        "        aux_tensors: Optional[list] = None,\n"
        "        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,\n"
        "        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).\n"
        "        stream: cuda.CUstream = None,\n"
        "    ):\n"
        "        \"\"\"Configures and launches the flash attention kernel.",
        "        aux_tensors: Optional[list] = None,\n"
        "        num_splits_dynamic_ptr: Optional[cute.Tensor] = None,\n"
        "        mBias: Optional[cute.Tensor] = None,\n"
        "        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).\n"
        "        stream: cuda.CUstream = None,\n"
        "    ):\n"
        "        \"\"\"Configures and launches the flash attention kernel.",
    ),
    (
        "        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]\n"
        "        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]\n"
        "        mQ, mO = [layout_utils.select(t, QO_layout_transpose) for t in (mQ, mO)]",
        "        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]\n"
        "        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]\n"
        "        mQ, mO = [layout_utils.select(t, QO_layout_transpose) for t in (mQ, mO)]\n"
        "        if const_expr(mBias is not None):\n"
        "            mBias = layout_utils.select(assume_tensor_aligned(mBias), QO_layout_transpose)",
    ),
    (
        "        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(\n"
        "            softmax_scale, self.score_mod\n"
        "        )",
        "        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(\n"
        "            softmax_scale,\n"
        "            self.score_mod if self.score_mod is not None\n"
        "            else (True if self.has_bias else None),\n"
        "        )",
    ),
    # 3. launch threading
    (
        "            SharedStorage,\n"
        "            aux_tensors,\n"
        "            fastdiv_mods,\n"
        "        ).launch(",
        "            SharedStorage,\n"
        "            aux_tensors,\n"
        "            fastdiv_mods,\n"
        "            mBias,\n"
        "            self.sBias_layout,\n"
        "            self.gmem_tiled_copy_Bias,\n"
        "        ).launch(",
    ),
    # 4. kernel signature
    (
        "        aux_tensors=Optional[list[cute.Tensor]],\n"
        "        fastdiv_mods=None,\n"
        "    ):\n"
        "        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())",
        "        aux_tensors=Optional[list[cute.Tensor]],\n"
        "        fastdiv_mods=None,\n"
        "        mBias: Optional[cute.Tensor] = None,\n"
        "        sBias_layout=None,\n"
        "        gmem_tiled_copy_Bias=None,\n"
        "    ):\n"
        "        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())",
    ),
    # 5. kernel -> mma threading (sBias from storage; args appended to call)
    (
        "            self.mma(\n"
        "                tiled_mma_qk,\n"
        "                tiled_mma_pv,\n"
        "                mO,\n"
        "                mLSE,\n"
        "                sQ,\n"
        "                sK,",
        "            sBias = (\n"
        "                storage.sBias.get_tensor(sBias_layout)\n"
        "                if const_expr(self.has_bias and mBias is not None)\n"
        "                else None\n"
        "            )\n"
        "            self.mma(\n"
        "                tiled_mma_qk,\n"
        "                tiled_mma_pv,\n"
        "                mO,\n"
        "                mLSE,\n"
        "                sQ,\n"
        "                sK,",
    ),
    # mma() signature + call-site tail (both end with fastdiv_mods)
    (
        "        blocksparse_tensors: Optional[BlockSparseTensors],\n"
        "        aux_tensors: Optional[list],\n"
        "        fastdiv_mods=None,\n"
        "    ):\n"
        "        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)",
        "        blocksparse_tensors: Optional[BlockSparseTensors],\n"
        "        aux_tensors: Optional[list],\n"
        "        fastdiv_mods=None,\n"
        "        mBias: Optional[cute.Tensor] = None,\n"
        "        sBias: Optional[cute.Tensor] = None,\n"
        "        gmem_tiled_copy_Bias=None,\n"
        "    ):\n"
        "        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)",
    ),
    # 6. per-work-tile: bias slice + partitions + bias_fn; ride score_mod slot
    (
        "            score_mod_fn = None\n"
        "            if const_expr(self.score_mod is not None):\n"
        "                score_mod_fn = partial(\n"
        "                    self.apply_score_mod,\n"
        "                    thr_mma_qk,\n"
        "                    batch_idx,\n"
        "                    head_idx,\n"
        "                    m_block,\n"
        "                    softmax_scale=softmax_scale,\n"
        "                    aux_tensors=aux_tensors,\n"
        "                    fastdiv_mods=fastdiv_mods,\n"
        "                )",
        "            score_mod_fn = None\n"
        "            if const_expr(self.score_mod is not None):\n"
        "                score_mod_fn = partial(\n"
        "                    self.apply_score_mod,\n"
        "                    thr_mma_qk,\n"
        "                    batch_idx,\n"
        "                    head_idx,\n"
        "                    m_block,\n"
        "                    softmax_scale=softmax_scale,\n"
        "                    aux_tensors=aux_tensors,\n"
        "                    fastdiv_mods=fastdiv_mods,\n"
        "                )\n"
        "            elif const_expr(self.has_bias and mBias is not None):\n"
        "                if const_expr(not seqlen.has_cu_seqlens_q):\n"
        "                    mBias_sliced = mBias[None, None, head_idx, batch_idx]\n"
        "                else:\n"
        "                    mBias_sliced = cute.domain_offset(\n"
        "                        (seqlen.offset_q, 0), mBias[None, None, head_idx]\n"
        "                    )\n"
        "                _bias_ptr = cute.make_ptr(\n"
        "                    mBias.element_type,\n"
        "                    mBias_sliced.iterator.toint(),\n"
        "                    cute.AddressSpace.gmem,\n"
        "                    assumed_align=16,\n"
        "                )\n"
        "                mBias_cur = assume_tensor_aligned(cute.make_tensor(\n"
        "                    _bias_ptr,\n"
        "                    cute.make_layout(\n"
        "                        mBias_sliced.shape, stride=mBias_sliced.layout.stride\n"
        "                    ),\n"
        "                ))\n"
        "                padded_bias = mBias_cur.shape[1]\n"
        "                gBias_tiles = cute.local_tile(\n"
        "                    mBias_cur, (self.tile_m, self.tile_n), (m_block, None)\n"
        "                )\n"
        "                gmem_thr_copy_Bias = gmem_tiled_copy_Bias.get_slice(\n"
        "                    tidx % self.num_mma_threads\n"
        "                )\n"
        "                tBgBias = gmem_thr_copy_Bias.partition_S(gBias_tiles)\n"
        "                tBsBias = gmem_thr_copy_Bias.partition_D(sBias)\n"
        "                bias_tile_shift = (\n"
        "                    padded_bias // self.tile_n\n"
        "                    - (128 * (m_block + 1)) // self.tile_n\n"
        "                )\n"
        "                score_mod_fn = partial(\n"
        "                    self.apply_rel_bias_sm90,\n"
        "                    thr_mma_qk,\n"
        "                    gmem_tiled_copy_Bias,\n"
        "                    tBgBias,\n"
        "                    tBsBias,\n"
        "                    sBias,\n"
        "                    bias_tile_shift,\n"
        "                    softmax_scale,\n"
        "                )",
    ),
    # 7. the sm_90 bias method (append before apply_score_mod def)
    (
        "    @cute.jit\n"
        "    def apply_score_mod(\n"
        "        self,\n"
        "        thr_mma_qk,\n"
        "        batch_idx,\n"
        "        head_idx,\n"
        "        m_block,",
        "    @cute.jit\n"
        "    def apply_rel_bias_sm90(\n"
        "        self,\n"
        "        thr_mma_qk,\n"
        "        gmem_tiled_copy_Bias,\n"
        "        tBgBias,\n"
        "        tBsBias,\n"
        "        sBias,\n"
        "        bias_tile_shift,\n"
        "        softmax_scale,\n"
        "        acc_S,\n"
        "        n_block=None,\n"
        "        seqlen=None,\n"
        "    ):\n"
        "        \"\"\"Consumer-side sheared-bias tile: cp.async load -> barrier 7\n"
        "        (MMA warpgroups) -> fragment-coord smem apply. Tiles are never\n"
        "        partial (tile_mn forced (128,128) with bias; k_min % tile_n == 0).\n"
        "        acc = acc*scale + bias, or scale-only for out-of-range tiles.\"\"\"\n"
        "        tile_valid = (n_block + bias_tile_shift) >= 0\n"
        "        if tile_valid:\n"
        "            cute.copy(\n"
        "                gmem_tiled_copy_Bias,\n"
        "                tBgBias[None, None, None, n_block + bias_tile_shift],\n"
        "                tBsBias,\n"
        "            )\n"
        "        cute.arch.cp_async_commit_group()\n"
        "        cute.arch.cp_async_wait_group(0)\n"
        "        cute.arch.barrier(\n"
        "            barrier_id=7, number_of_threads=self.num_mma_threads\n"
        "        )\n"
        "        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))\n"
        "        tScS = thr_mma_qk.partition_C(cS)\n"
        "        n_vals = cutlass.const_expr(cute.size(acc_S.shape))\n"
        "        if tile_valid:\n"
        "            for i in cutlass.range(0, n_vals, 1, unroll_full=True):\n"
        "                acc_S[i] = acc_S[i] * softmax_scale + Float32(\n"
        "                    sBias[(tScS[i][0], tScS[i][1])]\n"
        "                )\n"
        "        else:\n"
        "            for i in cutlass.range(0, n_vals, 1, unroll_full=True):\n"
        "                acc_S[i] = acc_S[i] * softmax_scale\n"
        "\n"
        "    @cute.jit\n"
        "    def apply_score_mod(\n"
        "        self,\n"
        "        thr_mma_qk,\n"
        "        batch_idx,\n"
        "        head_idx,\n"
        "        m_block,",
    ),
]

E_IFACE = [
    # force square tiles on sm_90 when bias present (before tile resolution)
    (
        "    fwd_cfg = FwdConfig(128, 128, True, True)  # default\n"
        "    if tile_mn is None:",
        "    fwd_cfg = FwdConfig(128, 128, True, True)  # default\n"
        "    if rel_bias is not None and arch // 10 == 9 and tile_mn is None:\n"
        "        # bias tiles must never be partial: k_min is a multiple of 128,\n"
        "        # so tile_n must divide 128 (journal/u2-hopper-design.md)\n"
        "        tile_mn = (128, 128)\n"
        "    if tile_mn is None:",
    ),
    # sm_90 assert relaxation (tile_n==128 now guaranteed there)
    (
        "        assert tile_n == 128 or arch // 10 in [8, 12]",
        "        assert tile_n == 128 or arch // 10 in [8, 9, 12]",
    ),
    # sm_90 ctor: has_bias
    (
        "                has_aux_tensors=aux_tensors is not None,\n"
        "                q_subtile_factor=q_subtile_factor,\n"
        "                paged_kv_non_tma=paged_kv_non_tma,\n"
        "            )",
        "                has_aux_tensors=aux_tensors is not None,\n"
        "                q_subtile_factor=q_subtile_factor,\n"
        "                paged_kv_non_tma=paged_kv_non_tma,\n"
        "                has_bias=bias is not None,\n"
        "            )",
    ),
]


def apply(path: Path, edits) -> int:
    s = path.read_text()
    n = 0
    for old, new in edits:
        if new in s:
            continue
        assert old in s, f"anchor missing in {path.name}: {old[:70]!r}"
        s = s.replace(old, new, 1)
        n += 1
    path.write_text(s)
    return n


if __name__ == "__main__":
    print(f"flash_fwd_sm90.py: {apply(SM90, E_SM90)} edits")
    print(f"interface.py: {apply(IFACE, E_IFACE)} edits")
