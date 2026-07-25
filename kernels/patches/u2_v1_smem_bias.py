#!/usr/bin/env python3
"""U2 v1: smem-staged bias tile (replaces v0's per-element gmem reads).

Design (journal/u2-hopper-design.md): bias tile for attention tile (m,n) is
one contiguous (tile_m x tile_n) block of the sheared tensor at column
offset n*tile_n + padded - 128*(m+1). k_min = 128*(m+1) - padded is a
multiple of tile_n, so tiles are NEVER partial: fully valid -> one plain
cp.async 2D copy (issued with the V load, same commit group); fully
out-of-range -> skip copy AND skip smem read (scale-only). Apply is a
per-element smem read + FMA at fragment coords, no bounds checks, no
gmem, no divmods.

Usage: python3 u2_v1_smem_bias.py /path/to/vllm
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

EDITS = [
    # A. sBias layout + tiled copy, built host-side in SM80 __call__
    (
        "        self._setup_attributes()\n"
        "        SharedStorage = self._get_shared_storage_cls()",
        "        self._setup_attributes()\n"
        "        if const_expr(self.has_bias):\n"
        "            self.sBias_layout = cute.make_ordered_layout(\n"
        "                (self.tile_m, self.tile_n), order=(1, 0)\n"
        "            )\n"
        "            self.gmem_tiled_copy_Bias = copy_utils.tiled_copy_2d(\n"
        "                self.dtype, self.tile_n, self.num_threads, is_async=True\n"
        "            )\n"
        "        else:\n"
        "            self.sBias_layout = None\n"
        "            self.gmem_tiled_copy_Bias = None\n"
        "        SharedStorage = self._get_shared_storage_cls()"),
    # B. SharedStorage: bias variant
    (
        "        @cute.struct\n"
        "        class SharedStorageQKV:\n"
        "            sV: sV_struct\n"
        "            sQ: sQ_struct\n"
        "            sK: sK_struct\n"
        "\n"
        "        @cute.struct\n"
        "        class SharedStorageSharedQV:\n"
        "            sQ: sQV_struct\n"
        "            sK: sK_struct\n"
        "\n"
        "        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV",
        "        if const_expr(self.has_bias):\n"
        "            sBias_struct = cute.struct.Align[\n"
        "                cute.struct.MemRange[self.dtype, cute.cosize(self.sBias_layout)], 1024\n"
        "            ]\n"
        "\n"
        "            @cute.struct\n"
        "            class SharedStorageQKVBias:\n"
        "                sV: sV_struct\n"
        "                sQ: sQ_struct\n"
        "                sK: sK_struct\n"
        "                sBias: sBias_struct\n"
        "\n"
        "            return SharedStorageQKVBias\n"
        "\n"
        "        @cute.struct\n"
        "        class SharedStorageQKV:\n"
        "            sV: sV_struct\n"
        "            sQ: sQ_struct\n"
        "            sK: sK_struct\n"
        "\n"
        "        @cute.struct\n"
        "        class SharedStorageSharedQV:\n"
        "            sQ: sQV_struct\n"
        "            sK: sK_struct\n"
        "\n"
        "        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV"),
    # C. kernel: sBias tensor + k-space gmem view + thread partitions
    (
        "        smem = cutlass.utils.SmemAllocator()\n"
        "        storage = smem.allocate(SharedStorage)\n"
        "        sQ = storage.sQ.get_tensor(sQ_layout)\n"
        "        sK = storage.sK.get_tensor(sK_layout)",
        "        smem = cutlass.utils.SmemAllocator()\n"
        "        storage = smem.allocate(SharedStorage)\n"
        "        sQ = storage.sQ.get_tensor(sQ_layout)\n"
        "        sK = storage.sK.get_tensor(sK_layout)\n"
        "        if const_expr(self.has_bias and mBias is not None):\n"
        "            sBias = storage.sBias.get_tensor(self.sBias_layout)\n"
        "            padded_bias = mBias_cur.shape[1]\n"
        "            # k-space view: gBiasK[r, k] = sheared bias of global row\n"
        "            # (m_block*tile_m + r) for kv index k (layout contract).\n"
        "            bias_shift = padded_bias - 128 * (m_block + 1)\n"
        "            gBiasK = cute.domain_offset(\n"
        "                (m_block * self.tile_m, bias_shift), mBias_cur\n"
        "            )\n"
        "            gBias_tiles = cute.local_tile(\n"
        "                gBiasK, (self.tile_m, self.tile_n), (0, None)\n"
        "            )\n"
        "            gmem_thr_copy_Bias = self.gmem_tiled_copy_Bias.get_slice(tidx)\n"
        "            tBgBias = gmem_thr_copy_Bias.partition_S(gBias_tiles)\n"
        "            tBsBias = gmem_thr_copy_Bias.partition_D(sBias)\n"
        "            bias_k_min_tile = (128 * (m_block + 1) - padded_bias) // self.tile_n\n"
        "        else:\n"
        "            sBias = None\n"
        "            tBgBias = None\n"
        "            tBsBias = None\n"
        "            bias_k_min_tile = Int32(0)"),
    # D. thread bias objects into compute_one_n_block
    (
        "            mma_params=mma_params,\n"
        "            smem_copy_params=smem_copy_params,\n"
        "            softmax=softmax,\n"
        "            mBias_cur=mBias_cur,\n"
        "            load_K=load_K,",
        "            mma_params=mma_params,\n"
        "            smem_copy_params=smem_copy_params,\n"
        "            softmax=softmax,\n"
        "            sBias=sBias,\n"
        "            tBgBias=tBgBias,\n"
        "            tBsBias=tBsBias,\n"
        "            bias_k_min_tile=bias_k_min_tile,\n"
        "            load_K=load_K,"),
    # E. compute_one_n_block signature: swap mBias_cur for smem objects
    (
        "        is_first_n_block: cutlass.Constexpr = False,\n"
        "        check_inf: cutlass.Constexpr = True,\n"
        "        mBias_cur=None,\n"
        "    ):",
        "        is_first_n_block: cutlass.Constexpr = False,\n"
        "        check_inf: cutlass.Constexpr = True,\n"
        "        sBias=None,\n"
        "        tBgBias=None,\n"
        "        tBsBias=None,\n"
        "        bias_k_min_tile=None,\n"
        "    ):"),
    # F. issue the bias tile copy with the V load (same commit group)
    (
        "        load_V_next()\n"
        "        sm80_utils.gemm(",
        "        if const_expr(self.has_bias and sBias is not None):\n"
        "            if n_block >= bias_k_min_tile:\n"
        "                cute.copy(\n"
        "                    self.gmem_tiled_copy_Bias,\n"
        "                    tBgBias[None, None, None, n_block],\n"
        "                    tBsBias,\n"
        "                )\n"
        "        load_V_next()\n"
        "        sm80_utils.gemm("),
    # G. replace v0 gmem apply with smem apply after the stage-1 sync
    (
        "        if const_expr(self.has_bias and mBias_cur is not None):\n"
        "            self.apply_rel_bias(\n"
        "                mma_params.thr_mma_qk,\n"
        "                m_block,\n"
        "                n_block,\n"
        "                acc_S,\n"
        "                softmax.softmax_scale,\n"
        "                mBias_cur,\n"
        "                seqlen,\n"
        "            )\n"
        "\n"
        "        smem_pipe_write = self.advance_pipeline(smem_pipe_write)",
        "        smem_pipe_write = self.advance_pipeline(smem_pipe_write)"),
    (
        "        if const_expr(mask_fn is not None):\n"
        "            mask_fn(acc_S, n_block=n_block)\n"
        "        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)",
        "        if const_expr(self.has_bias and sBias is not None):\n"
        "            if const_expr(self.num_stages != 1):\n"
        "                cute.arch.cp_async_wait_group(0)\n"
        "                cute.arch.barrier()\n"
        "            self.apply_rel_bias_smem(\n"
        "                mma_params.thr_mma_qk,\n"
        "                acc_S,\n"
        "                softmax.softmax_scale,\n"
        "                sBias,\n"
        "                n_block >= bias_k_min_tile,\n"
        "            )\n"
        "        if const_expr(mask_fn is not None):\n"
        "            mask_fn(acc_S, n_block=n_block)\n"
        "        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)"),
    # H. the smem apply method (replaces v0's apply_rel_bias body use)
    (
        "    @cute.jit\n"
        "    def apply_rel_bias(\n",
        "    @cute.jit\n"
        "    def apply_rel_bias_smem(\n"
        "        self,\n"
        "        thr_mma_qk,\n"
        "        acc_S,\n"
        "        softmax_scale,\n"
        "        sBias,\n"
        "        tile_valid,\n"
        "    ):\n"
        "        \"\"\"acc = acc*scale + sBias[frag coords]; scale-only if tile invalid\n"
        "        (fully out-of-range tile => bias 0 everywhere, contract-guaranteed\n"
        "        never partial since k_min % tile_n == 0).\"\"\"\n"
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
        "    def apply_rel_bias(\n"),
]


def main() -> None:
    s = FWD.read_text()
    # ensure copy_utils import exists
    if "from vllm.third_party.tml_fa4 import copy_utils" not in s and \
       "import copy_utils" not in s:
        anchor = "from vllm.third_party.tml_fa4 import utils"
        assert anchor in s
        s = s.replace(anchor, anchor +
                      "\nfrom vllm.third_party.tml_fa4 import copy_utils", 1)
    n = 0
    for old, new in EDITS:
        if new in s:
            continue
        assert old in s, f"anchor missing: {old[:70]!r}"
        s = s.replace(old, new, 1)
        n += 1
    FWD.write_text(s)
    print(f"v1 edits applied: {n}")


if __name__ == "__main__":
    main()
