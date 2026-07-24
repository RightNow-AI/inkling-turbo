# SPDX-License-Identifier: Apache-2.0
"""U2-Hopper Design B, V1: register-resident relative bias as a score_mod.

Replaces the day-0 gather `rel_logits[q, h, dist]` (aux tensor (T, H, ext)
bf16, zero reuse, gmem-latency-bound, measured 3.2x overhead at kv64k
decode, journal/remote/h100-session1.md session 4) with an inline
16-term dot product:

    bias(q, h, k) = sum_d r[q, h, d] * proj[d, dist],  dist = q - k

aux_tensors = [r (T, H, 16) bf16, proj (16, ext) bf16]. proj is 32KB total
(ext=1024) with T*H/ext-fold reuse -> L1-resident; r is 32B per row reused
across every attended kv. No rel_logits materialization, no shear kernel.

Log scaling: fold tau into r upstream (r' = r * tau[token]), same math as
day-0's rel_logits *= tau, different bf16 rounding path (documented; parity
gate judges).

V2 (only if V1 misses <=1.1x plain-attn): hoist r into registers per-row and
smem-stage proj inside apply_score_mod_inner.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

D_REL = 16


@cache
def get_relproj_score_mod(rel_extent: int) -> Callable:
    """V1: proj stored (16, ext), per-element column walk (strided loads)."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32

    from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK

    @cute.jit
    def score_mod_relproj(
        scores: cute.TensorSSA,
        b_idx: cute.TensorSSA,
        h_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info: SeqlenInfoQK,
        aux_tensors: list[cute.Tensor]) -> cute.TensorSSA:
        r = aux_tensors[0]      # (total_q, H, 16)
        proj = aux_tensors[1]   # (16, rel_extent)

        seqlen_local_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        rel_dist = (q_idx + seqlen_local_offset) - kv_idx
        global_q_idx = seqlen_info.offset_q + q_idx

        d0 = rel_dist[0]
        d_clamped = d0 if d0 >= 0 else 0
        d_clamped = d_clamped if d_clamped < rel_extent else (rel_extent - 1)

        acc = Float32(0.0)
        for dd in cutlass.range_constexpr(D_REL):
            acc += Float32(r[global_q_idx[0], h_idx[0], dd]) * Float32(
                proj[dd, d_clamped]
            )
        bias = acc if d0 == d_clamped else Float32(0.0)
        return scores + bias

    return score_mod_relproj


@cache
def get_relproj_score_mod_v15(rel_extent: int) -> Callable:
    """V1.5: proj stored TRANSPOSED (ext, 16), the 16 loads per element are
    contiguous (32B span; compiler can merge into vector loads). Tests the
    load-coalescing half of the V1 diagnosis with zero kernel plumbing."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32

    from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK

    @cute.jit
    def score_mod_relproj_t(
        scores: cute.TensorSSA,
        b_idx: cute.TensorSSA,
        h_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info: SeqlenInfoQK,
        aux_tensors: list[cute.Tensor]) -> cute.TensorSSA:
        r = aux_tensors[0]      # (total_q, H, 16)
        projT = aux_tensors[1]  # (rel_extent, 16) contiguous rows

        seqlen_local_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        rel_dist = (q_idx + seqlen_local_offset) - kv_idx
        global_q_idx = seqlen_info.offset_q + q_idx

        d0 = rel_dist[0]
        d_clamped = d0 if d0 >= 0 else 0
        d_clamped = d_clamped if d_clamped < rel_extent else (rel_extent - 1)

        acc = Float32(0.0)
        for dd in cutlass.range_constexpr(D_REL):
            acc += Float32(r[global_q_idx[0], h_idx[0], dd]) * Float32(
                projT[d_clamped, dd]
            )
        bias = acc if d0 == d_clamped else Float32(0.0)
        return scores + bias

    return score_mod_relproj_t
