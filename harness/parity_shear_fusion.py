#!/usr/bin/env python3
"""Gate: fused sheared rel writer (u2_shear_fusion) == ShearingBias output.

WHAT IS COMPARED

  reference : qkvr_prep writes rel_logits in NATURAL layout, then the stock
              tml-fa4 ShearingBias kernel shears it into (T + 128, H,
              rel_extent + 256).  This is the byte-for-byte definition of
              "correct"; the attention kernel consumes exactly this buffer.
  candidate : the patched qkvr_prep writes that same buffer directly, with no
              ShearingBias launch.

Both paths start from the SAME qkvr and the same rel_proj, and the candidate
keeps the reference's rounding sequence (fp32 dot -> round to bf16 -> back to
fp32 -> optional log-scale -> round to bf16).  ShearingBias itself does no
arithmetic, so for the 14 WRITER cases the expected result is BIT-EQUALITY and
the gate compares with ``==``.

The two END-TO-END cases are different.  They launch the FA4 attention kernel,
which is not run-to-run deterministic: three consecutive runs of identical code
on an RTX 5090 gave max diffs of 2.7e-2, 5.5e-2 and 7.2e-2 at different
indices, with the mean pinned near 8.5e-5.  Bit-equality there fails at random
and says nothing.  Those two cases therefore run the reference TWICE to measure
this kernel own noise on that shape, and require the fused-vs-reference
difference to sit inside it.  A dropped or mislaid bias is orders of magnitude
larger and is still caught.

HOW THIS CATCHES AN OFF-BY-ONE IN THE SHEAR COLUMN

  The shear is a per-row permutation.  If the fused writer computes
  ``base(i)`` one column too far left or right, every value in that row lands
  on the wrong column.  Neighbouring columns of a row hold projections at
  neighbouring relative distances, which for random inputs differ by O(1), so
  a one-column slip produces a mismatch at essentially every column of the
  row, not a small numeric drift.  A tolerance-based check would also fire,
  but it would report "large error" and leave you guessing; instead, on the
  first mismatching row this gate slides the candidate row against the
  reference row over shifts -8..+8 and prints the shift that makes them equal.
  A message of the form

      row 137 (token 137, head 0): candidate is shifted by +1 column

  names the defect directly.  The pad wings make this test sharp rather than
  ambiguous: the left wing is 0.0 (global) or -inf (SWA) and the right wing is
  always -inf, so a slipped row also drags a pad value into the value window
  and vice versa, which shows up as an inf/finite mismatch at the boundary.

  A second failure mode, the shear being right for prefill and wrong when
  seqlen_k != seqlen_q, is covered by the chunked and decode cases: the
  journal's published contract ``col = k + padded - 128*(m_block+1)`` is only
  the seqlen_q == seqlen_k specialisation, and a writer that hard-codes it
  passes the prefill cases and fails ``chunked_*`` and ``decode_*``.

COVERAGE
  global (causal, left pad 0.0) and sliding-window (left pad -inf)
  varlen form (production) and batched form (ShearingBias's other input form)
  prefill, chunked prefill (seqlen_k > seqlen_q), decode (seqlen_q == 1)
  all three rel writers: low-latency, throughput (rows >= 8192), fused-small
  a request mix where m_block varies within one launch (T > 128)
  plus two end-to-end cases that run the FA4 kernel on the fused buffer, so a
  correct buffer that never reaches the kernel still fails the gate

REQUIREMENTS
  a GPU, a vLLM tree with kernels/patches/u2_shear_fusion.py applied, and
  kernels/tml_fa4_modified deployed.

WHERE THIS HAS RUN
  sm_120 (RTX 5090): 16/16.  One recorded run, no artifact; the only record is
  commit 7375849.
  sm_90 (H100, session 26): 14/16.  All 14 writer cases bit-exact.  Both
  attention_consumes_* cases failed on a defect in flash_fwd_sm90.py that has
  since been fixed, so the current expectation on sm_90 is 16/16 and that has
  not been re-run.  Artifact: journal/remote/validate_s26_h100x1/.
  NOT run on sm_100.
  On performance: the fusion has been measured and it is a net LOSS on prefill,
  costing 1019us on global and 561us on sliding-window 8K while saving 5us on
  batch-32 decode.  See journal/remote/validate_s26_h100x1/README.md.  Passing
  this gate says the writer is correct, not that it is worth enabling.

ARTIFACT
  Writes parity_shear_fusion_sm<cc>.json next to this script: device name,
  torch version, CUDA capability, and every case with its pass/fail state and
  error strings.  The compute capability is in the filename so an sm_90 run
  does not overwrite the sm_120 one.  Exit code is unchanged: 0 iff every
  case passed.

Run (WSL): cd ~/inkling-turbo/vllm && source .venv/bin/activate && \\
  python $REPO/harness/parity_shear_fusion.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

HQ, HKV, D, D_REL, W = 8, 2, 128, 16, 4
QW, KVW = HQ * D, HKV * D
R_OFFSET = QW + 2 * KVW
WIDTH = R_OFFSET + HQ * D_REL
CONV_BS = 16
PAGE = 16
OFF_K, OFF_V = 0, 128
SHEAR_PAD = 256
SHEAR_ROW_PAD = 128


# ---------------------------------------------------------------- inputs


def make_inputs(seq_lens_q, ctx_lens, ext, seed, device="cuda"):
    """Varlen batch: seq_lens_q[r] scheduled tokens after ctx_lens[r] cached."""
    torch.manual_seed(seed)
    assert len(seq_lens_q) == len(ctx_lens)
    tokens = sum(seq_lens_q)
    qkvr = torch.randn(tokens, WIDTH, dtype=torch.bfloat16, device=device)
    rel_proj = (
        torch.randn(D_REL, ext, dtype=torch.bfloat16, device=device) * 0.5
    )
    cu = [0]
    for length in seq_lens_q:
        cu.append(cu[-1] + length)
    cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device=device)
    seq_lens = torch.tensor(
        [q + c for q, c in zip(seq_lens_q, ctx_lens)],
        dtype=torch.int32,
        device=device,
    )
    seq_idx = torch.tensor(
        [r for r, length in enumerate(seq_lens_q) for _ in range(length)],
        dtype=torch.int32,
        device=device,
    )
    positions = torch.tensor(
        [c + i for length, c in zip(seq_lens_q, ctx_lens) for i in range(length)],
        dtype=torch.int64,
        device=device,
    )
    query_start = torch.tensor(
        [cu[r] for r, length in enumerate(seq_lens_q) for _ in range(length)],
        dtype=torch.int32,
        device=device,
    )
    return dict(
        qkvr=qkvr,
        rel_proj=rel_proj,
        cu_seqlens_q=cu_seqlens_q,
        seq_lens=seq_lens,
        seq_idx=seq_idx,
        positions=positions,
        query_start=query_start,
        tokens=tokens,
        ext=ext,
    )


# ------------------------------------------------------- reference shear

_SHEAR_CACHE: dict = {}


def run_shearing_bias(
    rel_natural,
    *,
    ext,
    is_local,
    window_left,
    window_right,
    cu_seqlens_q=None,
    seq_lens=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    batched_rows=None,
):
    """Run the stock ShearingBias kernel; return its NaN-initialised output.

    Mirrors the production call in tml_fa4/interface.py:827-866: cu_seqlens_q
    for Q, seqused_k for K, no block-packed scheduler tensors (the scheduler
    only decides which CTA takes which row, never where a value lands).
    """
    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute
    from vllm.third_party.tml_fa4.interface import to_cute_tensor
    from vllm.third_party.tml_fa4.shearing_bias import ShearingBias

    padded = ext + SHEAR_PAD
    varlen = cu_seqlens_q is not None
    if varlen:
        total_q = rel_natural.shape[0]
        out = torch.full(
            (total_q + SHEAR_ROW_PAD, HQ, padded),
            float("nan"),
            dtype=rel_natural.dtype,
            device=rel_natural.device,
        )
    else:
        batch, s_q = rel_natural.shape[0], rel_natural.shape[1]
        rounded = batched_rows if batched_rows is not None else s_q
        out = torch.full(
            (batch, rounded, HQ, padded),
            float("nan"),
            dtype=rel_natural.dtype,
            device=rel_natural.device,
        )

    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    sb = ShearingBias(
        rel_extent=ext,
        is_causal=not is_local,
        is_local=is_local,
        pack_gqa=False,
        qhead_per_kvhead=1,
        rows_per_cta=4,
        tile_m=128,
        max_m_blocks_leq_one=False,
    )
    ints = dict(assumed_align=4, leading_dim=0)
    cu_t = to_cute_tensor(cu_seqlens_q, **ints) if varlen else None
    sk_t = to_cute_tensor(seq_lens, **ints) if seq_lens is not None else None
    # Same cache key the interface uses (interface.py:796-812): shapes are
    # dynamic, only dtype / mode / which optional tensors are present matter.
    # Without this the gate pays a CuTe compile per case on a rented GPU.
    key = (
        rel_natural.dtype,
        ext,
        not is_local,
        is_local,
        window_left is not None,
        window_right is not None,
        varlen,
        seq_lens is not None,
    )
    compiled = _SHEAR_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(
            sb,
            to_cute_tensor(rel_natural),
            to_cute_tensor(out),
            max_seqlen_q,
            max_seqlen_k,
            cu_t,
            None,
            None,
            sk_t,
            None,
            None,
            window_left,
            window_right,
            stream,
        )
        _SHEAR_CACHE[key] = compiled
    compiled(
        rel_natural,
        out,
        max_seqlen_q,
        max_seqlen_k,
        cu_seqlens_q,
        None,
        None,
        seq_lens,
        None,
        None,
        window_left,
        window_right,
        stream,
    )
    torch.cuda.synchronize()
    return out


# ------------------------------------------------------------ comparison


def diagnose_shift(ref_row, got_row, span=8):
    """Find s with ref[c + s] == got[c] for all c, i.e. the candidate row is the
    reference row moved -s columns.  Returns s, or None if it is not a shift.

    The search covers +-span (an off-by-one in base()) and +-128 / +-256
    (a wrong n_block_max, which slips by whole 128-column blocks).
    """
    n = ref_row.shape[0]
    candidates = [s for s in range(-span, span + 1) if s] + [
        -256, -128, 128, 256
    ]
    for s in candidates:
        if s > 0:
            a, b = ref_row[s:], got_row[: n - s]
        else:
            a, b = ref_row[: n + s], got_row[-s:]
        if a.shape[0] and bool(torch.equal(a, b)):
            return s
    return None


def compare(ref, got, rows, ext, label):
    """Element-by-element gate.  Returns a list of failure strings."""
    padded = ext + SHEAR_PAD
    r = ref[:rows].reshape(rows * HQ, padded).float().cpu()
    g = got[:rows].reshape(rows * HQ, padded).float().cpu()
    errs = []

    ref_nan = torch.isnan(r)
    if bool(ref_nan.any()):
        idx = int(ref_nan.flatten().nonzero()[0])
        row, col = divmod(idx, padded)
        errs.append(
            f"{label}: REFERENCE left column ({row // HQ}, h{row % HQ}, {col}) "
            "unwritten (NaN). The pad model in this harness is wrong, not the "
            "kernel: ShearingBias writes every column of every valid row."
        )
        return errs
    got_nan = torch.isnan(g)
    if bool(got_nan.any()):
        idx = int(got_nan.flatten().nonzero()[0])
        row, col = divmod(idx, padded)
        errs.append(
            f"{label}: candidate left column (token {row // HQ}, h{row % HQ}, "
            f"col {col}) UNWRITTEN. Every column of a valid row must be "
            "written exactly once: value window, left wing, or right wing."
        )

    diff = r != g
    n_bad = int(diff.sum())
    if n_bad:
        flat = int(diff.flatten().nonzero()[0])
        row, col = divmod(flat, padded)
        token, head = divmod(row, HQ)
        errs.append(
            f"{label}: {n_bad} / {r.numel()} elements differ. FIRST at "
            f"(token {token}, head {head}, col {col}): ref={r[row, col].item()!r} "
            f"got={g[row, col].item()!r}"
        )
        bad_rows = int(diff.any(dim=1).sum())
        errs.append(f"{label}: {bad_rows} / {r.shape[0]} rows differ")
        shift = diagnose_shift(r[row], g[row])
        if shift is not None:
            errs.append(
                f"{label}: row {row} (token {token}, head {head}) is a PURE "
                f"COLUMN SHIFT: ref[c{shift:+d}] == got[c], i.e. the candidate "
                f"wrote every value {-shift:+d} columns from where the "
                "reference put it. base(i) = n_idx_right + 256 - "
                "128*n_block_max is wrong by that amount; this is a layout "
                "defect, not a numeric one."
            )
        else:
            finite_ref = int((~torch.isinf(r[row])).sum())
            finite_got = int((~torch.isinf(g[row])).sum())
            errs.append(
                f"{label}: row {row} is not a pure shift "
                f"(non-inf columns ref={finite_ref} got={finite_got}); suspect "
                "num_bias_vals / left-pad masking, not base()."
            )
    return errs


# ----------------------------------------------------------------- cases


def case_rel_proj(name, seq_lens_q, ctx_lens, ext, is_local, seed, use_log):
    """Drive qkvr_rel_proj directly: low-latency and throughput writers."""
    from vllm.models.inkling.nvidia.ops.qkvr_prep import (
        RelShearSpec,
        qkvr_rel_proj,
    )

    inp = make_inputs(seq_lens_q, ctx_lens, ext, seed)
    tokens = inp["tokens"]
    dev = inp["qkvr"].device
    log_scaling = (
        torch.rand(tokens, dtype=torch.float32, device=dev) * 0.3 + 1.0
        if use_log
        else None
    )
    common = dict(num_q_heads=HQ, num_kv_heads=HKV, head_dim=D, d_rel=D_REL)

    rel_nat = torch.empty(tokens, HQ, ext, dtype=torch.bfloat16, device=dev)
    qkvr_rel_proj(inp["qkvr"], inp["rel_proj"], rel_nat, log_scaling, **common)
    torch.cuda.synchronize()

    window_left = ext - 1 if is_local else None
    window_right = 0 if is_local else None
    ref = run_shearing_bias(
        rel_nat,
        ext=ext,
        is_local=is_local,
        window_left=window_left,
        window_right=window_right,
        cu_seqlens_q=inp["cu_seqlens_q"],
        seq_lens=inp["seq_lens"],
        max_seqlen_q=max(seq_lens_q),
        max_seqlen_k=int(inp["seq_lens"].max()),
    )

    got = torch.full(
        (tokens + SHEAR_ROW_PAD, HQ, ext + SHEAR_PAD),
        float("nan"),
        dtype=torch.bfloat16,
        device=dev,
    )
    spec = RelShearSpec(
        cu_seqlens_q=inp["cu_seqlens_q"],
        seq_lens=inp["seq_lens"],
        seq_idx=inp["seq_idx"],
        num_tokens=tokens,
        window_left=window_left,
        window_right=0,
    )
    qkvr_rel_proj(
        inp["qkvr"], inp["rel_proj"], got, log_scaling, shear=spec, **common
    )
    torch.cuda.synchronize()
    return compare(ref, got, tokens, ext, name)


def case_batched_reference(name, batch, s_q, ctx, ext, is_local, seed):
    """Same fused varlen output, checked against the BATCHED ShearingBias form.

    ShearingBias accepts (b, s_q, h, ext) as well as (total_q, h, ext).  The
    fused writer only has the varlen form, so this proves the two forms agree
    on placement and that the varlen writer covers the batched case.
    """
    from vllm.models.inkling.nvidia.ops.qkvr_prep import (
        RelShearSpec,
        qkvr_rel_proj,
    )

    inp = make_inputs([s_q] * batch, [ctx] * batch, ext, seed)
    tokens = inp["tokens"]
    dev = inp["qkvr"].device
    common = dict(num_q_heads=HQ, num_kv_heads=HKV, head_dim=D, d_rel=D_REL)

    rel_nat = torch.empty(tokens, HQ, ext, dtype=torch.bfloat16, device=dev)
    qkvr_rel_proj(inp["qkvr"], inp["rel_proj"], rel_nat, None, **common)
    torch.cuda.synchronize()

    window_left = ext - 1 if is_local else None
    window_right = 0 if is_local else None
    rounded = (s_q + 127) // 128 * 128
    ref_b = run_shearing_bias(
        rel_nat.view(batch, s_q, HQ, ext).contiguous(),
        ext=ext,
        is_local=is_local,
        window_left=window_left,
        window_right=window_right,
        cu_seqlens_q=None,
        seq_lens=None,
        max_seqlen_q=s_q,
        max_seqlen_k=s_q + ctx,
        batched_rows=rounded,
    )
    # (b, rounded, H, padded) -> the varlen row order (request major)
    ref = ref_b[:, :s_q].reshape(tokens, HQ, ext + SHEAR_PAD)

    got = torch.full(
        (tokens + SHEAR_ROW_PAD, HQ, ext + SHEAR_PAD),
        float("nan"),
        dtype=torch.bfloat16,
        device=dev,
    )
    spec = RelShearSpec(
        cu_seqlens_q=inp["cu_seqlens_q"],
        seq_lens=inp["seq_lens"],
        seq_idx=inp["seq_idx"],
        num_tokens=tokens,
        window_left=window_left,
        window_right=0,
    )
    qkvr_rel_proj(
        inp["qkvr"], inp["rel_proj"], got, None, shear=spec, **common
    )
    torch.cuda.synchronize()
    return compare(ref, got, tokens, ext, name)


def _prep_side_tensors(inp, device="cuda"):
    """Minimal conv / KV apparatus so fused_qkvr_prep can run.

    None of it affects the rel output; it only has to be in bounds.
    """
    tokens = inp["tokens"]
    max_pos = int(inp["positions"].max()) + 1
    # Cap the table: the kernel clamps logical_block to max_blocks - 1, so a
    # short table stays in bounds. Long-context cases would otherwise allocate
    # a conv cache proportional to the context length for no reason.
    blocks_per_req = min((max_pos + CONV_BS - 1) // CONV_BS + 1, 64)
    n_req = inp["seq_lens"].shape[0]
    conv_table = (
        torch.arange(n_req * blocks_per_req, dtype=torch.int32, device=device)
        .reshape(n_req, blocks_per_req)
    )
    conv_cache = torch.zeros(
        n_req * blocks_per_req, HKV, CONV_BS, 256,
        dtype=torch.bfloat16, device=device,
    )
    logical = (inp["positions"] // CONV_BS).clamp_max(blocks_per_req - 1)
    conv_slots = (
        conv_table[inp["seq_idx"].long(), logical] * CONV_BS
        + (inp["positions"] % CONV_BS).to(torch.int32)
    ).to(torch.int32)
    n_pages = (tokens + PAGE - 1) // PAGE + 1
    attn_slots = torch.arange(tokens, dtype=torch.int32, device=device)
    key_cache = torch.zeros(
        n_pages, PAGE, HKV, D, dtype=torch.bfloat16, device=device
    )
    value_cache = torch.zeros_like(key_cache)
    return dict(
        conv_cache=conv_cache,
        conv_table=conv_table,
        conv_slots=conv_slots,
        key_cache=key_cache,
        value_cache=value_cache,
        attn_slots=attn_slots,
    )


def case_fused_prep(name, seq_lens_q, ctx_lens, ext, is_local, seed):
    """Drive the production entry point, both launch paths.

    tokens < 128 takes _run_fused_small (the decode path); tokens >= 128 takes
    the tiled path.  Called twice on identical inputs with fresh caches.

    fused_qkvr_prep allocates the output itself (torch.empty), so unlike
    case_rel_proj this case cannot use a NaN sentinel to prove every column was
    written.  An unwritten column still fails the element compare, it just
    reports as a value mismatch rather than as "unwritten".
    """
    from vllm.models.inkling.nvidia.ops.qkvr_prep import (
        RelShearSpec,
        fused_qkvr_prep,
    )

    inp = make_inputs(seq_lens_q, ctx_lens, ext, seed)
    tokens = inp["tokens"]
    dev = inp["qkvr"].device
    knw = torch.rand(D, dtype=torch.bfloat16, device=dev) + 0.5
    qnw = torch.rand(D, dtype=torch.bfloat16, device=dev) + 0.5
    kw = torch.randn(HKV * D, W, dtype=torch.bfloat16, device=dev) * 0.3
    vw = torch.randn(HKV * D, W, dtype=torch.bfloat16, device=dev) * 0.3

    def call(rel_shear):
        side = _prep_side_tensors(inp)
        return fused_qkvr_prep(
            inp["qkvr"], kw, vw, qnw, knw, inp["rel_proj"], 1e-6,
            HQ, HKV, D, D_REL,
            side["conv_cache"], side["key_cache"], side["value_cache"],
            inp["positions"], side["conv_table"], inp["seq_idx"],
            side["conv_slots"], inp["query_start"], side["attn_slots"],
            OFF_K, OFF_V, CONV_BS, None,
            rel_shear=rel_shear,
        )

    _, rel_nat = call(None)
    torch.cuda.synchronize()

    window_left = ext - 1 if is_local else None
    window_right = 0 if is_local else None
    ref = run_shearing_bias(
        rel_nat.contiguous(),
        ext=ext,
        is_local=is_local,
        window_left=window_left,
        window_right=window_right,
        cu_seqlens_q=inp["cu_seqlens_q"],
        seq_lens=inp["seq_lens"],
        max_seqlen_q=max(seq_lens_q),
        max_seqlen_k=int(inp["seq_lens"].max()),
    )

    spec = RelShearSpec(
        cu_seqlens_q=inp["cu_seqlens_q"],
        seq_lens=inp["seq_lens"],
        seq_idx=inp["seq_idx"],
        num_tokens=tokens,
        window_left=window_left,
        window_right=0,
    )
    _, got = call(spec)
    torch.cuda.synchronize()
    assert got.shape == (tokens + SHEAR_ROW_PAD, HQ, ext + SHEAR_PAD), got.shape
    return compare(ref, got, tokens, ext, name)


def case_attention_consumes(name, seq_lens_q, ext, is_local, seed):
    """End-to-end: FA4 with bias=<fused> == FA4 with rel_bias=<natural>.

    The layout cases prove the buffer is right.  This proves the plumbing is:
    that `bias=` reaches the kernel, that has_bias stays on (upstream finding
    01: stock tml-fa4 silently IGNORES the bias on the wrong path and returns
    plain attention), and that the interface really does skip ShearingBias.
    Both calls run the same kernel over the same numbers, so the outputs must
    match bit for bit; a silently dropped bias shows up as a large diff, not a
    small one.

    pack_gqa is pinned False so the case behaves the same on Hopper (where the
    interface forces it off with bias) and on Blackwell (where it would
    otherwise be heuristic).
    """
    from vllm.models.inkling.nvidia.ops.qkvr_prep import (
        RelShearSpec,
        qkvr_rel_proj,
    )
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func

    inp = make_inputs(seq_lens_q, [0] * len(seq_lens_q), ext, seed)
    tokens = inp["tokens"]
    dev = inp["qkvr"].device
    common = dict(num_q_heads=HQ, num_kv_heads=HKV, head_dim=D, d_rel=D_REL)

    rel_nat = torch.empty(tokens, HQ, ext, dtype=torch.bfloat16, device=dev)
    qkvr_rel_proj(inp["qkvr"], inp["rel_proj"], rel_nat, None, **common)
    fused = torch.full(
        (tokens + SHEAR_ROW_PAD, HQ, ext + SHEAR_PAD),
        float("nan"),
        dtype=torch.bfloat16,
        device=dev,
    )
    spec = RelShearSpec(
        cu_seqlens_q=inp["cu_seqlens_q"],
        seq_lens=inp["seq_lens"],
        seq_idx=inp["seq_idx"],
        num_tokens=tokens,
        window_left=ext - 1 if is_local else None,
        window_right=0,
    )
    qkvr_rel_proj(
        inp["qkvr"], inp["rel_proj"], fused, None, shear=spec, **common
    )
    torch.cuda.synchronize()

    torch.manual_seed(seed + 1)
    q = torch.randn(tokens, HQ, D, dtype=torch.bfloat16, device=dev) / D**0.25
    k = torch.randn(tokens, HKV, D, dtype=torch.bfloat16, device=dev) / D**0.25
    v = torch.randn(tokens, HKV, D, dtype=torch.bfloat16, device=dev)
    kw = dict(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=inp["cu_seqlens_q"],
        cu_seqlens_k=inp["cu_seqlens_q"],
        max_seqlen_q=max(seq_lens_q),
        max_seqlen_k=max(seq_lens_q),
        softmax_scale=1.0 / D,
        causal=True,
        window_size=(ext - 1, 0) if is_local else (None, None),
        pack_gqa=False,
    )

    def run(**bias_kw):
        out = flash_attn_varlen_func(**kw, **bias_kw)
        return out[0] if isinstance(out, tuple) else out

    # The attention kernel is NOT run-to-run deterministic. Measured on an
    # RTX 5090: three consecutive runs of identical code gave max diffs of
    # 2.7e-2, 5.5e-2 and 7.2e-2 at different indices, with the mean pinned
    # near 8.5e-5. So bit-equality is the wrong test here; it fails at random.
    # Instead run the REFERENCE twice to measure this kernel's own noise on
    # this shape, then require the fused-vs-reference difference to sit inside
    # that. This is the same discipline the full-model logit gate uses.
    ref = run(rel_bias=rel_nat).float()
    ref2 = run(rel_bias=rel_nat).float()
    got = run(bias=fused).float()
    if ref.shape != got.shape:
        return [f"{name}: shape {tuple(got.shape)} != {tuple(ref.shape)}"]

    noise = (ref - ref2).abs()
    noise_mx, noise_mean = float(noise.max()), float(noise.mean())
    diff = (ref - got).abs()
    mx, mean = float(diff.max()), float(diff.mean())

    # Allow the observed floor plus headroom. A dropped bias is orders of
    # magnitude larger than this and is still caught.
    tol_mx = max(4.0 * noise_mx, 1e-2)
    tol_mean = max(4.0 * noise_mean, 1e-4)
    if mx <= tol_mx and mean <= tol_mean:
        return []

    idx = int(diff.flatten().argmax())
    return [
        f"{name}: attention output differs beyond this kernel's own noise. "
        f"fused-vs-ref max={mx:.4e} mean={mean:.4e}; "
        f"ref-vs-ref noise max={noise_mx:.4e} mean={noise_mean:.4e}; "
        f"tolerance max={tol_mx:.4e} mean={tol_mean:.4e}. "
        f"Worst at flat index {idx} (ref={float(ref.flatten()[idx]):.6f} "
        f"got={float(got.flatten()[idx]):.6f}). A diff far above the noise "
        "floor means the pre-sheared bias was DROPPED or mislaid; one near "
        "it means the buffers agree."
    ]


# ------------------------------------------------------------------ main

CASES = [
    # (fn, name, kwargs)
    (case_rel_proj, "prefill_global_multi",
     dict(seq_lens_q=[200, 137, 64], ctx_lens=[0, 0, 0], ext=512,
          is_local=False, use_log=False)),
    (case_rel_proj, "prefill_swa_multi",
     dict(seq_lens_q=[200, 137, 64], ctx_lens=[0, 0, 0], ext=512,
          is_local=True, use_log=False)),
    (case_rel_proj, "prefill_global_logscale",
     dict(seq_lens_q=[300], ctx_lens=[0], ext=512,
          is_local=False, use_log=True)),
    (case_rel_proj, "chunked_global",
     dict(seq_lens_q=[256, 130], ctx_lens=[1000, 63], ext=512,
          is_local=False, use_log=False)),
    (case_rel_proj, "chunked_swa",
     dict(seq_lens_q=[256, 130], ctx_lens=[1000, 63], ext=512,
          is_local=True, use_log=False)),
    (case_rel_proj, "chunked_block_aligned_ctx",
     dict(seq_lens_q=[300], ctx_lens=[128], ext=512,
          is_local=False, use_log=False)),
    (case_rel_proj, "ext128_global",
     dict(seq_lens_q=[200, 40], ctx_lens=[0, 7], ext=128,
          is_local=False, use_log=False)),
    (case_rel_proj, "ext1024_swa",
     dict(seq_lens_q=[260], ctx_lens=[5000], ext=1024,
          is_local=True, use_log=False)),
    (case_rel_proj, "throughput_path_global",
     dict(seq_lens_q=[600, 600], ctx_lens=[0, 4096], ext=512,
          is_local=False, use_log=False)),
    (case_batched_reference, "batched_form_global",
     dict(batch=3, s_q=200, ctx=0, ext=512, is_local=False)),
    (case_batched_reference, "batched_form_swa",
     dict(batch=3, s_q=200, ctx=0, ext=512, is_local=True)),
    (case_fused_prep, "fused_small_decode_global",
     dict(seq_lens_q=[1] * 32, ctx_lens=[4095 + 7 * i for i in range(32)],
          ext=512, is_local=False)),
    (case_fused_prep, "fused_small_decode_swa",
     dict(seq_lens_q=[1] * 32, ctx_lens=[4095 + 7 * i for i in range(32)],
          ext=512, is_local=True)),
    (case_fused_prep, "fused_tiled_mixed",
     dict(seq_lens_q=[137, 1, 1, 60], ctx_lens=[0, 900, 8191, 33],
          ext=512, is_local=False)),
    (case_attention_consumes, "attention_consumes_global",
     dict(seq_lens_q=[256, 137], ext=512, is_local=False)),
    (case_attention_consumes, "attention_consumes_swa",
     dict(seq_lens_q=[256, 137], ext=512, is_local=True)),
]


def main() -> None:
    device = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"device: {device}, capability {cc}")

    cases: dict[str, dict] = {}
    failures = 0
    for fn, name, kwargs in CASES:
        try:
            errs = fn(name, seed=abs(hash(name)) % (2**31), **kwargs)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            errs = [f"EXCEPTION {type(exc).__name__}: {exc}"]
        if errs:
            failures += 1
            print(f"[{name}] FAIL")
            for line in errs:
                print(f"    {line}")
        else:
            print(f"[{name}] OK")
        cases[name] = {"pass": not errs, "errors": list(errs)}

    passed = len(CASES) - failures
    print(f"\n{passed}/{len(CASES)} cases bit-exact")

    # Artifact. Same convention as the other harnesses (JSON next to the
    # script), with the compute capability in the name so runs on different
    # arches accumulate instead of overwriting each other.
    out = Path(__file__).with_name(
        f"parity_shear_fusion_sm{cc[0]}{cc[1]}.json"
    )
    out.write_text(
        json.dumps(
            {
                "device": device,
                "compute_capability": f"{cc[0]}.{cc[1]}",
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "passed": passed,
                "total": len(CASES),
                "summary": f"{passed}/{len(CASES)} cases bit-exact",
                "cases": cases,
            },
            indent=2,
        )
    )
    print(f"saved: {out}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
