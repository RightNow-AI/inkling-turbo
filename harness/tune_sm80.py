#!/usr/bin/env python3
"""Parity-gated tile tuning for the generic (sm_80-family) rel-bias kernel.

Upstream ships FwdConfig(128, 64) for SM80 with a literal "should tune"
comment. This sweep times every legal bias-compatible config on the real
shapes and REFUSES to report a timing whose parity run is not green
(fast-and-wrong is a failed config, per the project rules).

Legal space with bias: tile_m fixed at 128 (the shear contract bakes
128-row blocks); tile_n must divide 128 and divide padded -> {32, 64, 128}.

Output: table + JSON (tune_sm80.json) with the winner per case.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

_TILE: list = [None]  # active config, injected below the public wrapper


def _install_tile_injector() -> None:
    """The public flash_attn_varlen_func does not forward tile_mn; inject it
    at the _flash_attn_fwd layer, which owns the (tile_m, tile_n) override."""
    import vllm.third_party.tml_fa4.interface as itf

    orig = itf._flash_attn_fwd

    def patched(*args, **kwargs):
        if _TILE[0] is not None:
            kwargs["tile_mn"] = _TILE[0]
        return orig(*args, **kwargs)

    # the interface stores its compile caches as attributes on the function
    # object and reads them through the module-global name; inherit them all
    patched.__dict__.update(orig.__dict__)
    itf._flash_attn_fwd = patched


CONFIGS = [(128, 32), (128, 64), (128, 128)]
CASES = [
    # (name, T_q, T_k, Hq, Hkv, ext, window_left)
    ("prefill_global_8k", 8192, 8192, 64, 8, 1024, None),
    ("prefill_swa_8k", 8192, 8192, 64, 16, 512, 511),
    ("decode_b1_global_kv64k", 1, 65536, 64, 8, 1024, None),
    ("decode_32seqs_global_kv64k", "batched", 65536, 64, 8, 1024, None),
]
PARITY_T = 512  # parity shape per config (small, fast, still multi-block)


def make_case(T_q, T_k, Hq, Hkv, ext, window_left):
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func

    D = 128
    dev = "cuda"
    if T_q == "batched":
        B, L = 32, T_k
        q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=dev)
        k = torch.randn(B * L, Hkv, D, dtype=torch.bfloat16, device=dev)
        v = torch.randn(B * L, Hkv, D, dtype=torch.bfloat16, device=dev)
        rel = torch.randn(B, Hq, ext, dtype=torch.bfloat16, device=dev)
        cu_q = torch.arange(B + 1, dtype=torch.int32, device=dev)
        cu_k = torch.arange(B + 1, dtype=torch.int32, device=dev) * L
        max_q, max_k = 1, L
    else:
        q = torch.randn(T_q, Hq, D, dtype=torch.bfloat16, device=dev)
        k = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=dev)
        v = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=dev)
        rel = torch.randn(T_q, Hq, ext, dtype=torch.bfloat16, device=dev)
        cu_q = torch.tensor([0, T_q], dtype=torch.int32, device=dev)
        cu_k = torch.tensor([0, T_k], dtype=torch.int32, device=dev)
        max_q, max_k = T_q, T_k
    window = (None, None) if window_left is None else (window_left, 0)

    def fn():
        return flash_attn_varlen_func(
            q=q, k=k, v=v, rel_bias=rel,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max_q, max_seqlen_k=max_k,
            softmax_scale=1.0 / D, causal=True, window_size=window,
        )
    return fn


def parity_ok() -> tuple[bool, float]:
    """Small-shape oracle check for this config (global + swa)."""
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func

    D = 128
    dev = "cuda"
    torch.manual_seed(1234)
    worst = 0.0
    for Hq, Hkv, ext, win in ((8, 8, 1024, None), (8, 8, 512, 511)):
        T = PARITY_T
        q = torch.randn(T, Hq, D, dtype=torch.bfloat16, device=dev)
        k = torch.randn(T, Hkv, D, dtype=torch.bfloat16, device=dev)
        v = torch.randn(T, Hkv, D, dtype=torch.bfloat16, device=dev)
        rel = torch.randn(T, Hq, ext, dtype=torch.bfloat16, device=dev)
        cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
        window = (None, None) if win is None else (win, 0)
        out = flash_attn_varlen_func(
            q=q, k=k, v=v, rel_bias=rel, cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=T, max_seqlen_k=T, softmax_scale=1.0 / D,
            causal=True, window_size=window)
        if isinstance(out, tuple):
            out = out[0]
        # float32 oracle
        qf, kf, vf = q.float(), k.float(), v.float()
        scores = torch.einsum("qhd,khd->hqk", qf.view(T, Hq, D),
                              kf.view(T, Hkv, D).repeat_interleave(Hq // Hkv, 1)
                              .view(T, Hq, D)) / D
        i = torch.arange(T, device=dev)
        d = i[:, None] - i[None, :]
        bias = torch.zeros(Hq, T, T, device=dev)
        valid = (d >= 0) & (d < ext)
        dc = d.clamp(0, ext - 1)
        for h in range(Hq):
            bias[h] = torch.where(valid, rel.float()[i[:, None].expand(T, T),
                                                     h, dc], 0.0)
        scores += bias
        causal = d >= 0
        if win is not None:
            causal &= d <= win
        scores = scores.masked_fill(~causal[None], float("-inf"))
        ref = torch.einsum("hqk,khd->qhd", torch.softmax(scores, -1),
                           vf.view(T, Hkv, D).repeat_interleave(Hq // Hkv, 1)
                           .view(T, Hq, D))
        diff = (out.float() - ref).abs().max().item()
        worst = max(worst, diff)

    # The cases above all use ONE cu_seqlens for both q and k, so every one of
    # them has seqlen_q == seqlen_k. The cases this file TIMES do not: they are
    # decode shapes, T_q=1 against T_k=65536. So the gate that is supposed to
    # stop a timing from being reported without green parity was checking a
    # different shape family from the one it timed, and the generic kernel's
    # shear shift was wrong on exactly the family it never checked. The Ampere
    # tile-sweep percentages were selected under that hole.
    # See journal/regression-sm90-bias-shift.md.
    #
    # Cover the timed family here. The reference lives in
    # parity_rel_chunked_decode because it needs a query that starts partway
    # into the sequence, and duplicating it invites the two copies to drift.
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func
    import parity_rel_chunked_decode as prcd

    for T_q, ctx, Hq, Hkv, ext, win in (
        (1, 4095, 8, 8, 1024, None),
        (1, 4095, 8, 8, 512, 511),
        (128, 1408, 8, 8, 1024, None),
    ):
        torch.manual_seed(4321)
        T_k = ctx + T_q
        q = torch.randn(T_q, Hq, D, dtype=torch.bfloat16, device=dev)
        k = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=dev)
        v = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=dev)
        rel = torch.randn(T_q, Hq, ext, dtype=torch.bfloat16, device=dev)
        cu_q = torch.tensor([0, T_q], dtype=torch.int32, device=dev)
        cu_k = torch.tensor([0, T_k], dtype=torch.int32, device=dev)
        window = (None, None) if win is None else (win, 0)
        out = flash_attn_varlen_func(
            q=q, k=k, v=v, rel_bias=rel,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=T_q, max_seqlen_k=T_k, softmax_scale=1.0 / D,
            causal=True, window_size=window)
        if isinstance(out, tuple):
            out = out[0]
        ref = prcd.reference(q, k, v, rel, 1.0 / D, ctx, win)
        diff = (out.float() - ref.float()).abs().max().item()
        print(f"    parity seqlen_q!=seqlen_k T_q={T_q} ctx={ctx} "
              f"ext={ext} win={win}: max={diff:.3e}")
        worst = max(worst, diff)

    return worst < 2e-2, worst


def time_case(fn, iters=15, warmup=5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters  # us


def main() -> None:
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"capability {torch.cuda.get_device_capability(0)}")
    _install_tile_injector()
    results: dict = {}
    for tile_mn in CONFIGS:
        _TILE[0] = tile_mn
        ok, worst = parity_ok()
        tag = f"tile{tile_mn[0]}x{tile_mn[1]}"
        print(f"\n[{tag}] parity max_diff={worst:.3e} "
              f"{'OK' if ok else 'FAIL -> timings SUPPRESSED'}")
        if not ok:
            results[tag] = {"parity": worst, "timings": None}
            continue
        timings = {}
        for name, *shape in CASES:
            try:
                us = time_case(make_case(*shape))
                timings[name] = round(us, 1)
                print(f"  {name}: {us:.1f} us")
            except Exception as exc:  # noqa: BLE001
                timings[name] = f"ERROR: {str(exc)[:80]}"
                print(f"  {name}: ERROR {str(exc)[:80]}")
        results[tag] = {"parity": worst, "timings": timings}

    best = {}
    for name, *_ in CASES:
        cands = [(cfg, r["timings"][name]) for cfg, r in results.items()
                 if r["timings"] and isinstance(r["timings"].get(name), float)]
        if cands:
            best[name] = min(cands, key=lambda kv: kv[1])
    print("\n=== winners ===")
    for name, (cfg, us) in best.items():
        print(f"  {name}: {cfg} @ {us} us")
    # Next to this script, with the capability in the name, matching every other
    # harness here. It used to write to Path.home(), which meant a remote runner
    # collecting artifacts from the harness directory never found it, and the
    # compute capability was nowhere in the file, so an sm_80 result and an
    # sm_120 result were indistinguishable once copied.
    cc = torch.cuda.get_device_capability(0)
    out = Path(__file__).with_name(f"tune_sm80_sm{cc[0]}{cc[1]}.json")
    out.write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "results": results,
        "best": best,
    }, indent=1))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
