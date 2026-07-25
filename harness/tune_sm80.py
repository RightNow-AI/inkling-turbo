#!/usr/bin/env python3
"""Parity-gated tile tuning for the generic (sm_80-family) rel-bias kernel.

Upstream ships FwdConfig(128, 64) for SM80 with a literal "should tune"
comment. This sweep times every legal bias-compatible config on the real
shapes and REFUSES to report a timing whose parity run is not green
(fast-and-wrong is a failed config, per the project rules).

Legal space with bias: tile_m fixed at 128 (the shear contract bakes
128-row blocks); tile_n must divide 128 and divide padded -> {32, 64, 128}.

It also refuses to name a winner it cannot defend. Configs are timed in
interleaved rounds and a winner has to beat the runner-up with a disjoint
sample interval, because the first version of this file reported single
samples and two A100 runs of it disagreed by up to 27.6% on an unchanged
config while the configs themselves differed by at most 7.2%.

Output: table + per-case verdicts + JSON with every raw sample.
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


def _parity_probes() -> list:
    """Parity shapes DERIVED from CASES, so the gate cannot drift off the
    timings again.

    This file's post-mortem (journal/regression-ampere-tile-sweep.md, lesson 2)
    said the original incident was possible only because `make_case()` and
    `parity_ok()` built their tensors independently, and that deriving one from
    the other would make the divergence impossible to write. It then happened a
    second time on a different axis: the hand-written parity shapes all used
    `Hq == Hkv == 8`, so `qhead_per_kvhead == 1`, while every timed case uses
    `Hq=64` against `Hkv=8` or `Hkv=16`. A `pack_gqa` shear defect found on
    2026-07-25 mis-sheared the bias by +128 columns for exactly
    `qhead_per_kvhead > 1`, and was therefore invisible to this gate and live on
    every shape it timed. See the `arch // 10 in [8, 12]` guard in
    kernels/tml_fa4_modified/interface.py.

    So the geometry now comes from CASES verbatim, and only the depth is reduced.
    Depth has to be reduced: TOL_MEAN is absolute, and a 64K global decode moves
    the output by less than the tolerance, so an oracle cannot see the bias
    arrive at the timed depth (parity_rel_bias_coverage.py's docstring has the
    dilution table). A green result here certifies the timed FAMILY and the
    timed HEAD GEOMETRY, not the timed DEPTH. The coverage probe is the
    instrument for 64K.
    """
    geoms = []
    for _name, _T_q, _T_k, Hq, Hkv, ext, win in CASES:
        g = (Hq, Hkv, ext, win)
        if g not in geoms:
            geoms.append(g)
    probes = []
    for Hq, Hkv, ext, win in geoms:
        gtag = f"hq{Hq}_hkv{Hkv}_ext{ext}_{'swa' if win else 'global'}"
        # ctx = 0 is full prefill, the family the old gate covered.
        # ctx = 1000 is 104 mod 128. Every gated shape in this repository used
        # to have seqlen_k a multiple of 128, which is precisely the condition
        # under which both shear defects found so far are exact. It is the
        # single most load-bearing number in this list.
        # ctx = 4095 is the decode family the timings actually use.
        probes.append((f"{gtag}/full_prefill_712", 712, 0, Hq, Hkv, ext, win))
        probes.append((f"{gtag}/chunked_200_on_1000", 200, 1000, Hq, Hkv, ext,
                       win))
        probes.append((f"{gtag}/decode_1_on_4095", 1, 4095, Hq, Hkv, ext, win))
    return probes


def parity_ok() -> tuple[bool, float]:
    """Oracle check for the active config, on the timed geometry.

    Reuses parity_rel_chunked_decode's run_case and its calibrated tolerances
    rather than carrying a second oracle. The tolerances there were calibrated
    against a deliberately broken kernel; a copy here would drift from them.
    """
    import parity_rel_chunked_decode as prcd

    worst = 0.0
    failures = []
    powerless = []
    for name, T_q, ctx, Hq, Hkv, ext, win in _parity_probes():
        r = prcd.run_case(name, T_q, ctx, Hq, Hkv, ext, win, seed=4321)
        mx, mn = r["max_abs_diff"], r["mean_abs_diff"]
        sig = r["signal_over_tol"]
        bad = mx > prcd.TOL_MAX or mn > prcd.TOL_MEAN
        # A probe the bias barely moves cannot fail on a dropped bias, so
        # passing it certifies nothing. Report that rather than bank it.
        weak = sig < prcd.SIGNAL_MARGIN
        flag = "FAIL" if bad else ("OK (NO POWER)" if weak else "OK")
        print(f"    {flag:>13s} {name}: max={mx:.3e} mean={mn:.3e} "
              f"signal={sig:.1f}x TOL_MEAN")
        if bad:
            failures.append(name)
        if weak and not bad:
            powerless.append(name)
        worst = max(worst, mx)
    if powerless:
        print(f"    note: {len(powerless)} probe(s) had no power to fail on a "
              f"dropped bias: {', '.join(powerless)}")
    return not failures, worst


def _parity_ok_legacy() -> tuple[bool, float]:
    """The pre-2026-07-25 gate, kept only so the incident is reproducible.

    Not called. It is what certified the withdrawn Ampere percentages: two
    512-on-512 shapes at Hq == Hkv == 8, checking neither the timed sequence
    family nor the timed head geometry.
    """
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

    # Every case above passes ONE cu_seqlens as both cu_seqlens_q and
    # cu_seqlens_k, so all of them have seqlen_q == seqlen_k, and all of them
    # have Hq == Hkv. The cases this file TIMES have neither property: they are
    # 1-against-65536 at Hq=64 over Hkv=8. Two separate bias defects lived in
    # the gap and this gate certified the kernel through both of them.
    # See journal/regression-ampere-tile-sweep.md.
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


# Two A100 runs of the single-sample version of this file disagreed with each
# other by up to 27.6% on the SAME config and the SAME shape, while the largest
# difference BETWEEN configs was 7.2%. A sweep whose run-to-run noise is four
# times its signal cannot rank anything, and the three Ampere percentages this
# file once produced (10.1%, 18.2%, 18.7%) were single samples read off exactly
# that noise. See journal/regression-ampere-tile-sweep.md.
#
# So: repeat, interleave, and refuse to name a winner unless the intervals are
# actually disjoint.
REPEATS = 5
# Interleaving matters as much as repeating. The old loop was
# `for config: for case: time`, so a GPU that heats up over the sweep charged
# the whole penalty to whichever config happened to run last. Rounds put every
# config under the same drift.
DOMINATED_FACTOR = 5.0  # stop re-timing a config this much worse than the best


def _summarize(samples: list) -> dict:
    ok = sorted(s for s in samples if isinstance(s, float))
    if not ok:
        return {"median": None, "lo": None, "hi": None, "n": 0}
    mid = ok[len(ok) // 2] if len(ok) % 2 else (ok[len(ok) // 2 - 1]
                                               + ok[len(ok) // 2]) / 2
    return {"median": round(mid, 1), "lo": round(ok[0], 1),
            "hi": round(ok[-1], 1), "n": len(ok)}


def main() -> None:
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"capability {torch.cuda.get_device_capability(0)}")
    _install_tile_injector()

    results: dict = {}
    green: list = []
    for tile_mn in CONFIGS:
        _TILE[0] = tile_mn
        ok, worst = parity_ok()
        tag = f"tile{tile_mn[0]}x{tile_mn[1]}"
        print(f"\n[{tag}] parity max_diff={worst:.3e} "
              f"{'OK' if ok else 'FAIL -> timings SUPPRESSED'}")
        # The value above is the same to every digit for all three configs and
        # was the same in a previous run against a different kernel. That is
        # expected, not broken: the output is bf16, so one ULP is ~6e-3 relative
        # while reduction-order perturbation from a different tile_n is ~1e-6,
        # and the max-abs-diff saturates on output quantization. It still fails
        # loudly if the bias is dropped or misaddressed, which is its job, but
        # do not read it as sensitive to the tile size.
        results[tag] = {"parity": worst, "samples": {}, "timings": {}}
        if ok:
            green.append((tag, tile_mn))
        else:
            results[tag]["timings"] = None

    dominated: set = set()
    for r in range(REPEATS):
        print(f"\n--- round {r + 1}/{REPEATS} ---")
        for tag, tile_mn in green:
            _TILE[0] = tile_mn
            for name, *shape in CASES:
                if (tag, name) in dominated:
                    continue
                try:
                    us = time_case(make_case(*shape))
                    results[tag]["samples"].setdefault(name, []).append(us)
                    print(f"  [{tag}] {name}: {us:.1f} us")
                except Exception as exc:  # noqa: BLE001
                    results[tag]["samples"].setdefault(name, []).append(
                        f"ERROR: {str(exc)[:80]}")
                    print(f"  [{tag}] {name}: ERROR {str(exc)[:80]}")
                    dominated.add((tag, name))
            # A config 5x off the pace is not a tuning candidate, it is a
            # pathology (tile128x128 on A100 runs ~50x slower). Re-timing it
            # four more times buys nothing and costs GPU minutes. Dropping it
            # is logged below rather than done silently.
            for name, *_ in CASES:
                if (tag, name) in dominated:
                    continue
                mine = _summarize(results[tag]["samples"].get(name, []))
                others = [_summarize(results[o]["samples"].get(name, []))
                          for o, _ in green if o != tag]
                floor = [o["median"] for o in others if o["median"]]
                if mine["median"] and floor and \
                        mine["median"] > DOMINATED_FACTOR * min(floor):
                    dominated.add((tag, name))
                    print(f"  [{tag}] {name}: DOMINATED "
                          f"({mine['median']:.0f} us vs best {min(floor):.0f} "
                          f"us), dropped after {mine['n']} of {REPEATS} rounds")

    for tag in results:
        if results[tag]["timings"] is None:
            continue
        for name, *_ in CASES:
            s = _summarize(results[tag]["samples"].get(name, []))
            results[tag]["stats"] = results[tag].get("stats", {})
            results[tag]["stats"][name] = s
            # keep the old float-per-case shape so existing readers still work
            results[tag]["timings"][name] = s["median"]

    # A winner is only a winner if its worst sample beats the runner-up's best
    # sample. Anything else is the noise that produced the withdrawn numbers.
    best: dict = {}
    verdicts: dict = {}
    for name, *_ in CASES:
        cands = [(cfg, r["stats"][name]) for cfg, r in results.items()
                 if r.get("stats") and r["stats"][name]["median"] is not None]
        if len(cands) < 2:
            verdicts[name] = "insufficient data"
            continue
        cands.sort(key=lambda kv: kv[1]["median"])
        (w, ws), (up, ups) = cands[0], cands[1]
        if ws["hi"] < ups["lo"]:
            best[name] = (w, ws["median"])
            verdicts[name] = (
                f"{w} wins: [{ws['lo']}, {ws['hi']}] is disjoint from "
                f"{up} [{ups['lo']}, {ups['hi']}]")
        else:
            verdicts[name] = (
                f"NO WINNER: {w} [{ws['lo']}, {ws['hi']}] overlaps {up} "
                f"[{ups['lo']}, {ups['hi']}]; run-to-run spread exceeds the "
                f"difference between configs, so no tile size is selectable "
                f"on this shape")

    print("\n=== verdicts ===")
    for name, v in verdicts.items():
        print(f"  {name}: {v}")
    print("\n=== winners (disjoint intervals only) ===")
    if not best:
        print("  none. No config separated from the runner-up on any shape.")
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
        "repeats": REPEATS,
        "results": results,
        "best": best,
        "verdicts": verdicts,
    }, indent=1))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
