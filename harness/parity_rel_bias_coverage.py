#!/usr/bin/env python3
"""Gate: prove the relative bias reaches EVERY KV block it is supposed to,
at the shapes the published decode numbers were measured on.

WHY THIS FILE EXISTS

The shear shift defect was not a numeric drift. `apply_rel_bias_sm90` guards on
`0 <= tile_idx < bias_num_tiles`, so a wrong shift did not corrupt the bias, it
DELETED it: at batch-1 decode with 64K of KV, exactly one KV block out of 512
received a bias tile and it was the oldest one, whose distances are all far
outside the relative extent. The learned relative-position term was absent from
decode on Hopper, and the kernel that skipped it was the kernel that produced
the 2.7x to 2.8x decode figures.

An oracle-based parity gate is the right instrument in principle, and
harness/parity_rel_varlen_batch.py and harness/parity_rel_paged.py add several.
It is not sufficient here, for two independent reasons.

First, an O(T_q * T_k) float32 oracle cannot be built at 64K KV with 64 heads,
and 64K KV with 64 heads is the exact shape the withdrawn numbers came from.

Second, and worse, at long context an oracle tolerance is not SENSITIVE enough
even where it fits. The relative term only touches the most recent rel_extent
keys, so with attention spread over tens of thousands of keys, deleting the bias
entirely barely moves the output. harness/parity_rel_chunked_decode.py measures
the dilution from the reference alone, global causal with rel_extent 1024,
against TOL_MEAN 5e-4:

    ctx 1023  1.26e-02  25.2x     ctx 8191   1.45e-03   2.9x  thin
    ctx 4095  3.48e-03   7.0x     ctx 32767  5.47e-04   1.1x  effectively blind
                                  ctx 65535  2.24e-04   0.4x  BLIND

so a pure oracle gate at 32K or 64K of KV reports OK whether the bias arrived or
not. Sliding window is exempt: the window caps the attended key count and stops
the dilution. Every oracle case in this file therefore reports its own
`signal_mean`, `signal_over_tol` and `can_detect_dropped_bias`, and the 64K case
is EXPECTED to come back blind. That is the whole reason the probe exists, and
why the probe, not the oracle, is what the pass verdict rests on here.

This file closes the hole two ways.

  1. EXACT ROW ORACLE. For a decode step there is only one query row per
     sequence, so the score vector is (heads, seqlen_k) and it fits. Computing
     it per kv-head group keeps the peak under about 100 MB even at 64K KV, so
     the production decode shape gets a real float32 oracle rather than a proxy.

  2. SPIKE PROBE, and this is the part that needs no reference at all. Set
     rel_logits to zero everywhere except one distance d, run, and compare
     against the all-zero run. The key at absolute position `q_pos - d` had its
     logit raised by a known amplitude, so the output MUST move. If it does not
     move, the bias never reached that key. Sweeping d across the extent maps
     which KV blocks are receiving bias and which are being skipped, which is
     precisely the quantity that was wrong, and it reports it as a count rather
     than as a single scalar difference.

     Under the old formula this probe answers 0 of 12 distances responded, on
     any decode shape, with no reference to be wrong about. Under a
     partially-wrong shift it answers which subset responded. That is a
     diagnostic, not just a verdict.

     The probe also carries negative controls: a distance whose target key does
     not exist must NOT produce a response. Without them, a gate that reports
     "everything responded" could be reporting nothing but noise.

WHAT A FAILURE MEANS

  exact_row_oracle FAIL: the decode output is wrong. Read the per-case max.
  responded < probed: the bias is missing from some KV blocks. `tiles_probed`
    lists the KV tile each distance targets, so the missing set names itself.
  negative control responded: the response threshold is too low for this shape
    and the positive results from the same case are not trustworthy.

NOT RUN ON A GPU. Written from the code, never executed. Every field it would
fill is absent until it runs on real hardware.

Run (WSL): cd ~/inkling-turbo/vllm && source .venv/bin/activate && \
  python $REPO/harness/parity_rel_bias_coverage.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch

D = 128
DEV = "cuda"
KV_TILE = 128  # tile_n is forced to 128 for the sm_90 bias path (interface.py)

# Amplitude of the single-distance spike, in logit units before softmax. One key
# out of 65536 at +8.0 takes about 4.4 percent of the softmax mass
# (e**8 / (65535 + e**8)), which moves the output by order 1e-2 per component.
# That is roughly an order of magnitude above the bf16 run-to-run floor this
# kernel shows, and the floor is measured per case below rather than assumed.
SPIKE_AMP = 8.0

# A response has to clear the measured noise floor by this factor, or 1e-3,
# whichever is larger. The floor is measured by running the zero-bias case
# twice: this kernel is not run-to-run deterministic, which
# harness/parity_shear_fusion.py documents with measured numbers.
RESPONSE_NOISE_FACTOR = 8.0
RESPONSE_FLOOR = 1e-3

# Oracle tolerances inherited from harness/parity_rel_chunked_decode.py, which
# calibrated them on an H100 against the defect put back on purpose: worst
# legitimate mean 6.96e-05, best defective mean 3.28e-03. The mean is the
# discriminator, TOL_MAX is a backstop for a NaN or a dropped output.
TOL_MAX = 0.05
TOL_MEAN = 5e-4

# The exact row oracle is only built where it is cheap. Above these bounds the
# case reports the spike probe alone and says so, rather than allocating an
# oracle that will not fit.
ORACLE_MAX_ROWS = 8
ORACLE_MAX_KV_WHEN_MANY_ROWS = 8192


def exact_oracle(q, k, v, rel_rows, softmax_scale, ctx, window_left,
                 bias_only_oldest_tile=False):
    """float32 oracle computed one query row and one kv-head group at a time.

    Never materializes a (T_q, Hq, T_k) score tensor, so it reaches 64K KV with
    64 query heads. q is (T_q, Hq, D); k, v are (T_k, Hkv, D); rel_rows is
    (T_q, Hq, rel_extent).
    """
    T_q, Hq, _ = q.shape
    T_k, Hkv, _ = k.shape
    rel_extent = rel_rows.shape[-1]
    rep = Hq // Hkv
    out = torch.empty(T_q, Hq, v.shape[-1], dtype=torch.float32, device=q.device)

    k_pos = torch.arange(T_k, device=q.device)
    for i in range(T_q):
        dist = (ctx + i) - k_pos                          # (T_k,)
        in_range = (dist >= 0) & (dist < rel_extent)
        if bias_only_oldest_tile:
            in_range = in_range & ((k_pos // KV_TILE) == 0)
        d_idx = dist.clamp(0, rel_extent - 1)
        keep = dist >= 0
        if window_left is not None:
            keep &= dist <= window_left
        for g in range(Hkv):
            hs, he = g * rep, (g + 1) * rep
            qg = q[i, hs:he].float()                      # (rep, D)
            kg = k[:, g].float()                          # (T_k, D)
            s = (qg @ kg.transpose(0, 1)) * softmax_scale  # (rep, T_k)
            b = rel_rows[i, hs:he].float()[:, d_idx]      # (rep, T_k)
            s = s + b * in_range
            s = s.masked_fill(~keep.expand_as(s), float("-inf"))
            p = torch.softmax(s, dim=-1)
            out[i, hs:he] = p @ v[:, g].float()
            del s, b, p, kg
    return out


def call_kernel(q, k, v, rel, cu_q, cu_k, T_q, T_k, window_left, scale):
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func

    window = (None, None) if window_left is None else (window_left, 0)
    out = flash_attn_varlen_func(
        q=q, k=k, v=v,
        rel_bias=rel,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=T_q, max_seqlen_k=T_k,
        softmax_scale=scale, causal=True, window_size=window,
    )
    return out[0] if isinstance(out, tuple) else out


def run_case(name, T_q, ctx, Hq, Hkv, rel_extent, is_local, distances,
             neg_distances, seed):
    torch.manual_seed(seed)
    T_k = ctx + T_q
    q = torch.randn(T_q, Hq, D, dtype=torch.bfloat16, device=DEV) / (D**0.25)
    k = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=DEV) / (D**0.25)
    v = torch.randn(T_k, Hkv, D, dtype=torch.bfloat16, device=DEV)
    cu_q = torch.tensor([0, T_q], dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0, T_k], dtype=torch.int32, device=DEV)
    window_left = rel_extent - 1 if is_local else None
    scale = 1.0 / D
    rec: dict = {
        "shape": {"T_q": T_q, "ctx": ctx, "T_k": T_k, "Hq": Hq, "Hkv": Hkv,
                  "rel_extent": rel_extent, "window_left": window_left,
                  "kv_tiles": (T_k + KV_TILE - 1) // KV_TILE},
    }

    # ---------------------------------------------------------- noise floor
    zero = torch.zeros(T_q, Hq, rel_extent, dtype=torch.bfloat16, device=DEV)
    base = call_kernel(q, k, v, zero, cu_q, cu_k, T_q, T_k, window_left,
                       scale).float()
    base2 = call_kernel(q, k, v, zero, cu_q, cu_k, T_q, T_k, window_left,
                        scale).float()
    noise = float((base - base2).abs().max())
    threshold = max(RESPONSE_NOISE_FACTOR * noise, RESPONSE_FLOOR)
    rec["zero_bias_noise_max"] = noise
    rec["response_threshold"] = threshold

    # ------------------------------------------------------ 1. exact oracle
    build_oracle = (T_q <= ORACLE_MAX_ROWS
                    or T_k <= ORACLE_MAX_KV_WHEN_MANY_ROWS)
    if build_oracle:
        r_small = torch.randn(T_q, Hq, 16, dtype=torch.bfloat16,
                              device=DEV) * 0.4
        proj = torch.randn(16, rel_extent, dtype=torch.bfloat16,
                           device=DEV) * 0.3
        rel_real = (r_small.float() @ proj.float()).to(torch.bfloat16)
        got = call_kernel(q, k, v, rel_real, cu_q, cu_k, T_q, T_k, window_left,
                          scale).float()
        ref = exact_oracle(q, k, v, rel_real, scale, ctx, window_left)
        diff = (got - ref).abs()
        mx, mean = float(diff.max()), float(diff.mean())
        # How far the defective output would sit from the correct one at THIS
        # shape. If that distance is below the tolerance, this oracle check
        # cannot fail on a dropped bias and only the probe below can.
        defective = exact_oracle(q, k, v, rel_real, scale, ctx, window_left,
                                 bias_only_oldest_tile=True)
        sep = float((ref - defective).abs().mean())
        rec["exact_row_oracle"] = {
            "max_abs_diff": mx, "mean_abs_diff": mean,
            "signal_mean": sep,
            "signal_over_tol": sep / TOL_MEAN,
            "can_detect_dropped_bias": bool(sep >= 4.0 * TOL_MEAN),
            "pass": bool(mx <= TOL_MAX and mean <= TOL_MEAN),
        }
        del got, ref, diff, rel_real, defective
    else:
        rec["exact_row_oracle"] = {
            "pass": None,
            "skipped": (f"T_q={T_q} rows over {ORACLE_MAX_ROWS} with "
                        f"T_k={T_k} over {ORACLE_MAX_KV_WHEN_MANY_ROWS}; "
                        "the spike probe below still covers this shape"),
        }

    # -------------------------------------------------------- 2. spike probe
    last_q_pos = ctx + T_q - 1
    probes = []
    responded = 0
    for d in distances:
        rel = torch.zeros(T_q, Hq, rel_extent, dtype=torch.bfloat16, device=DEV)
        rel[:, :, d] = SPIKE_AMP
        got = call_kernel(q, k, v, rel, cu_q, cu_k, T_q, T_k, window_left,
                          scale).float()
        delta = (got - base).abs()
        resp = float(delta.max())
        # Which query rows moved: a row-localized failure means the shift is
        # m_block dependent, which a single scalar would not reveal.
        rows_moved = int((delta.amax(dim=(1, 2)) > threshold).sum())
        target_kv = last_q_pos - d
        probes.append({
            "distance": d,
            "target_kv_index_for_last_row": target_kv,
            "target_kv_tile_for_last_row": target_kv // KV_TILE,
            "response_max": resp,
            "responded": bool(resp > threshold),
            "rows_moved": rows_moved,
            "rows_total": T_q,
        })
        if resp > threshold:
            responded += 1
        del rel, got, delta

    rec["probe"] = {
        "spike_amplitude": SPIKE_AMP,
        "probed": len(distances),
        "responded": responded,
        "distances": probes,
        "tiles_probed": sorted({p["target_kv_tile_for_last_row"]
                                for p in probes}),
        "pass": responded == len(distances),
    }

    # ---------------------------------------------------- 3. negative control
    negs = []
    neg_ok = True
    for d in neg_distances:
        rel = torch.zeros(T_q, Hq, rel_extent, dtype=torch.bfloat16, device=DEV)
        rel[:, :, d] = SPIKE_AMP
        got = call_kernel(q, k, v, rel, cu_q, cu_k, T_q, T_k, window_left,
                          scale).float()
        resp = float((got - base).abs().max())
        quiet = resp <= threshold
        negs.append({"distance": d, "response_max": resp, "quiet": bool(quiet),
                     "why_no_key": (f"last query is at absolute position "
                                    f"{last_q_pos}, so distance {d} points at "
                                    f"kv index {last_q_pos - d}")})
        neg_ok &= quiet
        del rel, got
    rec["negative_control"] = {"cases": negs, "pass": bool(neg_ok)}

    rec["pass"] = bool(
        (rec["exact_row_oracle"]["pass"] in (True, None))
        and rec["probe"]["pass"]
        and rec["negative_control"]["pass"]
    )
    return rec


# Distances are chosen to land on tile boundaries and just inside and outside
# them, so a shift that is wrong by whole tiles and a shift that is wrong by a
# fraction of a tile look different in the report.
_TILE_SWEEP_1K = [0, 1, 127, 128, 129, 255, 256, 383, 447, 512, 640, 768, 1023]
_TILE_SWEEP_512 = [0, 1, 127, 128, 129, 255, 256, 383, 511]

CASES = [
    # Every KV block is inside the relative extent, so the sweep touches ALL 8
    # KV tiles. This is the direct test of "how many blocks got bias".
    ("decode_extent_covers_all_kv",
     dict(T_q=1, ctx=1023, Hq=8, Hkv=1, rel_extent=1024, is_local=False,
          distances=_TILE_SWEEP_1K, neg_distances=[], seed=4001)),
    # Same, sliding window.
    ("decode_extent_covers_all_kv_swa",
     dict(T_q=1, ctx=511, Hq=8, Hkv=2, rel_extent=512, is_local=True,
          distances=_TILE_SWEEP_512, neg_distances=[], seed=4002)),
    # A chunked prefill chunk: 128 query rows over cached context. rows_moved
    # localizes any m_block dependence.
    ("chunked_extent_covers_all_kv",
     dict(T_q=128, ctx=896, Hq=8, Hkv=1, rel_extent=1024, is_local=False,
          distances=_TILE_SWEEP_1K, neg_distances=[], seed=4003)),
    # Negative control shape: the context is shorter than the probed distances,
    # so those distances point at keys that do not exist and must stay quiet.
    ("negative_control_short_context",
     dict(T_q=1, ctx=127, Hq=8, Hkv=1, rel_extent=1024, is_local=False,
          distances=[0, 1, 127], neg_distances=[128, 129, 512, 1023], seed=4004)),
    # THE published shape. 64K KV, batch 1, Inkling's global head geometry at
    # TP1. The exact row oracle fits here because there is one query row.
    ("decode_b1_kv64k_production_geometry",
     dict(T_q=1, ctx=65535, Hq=64, Hkv=8, rel_extent=1024, is_local=False,
          distances=_TILE_SWEEP_1K, neg_distances=[], seed=4005)),
    # The 8K decode shape, same geometry.
    ("decode_b1_kv8k_production_geometry",
     dict(T_q=1, ctx=8191, Hq=64, Hkv=8, rel_extent=1024, is_local=False,
          distances=_TILE_SWEEP_1K, neg_distances=[], seed=4006)),
]


def main() -> None:
    cc = torch.cuda.get_device_capability(0)
    device = torch.cuda.get_device_name(0)
    print(f"device: {device}, capability {cc}")
    print(f"spike amplitude {SPIKE_AMP}, response must clear "
          f"{RESPONSE_NOISE_FACTOR}x the measured zero-bias noise "
          f"or {RESPONSE_FLOOR}")
    print()

    results = {}
    failures = 0
    for name, kw in CASES:
        try:
            r = run_case(name, **kw)
            o = r["exact_row_oracle"]
            p = r["probe"]
            n = r["negative_control"]
            print(f"[{name}] {'OK' if r['pass'] else 'FAIL'}")
            if o["pass"] is None:
                print(f"    oracle:  skipped ({o['skipped']})")
            else:
                det = ("detector" if o["can_detect_dropped_bias"]
                       else "BLIND to a dropped bias at this shape, "
                            "the probe below is the check that matters")
                print(f"    oracle:  {'OK' if o['pass'] else 'FAIL'} "
                      f"max={o['max_abs_diff']:.4e} "
                      f"mean={o['mean_abs_diff']:.4e} "
                      f"| signal={o['signal_mean']:.3e} "
                      f"({o['signal_over_tol']:.1f}x TOL_MEAN, {det})")
            print(f"    probe:   {p['responded']}/{p['probed']} distances "
                  f"moved the output, kv tiles touched {p['tiles_probed']}")
            if not p["pass"]:
                for d in p["distances"]:
                    if not d["responded"]:
                        print(f"      NO RESPONSE at distance "
                              f"{d['distance']} (kv tile "
                              f"{d['target_kv_tile_for_last_row']}), "
                              f"max delta {d['response_max']:.3e}")
            if n["cases"]:
                print(f"    control: {'OK' if n['pass'] else 'FAIL'} "
                      f"({len(n['cases'])} distances expected quiet)")
            if not r["pass"]:
                failures += 1
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            r = {"error": f"{type(exc).__name__}: {exc}", "pass": False}
            print(f"[{name}] FAIL: {type(exc).__name__}: {exc}")
            failures += 1
        results[name] = r
        torch.cuda.empty_cache()

    out = Path(__file__).with_name(
        f"parity_rel_bias_coverage_sm{cc[0]}{cc[1]}.json")
    out.write_text(json.dumps({
        "device": device,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "kv_tile": KV_TILE,
        "spike_amplitude": SPIKE_AMP,
        "response_noise_factor": RESPONSE_NOISE_FACTOR,
        "response_floor": RESPONSE_FLOOR,
        "tol_max": TOL_MAX,
        "tol_mean": TOL_MEAN,
        "cases": results,
        "passed": len(CASES) - failures,
        "total": len(CASES),
    }, indent=2), encoding="utf-8")

    print()
    print(f"{len(CASES) - failures}/{len(CASES)} cases pass")
    print(f"saved: {out}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
