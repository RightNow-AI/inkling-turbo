#!/usr/bin/env python3
"""Regenerate docs/figures/fig1_latency.png from the measurement artifacts.

WHY THIS EXISTS

The figure it replaces was a binary with no generator. When the decode numbers
were withdrawn after the sm_90 shear-shift defect was found, the README prose
could be corrected but the figure could not, and a figure travels without its
caption. A published chart that cannot be regenerated from the artifacts is a
claim that cannot be corrected.

Every value plotted here is read out of a JSON file under journal/remote/.
Nothing is typed in. If an artifact is missing a case, that case is omitted and
named in the output rather than filled in.

USAGE
    py scripts/make_fig_latency.py
    py scripts/make_fig_latency.py --out docs/figures/fig1_latency.png

The defaults point at the session-27 pair, which is the run with the CORRECTED
decode bias, both builds timed in the same container.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter   # noqa: E402

REPO = Path(__file__).resolve().parent.parent
S27 = REPO / "journal/remote/validate_s27_decodefix"

DEFAULT_OURS = S27 / "microbench_ours_attn_shearfusion_OFF_modal_h100x1_route.json"
DEFAULT_BASE = S27 / "microbench_attn_scoremod_modal_h100x1_route.json"

# (artifact key, label, sub-label). Order is top to bottom in the figure.
WORKLOADS = [
    ("decode_b1_global_kv64k", "decode, batch 1", "64K KV"),
    ("decode_b32_global_kv64k", "decode, batch 32", "64K KV"),
    ("decode_b32_global_kv8k", "decode, batch 32", "8K KV"),
    ("prefill_global_8k", "prefill, 8K", "global"),
    ("prefill_swa_8k", "prefill, 8K", "sliding window"),
]

# The biasless floor exists only for the two 64K decode cases.
PLAIN = {
    "decode_b1_global_kv64k": "decode_b1_plain_kv64k",
    "decode_b32_global_kv64k": "decode_b32_plain_kv64k",
}

OURS_C = "#0B6E6A"
BASE_C = "#9A3412"
PLAIN_C = "#64748B"
INK = "#111827"
GRID = "#DDE1E6"
WIN = "#0B6E6A"
LOSE = "#B91C1C"


def load(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"missing artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"unreadable artifact {path}: {exc}")


def total(d: dict, key: str):
    rec = d.get(key)
    if not isinstance(rec, dict):
        return None
    v = rec.get("total_us_per_iter")
    return float(v) if isinstance(v, (int, float)) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, default=DEFAULT_OURS)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs/figures/fig1_latency.png")
    ap.add_argument("--dpi", type=int, default=320)
    args = ap.parse_args()

    ours_d, base_d = load(args.ours), load(args.baseline)

    rows, omitted = [], []
    for key, label, sub in WORKLOADS:
        a, b = total(ours_d, key), total(base_d, key)
        if a is None or b is None:
            omitted.append(f"{key} (ours={a}, baseline={b})")
            continue
        rows.append({"key": key, "label": label, "sub": sub, "ours": a,
                     "base": b, "plain": total(base_d, PLAIN.get(key, ""))})
    if not rows:
        sys.exit("no comparable cases found in the two artifacts")
    for o in omitted:
        print(f"OMITTED, not plotted: {o}")

    n = len(rows)
    fig, ax = plt.subplots(figsize=(11.0, 0.95 * n + 2.5))
    ax.set_xscale("log")

    ypos = list(range(n - 1, -1, -1))
    for y, r in zip(ypos, rows):
        ax.plot([r["ours"], r["base"]], [y, y], color=GRID, lw=6,
                solid_capstyle="round", zorder=1)
        if r["plain"]:
            ax.scatter(r["plain"], y, s=70, facecolors="none",
                       edgecolors=PLAIN_C, lw=1.8, zorder=3)
            ax.annotate(f"{r['plain']:.0f}", (r["plain"], y),
                        textcoords="offset points", xytext=(0, -16),
                        ha="center", fontsize=7.5, color=PLAIN_C)
        ax.scatter(r["base"], y, s=95, color=BASE_C, zorder=4)
        ax.scatter(r["ours"], y, s=95, color=OURS_C, zorder=4)
        # Label the LEFT point leftwards and the RIGHT point rightwards, decided
        # per row. Anchoring ours-left unconditionally collided on the one row
        # where ours is the larger value, which is the sliding-window case we
        # lose. The row that most needs to be legible was the row that broke.
        ours_is_left = r["ours"] <= r["base"]
        # The biasless floor marker sits just left of ours on the 64K decode
        # rows (736 against 868), so a left-anchored ours label lands on top of
        # that glyph. Lift it clear when the floor is within a factor of 1.6.
        crowded = bool(r["plain"]) and r["ours"] / r["plain"] < 1.6
        for val, col, is_left, dy in (
            (r["ours"], OURS_C, ours_is_left, 13 if crowded else 4),
            (r["base"], BASE_C, not ours_is_left, 4),
        ):
            ax.annotate(
                f"{val:.0f}", (val, y),
                textcoords="offset points",
                xytext=(0 if (crowded and col == OURS_C)
                        else (-10 if is_left else 10), dy),
                fontsize=8.5, color=col, va="center",
                ha="center" if (crowded and col == OURS_C)
                else ("right" if is_left else "left"),
            )

        ratio = r["base"] / r["ours"]
        if ratio >= 1.0:
            txt, col = f"{ratio:.2f}x faster", WIN
        else:
            txt, col = f"{1.0 / ratio:.2f}x SLOWER", LOSE
        ax.annotate(txt, (1.0, y), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(12, 0),
                    fontsize=9, fontweight="bold", color=col, va="center")

    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r['label']}\n{r['sub']}" for r in rows], fontsize=9)
    ax.tick_params(axis="y", length=0)
    # Headroom below the last row for the legend, which otherwise sat on top of
    # the sliding-window case.
    ax.set_ylim(-1.45, n - 0.35)

    lo = min(min(r["ours"], r["base"], r["plain"] or 1e9) for r in rows)
    hi = max(max(r["ours"], r["base"]) for r in rows)
    ax.set_xlim(lo * 0.55, hi * 1.5)
    ticks = [t for t in (100, 200, 500, 1000, 2000, 5000) if lo * 0.5 < t < hi * 2]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("attention kernel latency per iteration, microseconds "
                  "(log scale). Lower is better.", fontsize=9, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.scatter([], [], s=95, color=OURS_C, label="Inkling-turbo")
    ax.scatter([], [], s=95, color=BASE_C,
               label="day-0 score_mod, the path vLLM serves with")
    ax.scatter([], [], s=70, facecolors="none", edgecolors=PLAIN_C, lw=1.8,
               label="plain attention, no bias (a floor, not a baseline)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.10), ncol=3,
              frameon=False, fontsize=8.5, handletextpad=0.4,
              columnspacing=1.6)

    src = f"{args.ours.parent.name}/"
    fig.suptitle("Attention kernel latency against the path vLLM ships",
                 x=0.012, ha="left", fontsize=13, fontweight="bold",
                 color=INK, y=0.99)
    fig.text(0.012, 0.935,
             "One H100 SXM5, sm_90. Both builds timed in the same container. "
             "Every point has a passing parity run behind it, including the "
             f"seqlen_q != seqlen_k gate. Source: journal/remote/{src}",
             ha="left", fontsize=8, color="#6B7280")
    fig.text(0.012, 0.905,
             "The sliding-window case is shown because we lose it, and 55 of "
             "Inkling's 66 layers are sliding window.",
             ha="left", fontsize=8, color="#6B7280")

    fig.tight_layout(rect=(0, 0, 0.88, 0.89))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight",
                facecolor="white")
    print(f"wrote {args.out} at {args.dpi} dpi from")
    print(f"  ours     {args.ours}")
    print(f"  baseline {args.baseline}")
    for r in rows:
        print(f"  {r['key']:26s} ours {r['ours']:8.1f}  base {r['base']:8.1f}"
              f"  {r['base'] / r['ours']:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
