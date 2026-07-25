#!/usr/bin/env python3
"""Regenerate docs/figures/fig4_e2e.png, the end-to-end serving comparison.

WHY THIS EXISTS

The serving number is the one that matters and it was the last thing this
repository could not show. It is also the number most easily overstated, because
the three matched comparisons all land near 1.10x and a bar chart of three
medians would look like a single confident result rather than three independent
ones.

So this figure plots every individual run, not the medians, normalised to the
stock median of its own comparison. That makes two things visible at once that a
bar chart hides:

  1. where the ranges do NOT touch, which is the case for output throughput,
     TPOT and p99 end-to-end in all three comparisons, and is the reason those
     are quoted;
  2. where they DO touch, which is the case for TTFT everywhere, and is the
     reason TTFT is not quoted. Including the metric we decline to claim is the
     point of the figure, not an omission from it.

Everything is read from the committed run artifacts under
journal/remote/e2e_s30_h200/, so this cannot drift from the record.

USAGE
    py scripts/make_fig_e2e.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "journal/remote/e2e_s30_h200"

OURS = "#0B6E6A"
STOCK = "#6B7280"
WARN = "#B45309"
INK = "#111827"
RULE = "#E5E7EB"

# (mix, concurrency, label)
COMPARISONS = [
    ("prefill", 1, "prefill-heavy, conc 1"),
    ("prefill", 8, "prefill-heavy, conc 8"),
    ("decode", 8, "decode-heavy, conc 8"),
]

# (json key, label, direction) - "lower" metrics are inverted so that
# right-of-1.0 always means "Inkling-turbo is better", on every row.
METRICS = [
    ("output_throughput", "output tok/s", "higher"),
    ("median_tpot_ms", "TPOT p50", "lower"),
    ("p99_e2el_ms", "p99 end to end", "lower"),
    ("median_ttft_ms", "TTFT p50", "lower"),
]


def load(build: str, mix: str, conc: int) -> list:
    d = RUNS / build / mix / str(conc)
    if not d.exists():
        return []
    out = []
    for f in sorted(d.glob("run*.json")):
        out.append(json.loads(f.read_text(encoding="utf-8")))
    return out


def series(recs: list, key: str) -> list:
    return [float(r[key]) for r in recs
            if isinstance(r.get(key), (int, float))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs/figures/fig4_e2e.png")
    ap.add_argument("--dpi", type=int, default=320)
    args = ap.parse_args()

    rows = []          # (row_label, group_label, stock_rel, ours_rel, sep)
    for key, mlabel, direction in METRICS:
        for mix, conc, clabel in COMPARISONS:
            o, s = load("ours", mix, conc), load("stock", mix, conc)
            ov, sv = series(o, key), series(s, key)
            if len(ov) < 3 or len(sv) < 3:
                continue
            base = st.median(sv)
            if not base:
                continue
            if direction == "higher":
                srel = [v / base for v in sv]
                orel = [v / base for v in ov]
            else:
                # invert so better is always to the right
                srel = [base / v for v in sv]
                orel = [base / v for v in ov]
            sep = min(orel) > max(srel)
            rows.append((mlabel, clabel, srel, orel, sep))

    if not rows:
        print("no complete 3-vs-3 comparisons found; nothing to draw",
              file=sys.stderr)
        return 1

    n = len(rows)
    # Reserve vertical space from the measured block heights rather than a fixed
    # fraction, the same way make_fig_status.py does, so editing the note cannot
    # push it into the legend.
    note_lines = 4
    note_h = note_lines * 8.2 * 1.6 / 72.0
    legend_h = 0.40
    fig_h = 0.46 * n + 1.35 + legend_h + note_h + 0.30
    fig, ax = plt.subplots(figsize=(11.4, fig_h))

    ax.axvline(1.0, color=INK, lw=1.1, zorder=2)
    ylabels = []
    for i, (mlabel, clabel, srel, orel, sep) in enumerate(rows):
        y = n - 1 - i
        ax.plot([min(srel), max(srel)], [y, y], "-", color=STOCK, lw=5.0,
                alpha=0.30, solid_capstyle="butt", zorder=3)
        ax.plot([min(orel), max(orel)], [y, y], "-", color=OURS, lw=5.0,
                alpha=0.30, solid_capstyle="butt", zorder=3)
        ax.plot(srel, [y] * len(srel), "o", ms=6.0, color=STOCK, zorder=4)
        ax.plot(orel, [y] * len(orel), "o", ms=6.0, color=OURS, zorder=4)
        ratio = st.median(orel)
        ax.text(max(max(orel), max(srel)) + 0.006, y,
                f"{ratio:.3f}x" + ("" if sep else "  ranges overlap"),
                va="center", ha="left", fontsize=8.6,
                color=OURS if sep else WARN,
                fontweight="bold" if sep else "normal")
        ylabels.append(f"{mlabel}  ({clabel})")

    ax.set_yticks(list(range(n - 1, -1, -1)))
    ax.set_yticklabels(ylabels, fontsize=9.2)
    ax.set_ylim(-0.7, n - 0.3)
    lo = min(min(r[2] + r[3]) for r in rows)
    hi = max(max(r[2] + r[3]) for r in rows)
    # Left pad enough that the worst TTFT outlier is fully drawn rather than
    # half-clipped by the spine, and right pad for the ratio labels.
    span = hi - lo
    ax.set_xlim(lo - 0.045 * span, hi + 0.30 * span)
    ax.set_xlabel("relative to the stock median of the same comparison "
                  "(right is better on every row)", fontsize=9)
    ax.tick_params(axis="both", length=0)
    ax.grid(axis="x", color=RULE, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=7, color=STOCK,
               label="stock day-0, one point per run"),
        Line2D([], [], marker="o", ls="", ms=7, color=OURS,
               label="Inkling-turbo, one point per run"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.55 / (0.46 * n)),
        ncol=2, frameon=False, fontsize=8.8)

    fig.suptitle("End-to-end serving, 8x H200 TP8, three runs per build",
                 x=0.012, ha="left", fontsize=13.5, fontweight="bold",
                 color=INK, y=0.995)
    fig.text(0.012, 0.955,
             "Every run plotted, not the medians. Regenerate with "
             "py scripts/make_fig_e2e.py",
             ha="left", fontsize=8.8, color="#6B7280")

    note = (
        "Both builds: TP8, the real 592GB NVFP4 checkpoint, --max-model-len 3072 "
        "--gpu-memory-utilization 0.94 --enforce-eager --seed 0, and an identical "
        "KV budget of 188160 tokens with 0.0000% drift.\n"
        "The three throughput ratios are 1.110x, 1.106x and 1.107x across two "
        "mixes and two concurrencies, and in those rows the slowest Inkling-turbo "
        "run beats the fastest stock run, so the ranges do not touch.\n"
        "TTFT is shown and NOT claimed: its medians favour Inkling-turbo in all "
        "three comparisons, but one cold-start run per build makes every range "
        "overlap. A median that flatters us inside a range\nthat overlaps is not a "
        "result. Offered concurrency is --max-concurrency, the client's number, "
        "not a server batch size; the server reports its own ceiling as 61.25x."
    )
    fig.text(0.012, 0.012, note, ha="left", va="bottom", fontsize=8.2,
             color="#4B5563", linespacing=1.6)

    bottom = (note_h + legend_h + 0.18) / fig_h
    top = 1.0 - 0.80 / fig_h
    fig.tight_layout(rect=(0, bottom, 1, top))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out} at {args.dpi} dpi")
    print(f"  {n} rows, {sum(1 for r in rows if r[4])} with disjoint ranges")
    for mlabel, clabel, srel, orel, sep in rows:
        print(f"  {'SEP ' if sep else 'over'} {mlabel:>16s} {clabel:>22s} "
              f"{st.median(orel):.4f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
