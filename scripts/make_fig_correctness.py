#!/usr/bin/env python3
"""Regenerate docs/figures/fig2_correctness.png from the full-model gate artifact.

WHY THIS EXISTS

The figure it replaces was a binary with no generator, and it rendered
"32 of 32 prompts produced identical greedy tokens" in large type. That claim
needs a qualifier the image could not carry: the cross-build comparison ran
`max_tokens=0` with `echo=True`, so every one of its 2369 compared positions is
an echoed prompt token and none was generated. The gate's own docstring says a
token mismatch in an echoed prompt is impossible with one tokenizer, so the token
half of that headline is close to a tautology. A caption fixed the prose, but a
figure travels without its caption.

Every number here is read out of journal/remote/gate_logit_parity_8xh100.json.
Nothing is typed in.

THE ASYMMETRY THIS FIGURE HAS TO SHOW, because it is the actual finding:
the cross-build bar is measured on PREFILL positions only, while the two
same-build control bars ran `max_tokens=32`, so part of their positions are
generated. Reading one against the other is not apples to apples, and the figure
says so on its face rather than in a caption.

USAGE
    py scripts/make_fig_correctness.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SRC = REPO / "journal/remote/gate_logit_parity_8xh100.json"

OURS_C = "#0B6E6A"
CTRL_C = "#94A3B8"
INK = "#111827"
GRID = "#DDE1E6"
FAIL = "#B91C1C"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs/figures/fig2_correctness.png")
    ap.add_argument("--dpi", type=int, default=320)
    args = ap.parse_args()

    if not args.src.exists():
        sys.exit(f"missing artifact: {args.src}")
    d = json.loads(args.src.read_text(encoding="utf-8"))
    par, cfg, bc = d["parity"], d["config"], d["batch_consistency"]

    bars = [
        {
            "label": "ours vs stock\ncross-build",
            "mean": par["mean"],
            "n": par["tokens_compared"],
            "gen": 0,
            "color": OURS_C,
            "note": "prefill positions only",
        },
        {
            "label": "stock vs stock\nsame-build floor",
            "mean": bc["stock"]["mean"],
            "n": bc["stock"]["tokens_compared"],
            "gen": cfg["batch_prompts"] * cfg["batch_max_tokens"],
            "color": CTRL_C,
            "note": "includes generated",
        },
        {
            "label": "ours vs ours\nsame-build floor",
            "mean": bc["ours"]["mean"],
            "n": bc["ours"]["tokens_compared"],
            "gen": cfg["batch_prompts"] * cfg["batch_max_tokens"],
            "color": CTRL_C,
            "note": "includes generated",
        },
    ]

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    xs = range(len(bars))
    ax.bar(list(xs), [b["mean"] for b in bars],
           color=[b["color"] for b in bars], width=0.52, zorder=3)

    for x, b in zip(xs, bars):
        ax.annotate(f"{b['mean']:.3f}", (x, b["mean"]),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=11, fontweight="bold",
                    color=b["color"])

    tol = cfg["tol_mean"]
    ax.axhline(tol, color=FAIL, lw=1.4, ls="--", zorder=4)
    # Top left, which is the only region no bar occupies. Placing this near the
    # line itself ran the text straight through the two control bars.
    ax.text(
        0.015, 0.97,
        f"dashed line: a-priori tolerance {tol}\n"
        "every bar is above it, both same-build controls\n"
        "included, so the gate is recorded as a FAILURE\n"
        "about the platform rather than about our kernel",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=8.8, color=FAIL, zorder=5, linespacing=1.5,
    )

    # Position counts go INTO the tick label. Annotating below the axis collided
    # with the tick text, and on the bar that matters most.
    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        [f"{b['label']}\n{b['n']} positions, "
         + ("none generated" if b["gen"] == 0 else f"{b['gen']} generated")
         for b in bars],
        fontsize=9.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("mean absolute per-token logprob difference\n(lower is better)",
                  fontsize=9, color=INK)
    ax.set_ylim(0, max(b["mean"] for b in bars) * 1.55)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)

    fig.suptitle("Full-model agreement with the stock build, 8x H100 at TP8",
                 x=0.012, ha="left", fontsize=13, fontweight="bold",
                 color=INK, y=0.99)
    fig.text(0.012, 0.925,
             "The two comparisons are NOT measured on the same thing, and that "
             "asymmetry is the finding. The cross-build bar ran max_tokens=0 "
             "with echo=True, so all of its",
             ha="left", fontsize=8.2, color="#6B7280")
    fig.text(0.012, 0.897,
             f"{par['tokens_compared']} positions are echoed prompt tokens and "
             f"none was generated. The controls ran max_tokens="
             f"{cfg['batch_max_tokens']}, so part of theirs were. Token identity "
             "on echoed prompts is near-tautological.",
             ha="left", fontsize=8.2, color="#6B7280")
    fig.text(0.012, 0.869,
             f"Source: journal/remote/{args.src.name}. Regenerate with "
             "py scripts/make_fig_correctness.py",
             ha="left", fontsize=8.2, color="#9CA3AF")

    fig.tight_layout(rect=(0, 0.02, 1, 0.855))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out} at {args.dpi} dpi from {args.src}")
    for b in bars:
        print(f"  {b['label'].replace(chr(10), ' '):34s} mean {b['mean']:.6f} "
              f"n={b['n']} generated={b['gen']}")
    print(f"  tolerance {tol}, gate pass={par['pass']}, "
          f"tokens_match_all={par['tokens_match_all']} (echoed prompts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
