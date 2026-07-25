#!/usr/bin/env python3
"""Regenerate docs/figures/fig3_status.png, the validation matrix.

WHY THIS EXISTS

The figure it replaces was a binary with no generator, and by 2026-07-25 it was
wrong in four places: it showed U2 relative attention as validated on sm_80 and
sm_120 when the shear-shift fix has never executed on either, it showed the
Ampere tile tuning as validated when those percentages are withdrawn, it showed
the full-model gate as a clean pass when the token half is near-tautological and
the logprob half is a recorded failure, and it had no row for the
`seqlen_q != seqlen_k` gate that the whole incident turned on.

A status matrix with no generator drifts silently, because nothing recomputes it
when a claim is withdrawn. This one is a declarative spec plus a renderer, so
correcting it is a one-line edit with a reason attached.

STATUS VOCABULARY, and the reason there are five and not three: two of the
statuses this repository actually needs did not exist in the old figure. A green
dot cannot represent "green on one shape family and never run on the other", and
it cannot represent "measured, then withdrawn".

USAGE
    py scripts/make_fig_status.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

REPO = Path(__file__).resolve().parent.parent

OK = "#0B6E6A"
PART = "#B45309"
GONE = "#B91C1C"
NONE_C = "#9CA3AF"
INK = "#111827"
RULE = "#D1D5DB"

ARCHES = [
    ("sm_120", "RTX 5090"),
    ("sm_90", "H100"),
    ("sm_80", "A100"),
    ("sm_100", "B200"),
]

# status: ok | partial | withdrawn | todo | na
ROWS = [
    ("U2 relative attention, full prefill",
     ["ok", "ok", "ok", "todo"]),
    ("U2 relative attention, decode and chunked",
     ["todo", "ok", "todo", "todo"]),
    ("U2 shear-shift fix executed on silicon",
     ["todo", "ok", "todo", "na"]),
    ("U2 full-model gate, prefill positions",
     ["na", "partial", "na", "todo"]),
    ("U2 tile tuning",
     ["na", "na", "withdrawn", "todo"]),
    ("U3 FP8 paged KV, write path",
     ["ok", "ok", "ok", "todo"]),
    ("U3 FP8 paged KV, read path",
     ["todo", "todo", "todo", "todo"]),
    ("U2 split-KV decode",
     ["todo", "todo", "todo", "todo"]),
    ("U2 shear fusion, writer correct",
     ["ok", "ok", "todo", "todo"]),
    ("U2 shear fusion, worth enabling",
     ["todo", "withdrawn", "todo", "todo"]),
    ("End-to-end serving curves",
     ["na", "todo", "na", "todo"]),
    ("U1 MoE grouped GEMM",
     ["todo", "todo", "todo", "todo"]),
]

FOOTNOTES = [
    "Full prefill means seqlen_q == seqlen_k. That distinction is the whole "
    "reason this figure has two attention rows: for months the suite covered "
    "only the first and was read as covering the kernel.",
    "sm_80 and sm_120 attention is green on full prefill and open on decode. "
    "The shear-shift fix landed in flash_fwd.py but has never executed on "
    "either architecture, so neither has a decode-shape correctness result.",
    "The full-model gate is partial, not green: it ran max_tokens=0 with "
    "echo=True, so it compared echoed prompt positions and never generated a "
    "token, and its logprob half is a recorded failure against its a-priori "
    "tolerance.",
    "Ampere tile tuning is withdrawn: the sweep timed decode shapes while its "
    "parity gate checked full prefill, and the kernel was wrong on the family "
    "it timed.",
    "Shear fusion splits into two rows because the writer is bit-exact on two "
    "architectures while the feature is measured a net loss on prefill, so "
    "correct and worth enabling are not the same claim.",
    "Blackwell was never obtainable. No sm_100 result is claimed anywhere in "
    "this repository, and upstream's own Blackwell path already implements this "
    "bias correctly.",
]


def draw(ax, x, y, status):
    if status == "ok":
        ax.plot(x, y, "o", ms=11, color=OK, zorder=3)
    elif status == "partial":
        ax.plot(x, y, "o", ms=11, mfc="none", mec=PART, mew=2.6, zorder=3)
        ax.plot(x, y, "o", ms=4.5, color=PART, zorder=4)
    elif status == "withdrawn":
        ax.plot(x, y, "o", ms=11, mfc="none", mec=GONE, mew=2.0, zorder=3)
        ax.plot([x - 0.09, x + 0.09], [y, y], "-", color=GONE, lw=2.2, zorder=4)
    elif status == "todo":
        ax.plot(x, y, "o", ms=10, mfc="none", mec=NONE_C, mew=1.7, zorder=3)
    else:
        ax.plot([x - 0.055, x + 0.055], [y, y], "-", color=NONE_C, lw=1.6,
                zorder=3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs/figures/fig3_status.png")
    ap.add_argument("--dpi", type=int, default=320)
    args = ap.parse_args()

    n = len(ROWS)
    fig, ax = plt.subplots(figsize=(13.0, 0.52 * n + 3.1))

    for r, (label, states) in enumerate(ROWS):
        y = n - 1 - r
        for c, st in enumerate(states):
            draw(ax, c, y, st)

    ax.set_xlim(-0.62, len(ARCHES) - 0.38)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([n - 1 - r for r in range(n)])
    ax.set_yticklabels([lbl for lbl, _ in ROWS], fontsize=10)
    ax.set_xticks(range(len(ARCHES)))
    ax.set_xticklabels([f"{a}\n{d}" for a, d in ARCHES], fontsize=10.5,
                       fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(n - 0.32, color=RULE, lw=1.0)
    ax.axhline(-0.55, color=RULE, lw=1.0)

    handles = [
        Line2D([], [], marker="o", ls="", ms=10, color=OK,
               label="validated on that silicon"),
        Line2D([], [], marker="o", ls="", ms=10, mfc="none", mec=PART, mew=2.4,
               label="partial, read the note"),
        Line2D([], [], marker="o", ls="", ms=10, mfc="none", mec=GONE, mew=2.0,
               label="measured, then withdrawn"),
        Line2D([], [], marker="o", ls="", ms=10, mfc="none", mec=NONE_C,
               mew=1.7, label="not done"),
        Line2D([], [], marker="_", ls="", ms=12, color=NONE_C,
               label="not applicable"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.055),
              ncol=5, frameon=False, fontsize=9.2, handletextpad=0.5,
              columnspacing=1.5)

    fig.suptitle("Validation status by unit and architecture",
                 x=0.012, ha="left", fontsize=14, fontweight="bold",
                 color=INK, y=0.995)
    fig.text(0.012, 0.952,
             "Nothing is marked validated on hardware it has never run on. "
             "Regenerate with py scripts/make_fig_status.py",
             ha="left", fontsize=9, color="#6B7280")

    body = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(FOOTNOTES))
    fig.text(0.012, 0.012, body, ha="left", va="bottom", fontsize=8.4,
             color="#4B5563", linespacing=1.65, wrap=True)

    fig.tight_layout(rect=(0, 0.205, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")

    print(f"wrote {args.out} at {args.dpi} dpi")
    tally: dict = {}
    for _, states in ROWS:
        for st in states:
            tally[st] = tally.get(st, 0) + 1
    print("  cells:", ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
