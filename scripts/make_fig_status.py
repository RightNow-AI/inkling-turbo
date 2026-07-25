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

STATUS VOCABULARY, and the reason there are six and not three: three of the
statuses this repository actually needs did not exist in the old figure. A green
dot cannot represent "green on one shape family and never run on the other", it
cannot represent "measured, then withdrawn", and it cannot represent "the gate
ran and the kernel faulted", which is a different state from "not done" and was
added on 2026-07-25 for the multi-sequence varlen row.

USAGE
    py scripts/make_fig_status.py
"""

from __future__ import annotations

import argparse
import textwrap
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
     ["ok", "ok", "ok", "todo"]),
    ("U2 relative attention, multi-sequence varlen batching",
     ["ok", "partial", "partial", "todo"]),
    ("U2 shear-shift fix executed on silicon",
     ["ok", "ok", "ok", "na"]),
    ("U2 bias coverage at 64K decode depth",
     ["partial", "todo", "partial", "todo"]),
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
    "reason this figure has separate attention rows: for months the suite "
    "covered only the first and was read as covering the kernel.",
    "The generic kernel's corrected shear shift now has silicon behind it on "
    "both architectures that use it. sm_120, local RTX 5090, 2026-07-25: "
    "chunked-and-decode parity 7/7 with per-case signal 7.5x to 37.8x above "
    "tolerance, full-prefill 3/3, bias coverage 6/6, writer 5/5. sm_80, Modal "
    "A100-40GB with capability (8,0) asserted, same day, $0.99: chunked-and-"
    "decode 7/7 with 6.7x to 37.7x headroom, full-prefill 3/3, coverage 6/6, "
    "writer 5/5. The sm_120 counts were journal-only at first because that "
    "session's JSON files were zero bytes; they are backed from "
    "journal/remote/local_sm120_s30 onward.",
    "Multi-sequence varlen batching is the call shape vLLM serving constructs on "
    "every step, and until 2026-07-25 no gate here had ever built it. Two "
    "defects were in it. An unpredicated bias copy faulted with "
    "cudaErrorIllegalAddress, and the shear writer packed GQA heads the generic "
    "reader did not, which mis-sheared the bias by 128 columns at the production "
    "Hq=64 over Hkv=8 geometry. Both are fixed: sm_120 is 12 of 12. sm_80 is 11 "
    "of 12, the crash gone and the twelfth case awaiting one more A100 step. On "
    "sm_90 32-sequence batches run and have been timed repeatedly, but no parity "
    "gate has ever checked a multi-sequence batch there, so that cell stays "
    "partial rather than green.",
    "Bias coverage is partial, not green, for a reason worth reading, and it "
    "reads the same on both architectures. The gate scores 6 of 6 on sm_120 and "
    "on sm_80, but at production decode geometry its oracle comparison is blind "
    "to a completely dropped bias, signal 0.5x of its own tolerance. What "
    "discriminates is the probe that walks individual distances: 13 of 13 move "
    "the output, and the tiles it names as touched at 64K are [504..511], the "
    "newest blocks, which is what the corrected shift should select.",
    "The full-model gate is partial, not green: it ran max_tokens=0 with "
    "echo=True, so it compared echoed prompt positions and never generated a "
    "token, and its logprob half is a recorded failure against its a-priori "
    "tolerance.",
    "Ampere tile tuning is withdrawn and, since 2026-07-25, refuted. The sweep "
    "timed decode shapes at Hq=64 over Hkv=8 while its parity gate checked full "
    "prefill at Hq==Hkv, and the kernel was wrong on both of those axes. Re-run "
    "on verified Ampere, the same configuration moved by up to 27.6% between "
    "runs while the configurations differ by at most 7.2%, so one sample per "
    "cell cannot rank them at all.",
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
    elif status == "broken":
        ax.plot(x, y, "X", ms=12.5, color=GONE, zorder=3)
    elif status == "todo":
        ax.plot(x, y, "o", ms=10, mfc="none", mec=NONE_C, mew=1.7, zorder=3)
    else:
        ax.plot([x - 0.055, x + 0.055], [y, y], "-", color=NONE_C, lw=1.6,
                zorder=3)


# Layout constants. The footnote block used to get a fixed 20.5% of the figure
# height via tight_layout(rect=...), which meant editing a footnote could push it
# up into the legend, and did: adding the sm_80 results to note 2 overlapped the
# two in the committed PNG. Wrap the text here rather than leaving it to
# matplotlib's wrap=True, so the line count is known before the figure is sized
# and the reservation is computed from it.
FIG_W = 13.0
NOTE_PT = 8.4
NOTE_LINESPACING = 1.65
NOTE_COLS = 208           # calibrated for NOTE_PT at FIG_W, see _note_lines
ROW_H = 0.52              # inches per matrix row
HEAD_H = 1.55             # title, subtitle, column headers
LEGEND_H = 0.42


def _note_lines() -> list:
    """Footnotes as explicit lines, so the caller can count them."""
    out = []
    for i, t in enumerate(FOOTNOTES):
        body = f"{i + 1}. {t}"
        wrapped = textwrap.wrap(body, width=NOTE_COLS) or [body]
        # continuation lines indent under the number
        out.append(wrapped[0])
        out.extend("   " + w for w in wrapped[1:])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs/figures/fig3_status.png")
    ap.add_argument("--dpi", type=int, default=320)
    args = ap.parse_args()

    n = len(ROWS)
    notes = _note_lines()
    note_h = len(notes) * NOTE_PT * NOTE_LINESPACING / 72.0
    fig_h = ROW_H * n + HEAD_H + LEGEND_H + note_h + 0.30
    fig, ax = plt.subplots(figsize=(FIG_W, fig_h))

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

    # Only legend the states that actually appear. A key advertising "open
    # defect" when no cell is one is a small dishonesty in the other direction.
    used = {st for _, states in ROWS for st in states}
    all_handles = [
        ("ok", Line2D([], [], marker="o", ls="", ms=10, color=OK,
                      label="validated on that silicon")),
        ("partial", Line2D([], [], marker="o", ls="", ms=10, mfc="none",
                           mec=PART, mew=2.4, label="partial, read the note")),
        ("withdrawn", Line2D([], [], marker="o", ls="", ms=10, mfc="none",
                             mec=GONE, mew=2.0,
                             label="measured, then withdrawn")),
        ("broken", Line2D([], [], marker="X", ls="", ms=11, color=GONE,
                          label="ran, and faults: open defect")),
        ("todo", Line2D([], [], marker="o", ls="", ms=10, mfc="none",
                        mec=NONE_C, mew=1.7, label="not done")),
        ("na", Line2D([], [], marker="_", ls="", ms=12, color=NONE_C,
                      label="not applicable")),
    ]
    handles = [h for st, h in all_handles if st in used]
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(0.0, -0.4 / (ROW_H * n)),
              ncol=len(handles), frameon=False, fontsize=8.6,
              handletextpad=0.45, columnspacing=1.05)

    fig.suptitle("Validation status by unit and architecture",
                 x=0.012, ha="left", fontsize=14, fontweight="bold",
                 color=INK, y=0.995)
    fig.text(0.012, 0.952,
             "Nothing is marked validated on hardware it has never run on. "
             "Regenerate with py scripts/make_fig_status.py",
             ha="left", fontsize=9, color="#6B7280")

    fig.text(0.012, 0.012, "\n".join(notes), ha="left", va="bottom",
             fontsize=NOTE_PT, color="#4B5563",
             linespacing=NOTE_LINESPACING)

    # Both bounds come from the measured block heights, so a longer footnote
    # grows the canvas instead of colliding with the legend.
    bottom = (note_h + LEGEND_H + 0.20) / fig_h
    top = 1.0 - (HEAD_H - 0.55) / fig_h
    fig.tight_layout(rect=(0, bottom, 1, top))
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
