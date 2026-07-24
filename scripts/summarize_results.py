#!/usr/bin/env python3
"""Turn two raw microbench JSON artifacts into the repo's published claim table.

Purpose: this repo has already published a wrong headline once, by quoting a
hand-picked subset of the measured cases and by comparing against one of its
own abandoned prototypes. Both of those mistakes are mechanical, so a script
can refuse them.

Intended process, NOT an enforced one. Every latency claim in the README should
be derived from an artifact by this script rather than hand-copied. Nothing
enforces that today: as of this writing no file in the repo references this
script, and there is no .github/workflows/ directory, so no CI runs it. That is
the state of the world, not a description of a pipeline. Wiring it up means
running this script in CI and diffing its "Kernel latency" table against
README.md. Until that exists the derivation is done by hand, and this script is
only the thing that makes the hand step checkable.

Usage:
    py scripts/summarize_results.py OURS.json BASELINE.json

    OURS.json      the build being claimed for
    BASELINE.json  what it is claimed against

It compares any two runs with the same schema, so it also serves
fusion-off vs fusion-on (pass the fusion-off run as BASELINE).

Input schema, verified against
journal/remote/microbench_attn_day0_session25_h100.json and
journal/remote/microbench_attn_scoremod_session25_h100.json:

    {"<case_name>": {"total_us_per_iter": float,
                     "kernels_us": {"<mangled kernel name>": float}}, ...}

with one observed variant: a case that failed to run carries a single "error"
key holding a traceback and no timings (gate_select_T1, gate_select_T4096).
Those are reported as recorded failures, never dropped, never counted as
measurements.

Refusals, each one a past or plausible failure mode:
  1. relproj / relprojT cases are ours, abandoned prototypes living in
     kernels/relproj_score_mod.py. They are NOT a baseline. They are
     quarantined out of every ratio and labelled in the output.
  2. A case in one file and not the other is reported as missing. Silently
     dropping the unmatched cases is exactly what produced the wrong prefill
     claim.
  3. Cases where we are slower get their own section, printed even when
     empty, so a regression cannot hide by absence.
  4. Unreadable file or missing expected key exits non-zero.
  5. sum(kernels_us) must not EXCEED total_us_per_iter. Editing one float is
     the cheapest possible artifact forgery. Note this is an inequality, not an
     equality: both harnesses sum every kernel into the total but truncate
     kernels_us to the top 8 (microbench_attn_day0.py:48-50), so a case with
     more than 8 distinct kernels legitimately has sum(kernels_us) < total.
  6. --tie-band is capped at TIE_BAND_MAX. A band wide enough to reclassify a
     real regression as a tie is refused, any non-zero band is named in the
     section heading and in every verdict it touches, and --fail-on-slower
     counts the workloads a band reclassified, so no band can turn that gate
     green. A tie band cannot launder a loss here.

Ratio convention: ratio = baseline_us / ours_us. Above 1.0 means ours is
faster. A regression is also printed as its reciprocal, "1.28x slower", the
form README.md uses, so deriving the README needs no reciprocal by hand. No
tie band by default, so 1.001x prints as "faster"; pass --tie-band to widen
the noise floor if a run needs it, up to TIE_BAND_MAX.

Exactly one column in the report is a projection rather than a measurement:
"ex-shear ratio" in the ShearingBias section. It is labelled PROJECTED in the
column header and in every cell it prints, because the shear fusion folds that
kernel into qkvr_prep, it does not delete it. Nothing else in the report is
derived.

Exit codes, five of them, unchanged:
    0  report produced, no schema problem
    1  usage error, including a --tie-band above TIE_BAND_MAX
    2  a file is unreadable, is not JSON, or is not a JSON object
    3  schema violation, including a total that is not the sum of its kernels,
       or zero comparable cases between the two files
    4  --strict-errors and the artifacts contain recorded failures
    5  --fail-on-slower and at least one workload is slower, tie band or not
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_UNREADABLE = 2
EXIT_SCHEMA = 3
EXIT_RECORDED_ERRORS = 4
EXIT_SLOWER = 5

# Both keys are required of any case that claims to be a measurement.
REQUIRED_CASE_KEYS = ("total_us_per_iter", "kernels_us")

# sum(kernels_us) must not exceed total_us_per_iter. The harnesses set
# total = sum over ALL kernels but store only the top 8 in kernels_us
# (microbench_attn_day0.py:48-50), so this is an upper bound, not an equality:
# demanding equality would reject any case with more than 8 kernels as a
# forgery. All 7 real cases in
# journal/remote/microbench_attn_*_session25_h100.json satisfy it to float
# exactness (2478.994 + 827.224 + 1.333 + 1.286 = 3308.837). The tolerance is
# loose enough for float accumulation and profiler rounding and far too tight
# to hide a hand-edited total.
TOTAL_SUM_REL_TOL = 0.01   # 1% of the total
# harness/microbench_attn_day0.py:48 keeps only the top 8 kernels in
# kernels_us while total_us_per_iter sums all of them.
KERNELS_US_TOP_N = 8
TOTAL_SUM_ABS_TOL = 0.5    # us, floor for the smallest cases

# Ceiling on --tie-band. Do NOT justify this with a run-to-run spread figure
# unless one is measured: the cross-session spread actually present under
# journal/remote/ reaches 6.2% (decode_b1_global_kv64k, 905.6 us in session 24
# vs 852.6 in session 25) and 11.8% on the score_mod prefill, and those are
# different machines and toolchains rather than a same-box noise floor, so they
# do not license a wide band either. 5% is a convention, not a measurement.
# Anything wider is not a noise floor, it is a regression being reclassified as
# a tie, which is the exact thing this script exists to refuse.
TIE_BAND_MAX = 0.05

# Marker printed in the header and in every cell of the one derived column.
PROJECTED_TAG = "PROJECTED"

# Substring test, case-insensitive, matches relproj_* and relprojT_*.
PROTOTYPE_MARKER = "relproj"
PROTOTYPE_LABEL = "ours, abandoned prototype, NOT a baseline"

# The ShearingBias pre-kernel. Matched on "hearing" so it survives both the
# CamelCase class name and any snake_case spelling in the mangled symbol.
SHEAR_MARKER = "hearing"

# Baseline-only reference points: attention with no bias term at all.
FLOOR_MARKER = "plain"
FLOOR_NOTE = "biasless floor, reference point, not a coverage gap"

# A case we measured and the baseline file does not contain. Nobody ran the
# baseline harness on that shape, so no comparison exists and none can be
# quoted. That is a gap in our own coverage, not a baseline limitation.
OURS_ONLY_NOTE = ("coverage gap on our side: we never ran the baseline on this "
                  "shape, so no ratio exists for it")

# A baseline case that is not the biasless floor and that we never ran.
BASE_ONLY_NOTE = ("coverage gap on our side: we have no measurement of this "
                  "shape, so no ratio exists for it")


class Artifact:
    """One parsed microbench JSON file."""

    def __init__(self, path: Path, label: str, digest: str):
        self.path = path
        self.label = label
        self.digest = digest
        self.measured: dict[str, dict] = {}   # case -> {"total": float, "kernels": dict}
        self.failed: dict[str, str] = {}      # case -> recorded error text
        self.violations: list[str] = []       # schema problems, fatal


def die(code: int, message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(code)


def load_artifact(path_str: str, label: str | None) -> Artifact:
    """Read and validate one artifact. Exits non-zero if it cannot be read."""
    path = Path(path_str).resolve()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        die(EXIT_UNREADABLE, f"cannot read {path}: {exc}")
    digest = hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        die(EXIT_UNREADABLE, f"{path} is not valid JSON: {exc}")
    if not isinstance(data, dict):
        die(EXIT_UNREADABLE,
            f"{path} top level is {type(data).__name__}, expected a JSON object "
            "of case_name -> case")

    art = Artifact(path, label or path.name, digest)
    for case, body in data.items():
        if not isinstance(body, dict):
            art.violations.append(
                f"{art.label}:{case}: case body is {type(body).__name__}, "
                "expected an object")
            continue

        # A case that failed to run carries "error" and no timings. That is a
        # recorded failure, not a broken artifact.
        if "error" in body and not any(k in body for k in REQUIRED_CASE_KEYS):
            text = str(body["error"]).strip()
            art.failed[case] = text.splitlines()[-1] if text else "(empty error)"
            continue

        missing = [k for k in REQUIRED_CASE_KEYS if k not in body]
        if missing:
            art.violations.append(
                f"{art.label}:{case}: missing expected key(s) {', '.join(missing)}")
            continue

        total = body["total_us_per_iter"]
        if isinstance(total, bool) or not isinstance(total, (int, float)):
            art.violations.append(
                f"{art.label}:{case}: total_us_per_iter is "
                f"{total!r}, expected a number")
            continue
        if not total > 0:
            art.violations.append(
                f"{art.label}:{case}: total_us_per_iter is {total}, "
                "expected a positive number")
            continue

        kernels = body["kernels_us"]
        if not isinstance(kernels, dict):
            art.violations.append(
                f"{art.label}:{case}: kernels_us is "
                f"{type(kernels).__name__}, expected an object")
            continue
        bad = [k for k, v in kernels.items()
               if isinstance(v, bool) or not isinstance(v, (int, float))]
        if bad:
            art.violations.append(
                f"{art.label}:{case}: kernels_us has non-numeric value(s) for "
                f"{', '.join(sorted(bad)[:3])}")
            continue

        if not kernels:
            art.violations.append(
                f"{art.label}:{case}: kernels_us is empty, so the reported "
                f"total_us_per_iter of {float(total)} is not accounted for by "
                "any kernel")
            continue

        # sum(kernels_us) must not EXCEED the total. Editing either side is the
        # cheapest possible artifact forgery, and a total smaller than the
        # kernels it is supposed to contain is impossible.
        #
        # An upper bound, deliberately, not an equality: the harnesses set
        # total = sum over ALL kernels but keep only the top 8 in kernels_us
        # (harness/microbench_attn_day0.py:48-50). A case with more than 8
        # distinct kernels therefore has sum(kernels_us) < total legitimately,
        # and demanding equality would reject a real measurement as a forgery.
        # Equality IS checked when the case has fewer than 8 kernels, which is
        # where truncation cannot be the explanation.
        ksum = sum(float(v) for v in kernels.values())
        allowed = max(TOTAL_SUM_ABS_TOL, TOTAL_SUM_REL_TOL * float(total))
        truncated = len(kernels) >= KERNELS_US_TOP_N
        over = ksum - float(total)
        under = float(total) - ksum
        bad = over > allowed or (not truncated and under > allowed)
        if bad:
            art.violations.append(
                f"{art.label}:{case}: total_us_per_iter is {float(total):.4f} "
                f"but kernels_us ({len(kernels)} kernels) sums to {ksum:.4f}, "
                f"off by {abs(over):.4f} us against an allowed {allowed:.4f} us "
                f"(max of {TOTAL_SUM_ABS_TOL} us and "
                f"{TOTAL_SUM_REL_TOL:.0%} of the total). "
                + (
                    "sum(kernels_us) exceeds the total, which is impossible."
                    if over > allowed
                    else "The case has fewer than "
                    f"{KERNELS_US_TOP_N} kernels, so top-N truncation cannot "
                    "explain the shortfall."
                )
                + " This case is not a usable measurement")
            continue

        art.measured[case] = {"total": float(total),
                              "kernels": {k: float(v) for k, v in kernels.items()}}
    return art


def is_prototype(case: str) -> bool:
    return PROTOTYPE_MARKER in case.lower()


def is_floor(case: str) -> bool:
    return FLOOR_MARKER in case.lower()


def shear_us(kernels: dict[str, float]) -> float:
    """Microseconds spent in the ShearingBias pre-kernel, 0.0 if it is absent."""
    return sum(v for k, v in kernels.items() if SHEAR_MARKER in k.lower())


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Markdown table, padded so the raw stdout is readable too."""
    if not rows:
        rows = [["(none)"] + [""] * (len(headers) - 1)]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |",
           "|-" + "-|-".join("-" * w for w in widths) + "-|"]
    for row in rows:
        out.append("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |")
    return out


def fmt_us(x: float) -> str:
    return f"{x:.1f}"


def slower_form(ratio: float) -> str:
    """A below-1.0 ratio in the reciprocal form README.md publishes.

    0.78x here is 1.28x slower there. Both are correct and the README uses the
    second, so emitting only the first would leave a hand-taken reciprocal
    between this script and the published claim.
    """
    return f"{1.0 / ratio:.2f}x slower"


def ratio_cell(ratio: float) -> str:
    """Ratio for a table cell, carrying its own reciprocal when it is a loss."""
    if ratio < 1.0:
        return f"{ratio:.2f}x = {slower_form(ratio)}"
    return f"{ratio:.2f}x"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="summarize_results.py",
        description="Derive the published kernel latency claims from two raw "
                    "microbench JSON artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ours", help="JSON artifact for the build being claimed for")
    ap.add_argument("baseline", help="JSON artifact it is claimed against")
    ap.add_argument("--ours-label", default=None,
                    help="column label for the ours file (default: filename)")
    ap.add_argument("--baseline-label", default=None,
                    help="column label for the baseline file (default: filename)")
    ap.add_argument("--tie-band", type=float, default=0.0, metavar="FRAC",
                    help=f"ratios within +/- FRAC of 1.0 print as 'tie' instead "
                         f"of faster/SLOWER (default 0.0, no tie band; capped "
                         f"at {TIE_BAND_MAX}, and any non-zero value is named "
                         f"in the SLOWER heading and in every verdict it "
                         f"changes)")
    ap.add_argument("--strict-errors", action="store_true",
                    help="exit 4 if either artifact contains a recorded "
                         "per-case error")
    ap.add_argument("--fail-on-slower", action="store_true",
                    help="exit 5 if any comparable workload is slower, "
                         "including workloads a --tie-band reclassified as "
                         "ties")
    args = ap.parse_args(argv)

    if args.tie_band < 0:
        die(EXIT_USAGE, "--tie-band must be >= 0")
    if args.tie_band > TIE_BAND_MAX:
        die(EXIT_USAGE,
            f"--tie-band {args.tie_band} is wider than the {TIE_BAND_MAX} cap. "
            f"The {TIE_BAND_MAX:.0%} cap is a convention, not a measured noise "
            "floor, and no same-box run-to-run spread has been measured for "
            "these kernels. A band above it is not a noise floor, it is a "
            "regression being reclassified as a tie. Report the regression "
            "instead")
    band = args.tie_band

    ours = load_artifact(args.ours, args.ours_label)
    base = load_artifact(args.baseline, args.baseline_label)

    if ours.path == base.path:
        die(EXIT_USAGE, f"both arguments point at the same file: {ours.path}")

    # Quarantine the abandoned prototypes before anything computes a ratio.
    proto = sorted({c for c in list(ours.measured) + list(base.measured)
                    if is_prototype(c)})
    ours_cases = {c for c in ours.measured if not is_prototype(c)}
    base_cases = {c for c in base.measured if not is_prototype(c)}

    shared = sorted(ours_cases & base_cases)
    only_ours = sorted(ours_cases - base_cases)
    only_base = sorted(base_cases - ours_cases)

    rows, slower, ties, banded_away = [], [], [], []
    measured_verdict: dict[str, tuple[float, str]] = {}
    for case in shared:
        o = ours.measured[case]["total"]
        b = base.measured[case]["total"]
        ratio = b / o
        raw = "faster" if ratio >= 1.0 else "SLOWER"
        if abs(ratio - 1.0) <= band:
            if band > 0:
                verdict = (f"tie, TIE BAND +/-{band:.3f} APPLIED "
                           f"(raw {ratio:.2f}x {raw})")
            else:
                verdict = "tie"
            ties.append((case, o, b, ratio))
            if raw == "SLOWER":
                banded_away.append((case, o, b, ratio))
        elif raw == "faster":
            verdict = "faster"
        else:
            verdict = "SLOWER"
            slower.append((case, o, b, ratio))
        measured_verdict[case] = (ratio, raw)
        rows.append([case, fmt_us(o), fmt_us(b), ratio_cell(ratio), verdict])

    out: list[str] = []
    add = out.append

    add("# Kernel latency summary")
    add("")
    add("Generated by `scripts/summarize_results.py` from the raw artifacts "
        "below.")
    add("")
    add("**Every number in this report is read out of those two files, with "
        "exactly one exception, and the exception is labelled.** The exception "
        f"is the `ex-shear ratio ({PROJECTED_TAG})` column in the ShearingBias "
        "section: it is derived by subtracting a kernel that still runs, so it "
        f"is a projection, not a measurement. It carries `{PROJECTED_TAG}` in "
        "its header and in every cell it prints, so a row lifted out of that "
        "table still says so. Nothing else here is estimated, projected, or "
        "hand-entered.")
    add("")
    add("## Sources")
    add("")
    out.extend(table(
        ["role", "label", "path", "sha256 (first 16)"],
        [["ours", ours.label, str(ours.path), ours.digest[:16]],
         ["baseline", base.label, str(base.path), base.digest[:16]]]))
    add("")
    add("Ratio = baseline us / ours us. Above 1.00x means ours is faster. "
        "Below 1.00x is a regression, and every such ratio also prints its "
        "reciprocal in the `Nx slower` form README.md uses, so no reciprocal "
        "is taken by hand.")
    add("")
    if band > 0:
        add(f"**A tie band of +/-{band:.3f} is in effect** (cap "
            f"{TIE_BAND_MAX:.3f}). Ratios inside it print as `tie` and are "
            "excluded from the SLOWER section below. Every verdict it changes "
            "names it. Default is 0.000, which reports every ratio as "
            "measured.")
        if banded_away:
            add("")
            add(f"**WARNING: that band reclassified {len(banded_away)} "
                "measured regression(s) as ties.** Without it the SLOWER "
                "section would contain: "
                + "; ".join(f"`{c}` at {ratio_cell(r)}"
                            for c, _, _, r in banded_away) + ".")
    else:
        add("Tie band: +/- 0.000, none applied. Every ratio is reported as "
            "measured.")
    add("")

    add("## Kernel latency, us per iter")
    add("")
    out.extend(table(
        ["workload", f"ours ({ours.label})", f"baseline ({base.label})",
         "ratio", "verdict"],
        rows))
    add("")
    if not shared:
        add("No workload appears in both files. There is nothing to compare.")
        add("")

    # Slower section, printed even when empty so a regression cannot hide.
    # A non-zero tie band is named in the heading, because this section is
    # where such a band would otherwise do its laundering.
    add("## Workloads where we are SLOWER"
        + (f" (TIE BAND +/-{band:.3f} APPLIED)" if band > 0 else ""))
    add("")
    if slower:
        out.extend(table(
            ["workload", "ours us", "baseline us", "ratio", "same, as README "
             "publishes it"],
            [[c, fmt_us(o), fmt_us(b), f"{r:.2f}x", slower_form(r)]
             for c, o, b, r in slower]))
        add("")
        add(f"{len(slower)} of {len(shared)} comparable workloads are slower. "
            "This section is part of the published claim, not a footnote.")
        if band > 0:
            add("")
            add(f"A tie band of +/-{band:.3f} was applied, so this section is "
                f"already filtered. {len(banded_away)} further workload(s) are "
                "slower than the baseline and were reclassified as ties by "
                "that band.")
    elif banded_away:
        add(f"**Empty only because of the +/-{band:.3f} tie band.** "
            f"{len(banded_away)} of {len(shared)} comparable workloads are "
            "measurably slower than the baseline and were reclassified as "
            "ties: "
            + "; ".join(f"`{c}` at {ratio_cell(r)}"
                        for c, _, _, r in banded_away)
            + ". Re-run with the default `--tie-band 0.0` to see this section "
              "unfiltered. Do not quote this section as a parity result.")
    else:
        add("None. All " + str(len(shared)) + " comparable workloads are at or "
            "above parity"
            + (f", with a tie band of +/-{band:.3f} applied that reclassified "
               "no slower workload" if band > 0 else "")
            + ". This section prints even when empty, so its absence never "
              "means 'not checked'.")
    add("")
    if ties:
        add(f"Inside the +/-{band:.3f} tie band: "
            + ", ".join(f"{c} ({ratio_cell(r)})" for c, _, _, r in ties) + ".")
        add("")

    # ShearingBias share, ours only. This is the cost the shear fusion targets.
    add("## ShearingBias share of our total")
    add("")
    base_has_shear = any(shear_us(v["kernels"]) > 0 for v in base.measured.values())
    add("The ShearingBias pre-kernel is a separate launch in our path. "
        + ("The baseline file also contains it, so the ratio column above "
           "already nets it out on both sides. "
           if base_has_shear else
           "The baseline file does not contain it at all, so it is pure "
           "overhead on our side of every ratio above. ")
        + "`ex-shear` is our total minus that kernel.")
    add("")
    add(f"**The last TWO columns are {PROJECTED_TAG}, not results.** "
        f"`ex-shear us` is the latency of a hypothetical build in which "
        f"ShearingBias is free; it appears in neither artifact. "
        f"`ex-shear ratio ({PROJECTED_TAG})` is the ceiling the "
        "shear fusion could reach, not a measured result: the fusion folds "
        "this work into qkvr_prep, it does not delete it. No build has ever "
        "produced these numbers. The measured verdict for each workload sits "
        "in the column immediately to its left, so the two are never read "
        f"apart. `ex-shear` also still contains any small helper kernels in "
        "the case, so it sits slightly above the attention kernel on its own. "
        "Do not quote it as attention-alone time, and do not quote the "
        f"{PROJECTED_TAG} column as a result.")
    add("")
    shear_rows, flips = [], []
    for case in sorted(ours.measured):
        if is_prototype(case):
            continue
        total = ours.measured[case]["total"]
        s = shear_us(ours.measured[case]["kernels"])
        ex = total - s
        if case in measured_verdict:
            m_ratio, m_raw = measured_verdict[case]
            m_cell = (f"{m_ratio:.2f}x SLOWER = {slower_form(m_ratio)}"
                      if m_raw == "SLOWER" else f"{m_ratio:.2f}x faster")
        else:
            m_ratio, m_raw, m_cell = None, None, "n/a, not in baseline"
        if case in base_cases and ex > 0:
            p_ratio = base.measured[case]["total"] / ex
            p_cell = f"{PROJECTED_TAG} {p_ratio:.2f}x"
            if m_raw == "SLOWER" and p_ratio >= 1.0:
                flips.append((case, m_ratio, p_ratio))
        else:
            p_ratio, p_cell = None, "n/a"
        shear_rows.append([case, fmt_us(total),
                           fmt_us(s) if s else "0.0 (absent)",
                           f"{100.0 * s / total:.1f}%",
                           f"{fmt_us(ex)} ({PROJECTED_TAG})",
                           m_cell, p_cell])
    out.extend(table(
        ["workload", "our total us", "ShearingBias us", "share",
         f"ex-shear us ({PROJECTED_TAG}, NOT MEASURED)",
         "measured ratio (RESULT)",
         f"ex-shear ratio ({PROJECTED_TAG}, NOT MEASURED)"],
        shear_rows))
    add("")
    if flips:
        add(f"**Verdict flip warning.** On the following workload(s) the "
            f"{PROJECTED_TAG} column crosses 1.00x while the measured result "
            "does not. A row lifted out of this table would read as a win on "
            "a workload this repo currently loses:")
        add("")
        for case, m_ratio, p_ratio in flips:
            add(f"- `{case}`: measured **{m_ratio:.2f}x, "
                f"{slower_form(m_ratio)}**. Projected ex-shear "
                f"{p_ratio:.2f}x, which is a ceiling for work not yet done. "
                "The measured number is the result.")
        add("")

    # Prototype quarantine.
    add("## Quarantined: relproj / relprojT")
    add("")
    if proto:
        add(f"These cases are **{PROTOTYPE_LABEL}** (kernels/relproj_score_mod.py). "
            "They are excluded from every ratio above. Comparing against them "
            "is comparing against ourselves, which this repo has published "
            "once by mistake.")
        add("")
        out.extend(table(
            ["case", "label", "appears in", "total us"],
            [[c, PROTOTYPE_LABEL,
              ", ".join(x.label for x in (ours, base) if c in x.measured),
              ", ".join(f"{x.label}={fmt_us(x.measured[c]['total'])}"
                        for x in (ours, base) if c in x.measured)]
             for c in proto]))
    else:
        add("No relproj or relprojT case found in either file.")
    add("")

    # Missing cases. Never silently dropped.
    add("## Cases present in one file only")
    add("")
    add("Reported, not dropped. A dropped unmatched case is what produced the "
        "wrong prefill claim. Prototype cases are listed in the quarantine "
        "section above instead.")
    add("")
    miss_rows = []
    for c in only_ours:
        miss_rows.append([c, ours.label, f"missing from {base.label}",
                          fmt_us(ours.measured[c]["total"]), OURS_ONLY_NOTE])
    for c in only_base:
        miss_rows.append([c, base.label, f"missing from {ours.label}",
                          fmt_us(base.measured[c]["total"]),
                          FLOOR_NOTE if is_floor(c) else BASE_ONLY_NOTE])
    out.extend(table(["case", "present in", "status", "total us", "note"],
                     miss_rows))
    add("")

    # Recorded per-case failures.
    add("## Recorded failures in the artifacts")
    add("")
    failures = [(x.label, c, msg)
                for x in (ours, base) for c, msg in sorted(x.failed.items())]
    if failures:
        add("These cases ran and failed. They carry an `error` key and no "
            "timings. They are not measurements and are not comparable.")
        add("")
        out.extend(table(["file", "case", "last line of recorded error"],
                         [[lbl, c, msg] for lbl, c, msg in failures]))
    else:
        add("None.")
    add("")

    violations = ours.violations + base.violations
    if violations:
        add("## Schema violations")
        add("")
        add("A case is neither a valid measurement nor a recorded error. The "
            "run exits non-zero.")
        add("")
        for v in violations:
            add(f"- {v}")
        add("")

    add("## Counts")
    add("")
    out.extend(table(
        ["file", "measured", "failed", "prototype", "schema violations"],
        [[x.label, str(sum(1 for c in x.measured if not is_prototype(c))),
          str(len(x.failed)),
          str(sum(1 for c in x.measured if is_prototype(c))),
          str(len(x.violations))] for x in (ours, base)]))
    add("")

    print("\n".join(out))

    # Exit status. Loud in a pipeline, in severity order.
    if violations:
        print(f"ERROR: {len(violations)} schema violation(s), see report",
              file=sys.stderr)
        return EXIT_SCHEMA
    if not shared:
        print("ERROR: zero comparable workloads between the two files; check "
              "the arguments are two runs of the same benchmark",
              file=sys.stderr)
        return EXIT_SCHEMA
    if args.strict_errors and failures:
        print(f"ERROR: {len(failures)} recorded per-case failure(s) and "
              "--strict-errors is set", file=sys.stderr)
        return EXIT_RECORDED_ERRORS
    # A tie band must not be able to turn this gate green. It filters what the
    # SLOWER section prints, it does not change which workloads are slower, so
    # the gate counts the banded ones too. With the default band of 0.0 there
    # are none and this is exactly the previous behaviour.
    if args.fail_on_slower and (slower or banded_away):
        detail = ""
        if banded_away:
            detail = (f" ({len(banded_away)} of them inside the +/-{band:.3f} "
                      "tie band; the band does not clear this gate)")
        print(f"ERROR: {len(slower) + len(banded_away)} workload(s) slower and "
              f"--fail-on-slower is set{detail}", file=sys.stderr)
        return EXIT_SLOWER
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
