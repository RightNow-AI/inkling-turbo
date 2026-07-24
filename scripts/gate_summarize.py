#!/usr/bin/env python3
"""Summarize gate_e2e_bench.sh results: stock vs ours, median of 5 + best.

Reads the ~/bench_results/<build>/<mix>/<concurrency>/run<N>.json tree
written by `vllm bench serve --save-result` and emits a markdown comparison
table per mix/concurrency to ~/gate_summary.md.

Metric keys, verified against the pinned fork @850295881
($REPO/vllm/vllm/benchmarks/serve.py):
  - request_throughput, output_throughput: result dict, lines 1200-1203
  - median_ttft_ms / median_tpot_ms / median_itl_ms and the p99_*_ms
    variants: process_one_metric writes mean_/median_/std_/p<P>_ keys per
    selected metric, lines 1276-1288 (gate_e2e_bench.sh passes
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 99)
  - completed / failed request counts: lines 1196-1197

Evidence rules (docs/METHODOLOGY.md, measured-or-null ledger):
  - median of the 5 runs AND best, never best-only
  - a missing or unreadable run is simply absent; if a config has zero
    runs its cells render as null. Nothing is estimated or fabricated.
"""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

BUILDS = ["stock", "ours"]

# Deliberately NOT a hardcoded table. It used to read
#   "prefill": "prefill-heavy (random, 8192 in / 128 out)"
# which was the old Lambda script's shape. scripts/modal_e2e_bench.py runs
# 2048 in, and 8192 is impossible there because _parse_mixes rejects any mix
# over MAX_MODEL_LEN = 3072. So the summarizer this repo tells you to run would
# have published an 8192-in label over 2048-in measurements. Labels now come
# from the run's own manifest.json, and if that is missing the length is left
# unstated rather than guessed.
MIX_KINDS = {"prefill": "prefill-heavy", "decode": "decode-heavy"}


def mix_labels(root: Path) -> dict:
    """Build mix labels from the run's manifest, never from a constant."""
    labels = {k: f"{v} (shape not recorded)" for k, v in MIX_KINDS.items()}
    manifest = root / "manifest.json"
    if not manifest.exists():
        print(
            f"WARNING: no manifest.json under {root}. Input and output lengths "
            "will be left unstated rather than assumed.",
            file=sys.stderr,
        )
        return labels
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: unreadable manifest.json: {exc}", file=sys.stderr)
        return labels
    # modal_e2e_bench writes {"matrix": {"mixes": [{"name","input_len",
    # "output_len"}, ...]}}. Accept a bare list too, so an older manifest still
    # resolves instead of silently falling back to "shape not recorded".
    matrix = data.get("matrix")
    mixes = matrix.get("mixes") if isinstance(matrix, dict) else matrix
    if mixes is None:
        mixes = data.get("mixes")
    found = 0
    for m in mixes or []:
        if not isinstance(m, dict):
            continue
        name = m.get("name") or m.get("mix")
        ilen, olen = m.get("input_len"), m.get("output_len")
        if name in labels and ilen and olen:
            labels[name] = (
                f"{MIX_KINDS[name]} (random, {ilen} in / {olen} out)"
            )
            found += 1
    if not found:
        print(
            f"WARNING: manifest.json at {manifest} carried no mix shapes; "
            "lengths left unstated.",
            file=sys.stderr,
        )
    return labels

# (json_key, label, unit, better) -- better: "high" or "low"
METRICS = [
    ("request_throughput", "Request throughput", "req/s", "high"),
    ("output_throughput", "Output token throughput", "tok/s", "high"),
    ("median_ttft_ms", "Median TTFT", "ms", "low"),
    ("p99_ttft_ms", "P99 TTFT", "ms", "low"),
    ("median_tpot_ms", "Median TPOT", "ms", "low"),
    ("p99_tpot_ms", "P99 TPOT", "ms", "low"),
    ("p99_itl_ms", "P99 ITL", "ms", "low"),
    ("p99_e2el_ms", "P99 E2E latency", "ms", "low"),
]


def load_runs(cfg_dir: Path) -> list[dict]:
    runs = []
    for f in sorted(cfg_dir.glob("run*.json")):
        try:
            runs.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARN: skipping unreadable {f}: {exc}", file=sys.stderr)
    return runs


def med_best(runs: list[dict], key: str, better: str):
    """(median, best) across runs for one metric key; (None, None) if no
    run carries the key. Missing stays null, never fabricated."""
    vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
    if not vals:
        return None, None
    med = statistics.median(vals)
    best = max(vals) if better == "high" else min(vals)
    return med, best


def fmt(v) -> str:
    if v is None:
        return "null"
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.3f}"


def ratio(ours, stock) -> str:
    if ours is None or stock is None or stock == 0:
        return "null"
    return f"{ours / stock:.3f}x"


def discover_concs(root: Path, mix: str) -> list[int]:
    concs = set()
    for build in BUILDS:
        d = root / build / mix
        if d.is_dir():
            for sub in d.iterdir():
                if sub.is_dir() and sub.name.isdigit():
                    concs.add(int(sub.name))
    return sorted(concs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.home() / "bench_results")
    ap.add_argument("--out", type=Path, default=Path.home() / "gate_summary.md")
    ap.add_argument("--expected-runs", type=int, default=5)
    args = ap.parse_args()

    if not args.root.is_dir():
        print(f"FATAL: results root {args.root} does not exist", file=sys.stderr)
        return 2

    lines = [
        "# Gate summary: stock day-0 vs Inkling-turbo kernels",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Results root: `{args.root}`",
        "",
        "Rules: median of "
        f"{args.expected_runs} runs plus best per config "
        "(docs/METHODOLOGY.md). Cells are measured-or-null; `null` means "
        "the runs do not exist or the metric was absent, never an estimate. "
        "Run counts below the expected "
        f"{args.expected_runs} are flagged inline.",
        "",
    ]

    any_data = False
    for mix, mix_desc in MIXES.items():
        lines.append(f"## Mix: {mix_desc}")
        lines.append("")
        concs = discover_concs(args.root, mix)
        if not concs:
            lines.append("No results for this mix (null).")
            lines.append("")
            continue
        for conc in concs:
            runs = {b: load_runs(args.root / b / mix / str(conc))
                    for b in BUILDS}
            counts = {b: len(runs[b]) for b in BUILDS}
            note = ", ".join(
                f"{b}: {counts[b]}/{args.expected_runs} runs"
                + (" (INCOMPLETE)" if counts[b] < args.expected_runs else "")
                for b in BUILDS)
            lines.append(f"### max-concurrency {conc}  ({note})")
            lines.append("")
            lines.append(
                "| Metric | stock median | stock best | ours median | "
                "ours best | ours/stock (median) |")
            lines.append("|---|---|---|---|---|---|")
            for key, label, unit, better in METRICS:
                s_med, s_best = med_best(runs["stock"], key, better)
                o_med, o_best = med_best(runs["ours"], key, better)
                if any(v is not None for v in (s_med, o_med)):
                    any_data = True
                lines.append(
                    f"| {label} ({unit}) | {fmt(s_med)} | {fmt(s_best)} | "
                    f"{fmt(o_med)} | {fmt(o_best)} | {ratio(o_med, s_med)} |")
            failed = {
                b: sum(int(r.get("failed") or 0) for r in runs[b])
                for b in BUILDS}
            if any(failed.values()):
                lines.append("")
                lines.append(
                    f"WARNING: failed requests present "
                    f"(stock: {failed['stock']}, ours: {failed['ours']}); "
                    "treat affected medians as suspect.")
            lines.append("")
        lines.append("")

    lines.append("---")
    lines.append(
        "Provenance: raw per-run JSON written by `vllm bench serve "
        "--save-result` (vllm/benchmarks/serve.py); one file per run under "
        "`<build>/<mix>/<concurrency>/run<N>.json`. Same checkpoint, same "
        "GPUs, same SLO across builds (docs/METHODOLOGY.md).")
    lines.append("")

    args.out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"summary written: {args.out}")
    if not any_data:
        print("NOTE: no measured data found; summary is all null.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
