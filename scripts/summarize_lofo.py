#!/usr/bin/env python3
"""Average a confidence sweep's leave-one-fold-out results (score_sweep.py's
*_lofo.csv) over folds and runs.

Each run's metrics are first averaged over its folds, then reported as mean ±
sd across runs, the same convention as the other tables. One block per
selection metric (the selected_on column). Standard library only.

    python3 scripts/summarize_lofo.py article/conf_sweep/seg_lofo.csv
    python3 scripts/summarize_lofo.py article/conf_sweep/*_lofo.csv -o article/conf_sweep/summary.txt
"""
import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ID_COLUMNS = {"framework", "run", "fold", "selected_on"}


def summarize(path: Path) -> str:
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        return f"{path}: empty\n"
    metrics = [c for c in rows[0] if c not in ID_COLUMNS]
    framework = rows[0]["framework"]

    blocks = []
    by_selection = defaultdict(list)
    for row in rows:
        by_selection[row["selected_on"]].append(row)
    for selected_on, sel_rows in by_selection.items():
        per_run = defaultdict(list)
        for row in sel_rows:
            per_run[row["run"]].append(row)
        runs = sorted(per_run)
        folds = sorted({int(r["fold"]) for r in sel_rows})
        width = max(len(m) for m in metrics)
        how = (
            "stored operating point, no threshold selection" if selected_on == "stored"
            else f"thresholds selected on {selected_on}, leave-one-fold-out"
        )
        lines = [
            f"{framework}: {how} "
            f"({len(runs)} runs x {len(folds)} folds; fold mean per run, then mean ± sd across runs)",
            f"{'metric':<{width}}  {'mean':>8}  {'sd':>7}",
            "-" * (width + 19),
        ]
        for m in metrics:
            values = [r[m] for r in sel_rows if r[m] not in ("", "nan")]
            if not values:
                continue
            run_means = [
                st.fmean(float(r[m]) for r in per_run[run] if r[m] not in ("", "nan")) for run in runs
            ]
            sd = st.stdev(run_means) if len(run_means) > 1 else 0.0
            lines.append(f"{m:<{width}}  {st.fmean(run_means):>8.4f}  {sd:>7.4f}")
        thresholds = defaultdict(int)
        for row in sel_rows:
            thresholds[float(row["threshold"])] += 1
        lines.append("chosen thresholds (count over runs x folds): "
                     + ", ".join(f"{t:.2f}: {n}" for t, n in sorted(thresholds.items())))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="*_lofo.csv files written by scripts/score_sweep.py")
    ap.add_argument("-o", "--output", default=None, help="also write the report here")
    args = ap.parse_args()

    report = "\n".join(summarize(Path(f)) for f in args.files)
    print(report, end="")
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
