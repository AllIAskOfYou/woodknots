#!/usr/bin/env python3
"""Collect every evaluated experiment of one framework into a single table.

One row per experiment under results/<framework>/, one column per metric: the
mean over that experiment's evaluated folds (fold_*/eval_results.json). The
"folds" column shows how many of the k folds were evaluated, so incomplete
experiments stand out. Standard library only.

    python3 scripts/collect_results.py seg
    python3 scripts/collect_results.py yolo --std -o article/yolo_results.txt
    python3 scripts/collect_results.py nnunet --metrics dice_fg_pooled mAP50 --sort mAP50
    python3 scripts/collect_results.py seg --exclude-reference
"""
import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def num_folds() -> int:
    text = (PROJECT_ROOT / "common/conf/dataset/woodknots.yaml").read_text()
    return int(re.search(r"^\s+k:\s*(\d+)", text, re.MULTILINE).group(1))


def load_experiment(exp_dir: Path) -> dict[int, dict]:
    """{fold: metrics} for every evaluated fold of one experiment."""
    per_fold = {}
    for path in sorted(exp_dir.glob("fold_*/eval_results.json")):
        per_fold[int(path.parent.name.split("_")[1])] = json.loads(path.read_text())["metrics"]
    return per_fold


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("framework", help="seg, yolo or nnunet")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--metrics", nargs="+", default=None, help="default: every metric found")
    ap.add_argument("--std", action="store_true", help="add the standard deviation across folds")
    ap.add_argument("--sort", default=None, help="sort rows by this metric, descending")
    ap.add_argument("--exclude-reference", action="store_true",
                    help="leave out every experiment whose name starts with exp_reference")
    ap.add_argument("-o", "--output", default=None,
                    help="default: <results-root>/<framework>/all_results.txt")
    args = ap.parse_args()

    base = PROJECT_ROOT / args.results_root / args.framework
    if not base.is_dir():
        raise SystemExit(f"no such results dir: {base}")

    experiments = {}
    for exp_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if args.exclude_reference and exp_dir.name.startswith("exp_reference"):
            continue
        per_fold = load_experiment(exp_dir)
        if per_fold:
            experiments[exp_dir.name] = per_fold
    if not experiments:
        raise SystemExit(f"no evaluated experiments under {base}")

    # Metric order as the evaluation writes it; metrics only some runs have go last.
    metrics = args.metrics or list(dict.fromkeys(
        m for per_fold in experiments.values() for fold in per_fold.values() for m in fold
    ))

    rows = []
    for name, per_fold in experiments.items():
        cells = {}
        for m in metrics:
            values = [fold[m] for fold in per_fold.values() if m in fold]
            if values:
                cells[m] = (st.fmean(values), st.stdev(values) if len(values) > 1 else 0.0)
        rows.append((name, len(per_fold), cells))
    if args.sort:
        rows.sort(key=lambda r: r[2].get(args.sort, (float("-inf"),))[0], reverse=True)

    k = num_folds()
    name_width = max(len("experiment"), *(len(r[0]) for r in rows))
    cell = lambda mean, sd: f"{mean:.4f}" + (f" ± {sd:.4f}" if args.std else "")
    widths = {m: max(len(m), len(cell(0.0, 0.0))) for m in metrics}

    header = f"{'experiment':<{name_width}}  {'folds':>5}  " + "  ".join(f"{m:>{widths[m]}}" for m in metrics)
    lines = [
        f"{args.framework}: {len(rows)} experiments, mean over evaluated folds"
        + (" ± sd across folds" if args.std else ""),
        "",
        header,
        "-" * len(header),
    ]
    for name, n, cells in rows:
        values = "  ".join(
            f"{cell(*cells[m]) if m in cells else '-':>{widths[m]}}" for m in metrics
        )
        lines.append(f"{name:<{name_width}}  {f'{n}/{k}':>5}  {values}")
    table = "\n".join(lines) + "\n"

    out = Path(args.output) if args.output else base / "all_results.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table)
    print(table, end="")
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
