"""Instance recall and matched Dice of the seg pair ensembles by knot size and
annotator confidence together: a size x confidence grid.

Size and confidence are correlated (low-confidence knots are mostly small), so
recall per size or per confidence alone cannot tell which of the two matters.
Within one size bin, confidence no longer varies with size, and vice versa.

The same analysis as size_bin_eval.py (cached predictions, default and tuned
leave-one-fold-out threshold, one matching per image split by group), with the
9 cells as groups. Besides the usual outputs it writes <prefix>_grid.txt: per
setting, a recall and a matched-Dice grid, each cell mean ± sd across the runs
(per fold, mean over folds per run) and its knot count over all folds. The
"all" row and column pool the cells' counts per fold, so they reproduce the
size-only and confidence-only results.

    uv run scripts/article/size_confidence_eval.py
"""
import argparse
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from size_bin_eval import (
    CONFIDENCE_TOLERANCE,
    CONFIDENCES,
    PROJECT_ROOT,
    add_common_args,
    bin_index,
    confidence,
    evaluate,
    mean_sd,
    radius,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bins", default=str(PROJECT_ROOT / "article/knot_size_bins.json"))
    add_common_args(ap, "article/size_confidence/seg")
    args = ap.parse_args()

    spec = json.loads(Path(args.bins).read_text())
    edges, sizes = spec["edges"], spec["names"]
    levels = sorted(CONFIDENCES)
    cells = [(s, c) for s in range(len(sizes)) for c in range(len(levels))]
    names = [f"{sizes[s]}|{levels[c]:g}" for s, c in cells]

    def group_of(ann: dict) -> int:
        s = bin_index(radius(ann), edges)
        value = confidence(ann)
        c = next(i for i, level in enumerate(levels) if abs(value - level) < CONFIDENCE_TOLERANCE)
        return cells.index((s, c))

    bounds = [0.0, *edges, math.inf]
    evaluate(
        args, names, group_of, "by knot size x annotator confidence",
        "cells: size bin on r = sqrt(area / pi), px ("
        + ", ".join(f"{n} [{lo:g}, {hi:g})" for n, lo, hi in zip(sizes, bounds[:-1], bounds[1:]))
        + ") x confidence (" + ", ".join(f"{c:g}" for c in levels) + ")",
    )

    # The grids, from the per-fold counts evaluate() wrote.
    counts = defaultdict(lambda: [0, 0, 0.0])  # (setting, run, fold, s, c) -> knots, found, dice_sum
    for row in csv.DictReader(open(f"{args.output}_per_fold.csv", newline="")):
        s, c = cells[names.index(row["group"])]
        found = int(row["found"])
        entry = counts[(row["setting"], row["run"], int(row["fold"]), s, c)]
        entry[0] += int(row["knots"])
        entry[1] += found
        entry[2] += float(row["matched_dice"]) * found if found else 0.0

    def value(setting, rows, cols, metric):
        """mean ± sd across runs of the fold mean, cells in rows x cols pooled per fold."""
        per_run = []
        for run in args.runs:
            folds = []
            for f in range(args.folds):
                knots = found = dice = 0
                for s in rows:
                    for c in cols:
                        k, n, d = counts[(setting, run, f, s, c)]
                        knots, found, dice = knots + k, found + n, dice + d
                if metric == "recall" and knots:
                    folds.append(found / knots)
                elif metric == "dice" and found:
                    folds.append(dice / found)
            per_run.append(st.fmean(folds) if folds else math.nan)
        return mean_sd(per_run)

    def total_knots(rows, cols):
        run = args.runs[0]
        return sum(counts[("default", run, f, s, c)][0]
                   for f in range(args.folds) for s in rows for c in cols)

    all_s, all_c = range(len(sizes)), range(len(levels))
    col_names = [f"conf {c:g}" for c in levels] + ["all"]
    col_sets = [[c] for c in all_c] + [list(all_c)]
    row_sets = [([s], sizes[s]) for s in all_s] + [(list(all_s), "all")]
    width = 26
    lines = [
        f"seg pair ensembles ({len(args.runs)} runs x {args.folds} folds) by knot size (rows) and "
        f"annotator confidence (columns)",
        "each cell: mean ± sd across runs of the fold mean, (knots in the cell over all folds)",
    ]
    for setting in ("default", "tuned"):
        for metric, label in (("recall", "instance recall"), ("dice", "matched Dice")):
            lines += ["", f"=== {setting} threshold, {label}",
                      f"{'':<8}" + "".join(f"{n:>{width}}" for n in col_names)]
            for rows, rname in row_sets:
                cells_text = [
                    f"{value(setting, rows, cols, metric)} ({total_knots(rows, cols)})" for cols in col_sets
                ]
                lines.append(f"{rname:<8}" + "".join(f"{t:>{width}}" for t in cells_text))
    grid = "\n".join(lines) + "\n"
    Path(f"{args.output}_grid.txt").write_text(grid)
    print("\n" + grid, end="")
    print(f"\nSaved {args.output}_grid.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
