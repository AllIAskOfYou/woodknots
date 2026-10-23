"""Bootstrap over test images for the per-group results of size_bin_eval.py,
confidence_bin_eval.py, size_confidence_eval.py (and tree_bin_eval.py).

Reads <prefix>_per_image.csv, which those scripts write: per setting, run, fold,
image and group, the knots, found knots and summed matched Dice / IoU. Each
replicate resamples every fold's test images with replacement (keeping the
fold's size), applies the same draw to every run and setting, and recomputes
the statistic exactly as the summaries do: pooled over the fold's images per
group, mean over folds per run, mean over runs. The spread of the replicates is
the uncertainty from which images happen to be in the test set, with the seed
variation averaged out; the summaries' ± is the seed variation alone.

Images of one log are correlated, so resampling single images treats them as
more independent than they are and the intervals are on the narrow side.

For group names of the form "<row>|<column>" (size_confidence_eval.py) the
output is a grid with pooled "all" margins; otherwise one row per group.

    uv run scripts/article/bootstrap_groups.py article/size_confidence/seg_per_image.csv
    uv run scripts/article/bootstrap_groups.py article/size_bins/seg_per_image.csv -B 5000 --seed 1
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

KNOTS, FOUND, DICE, IOU = range(4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("per_image", help="<prefix>_per_image.csv")
    ap.add_argument("-B", "--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", default=None, help="default: <prefix>_bootstrap.txt")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.per_image, newline="")))
    groups = list(dict.fromkeys(r["group"] for r in rows if r["group"]))
    gindex = {g: i for i, g in enumerate(groups)}
    settings = list(dict.fromkeys(r["setting"] for r in rows))
    runs = sorted({r["run"] for r in rows})
    folds = sorted({int(r["fold"]) for r in rows})
    cases = {f: sorted({r["case"] for r in rows if int(r["fold"]) == f}) for f in folds}
    cindex = {f: {c: i for i, c in enumerate(cs)} for f, cs in cases.items()}

    # V[(setting, run, fold)]: (images, groups, 4) counts.
    V = {(s, r, f): np.zeros((len(cases[f]), len(groups), 4))
         for s in settings for r in runs for f in folds}
    for row in rows:
        if not row["group"]:
            continue
        f = int(row["fold"])
        v = V[(row["setting"], row["run"], f)][cindex[f][row["case"]], gindex[row["group"]]]
        v += [int(row["knots"]), int(row["found"]), float(row["dice_sum"]), float(row["iou_sum"])]

    # One resampling per fold, shared by every run and setting: multiplicities (B, images).
    rng = np.random.default_rng(args.seed)
    M = {f: rng.multinomial(len(cases[f]), np.full(len(cases[f]), 1 / len(cases[f])), size=args.replicates)
         for f in folds}
    ones = {f: np.ones((1, len(cases[f]))) for f in folds}
    # S[(setting, run, fold)]: (B, groups, 4) resampled sums; P: the same on the data itself.
    S = {k: np.einsum("bi,igc->bgc", M[k[2]], v) for k, v in V.items()}
    P = {k: np.einsum("bi,igc->bgc", ones[k[2]], v) for k, v in V.items()}

    def statistic(sums, setting, members, metric) -> np.ndarray:
        """Per fold pooled over members, mean over folds per run, mean over runs."""
        per_run = []
        for r in runs:
            per_fold = []
            for f in folds:
                t = sums[(setting, r, f)][:, members].sum(axis=1)  # (B, 4)
                num, den = (t[:, FOUND], t[:, KNOTS]) if metric == "recall" else (t[:, DICE], t[:, FOUND])
                with np.errstate(invalid="ignore", divide="ignore"):
                    per_fold.append(np.where(den > 0, num / den, np.nan))
            per_run.append(np.nanmean(np.stack(per_fold), axis=0))
        return np.nanmean(np.stack(per_run), axis=0)

    def cell(setting, members, metric) -> str:
        point = statistic(P, setting, members, metric)[0]
        boot = statistic(S, setting, members, metric)
        boot = boot[~np.isnan(boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        knots = int(sum(P[(setting, runs[0], f)][0, members, KNOTS].sum() for f in folds))
        return f"{point:.3f} ± {boot.std(ddof=1):.3f} [{lo:.3f}, {hi:.3f}] ({knots})"

    grid = all("|" in g for g in groups)
    lines = [
        f"bootstrap over test images ({args.replicates} replicates, seed {args.seed}; "
        f"{len(runs)} runs x {len(folds)} folds) from {args.per_image}",
        "each cell: estimate ± bootstrap sd [95% percentile interval] (knots over all folds); "
        "the estimate is the summaries' statistic, the spread is image sampling with seeds averaged out",
    ]
    width = 36
    for setting in settings:
        for metric, label in (("recall", "instance recall"), ("dice", "matched Dice")):
            lines += ["", f"=== {setting} threshold, {label}"]
            if grid:
                row_names = list(dict.fromkeys(g.split("|")[0] for g in groups))
                col_names = list(dict.fromkeys(g.split("|")[1] for g in groups))
                members = lambda rs, cs: [gindex[f"{r}|{c}"] for r in rs for c in cs if f"{r}|{c}" in gindex]
                lines.append(f"{'':<8}" + "".join(f"{c:>{width}}" for c in [*col_names, "all"]))
                for rname, rs in [*((r, [r]) for r in row_names), ("all", row_names)]:
                    lines.append(f"{rname:<8}" + "".join(
                        f"{cell(setting, members(rs, cs), metric):>{width}}"
                        for cs in [*([c] for c in col_names), col_names]))
            else:
                w0 = max(8, *(len(g) + 2 for g in groups))
                for name, members in [*((g, [gindex[g]]) for g in groups), ("all", list(range(len(groups))))]:
                    lines.append(f"{name:<{w0}}{cell(setting, members, metric):>{width}}")
    report = "\n".join(lines) + "\n"
    out = Path(args.output) if args.output else Path(str(args.per_image).replace("_per_image.csv", "_bootstrap.txt"))
    out.write_text(report)
    print(report, end="")
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
