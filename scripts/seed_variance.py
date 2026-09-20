#!/usr/bin/env python3
"""Report how much a framework's scores move when only the seed changes.

Reads results/<framework>/<prefix><seed>/fold_<i>/eval_results.json for every
seed run of one experiment (by default exp_reference_0, exp_reference_1, ...),
and reports two spreads: across seeds of the k-fold mean (the number to quote
when comparing experiments) and within a single fold (what a single-fold
comparison has to beat). Framework-agnostic: metrics come from whatever the
eval files contain. Standard library only.

    python3 scripts/seed_variance.py seg -o article/seed_variance_seg.txt
    python3 scripts/seed_variance.py yolo --prefix exp_seed_ -o seeds.json
"""
import argparse
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_runs(results_root: Path, framework: str, prefix: str) -> dict[int, dict[int, dict]]:
    """{seed: {fold: metrics}} for every evaluated <prefix><seed>/fold_<i> run."""
    base = results_root / framework
    if not base.is_dir():
        raise SystemExit(f"no such results dir: {base}")

    runs: dict[int, dict[int, dict]] = {}
    for exp_dir in sorted(base.iterdir()):
        match = re.fullmatch(re.escape(prefix) + r"(\d+)", exp_dir.name)
        if not (match and exp_dir.is_dir()):
            continue
        per_fold = {}
        for fold_dir in sorted(exp_dir.glob("fold_*")):
            eval_path = fold_dir / "eval_results.json"
            if eval_path.exists():
                per_fold[int(fold_dir.name.split("_")[1])] = json.loads(eval_path.read_text())["metrics"]
        if per_fold:
            runs[int(match.group(1))] = per_fold
    return runs


def summarize(runs: dict[int, dict[int, dict]]) -> dict:
    seeds = sorted(runs)
    folds = sorted({f for per_fold in runs.values() for f in per_fold})
    complete = [f for f in folds if all(f in runs[s] for s in seeds)]
    metrics = list(runs[seeds[0]][complete[0]])

    out = {"seeds": seeds, "folds": complete, "metrics": {}}
    for m in metrics:
        table = {s: {f: runs[s][f][m] for f in complete} for s in seeds}
        fold_means = {s: st.fmean(table[s].values()) for s in seeds}
        per_fold_sd = {f: st.stdev([table[s][f] for s in seeds]) for f in complete}
        means = list(fold_means.values())
        out["metrics"][m] = {
            "per_seed": table,
            "fold_mean_per_seed": fold_means,
            "mean": st.fmean(means),
            "sd": st.stdev(means),
            "variance": st.variance(means),
            "range": max(means) - min(means),
            "per_fold_sd_across_seeds": per_fold_sd,
            "mean_within_fold_sd": st.fmean(per_fold_sd.values()),
        }
    return out


def render(summary: dict, framework: str, prefix: str) -> str:
    seeds, folds = summary["seeds"], summary["folds"]
    lines = [
        f"seed variance — {framework}, {prefix}[{min(seeds)}-{max(seeds)}]",
        f"{len(seeds)} seeds x {len(folds)} folds = {len(seeds) * len(folds)} runs",
    ]
    for m, s in summary["metrics"].items():
        lines.append(f"\n=== {m}")
        lines.append("      " + " ".join(f"fold{f}".rjust(8) for f in folds) + f"   {len(folds)}-fold mean")
        for seed in seeds:
            row = " ".join(f"{s['per_seed'][seed][f]:8.4f}" for f in folds)
            lines.append(f"seed{seed:<2}{row}   {s['fold_mean_per_seed'][seed]:.4f}")
        lines.append(
            " per-fold sd across seeds: " + " ".join(f"{s['per_fold_sd_across_seeds'][f]:.4f}" for f in folds)
        )
        lines.append(
            f" across seeds (of the {len(folds)}-fold mean): mean={s['mean']:.4f} sd={s['sd']:.4f} "
            f"var={s['variance']:.2e} range={s['range']:.4f}"
        )
        lines.append(f" mean within-fold sd across seeds: {s['mean_within_fold_sd']:.4f}")
        lines.append(
            f" -> differences below ~{2 * s['sd']:.4f} ({m}, {len(folds)}-fold mean) or "
            f"~{2 * s['mean_within_fold_sd']:.4f} (single fold) are seed noise"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("framework", help="seg, yolo, nnunet, ...")
    ap.add_argument(
        "-o", "--output", required=True,
        help="report file; .json writes the full numbers, anything else the text report",
    )
    ap.add_argument("--prefix", default="exp_reference_", help="experiment prefix before the seed (default: exp_reference_)")
    ap.add_argument("--results-root", default=str(PROJECT_ROOT / "results"), help="default: <project>/results")
    args = ap.parse_args()

    runs = load_runs(Path(args.results_root), args.framework, args.prefix)
    if len(runs) < 2:
        raise SystemExit(
            f"need at least 2 seed runs matching {args.prefix}<seed> under "
            f"{Path(args.results_root) / args.framework}, found {len(runs)}"
        )

    summary = summarize(runs)
    report = render(summary, args.framework, args.prefix)
    print(report, end="")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".json":
        out.write_text(json.dumps({"framework": args.framework, "prefix": args.prefix, **summary}, indent=2) + "\n")
    else:
        out.write_text(report)
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
