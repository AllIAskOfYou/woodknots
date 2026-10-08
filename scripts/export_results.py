#!/usr/bin/env python3
"""Write already-evaluated runs as a CSV in score_sweep.py's *_lofo.csv layout,
so the stored operating point can be compared with (and summarised like) the
confidence-swept one. Nothing is evaluated: the metrics are read from

  - each run's results/<framework>/<run>/fold_<i>/eval_results.json, or
  - with --pairs-json, an ensemble_pairs.py result file (pair_0, pair_1, ...).

The threshold column is 0.0 (every stored instance kept) and selected_on is
"stored". The sweep-only columns (instances, instance_f1_knots, dice_fg_knots,
clean_empty, empty_cases) are not in the stored results and are left out;
dice_nefg and ele cover the same ground. Standard library only.

    python3 scripts/export_results.py nnunet exp_res_1_{0..4} -o article/conf_sweep/nnunet_stored.csv
    python3 scripts/export_results.py yolo exp_union_{0..4} -o article/conf_sweep/yolo_stored.csv
    python3 scripts/export_results.py seg --pairs-json results/ensembles/seg/pairs_exp_reference+exp_union.json \\
        --resolution 512 -o article/conf_sweep/seg_stored.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def from_runs(framework: str, runs: list[str]) -> dict[str, dict[int, dict]]:
    out = {}
    for run in runs:
        per_fold = {}
        for path in sorted((PROJECT_ROOT / "results" / framework / run).glob("fold_*/eval_results.json")):
            per_fold[int(path.parent.name.split("_")[1])] = json.loads(path.read_text())["metrics"]
        if not per_fold:
            raise SystemExit(f"no evaluated folds under results/{framework}/{run}")
        out[run] = per_fold
    return out


def from_pairs(path: Path, resolution: str) -> dict[str, dict[int, dict]]:
    saved = json.loads(path.read_text())
    per_res = saved.get("per_resolution") or {"?": saved["per_fold"]}
    if resolution not in per_res:
        raise SystemExit(f"{path}: no resolution {resolution}, has {sorted(per_res)}")
    per_fold = per_res[resolution]
    pairs = sorted({c for fold in per_fold.values() for c in fold if c.startswith("pair_")})
    return {p: {int(f): cols[p] for f, cols in per_fold.items()} for p in pairs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("framework", help="seg, yolo or nnunet")
    ap.add_argument("runs", nargs="*", help="run names under results/<framework>/")
    ap.add_argument("--pairs-json", default=None, help="an ensemble_pairs.py .json instead of runs")
    ap.add_argument("--resolution", default="512", help="with --pairs-json: which scoring resolution")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    if args.pairs_json:
        runs = from_pairs(Path(args.pairs_json), args.resolution)
    elif args.runs:
        runs = from_runs(args.framework, args.runs)
    else:
        raise SystemExit("give run names or --pairs-json")

    metrics = list(dict.fromkeys(m for pf in runs.values() for ms in pf.values() for m in ms))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["framework", "run", "fold", "selected_on", "threshold", *metrics])
        for run, per_fold in runs.items():
            for fold, ms in sorted(per_fold.items()):
                writer.writerow([args.framework, run, fold, "stored", 0.0, *(ms.get(m, "") for m in metrics)])
    print(f"{len(runs)} runs, {sum(len(pf) for pf in runs.values())} folds -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
