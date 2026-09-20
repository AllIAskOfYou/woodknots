#!/usr/bin/env python3
"""Report which runs finished training and prediction.

Each framework's train() touches .train_complete and predict() touches
.pred_complete in the run dir (results/<framework>/<experiment>/fold_<i>/),
as the very last step of each — so a killed run leaves no marker and is
reported as unfinished. This script only looks for those two files, which
keeps it standard-library only: it runs with a plain `python3`, no venv.

    python3 scripts/check_runs.py seg                   # every experiment
    python3 scripts/check_runs.py yolo exp_reference
    python3 scripts/check_runs.py nnunet --results-root results_sc
"""
import argparse
import re
import sys
from pathlib import Path

TRAIN_COMPLETE = ".train_complete"
PRED_COMPLETE = ".pred_complete"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_CFG = PROJECT_ROOT / "common" / "conf" / "dataset" / "woodknots.yaml"


def get_experiment_names(framework: str) -> list[str]:
    exp_dir = PROJECT_ROOT / framework / "conf" / "experiment"
    if not exp_dir.is_dir():
        raise SystemExit(f"no such experiment dir: {exp_dir}")
    return sorted(f.stem for f in exp_dir.glob("*.yaml"))


def get_num_folds() -> int:
    """dataset.folds.k, read with a regex — it's the only `k:` in that file."""
    match = re.search(r"^\s+k:\s*(\d+)\s*(#.*)?$", DATASET_CFG.read_text(), re.MULTILINE)
    if not match:
        raise SystemExit(f"could not find folds.k in {DATASET_CFG}; pass --folds.")
    return int(match.group(1))


def check_run(run_dir: Path) -> tuple[bool, bool, bool]:
    """(started, trained, predicted)"""
    return (
        run_dir.is_dir(),
        (run_dir / TRAIN_COMPLETE).exists(),
        (run_dir / PRED_COMPLETE).exists(),
    )


def state(started: bool, trained: bool, predicted: bool) -> str:
    if not started:
        return "not started"
    if not trained:
        return "training"
    if not predicted:
        return "predicting"
    return "complete"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("framework", help="seg, yolo or nnunet")
    ap.add_argument("experiments", nargs="*", help="default: every <framework>/conf/experiment/*.yaml")
    ap.add_argument(
        "--results-root",
        default=str(PROJECT_ROOT / "results"),
        help="paths.results_root (default: <project>/results)",
    )
    ap.add_argument("--folds", type=int, default=None, help="number of folds (default: dataset folds.k)")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    num_folds = args.folds if args.folds is not None else get_num_folds()
    experiments = args.experiments or get_experiment_names(args.framework)

    width = max([len(e) for e in experiments] + [len("experiment")])
    print(f"{'experiment':<{width}}  " + "  ".join(f"fold {f}".ljust(12) for f in range(num_folds)))
    print("-" * (width + 2 + 14 * num_folds))

    incomplete = []
    for exp in experiments:
        cells = []
        for fold in range(num_folds):
            run_dir = results_root / args.framework / exp / f"fold_{fold}"
            started, trained, predicted = check_run(run_dir)
            label = state(started, trained, predicted)
            cells.append(label.ljust(12))
            if label != "complete":
                incomplete.append((exp, fold, label, run_dir))
        print(f"{exp:<{width}}  " + "  ".join(cells))

    total = len(experiments) * num_folds
    print(f"\n{total - len(incomplete)}/{total} runs trained and predicted.")
    if incomplete:
        print("Unfinished:")
        for exp, fold, label, run_dir in incomplete:
            print(f"  {exp} fold {fold}: {label} ({run_dir})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
