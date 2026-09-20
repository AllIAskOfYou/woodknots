"""Report which nnU-Net experiment/fold runs completed. Standard library only,
so it runs with a plain `python3` (no venv, no hydra):

    python3 nnunet/check_training.py                    # every experiment
    python3 nnunet/check_training.py exp_0 exp_2
    python3 nnunet/check_training.py --results-root results_sc

Instead of composing the hydra config, paths follow the layout it produces:
<results_root>/nnunet/<experiment>/fold_<i>/nnunet_native/Dataset*/<trainer>__<plans>__2d/fold_<i>.
"""
import argparse
import re
import sys
from pathlib import Path

MIN_CHECKPOINT_BYTES = 1024  # anything smaller than this is obviously truncated/corrupted

FRAMEWORK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRAMEWORK_DIR.parent
DATASET_CFG = PROJECT_ROOT / "common" / "conf" / "dataset" / "woodknots.yaml"


def get_experiment_names() -> list[str]:
    return sorted(f.stem for f in (FRAMEWORK_DIR / "conf" / "experiment").glob("*.yaml"))


def get_num_folds() -> int:
    """dataset.folds.k from the shared dataset config, read with a regex
    (no yaml parser) — it's the only `k:` key in that file."""
    match = re.search(r"^\s+k:\s*(\d+)\s*(#.*)?$", DATASET_CFG.read_text(), re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find folds.k in {DATASET_CFG}; pass --folds.")
    return int(match.group(1))


def _fold_paths(results_root: Path, exp_name: str, fold: int) -> dict:
    run_dir = results_root / "nnunet" / exp_name / f"fold_{fold}"
    # Dataset id/name and trainer/plans only appear in the directory names, so
    # find them instead of re-deriving them from the config.
    candidates = sorted(
        (run_dir / "nnunet_native").glob(f"Dataset*/*__2d/fold_{fold}"),
        key=lambda p: p.stat().st_mtime,
    )
    fold_dir = candidates[-1] if candidates else run_dir / "nnunet_native" / f"fold_{fold}"
    return {
        "run_dir": run_dir,
        "fold_dir": fold_dir,
        "fold_dir_found": bool(candidates),
        "checkpoint_path": fold_dir / "checkpoint_final.pth",
        "best_checkpoint_path": fold_dir / "checkpoint_best.pth",
        # test-partition predictions (mltools.nnunet.train.predict)
        "predictions_dir": run_dir / "predictions",
    }


def _tail_has_error(fold_dir: Path) -> bool:
    log_files = sorted(fold_dir.glob("training_log_*.txt"))
    if not log_files:
        return False
    tail = log_files[-1].read_text(errors="ignore")[-2000:]
    return "Error" in tail or "Traceback" in tail


def check_fold(results_root: Path, exp_name: str, fold: int) -> dict:
    paths = _fold_paths(results_root, exp_name, fold)
    fold_dir = paths["fold_dir"]
    checkpoint_path = paths["checkpoint_path"]
    best_checkpoint_path = paths["best_checkpoint_path"]
    predictions_dir = paths["predictions_dir"]

    status = {
        "experiment": exp_name,
        "fold": fold,
        "run_dir": str(paths["run_dir"]),
        "checkpoint_path": str(checkpoint_path),
    }

    if not paths["fold_dir_found"]:
        status["state"] = "NOT_STARTED"
        return status

    if not checkpoint_path.exists():
        status["state"] = "FAILED" if _tail_has_error(fold_dir) else "INCOMPLETE"
        return status

    if any(
        not p.exists() or p.stat().st_size < MIN_CHECKPOINT_BYTES
        for p in (checkpoint_path, best_checkpoint_path)
    ):
        status["state"] = "CORRUPTED"
        return status

    if not predictions_dir.exists() or not any(predictions_dir.iterdir()):
        status["state"] = "NO_PREDICTIONS"
        return status

    status["state"] = "COMPLETED"
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="*", help="default: every nnunet/conf/experiment/*.yaml")
    ap.add_argument(
        "--results-root",
        default=str(PROJECT_ROOT / "results"),
        help="paths.results_root (default: <project>/results)",
    )
    ap.add_argument("--folds", type=int, default=None, help="number of folds (default: dataset folds.k)")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    num_folds = args.folds if args.folds is not None else get_num_folds()

    results = [
        check_fold(results_root, exp_name, fold)
        for exp_name in (args.experiments or get_experiment_names())
        for fold in range(num_folds)
    ]

    header = f"{'experiment':<12} {'fold':<5} {'state':<20} checkpoint"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['experiment']:<12} {r['fold']:<5} {r['state']:<20} {r['checkpoint_path']}")

    incomplete = [r for r in results if r["state"] != "COMPLETED"]
    print()
    print(f"{len(results) - len(incomplete)}/{len(results)} runs completed correctly.")
    if incomplete:
        print("Not completed:")
        for r in incomplete:
            print(f"  {r['experiment']} fold {r['fold']}: {r['state']} ({r['run_dir']})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
