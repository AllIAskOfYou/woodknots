import sys
from pathlib import Path

from hydra import initialize_config_dir

from mltools.common.setup import compose_with_hydra_config, get_experiment_names

MIN_CHECKPOINT_BYTES = 1024  # anything smaller than this is obviously truncated/corrupted


def _fold_paths(cfg) -> dict:
    dataset_name = f"Dataset{cfg.dataset.id:03d}_{cfg.dataset.name}"
    trainer = cfg.experiment.get("trainer", "nnUNetTrainer")
    plans = cfg.experiment.get("plans", "nnUNetPlans")
    fold_dir = (
        Path(cfg.paths.nnunet.results)
        / dataset_name
        / f"{trainer}__{plans}__2d"
        / f"fold_{cfg.experiment.fold}"
    )
    return {
        "fold_dir": fold_dir,
        "checkpoint_path": fold_dir / "checkpoint_final.pth",
        "validation_dir": fold_dir / "validation",
    }


def _tail_has_error(fold_dir: Path) -> bool:
    log_files = sorted(fold_dir.glob("training_log_*.txt"))
    if not log_files:
        return False
    tail = log_files[-1].read_text(errors="ignore")[-2000:]
    return "Error" in tail or "Traceback" in tail


def check_fold(cfg) -> dict:
    paths = _fold_paths(cfg)
    fold_dir = paths["fold_dir"]
    checkpoint_path = paths["checkpoint_path"]
    validation_dir = paths["validation_dir"]

    status = {
        "experiment": cfg.experiment.name,
        "fold": cfg.experiment.fold,
        "run_dir": str(cfg.paths.run_dir),
        "checkpoint_path": str(checkpoint_path),
    }

    if not fold_dir.exists():
        status["state"] = "NOT_STARTED"
        return status

    if not checkpoint_path.exists():
        status["state"] = "FAILED" if _tail_has_error(fold_dir) else "INCOMPLETE"
        return status

    if checkpoint_path.stat().st_size < MIN_CHECKPOINT_BYTES:
        status["state"] = "CORRUPTED"
        return status

    if not validation_dir.exists() or not any(validation_dir.iterdir()):
        status["state"] = "NO_VALIDATION_OUTPUT"
        return status

    status["state"] = "COMPLETED"
    return status


def main() -> int:
    framework_dir = Path(__file__).parent
    config_dir = str((framework_dir / "conf").resolve())

    results = []
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for exp_name in get_experiment_names(framework_dir):
            probe_cfg = compose_with_hydra_config(
                [f"experiment={exp_name}", "experiment.fold=0"]
            )
            num_folds = probe_cfg.dataset.folds.k
            for fold in range(num_folds):
                cfg = compose_with_hydra_config(
                    [f"experiment={exp_name}", f"experiment.fold={fold}"]
                )
                results.append(check_fold(cfg))

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
