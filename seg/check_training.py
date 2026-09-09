import sys
from pathlib import Path

from hydra import initialize_config_dir

from mltools.common.setup import compose_with_hydra_config, get_experiment_names

MIN_CHECKPOINT_BYTES = 1024  # anything smaller than this is obviously truncated/corrupted


def _run_paths(cfg) -> dict:
    run_dir = Path(cfg.paths.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    return {
        "run_dir": run_dir,
        "marker_path": checkpoint_dir / ".complete",
        "checkpoint_dir": checkpoint_dir,
        "predictions_dir": run_dir / "predictions",
    }


def _latest_slurm_log(framework: str, exp_name: str, fold: int) -> Path | None:
    log_dir = Path("slurm_logs") / f"{framework}-{exp_name}-fold_{fold}"
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("*.out"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _log_has_error(log_path: Path) -> bool:
    tail = log_path.read_text(errors="ignore")[-2000:]
    return "Error" in tail or "Traceback" in tail


def check_fold(cfg, framework: str) -> dict:
    paths = _run_paths(cfg)
    run_dir = paths["run_dir"]
    marker_path = paths["marker_path"]
    checkpoint_dir = paths["checkpoint_dir"]
    predictions_dir = paths["predictions_dir"]

    status = {
        "experiment": cfg.experiment.name,
        "fold": cfg.experiment.fold,
        "run_dir": str(run_dir),
        "marker_path": str(marker_path),
    }

    if not run_dir.exists():
        status["state"] = "NOT_STARTED"
        return status

    if not marker_path.exists():
        log_path = _latest_slurm_log(framework, cfg.experiment.name, cfg.experiment.fold)
        status["state"] = "FAILED" if log_path and _log_has_error(log_path) else "INCOMPLETE"
        return status

    ckpts = list(checkpoint_dir.glob("best-model-*.ckpt"))
    if len(ckpts) != 1 or ckpts[0].stat().st_size < MIN_CHECKPOINT_BYTES:
        status["state"] = "CORRUPTED"
        return status

    if not predictions_dir.exists() or not any(predictions_dir.iterdir()):
        status["state"] = "NO_PREDICTIONS"
        return status

    status["state"] = "COMPLETED"
    return status


def main() -> int:
    framework_dir = Path(__file__).parent
    framework = framework_dir.name
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
                results.append(check_fold(cfg, framework))

    header = f"{'experiment':<12} {'fold':<5} {'state':<15} marker"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['experiment']:<12} {r['fold']:<5} {r['state']:<15} {r['marker_path']}")

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
