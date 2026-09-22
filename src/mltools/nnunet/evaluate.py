import json
from omegaconf import DictConfig
from pathlib import Path

from mltools.common.evaluate import check_dir_ready, evaluate_run
from mltools.common.logger import get_logger

logger = get_logger(__name__)


def evaluate(cfg: DictConfig) -> None:
    dataset_id = cfg.dataset.id
    fold = cfg.experiment.fold

    dataset_name = f"Dataset{dataset_id:03d}_{cfg.dataset.name}"

    # Test-partition predictions from checkpoint_best (see train.predict), not
    # nnU-Net's own validation/ export, which covers the val partition.
    pred_dir = Path(cfg.paths.run_dir) / "predictions"
    check_dir_ready(pred_dir, "Prediction", logger=logger)

    logger.info(
        f"[{cfg.experiment.name} | {dataset_name} | fold {fold}] Evaluating predictions..."
    )
    results = evaluate_run(cfg, pred_dir)
    
    rounded = {k: round(v, 2) for k, v in results["metrics"].items()}
    logger.info(f"Results: {rounded}")

    out_path = Path(cfg.paths.run_dir) / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "framework": "nnunet",
                "experiment": cfg.experiment.name,
                "fold": fold,
                **results,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved evaluation results to: {out_path}")
