import json
from omegaconf import DictConfig
from pathlib import Path

from mltools.common.evaluate import evaluate_segmentation, check_dir_ready
from mltools.common.logger import get_logger

logger = get_logger(__name__)


def evaluate(cfg: DictConfig) -> None:
    dataset_id = cfg.dataset.id
    fold = cfg.experiment.fold

    dataset_name = f"Dataset{dataset_id:03d}_{cfg.dataset.name}"
    trainer = cfg.experiment.get("trainer", "nnUNetTrainer")
    plans = cfg.experiment.get("plans", "nnUNetPlans")

    pred_dir = (
        Path(cfg.paths.nnunet.results)
        / dataset_name
        / f"{trainer}__{plans}__2d"
        / f"fold_{fold}"
        / "validation"
    )
    gt_dir = Path(cfg.paths.nnunet.raw) / dataset_name / "labelsTr"

    check_dir_ready(pred_dir, "Prediction", logger=logger)
    check_dir_ready(gt_dir, "Ground-truth", logger=logger)

    logger.info(
        f"[{cfg.experiment.name} | {dataset_name} | fold {fold}] Evaluating predictions..."
    )
    results = evaluate_segmentation(
        pred_dir=str(pred_dir),
        gt_dir=str(gt_dir),
        class_names=cfg.dataset.class_names,
    )
    
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
