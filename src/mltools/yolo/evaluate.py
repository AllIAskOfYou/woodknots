import json
from pathlib import Path

from omegaconf import DictConfig

from mltools.common.evaluate import check_dir_ready, evaluate_segmentation
from mltools.common.logger import get_logger
from mltools.yolo.dataset import get_class_names, get_dataset_dir

logger = get_logger(__name__)


def evaluate(cfg: DictConfig) -> None:
    dataset_dir = get_dataset_dir(cfg.dataset.data_root, cfg.dataset.id, cfg.dataset.name)
    pred_dir = Path(cfg.paths.run_dir) / "predictions"
    gt_dir = Path(dataset_dir) / "gt_masks"

    check_dir_ready(pred_dir, "Prediction", logger=logger)
    check_dir_ready(gt_dir, "Ground-truth", logger=logger)

    logger.info(
        f"[{cfg.experiment.name} | fold {cfg.experiment.fold}] Evaluating predictions..."
    )
    class_names = list(get_class_names(cfg.dataset.label).values())
    results = evaluate_segmentation(
        pred_dir=str(pred_dir), gt_dir=str(gt_dir), class_names=class_names
    )
    logger.info(f"Results: {results['metrics']}")

    out_path = Path(cfg.paths.run_dir) / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "framework": "yolo",
                "experiment": cfg.experiment.name,
                "fold": cfg.experiment.fold,
                **results,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved evaluation results to: {out_path}")
