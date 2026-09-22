import json
from pathlib import Path

from omegaconf import DictConfig

from mltools.common.evaluate import check_dir_ready, evaluate_run
from mltools.common.logger import get_logger
from mltools.yolo.dataset import get_class_names
from mltools.yolo.train import get_mode

logger = get_logger(__name__)


def evaluate(cfg: DictConfig) -> None:
    pred_dir = Path(cfg.paths.run_dir) / "predictions"
    check_dir_ready(pred_dir, "Prediction", logger=logger)
    # The evaluation scores against the benchmark classes; YOLO's own classes
    # must be the same ones, or the class ids in its predictions mean
    # something else (e.g. 'separate' categorisation gives 3 knot classes).
    yolo_classes = list(get_class_names(cfg.dataset.label).values())
    if yolo_classes != list(cfg.dataset.class_names):
        raise NotImplementedError(
            f"YOLO classes {yolo_classes} differ from the benchmark classes "
            f"{list(cfg.dataset.class_names)}; mapping them is not implemented."
        )

    logger.info(
        f"[{cfg.experiment.name} | fold {cfg.experiment.fold}] Evaluating predictions..."
    )
    results = evaluate_run(cfg, pred_dir)
    logger.info(f"Results: {results['metrics']}")

    out_path = Path(cfg.paths.run_dir) / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "framework": "yolo",
                "mode": get_mode(cfg),
                "experiment": cfg.experiment.name,
                "fold": cfg.experiment.fold,
                **results,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved evaluation results to: {out_path}")
