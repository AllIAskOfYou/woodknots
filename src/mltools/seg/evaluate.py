import json
from pathlib import Path

from omegaconf import DictConfig

from mltools.common.evaluate import check_dir_ready, evaluate_run
from mltools.common.logger import get_logger

logger = get_logger(__name__)


def evaluate(cfg: DictConfig) -> None:
    pred_dir = Path(cfg.paths.run_dir) / "predictions"
    check_dir_ready(pred_dir, "Prediction", logger=logger)
    if cfg.train.class_name not in cfg.dataset.class_names:
        raise RuntimeError(
            f"train.class_name {cfg.train.class_name!r} is not a benchmark class {list(cfg.dataset.class_names)}"
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
                "framework": "seg",
                "experiment": cfg.experiment.name,
                "fold": cfg.experiment.fold,
                **results,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved evaluation results to: {out_path}")
