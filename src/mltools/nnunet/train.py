from omegaconf import DictConfig
from pathlib import Path

from mltools.common.logger import get_logger
from mltools.common.utils import run_command
from .utils import build_nnunet_env

logger = get_logger(__name__)


def train(cfg: DictConfig) -> None:
    dataset_id = cfg.dataset.id
    fold = cfg.experiment.fold

    dataset_name = f"Dataset{dataset_id:03d}_{cfg.dataset.name}"
    train_args = [
        "nnUNetv2_train",
        f"{dataset_id:03d}",
        "2d",
        str(fold),
        "-tr",
        cfg.experiment.get("trainer", "nnUNetTrainer"),
        "-p",
        cfg.experiment.get("plans", "nnUNetPlans"),
        "-device",
        cfg.experiment.get("device", "cuda"),
        "--c",
    ]

    env = build_nnunet_env(
        nnunet_raw=str(cfg.paths.nnunet.raw),
        nnunet_preprocessed=str(cfg.paths.nnunet.preprocessed),
        nnunet_results=str(cfg.paths.nnunet.results),
        nnunet_seed=cfg.get("seed", 42)
    )
    logger.info(
        f"[{cfg.experiment.name} | {dataset_name} | fold {fold}] Starting training..."
    )
    run_command(train_args, env=env, logger=logger)
