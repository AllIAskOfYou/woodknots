from omegaconf import DictConfig
from pathlib import Path

from mltools.common.logger import get_logger
from mltools.common.utils import run_command
from .utils import build_nnunet_env

logger = get_logger(__name__)


def train(cfg: DictConfig) -> None:
    dataset_id = cfg.dataset.id
    fold = cfg.experiment.fold
    trainer = cfg.experiment.get("trainer", "nnUNetTrainer")
    plans = cfg.experiment.get("plans", "nnUNetPlans")

    dataset_name = f"Dataset{dataset_id:03d}_{cfg.dataset.name}"

    checkpoint_path = (
        Path(cfg.paths.nnunet.results)
        / dataset_name
        / f"{trainer}__{plans}__2d"
        / f"fold_{fold}"
        / "checkpoint_final.pth"
    )
    if checkpoint_path.exists():
        logger.info(
            f"[{cfg.experiment.name} | {dataset_name} | fold {fold}] "
            f"Training already completed (found {checkpoint_path}), skipping."
        )
        return

    train_args = [
        "nnUNetv2_train",
        f"{dataset_id:03d}",
        "2d",
        str(fold),
        "-tr",
        trainer,
        "-p",
        plans,
        "-device",
        cfg.experiment.get("device", "cuda"),
        "--c",
    ]

    env = build_nnunet_env(
        nnunet_raw=str(cfg.paths.nnunet.raw),
        nnunet_preprocessed=str(cfg.paths.nnunet.preprocessed),
        nnunet_results=str(cfg.paths.nnunet.results),
    )
    logger.info(
        f"[{cfg.experiment.name} | {dataset_name} | fold {fold}] Starting training..."
    )
    run_command(train_args, env=env, logger=logger)
