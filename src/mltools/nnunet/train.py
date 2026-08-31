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

    nnunet_root = Path(cfg.dataset.data_root) / "nnunet_format"
    nnunet_raw_root = nnunet_root / "raw"
    nnunet_preprocessed_root = nnunet_root / "preprocessed"
    nnunet_results_root = nnunet_root / "results"

    env = build_nnunet_env(
        nnunet_raw=str(nnunet_raw_root),
        nnunet_preprocessed=str(nnunet_preprocessed_root),
        nnunet_results=str(nnunet_results_root),
    )
    logger.info(
        f"[{cfg.experiment.name} | {dataset_name} | fold {fold}] Starting training..."
    )
    run_command(train_args, env=env, logger=logger)
