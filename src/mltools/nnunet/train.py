from omegaconf import DictConfig
from pathlib import Path

from mltools.common.splits import get_split
from mltools.common.utils import symlink_relative

from mltools.common.logger import get_logger
from mltools.common.utils import (
    pred_complete_marker,
    run_command,
    train_complete_marker,
)
from .utils import build_nnunet_env

logger = get_logger(__name__)


def _fold_dir(cfg: DictConfig) -> Path:
    trainer = cfg.experiment.get("trainer", "nnUNetTrainer")
    plans = cfg.experiment.get("plans", "nnUNetPlans")
    return (
        Path(cfg.paths.nnunet.results)
        / f"Dataset{cfg.dataset.id:03d}_{cfg.dataset.name}"
        / f"{trainer}__{plans}__2d"
        / f"fold_{cfg.experiment.fold}"
    )


def train(cfg: DictConfig) -> None:
    dataset_id = cfg.dataset.id
    fold = cfg.experiment.fold
    trainer = cfg.experiment.get("trainer", "nnUNetTrainer")
    plans = cfg.experiment.get("plans", "nnUNetPlans")

    dataset_name = f"Dataset{dataset_id:03d}_{cfg.dataset.name}"

    marker = train_complete_marker(cfg.paths.run_dir)
    if marker.exists():
        logger.info(
            f"[{cfg.experiment.name} | {dataset_name} | fold {fold}] "
            f"Training already completed (found {marker}), skipping."
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

    marker.touch()


def predict(cfg: DictConfig) -> None:
    """Predict the split's held-out test partition into run_dir/predictions
    with checkpoint_best.pth — the checkpoint nnU-Net selected on the val
    partition, matching seg's best checkpoint and YOLO's best.pt. nnU-Net's
    own end-of-training validation/ export covers val, not test, so it isn't
    used for scoring. nnUNetv2_predict's defaults (step 0.5, Gaussian
    weighting, mirroring) are the same predictor settings that export uses."""
    fold = cfg.experiment.fold
    dataset_name = f"Dataset{cfg.dataset.id:03d}_{cfg.dataset.name}"
    tag = f"[{cfg.experiment.name} | {dataset_name} | fold {fold}]"

    pred_marker = pred_complete_marker(cfg.paths.run_dir)
    if pred_marker.exists():
        logger.info(f"{tag} Test predictions already generated (found {pred_marker}), skipping.")
        return

    test_cases = get_split(cfg.dataset.folds.file_path, fold)["test"]
    pred_dir = Path(cfg.paths.run_dir) / "predictions"

    checkpoint_path = _fold_dir(cfg) / "checkpoint_best.pth"
    if not train_complete_marker(cfg.paths.run_dir).exists() or not checkpoint_path.exists():
        raise RuntimeError(
            f"Cannot generate predictions: training has not completed "
            f"(no {train_complete_marker(cfg.paths.run_dir)} or {checkpoint_path}). "
            f"Run training first."
        )

    # nnUNetv2_predict takes a folder, so gather the test images into one.
    images_tr_dir = Path(cfg.paths.nnunet.raw) / dataset_name / "imagesTr"
    input_dir = Path(cfg.paths.run_dir) / "test_images"
    input_dir.mkdir(parents=True, exist_ok=True)
    for case_id in test_cases:
        link = input_dir / f"{case_id}_0000.png"
        if not link.exists():
            symlink_relative(link, images_tr_dir / f"{case_id}_0000.png")

    predict_args = [
        "nnUNetv2_predict",
        "-i",
        str(input_dir),
        "-o",
        str(pred_dir),
        "-d",
        f"{cfg.dataset.id:03d}",
        "-c",
        "2d",
        "-f",
        str(fold),
        "-tr",
        cfg.experiment.get("trainer", "nnUNetTrainer"),
        "-p",
        cfg.experiment.get("plans", "nnUNetPlans"),
        "-chk",
        checkpoint_path.name,
        "-device",
        cfg.experiment.get("device", "cuda"),
    ]
    env = build_nnunet_env(
        nnunet_raw=str(cfg.paths.nnunet.raw),
        nnunet_preprocessed=str(cfg.paths.nnunet.preprocessed),
        nnunet_results=str(cfg.paths.nnunet.results),
    )
    logger.info(f"{tag} Generating test predictions with {checkpoint_path.name}...")
    run_command(predict_args, env=env, logger=logger)

    pred_marker.touch()
