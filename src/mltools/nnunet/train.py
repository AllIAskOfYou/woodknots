import numpy as np
from omegaconf import DictConfig
from pathlib import Path

from mltools.common.instances import probs_to_instances, save_instances
from mltools.common.splits import SPLIT_MODES, TRAIN_TEST, TRAIN_VAL_TEST, get_split
from mltools.common.utils import symlink_relative

from mltools.common.logger import get_logger
from mltools.common.utils import (
    pred_complete_marker,
    run_command,
    train_complete_marker,
)
from .utils import build_nnunet_env

logger = get_logger(__name__)

CHECKPOINTS = {"best": "checkpoint_best.pth", "final": "checkpoint_final.pth"}


def _split_mode(cfg: DictConfig) -> str:
    """Set on dataset, not train: nnU-Net reads the split from the formatted
    dataset, so it is a property of dataset.id and each mode needs its own."""
    mode = cfg.dataset.get("split_mode", TRAIN_VAL_TEST)
    if mode not in SPLIT_MODES:
        raise ValueError(f"Unknown dataset.split_mode {mode!r}, expected one of {SPLIT_MODES}.")
    return mode


def _checkpoint(cfg: DictConfig) -> str:
    """Which of nnU-Net's two checkpoints to predict from. best, the default, is
    chosen by the EMA of the val pseudo-dice; under train_test that partition is
    the test set, so only final is allowed there."""
    mode = _split_mode(cfg)
    choice = cfg.experiment.get("checkpoint", "best" if mode == TRAIN_VAL_TEST else "final")
    if choice not in CHECKPOINTS:
        raise ValueError(
            f"Unknown experiment.checkpoint {choice!r}, expected one of {sorted(CHECKPOINTS)}."
        )
    if choice == "best" and mode == TRAIN_TEST:
        raise ValueError(
            "experiment.checkpoint=best is not allowed with dataset.split_mode=train_test: "
            "checkpoint_best.pth is selected on the val partition, which in that mode is the "
            "test partition. Use checkpoint=final, or split_mode=train_val_test."
        )
    return CHECKPOINTS[choice]


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

    _checkpoint(cfg)  # reject an invalid split_mode/checkpoint pair before training, not after

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
    """Predict the split's held-out test partition with the checkpoint
    _checkpoint() selects. nnU-Net's own end-of-training validation/ export
    covers val, not test, so it isn't used for scoring.
    nnUNetv2_predict's defaults (step 0.5, Gaussian weighting, mirroring) are
    the same predictor settings that export uses.

    nnUNetv2_predict writes its own output (segmentations + softmax
    probabilities) to run_dir/nnunet_predictions; each case is then converted
    to the shared instance format in run_dir/predictions/<case>.npz — the
    connected components of the probabilities, scored by mean probability
    (mltools.common.instances.probs_to_instances)."""
    fold = cfg.experiment.fold
    dataset_name = f"Dataset{cfg.dataset.id:03d}_{cfg.dataset.name}"
    tag = f"[{cfg.experiment.name} | {dataset_name} | fold {fold}]"

    pred_marker = pred_complete_marker(cfg.paths.run_dir)
    if pred_marker.exists():
        logger.info(f"{tag} Test predictions already generated (found {pred_marker}), skipping.")
        return

    test_cases = get_split(cfg.dataset.folds.file_path, fold, mode=_split_mode(cfg))["test"]
    raw_dir = Path(cfg.paths.run_dir) / "nnunet_predictions"
    pred_dir = Path(cfg.paths.run_dir) / "predictions"

    checkpoint_path = _fold_dir(cfg) / _checkpoint(cfg)
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
        str(raw_dir),
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
        "--save_probabilities",
    ]
    env = build_nnunet_env(
        nnunet_raw=str(cfg.paths.nnunet.raw),
        nnunet_preprocessed=str(cfg.paths.nnunet.preprocessed),
        nnunet_results=str(cfg.paths.nnunet.results),
    )
    logger.info(f"{tag} Generating test predictions with {checkpoint_path.name}...")
    run_command(predict_args, env=env, logger=logger)

    pred_dir.mkdir(parents=True, exist_ok=True)
    for case_id in test_cases:
        # (num_classes + 1, [1,] H, W) softmax, background first; 2D natural
        # images carry a singleton through-plane axis.
        with np.load(raw_dir / f"{case_id}.npz") as data:
            probs = data["probabilities"][1:].astype(np.float32)
        if probs.ndim == 4:
            probs = probs[:, 0]
        save_instances(pred_dir / f"{case_id}.npz", probs_to_instances(probs))

    pred_marker.touch()
