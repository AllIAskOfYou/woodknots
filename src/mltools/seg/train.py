import json
from pathlib import Path

import cv2
import lightning as L
import numpy as np
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger

from mltools.common.logger import get_logger
from mltools.common.utils import hash_cfg
from mltools.seg.dataset import (
    SETUP_IRRELEVANT_DATASET_KEYS,
    SegDataset,
    get_dataset_dir,
    load_case,
    write_fold_split,
)
from mltools.seg.seg_module import SegModule, denormalize_image
from mltools.seg.transforms import (
    INFERENCE_TRANSFORM,
    get_preprocessing_transform,
    get_train_transform,
    get_val_transform,
)

logger = get_logger(__name__)

# scripts/run.sbatch exports RANK=-1/LOCAL_RANK=-1 (nnU-Net's own "not
# distributed" convention), but Lightning reads RANK literally as the
# process rank and caches it into rank_zero_only.rank at import time —
# TensorBoardLogger then asserts rank == 0 and crashes. This project only
# ever runs single-GPU/single-process (scripts/run.sbatch: --ntasks=1,
# --gres=gpu:1), so it always genuinely is rank 0; force the cached value
# directly since an env-var fix here would be too late (lightning is
# already imported by the time this module runs).
rank_zero_only.rank = 0


def _checkpoint_dir(cfg: DictConfig) -> Path:
    return Path(cfg.paths.run_dir) / "checkpoints"


def _train_complete_marker(cfg: DictConfig) -> Path:
    """Lightning's ModelCheckpoint filename embeds the epoch/metric value
    (best-model-{epoch}-{val_dice}.ckpt), so unlike yolo's fixed best.pt
    there's no single fixed filename to check for completion. This marker
    is touched only as the very last step of a successful train() call, so
    an interrupted run is correctly detected as incomplete and retrained
    from scratch rather than silently treated as done."""
    return _checkpoint_dir(cfg) / ".complete"


def train(cfg: DictConfig) -> None:
    marker = _train_complete_marker(cfg)
    if marker.exists():
        logger.info(
            f"[{cfg.experiment.name} | fold {cfg.experiment.fold}] "
            f"Training already completed (found {marker}), skipping."
        )
        return

    if cfg.dataset.resolution < cfg.train.resolution:
        raise ValueError(
            f"Dataset resolution ({cfg.dataset.resolution}) must be >= "
            f"train resolution ({cfg.train.resolution})"
        )

    dataset_dir = get_dataset_dir(
        cfg.dataset.data_root,
        cfg.dataset.name,
        hash_cfg(cfg.dataset, exclude_keys=SETUP_IRRELEVANT_DATASET_KEYS),
    )
    with open(Path(dataset_dir) / "metadata.json") as f:
        metadata = json.load(f)
    channel_index = metadata["channel_names"].index(cfg.train.class_name)

    train_cases, val_cases = write_fold_split(
        dataset_dir, cfg.dataset.folds.file_path, cfg.experiment.fold
    )

    preprocessing = get_preprocessing_transform(cfg.train.preprocessing)
    train_transform = get_train_transform(cfg.train.augmentation)
    val_transform = get_val_transform(cfg.train.augmentation)

    train_dataset = SegDataset(
        dataset_dir,
        channel_index=channel_index,
        cases=train_cases,
        preprocessing=preprocessing,
        transform=train_transform,
        class_groups=cfg.train.class_groups,
        cache_amount=1.0,
    )
    val_dataset = SegDataset(
        dataset_dir,
        channel_index=channel_index,
        cases=val_cases,
        preprocessing=preprocessing,
        transform=val_transform,
        class_groups=cfg.train.class_groups,
        cache_amount=1.0,
    )

    tag = f"[{cfg.experiment.name} | fold {cfg.experiment.fold}]"
    logger.info(
        f"{tag} Train dataset: {len(train_dataset)} samples, "
        f"val dataset: {len(val_dataset)} samples"
    )

    # store 8 samples from the training dataset for visualization
    sample_dir = Path(cfg.paths.run_dir) / "samples" / "train"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(8, len(train_dataset))):
        image, mask = train_dataset[i]
        image_denorm = (denormalize_image(image) * 255).astype(np.uint8)
        image_denorm = cv2.cvtColor(image_denorm, cv2.COLOR_RGB2BGR)
        mask_long = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(sample_dir / f"image_{i}.png"), image_denorm)
        cv2.imwrite(str(sample_dir / f"mask_{i}.png"), mask_long[0])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    module = SegModule(cfg.train)

    tb_logger = TensorBoardLogger(
        save_dir=str(cfg.paths.run_dir),
        name="",
        version="",
        default_hp_metric=False,
    )
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=str(_checkpoint_dir(cfg)),
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        filename="best-model-{epoch:02d}-{val_dice:.4f}",
        save_last=True,
    )

    device_str = cfg.experiment.get("device", "cuda:0")
    device_id = int(device_str.split(":")[1]) if ":" in device_str else 0

    logger.info(f"{tag} Starting training...")
    trainer = L.Trainer(
        max_epochs=cfg.train.total_epochs,
        accelerator="cuda",
        devices=[device_id],
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
        logger=tb_logger,
    )
    trainer.fit(module, train_dataloader, val_dataloader)
    logger.info(f"{tag} Training complete.")

    marker.touch()


def predict(cfg: DictConfig) -> None:
    pred_dir = Path(cfg.paths.run_dir) / "predictions"
    if pred_dir.exists() and any(pred_dir.iterdir()):
        logger.info(
            f"[{cfg.experiment.name} | fold {cfg.experiment.fold}] "
            f"Predictions already generated (found {pred_dir}), skipping."
        )
        return

    marker = _train_complete_marker(cfg)
    if not marker.exists():
        raise RuntimeError(
            f"Cannot generate predictions: training has not completed (no {marker}). "
            f"Run training first."
        )

    dataset_dir = get_dataset_dir(
        cfg.dataset.data_root,
        cfg.dataset.name,
        hash_cfg(cfg.dataset, exclude_keys=SETUP_IRRELEVANT_DATASET_KEYS),
    )
    with open(Path(dataset_dir) / "metadata.json") as f:
        metadata = json.load(f)
    channel_index = metadata["channel_names"].index(cfg.train.class_name)

    _, val_cases = write_fold_split(dataset_dir, cfg.dataset.folds.file_path, cfg.experiment.fold)

    ckpts = sorted(_checkpoint_dir(cfg).glob("best-model-*.ckpt"))
    if len(ckpts) != 1:
        raise RuntimeError(
            f"Expected exactly one best checkpoint in {_checkpoint_dir(cfg)}, found {len(ckpts)}."
        )

    device = torch.device(cfg.experiment.get("device", "cuda:0"))
    module = SegModule.load_from_checkpoint(str(ckpts[0]), weights_only=False)
    module.eval()
    module.to(device)

    gt_dir = Path(cfg.paths.run_dir) / "gt_masks"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    class_groups = cfg.train.class_groups
    logger.info(
        f"[{cfg.experiment.name} | fold {cfg.experiment.fold}] Generating validation predictions..."
    )
    for case_id in val_cases:
        image, label = load_case(dataset_dir, case_id)

        transformed = INFERENCE_TRANSFORM(image=image)
        image_t = transformed["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = module.forward(image_t)
        pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        cv2.imwrite(str(pred_dir / f"{case_id}.png"), pred_mask)

        gt_channel = label[..., channel_index]
        if class_groups:
            mapped = np.zeros_like(gt_channel)
            for group in class_groups:
                for class_value in group.classes:
                    mapped[gt_channel == class_value] = group.value
            gt_channel = mapped
        gt_mask = (gt_channel > 0.5).astype(np.uint8)
        cv2.imwrite(str(gt_dir / f"{case_id}.png"), gt_mask)
