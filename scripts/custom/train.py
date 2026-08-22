import cv2
import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
import numpy as np
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import argparse

from mltools.seg.dataset import get_dataset_by_config, SegDataset
from mltools.seg.schema import TrainConfig
from mltools.seg.transform import (
    get_preprocessing_transform,
    get_train_transform,
    get_val_transform,
    INFERENCE_TRANSFORM,
)
from mltools.seg.module import SegModule
from mltools.seg.module.seg_module import denormalize_image
from mltools.common.data import get_metadata, get_fold


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: TrainConfig):
    hydra_dir = HydraConfig.get().runtime.output_dir
    if cfg.dataset.resolution < cfg.augmentation.resolution:
        raise ValueError(
            f"Dataset resolution ({cfg.dataset.resolution}) must be >= augmentation resolution ({cfg.augmentation.resolution})"
        )
    print(f"Storing logs in: {hydra_dir}")

    with open(Path(hydra_dir) / cfg.comment, "w") as f:
        f.write(".")

    root_path = get_dataset_by_config(cfg.dataset)
    metadata = get_metadata(root_path)
    channel_index = metadata["channel_names"].index(cfg.class_name)
    train_cases, val_cases = get_fold(root_path, fold=cfg.fold, k=5)
    preprocessing = get_preprocessing_transform(cfg.preprocessing)
    train_transform = get_train_transform(cfg.augmentation)
    # val_transform = INFERENCE_TRANSFORM
    val_transform = get_val_transform(cfg.augmentation)

    train_dataset = SegDataset(
        root_path,
        channel_index=channel_index,
        cases=train_cases,
        preprocessing=preprocessing,
        transform=train_transform,
        class_groups=cfg.class_groups,
        cache_amount=1.0,
    )
    val_dataset = SegDataset(
        root_path,
        channel_index=channel_index,
        cases=val_cases,
        preprocessing=preprocessing,
        transform=val_transform,
        class_groups=cfg.class_groups,
        cache_amount=1.0,
    )

    # store 8 samples from the training dataset for visualization during training
    sample_dir = Path(hydra_dir) / "samples" / "train"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for i in range(8):
        image, mask = train_dataset[i]
        image_denorm = (denormalize_image(image) * 255).astype(np.uint8)
        image_denorm = cv2.cvtColor(image_denorm, cv2.COLOR_RGB2BGR)
        mask_long = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(sample_dir / f"image_{i}.png", image_denorm)
        cv2.imwrite(sample_dir / f"mask_{i}.png", mask_long[0])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    print(f"Train Dataset: {len(train_dataset)} samples")
    print(f"Val Dataset: {len(val_dataset)} samples")

    module = SegModule(cfg.module)
    # ckpt_path = "experiments/light/2026-06-15/21-08-13/checkpoints/best-model-epoch=33-val_dice=0.8271.ckpt"
    # ckpt_path = "experiments/light/2026-06-15/21-08-13/checkpoints/last.ckpt"
    # module = SegModule.load_from_checkpoint(ckpt_path, weights_only=False)
    # module.cfg = cfg.module

    logger = TensorBoardLogger(
        save_dir=hydra_dir,
        name="",
        version="",
        default_hp_metric=False,
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val_dice",  # Name of the metric you logged in validation_step
        mode="max",  # We want the highest (maximum) value
        save_top_k=1,  # Only keep the single best model
        filename="best-model-{epoch:02d}-{val_dice:.4f}",
        save_last=True,  # Also keep a 'last.ckpt' for resuming
    )
    module.freeze_encoder()
    # module.model.encoder.train()
    # module.model.unfreeze_encoder_stages([2, 3])
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="cuda",
        devices=[0],
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
        logger=logger,
    )
    trainer.fit(module, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
