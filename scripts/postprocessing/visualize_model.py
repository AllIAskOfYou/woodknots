import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from mltools.seg.dataset import get_dataset_by_config, SegDataset
from mltools.seg.schema import TrainConfig
from mltools.seg.transform import (
    get_preprocessing_transform,
    get_train_transform,
    get_val_transform,
    INFERENCE_TRANSFORM,
)
from mltools.seg.module import SegModule
from mltools.common.data import get_metadata, get_fold
from matplotlib import pyplot as plt
import numpy as np
import cv2


def main():
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg: TrainConfig = hydra.compose(
            config_name="config",
        )
    if cfg.dataset.resolution < cfg.augmentation.resolution:
        raise ValueError(
            f"Dataset resolution ({cfg.dataset.resolution}) must be >= augmentation resolution ({cfg.augmentation.resolution})"
        )

    root_path = get_dataset_by_config(cfg.dataset)
    metadata = get_metadata(root_path)
    channel_index = metadata["channel_names"].index(cfg.class_name)
    train_cases, val_cases = get_fold(root_path, fold=cfg.fold, k=5)
    preprocessing = get_preprocessing_transform(cfg.preprocessing)
    train_transform = get_train_transform(cfg.augmentation)
    val_transform = INFERENCE_TRANSFORM
    # val_transform = get_val_transform(cfg.augmentation)

    train_dataset = SegDataset(
        root_path,
        channel_index=channel_index,
        cases=train_cases,
        preprocessing=preprocessing,
        transform=train_transform,
        class_groups=cfg.class_groups,
        cache_amount=0.0,
    )
    val_dataset = SegDataset(
        root_path,
        channel_index=channel_index,
        cases=val_cases,
        preprocessing=preprocessing,
        transform=val_transform,
        class_groups=cfg.class_groups,
        cache_amount=0.0,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    print(f"Train Dataset: {len(train_dataset)} samples")
    print(f"Val Dataset: {len(val_dataset)} samples")

    root_path = "experiments/light/2026-06-01/07-38-37"
    module = SegModule(cfg.module)
    checkpoint_path = (
        f"{root_path}/checkpoints/best-model-epoch=19-val_dice=0.7614.ckpt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    module.load_state_dict(checkpoint["state_dict"])

    # Move to GPU if needed
    module = module.to("cuda")

    # store the first 10 random samples from the train and validation sets each in their folder
    # overlay the true mask in green and the predicted mask in blue on the original image
    for i in range(10):
        sample = train_dataset[i]
        print(sample[0].shape, sample[1].shape)

        overlay = np.zeros_like(image)
        overlay[..., 1] = mask * 255
        overlay[..., 2] = pred_mask * 255
        blended = cv2.addWeighted(
            image.astype(np.uint8), 0.7, overlay.astype(np.uint8), 0.3, 0
        )

        plt.imshow(blended)
        plt.axis("off")
        plt.savefig(
            f"{root_path}/train_samples/{i}.png", bbox_inches="tight", pad_inches=0
        )

    for i in range(10):
        sample = val_dataset[i]
        image = sample[0].numpy().transpose(1, 2, 0)
        mask = sample[1].numpy().squeeze()
        pred = module(image[None, ...].transpose(0, 3, 1, 2)).squeeze().detach().numpy()
        pred_mask = pred > 0.5

        overlay = np.zeros_like(image)
        overlay[..., 1] = mask * 255
        overlay[..., 2] = pred_mask * 255
        blended = cv2.addWeighted(
            image.astype(np.uint8), 0.7, overlay.astype(np.uint8), 0.3, 0
        )

        plt.imshow(blended)
        plt.axis("off")
        plt.savefig(
            f"{root_path}/val_samples/{i}.png", bbox_inches="tight", pad_inches=0
        )


if __name__ == "__main__":
    main()
