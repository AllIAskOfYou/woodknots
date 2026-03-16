import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from mltools.seg.dataset import get_dataset_by_config, SegDataset
from mltools.seg.schema import TrainConfig
from mltools.seg.transform import get_train_transform, INFERENCE_TRANSFORM
from mltools.seg.dataset.utils import get_fold
from mltools.seg.module import SegModule


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: TrainConfig):
    hydra_dir = HydraConfig.get().runtime.output_dir
    print(f"Storing logs in: {hydra_dir}")

    root_path = get_dataset_by_config(cfg.dataset)
    train_cases, val_cases = get_fold(root_path, fold=cfg.fold, k=5)
    train_transform = get_train_transform(cfg.augmentation)
    val_transform = INFERENCE_TRANSFORM

    train_dataset = SegDataset(root_path, train_cases, transform=train_transform)
    val_dataset = SegDataset(root_path, val_cases, transform=val_transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    print(f"Train Dataset: {len(train_dataset)} samples")
    print(f"Val Dataset: {len(val_dataset)} samples")

    module = SegModule(cfg.module)

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

    trainer = L.Trainer(
        max_epochs=cfg.total_epochs,
        accelerator="cuda",
        devices=[1],
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=5,
        logger=logger,
    )

    trainer.fit(module, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
