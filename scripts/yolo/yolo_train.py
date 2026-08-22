import hydra
from hydra.core.hydra_config import HydraConfig
import os
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from ultralytics import YOLO

from mltools.yolo.dataset.utils import get_dataset_by_config
from mltools.yolo.schema import TrainConfig
from mltools.seg.metrics import STANDARD_METRICS


# define a callback for logging metrics to TensorBoard
#
class Validator:
    def __init__(self):
        self.inputs = []
        self.targets = []

    def on_val_batch_end(self, trainer):
        self.inputs.append(trainer.validator.preds)
        self.targets.append(trainer.validator.batch["masks"])

    def on_val_end(self, trainer):
        # Triggered after the full validation loop is done
        # Concatenate list of tensors into one large tensor
        full_gt = torch.cat(self.targets, dim=0)

        # Do your custom logic here
        print(f"Full validation tensor shape: {full_gt.shape}")

        # Clear for next epoch
        self.inputs.clear()
        self.targets.clear()

@hydra.main(version_base="1.3", config_path="../configs_yolo", config_name="config")
def main(cfg: TrainConfig):
    data_path = get_dataset_by_config(cfg.dataset)
    # model = YOLO(cfg.model_path)
    model = YOLO("runs/segment/experiments/yolo/2026-05-31/17-13-53/train/weights/best.pt")
    exp_path = HydraConfig.get().run.dir
    # logger = TensorBoardLogger(save_dir=exp_path, name="logs")
    # validator = Validator()
    # model.add_callback("on_val_batch_end", validator.on_val_batch_end)
    # model.add_callback("on_val_end", validator.on_val_end)
    model.val(
        data=data_path / "data.yaml",
        imgsz=cfg.imgsz,
        batch=4,
        device="cuda:0",
        save_json=True
    )
    return
    model.train(
        project=exp_path,
        data=data_path / "data.yaml",
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device="cuda:0",
        save_json=True,
        
        # Augmentation (Helps prevent overfitting on texture)
        hsv_h=0.015,     # Hue augmentation
        hsv_s=0.7,       # Saturation augmentation
        hsv_v=0.4,       # Value (brightness) augmentation
        degrees=10.0,    # Image rotation (+/- deg)
        translate=0.2,   # Image translation (+/- fraction)
        scale=0.9,       # Image scaling (+/- gain)
        mosaic=1.0,      # Use mosaic augmentation
        close_mosaic=20
    )


if __name__ == "__main__":
    main()
