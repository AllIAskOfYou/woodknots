import cv2
import lightning as L
from hydra.utils import instantiate
from monai.inferers import sliding_window_inference
import torch
import numpy as np

from mltools.seg.losses import SegLoss
from mltools.seg.schema import ModuleConfig
from mltools.seg.metrics import DiceScore, IoUScore, GMSScore
from pathlib import Path


class SegModule(L.LightningModule):
    def __init__(self, module_cfg: ModuleConfig):
        super().__init__()
        self.model = instantiate(module_cfg.model)
        # freeze encoder layers
        if module_cfg.freeze_encoder and hasattr(self.model, "encoder"):
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        self.cfg = module_cfg
        self.loss = SegLoss()
        self.save_hyperparameters(module_cfg)
        self.metrics = {
            "dice": DiceScore(soft=False, background=True, average="macro"),
            "dice_nobg": DiceScore(soft=False, background=False, average="macro"),
            "dice_soft": DiceScore(soft=True, background=True, average="macro"),
            "iou": IoUScore(soft=False, background=True, average="macro"),
            "gms": GMSScore(average="macro"),
        }

    def forward(self, x):
        res = self.cfg.resolution
        with torch.no_grad():
            outputs = sliding_window_inference(
                x, roi_size=(res, res), sw_batch_size=4, predictor=self.model, device=x.device, overlap=0.5, mode="gaussian"
            )
        preds = torch.sigmoid(outputs)
        return preds

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        images, masks = batch
        outputs = self.model(images)
        loss = self.loss(outputs, masks)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        for name in self.metrics:
            metric = self.metrics[name]
            self.log(
                f"train_{name}",
                metric.compute(outputs, masks),
                prog_bar=True,
                on_epoch=True,
                on_step=False,
            )
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        images, masks = batch
        res = self.cfg.resolution
        with torch.no_grad():
            outputs = sliding_window_inference(
                images,
                roi_size=(res, res),
                sw_batch_size=4,
                predictor=self.model,
                overlap=0.5,
                mode="gaussian",
            )
        outputs = outputs.detach()
        masks = masks.detach()

        if batch_idx < 4:
            # save the first batch of masks and outputs for visualization
            # save to the logging directory under "samples" subdirectory
            sample_dir = Path(self.logger.log_dir) / "samples"
            sample_dir.mkdir(exist_ok=True)
            mask_long = (masks.cpu().numpy() * 255).astype(np.uint8)
            output_long = (torch.sigmoid(outputs).cpu().numpy() * 255).astype(np.uint8)

            cv2.imwrite(sample_dir / f"mask_{batch_idx}.png", mask_long[0, 0])
            cv2.imwrite(sample_dir / f"output_{batch_idx}.png", output_long[0, 0])

        loss = self.loss(outputs, masks)
        self.log("val_loss", loss, prog_bar=True)

        for name in self.metrics:
            metric = self.metrics[name]
            self.log(
                f"val_{name}",
                metric.compute(outputs, masks),
                prog_bar=True,
                on_epoch=True,
                on_step=False,
            )

    def configure_optimizers(self):
        optimizer = instantiate(
            self.cfg.optimizer,
            params=filter(lambda p: p.requires_grad, self.parameters()),
        )
        scheduler = instantiate(
            self.cfg.scheduler,
            optimizer=optimizer,
            total_iters=int(self.cfg.total_epochs * 0.5),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
