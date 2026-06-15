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

from mltools.seg.module.dinov3 import DINOv3FPN, DINOv3MAnet

def denormalize_image(image_tensor):
    # Assuming the image was normalized using ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)  # CxHxW to HxWxC
    image_denorm = (image_np * std) + mean
    image_denorm = np.clip(image_denorm, 0, 1)
    return image_denorm


class SegModule(L.LightningModule):
    def __init__(self, module_cfg: ModuleConfig):
        super().__init__()
        # self.model = DINOv3FPN(out_channels=256, num_classes=1)
        self.model = DINOv3MAnet(num_classes=1)
        # self.model = instantiate(module_cfg.model)
        # freeze encoder layers
        if module_cfg.freeze_encoder:
            self.freeze_encoder()
        self.cfg = module_cfg
        self.loss = SegLoss(labels_smoothing=module_cfg.loss.labels_smoothing)
        self.save_hyperparameters(module_cfg)
        self.metrics = {
            "dice": DiceScore(soft=False, background=True, average="macro"),
            "dice_nobg": DiceScore(soft=False, background=False, average="macro"),
            "dice_soft": DiceScore(soft=True, background=True, average="macro"),
            "iou": IoUScore(soft=False, background=True, average="macro"),
            "gms": GMSScore(average="macro"),
        }
        self.should_validate = True

    def freeze_encoder(self):
        if hasattr(self.model, "encoder"):
            self.model.encoder.eval()
            for param in self.model.encoder.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        if hasattr(self.model, "encoder"):
            self.model.encoder.train()
            for param in self.model.encoder.parameters():
                param.requires_grad = True

    def forward(self, x):
        res = self.cfg.resolution
        with torch.no_grad():
            outputs = sliding_window_inference(
                x,
                roi_size=(res, res),
                sw_batch_size=4,
                predictor=self.model,
                device=x.device,
                overlap=0.5,
                mode="gaussian",
            )
        preds = torch.sigmoid(outputs)
        return preds

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        images, masks = batch
        outputs = self.model(images)
        self.loss.update_weight(batch_idx, self.trainer.estimated_stepping_batches)
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
    
    def on_validation_epoch_start(self):
        # Skip validation for first 2 epochs
        if self.current_epoch < 3:
            self.should_validate = False
        else:
            self.should_validate = True

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        if not self.should_validate:
            # return dummy val_dice for checkpointing to work        
            for name in self.metrics:
                self.log(
                    f"val_{name}",
                    0.0,
                    prog_bar=True,
                    on_epoch=True,
                    on_step=False,
                )
            return
        
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
        if batch_idx < 8:
            # save the first batch of masks and outputs for visualization
            # save to the logging directory under "samples" subdirectory
            sample_dir = Path(self.logger.log_dir) / "samples" / "val"
            sample_dir.mkdir(parents=True, exist_ok=True)
            # denormalize the images from imagenet and convert to uint8
            image_denorm = denormalize_image(images[0])
            image_long = (image_denorm * 255).astype(np.uint8)
            image_long = cv2.cvtColor(
                image_long, cv2.COLOR_RGB2BGR
            )  # convert to BGR for OpenCV
            mask_long = (masks.cpu().numpy() * 255).astype(np.uint8)
            # generate contour/edges from the mask
            contours, _ = cv2.findContours(
                mask_long[0, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            output_long = (torch.sigmoid(outputs).cpu().numpy() * 255).astype(np.uint8)
            output_long = cv2.cvtColor(output_long[0, 0], cv2.COLOR_GRAY2BGR)
            cv2.drawContours(output_long, contours, -1, (0, 0, 255), 2)

            cv2.imwrite(sample_dir / f"image_{batch_idx}.png", image_long)
            cv2.imwrite(sample_dir / f"output_{batch_idx}.png", output_long)

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
        total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = instantiate(
            self.cfg.scheduler,
            optimizer=optimizer,
            total_iters=total_steps,
            # total_iters=int(self.cfg.total_epochs * 0.5),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
