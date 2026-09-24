import cv2
import lightning as L
from hydra.utils import instantiate
from monai.inferers import sliding_window_inference
import torch
import torch.nn.functional as F
import numpy as np

from mltools.seg.losses import SegLoss
from mltools.common.logger import get_logger
from mltools.common.metrics import DiceScore, IoUScore, GMSScore
from pathlib import Path

from omegaconf import DictConfig

logger = get_logger(__name__)

# train.inference: how a whole image is predicted, at validation (checkpoint
# selection) and for the final predictions alike.
SLIDING_WINDOW = "sliding_window"  # resolution x resolution windows, overlap 0.5, Gaussian blending
WHOLE_IMAGE = "whole_image"  # one pass over the image, padded to a multiple of INFERENCE_STRIDE
INFERENCE_MODES = (SLIDING_WINDOW, WHOLE_IMAGE)
# Output stride of every backbone here (ConvNeXt, ResNet, MiT): inputs whose
# sides aren't multiples of it would misalign the decoder's skip connections.
INFERENCE_STRIDE = 32


def denormalize_image(image_tensor):
    # Assuming the image was normalized using ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)  # CxHxW to HxWxC
    image_denorm = (image_np * std) + mean
    image_denorm = np.clip(image_denorm, 0, 1)
    return image_denorm


class SegModule(L.LightningModule):
    def __init__(self, module_cfg: DictConfig):
        super().__init__()
        self.model = instantiate(module_cfg.model)
        # freeze encoder layers
        if module_cfg.freeze_encoder:
            self.freeze_encoder()
        # Selectively re-enable the deepest stages for fine-tuning. Applied
        # after freeze_encoder so `unfreeze_stages: [3]` means "freeze
        # everything except the last stage"; the encoder is put back in train
        # mode, since freeze_encoder() also called .eval() on it.
        unfreeze_stages = module_cfg.get("unfreeze_stages", None)
        if unfreeze_stages:
            self.model.encoder.train()
            self.model.unfreeze_encoder_stages(list(unfreeze_stages))
        self.cfg = module_cfg
        self.inference = module_cfg.get("inference", SLIDING_WINDOW)
        if self.inference not in INFERENCE_MODES:
            raise ValueError(f"Unknown train.inference {self.inference!r}, expected one of {INFERENCE_MODES}.")
        self.loss = SegLoss(
            start_weight=0.5,
            end_weight=0.5,
            labels_smoothing=module_cfg.loss.labels_smoothing,
            smooth_dice_target=module_cfg.loss.get("smooth_dice_target", True),
        )
        self.save_hyperparameters(module_cfg)
        # This module emits a single foreground channel, so background=True
        # synthesises the background channel these metrics average over —
        # reproducing the metric definitions the pre-restructure runs used
        # (val_dice here is background-inclusive).
        self.metrics = {
            "dice": DiceScore(background=True, average="macro"),
            "iou": IoUScore(background=True, average="macro"),
            "gms": GMSScore(background=True, average="macro"),
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

    @torch.no_grad()
    def infer_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Logits for whole images, per train.inference (see INFERENCE_MODES).
        whole_image zero-pads the bottom/right edges — zero is the mean colour
        after normalisation, the same fill MONAI's sliding window uses — and
        crops the logits back to the image."""
        if self.inference == SLIDING_WINDOW:
            res = self.cfg.resolution
            return sliding_window_inference(
                x,
                roi_size=(res, res),
                sw_batch_size=4,
                predictor=self.model,
                device=x.device,
                overlap=0.5,
                mode="gaussian",
            )
        h, w = x.shape[-2:]
        padded = F.pad(x, (0, (-w) % INFERENCE_STRIDE, 0, (-h) % INFERENCE_STRIDE))
        return self.model(padded)[..., :h, :w]

    def forward(self, x):
        return torch.sigmoid(self.infer_logits(x))

    @property
    def _model_computes_loss(self) -> bool:
        """Set-prediction models (Mask2Former) compute a Hungarian-matched
        loss internally from per-instance targets, so SegLoss doesn't apply
        to them; they expose compute_loss() and return dense logits from
        forward() for the metrics."""
        return getattr(self.model, "computes_own_loss", False)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        images, masks = batch
        if self._model_computes_loss:
            loss, outputs = self.model.compute_loss(images, masks)
        else:
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

    def on_validation_epoch_end(self):
        """Lightning's progress bar is the only thing that normally surfaces
        these numbers, and rich suppresses it entirely when stdout isn't a
        TTY (i.e. every sbatch job), so log one line per epoch instead —
        stdlib logging reaches both the SLURM .out and Hydra's run.log.
        Built from callback_metrics so it follows whatever self.metrics holds."""
        if self.trainer.sanity_checking:
            return
        metrics = self.trainer.callback_metrics

        def fmt(keys):
            return " ".join(
                f"{k}={float(metrics[k]):.4f}" for k in keys if k in metrics
            )

        parts = [f"epoch={self.current_epoch}"]
        for prefix in ("train", "val"):
            line = fmt([f"{prefix}_loss"] + [f"{prefix}_{n}" for n in self.metrics])
            if line:
                parts.append(line)
        if "lr" in metrics:
            parts.append(f"lr={float(metrics['lr']):.2e}")
        logger.info(" | ".join(parts))

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
        outputs = self.infer_logits(images).detach()
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

        if self._model_computes_loss:
            # The sliding-window output is a stitched dense map, so the
            # query-based loss can't be recovered from it — run the model's
            # own loss over the full image instead. Metrics still come from
            # the sliding-window logits above, matching inference.
            loss, _ = self.model.compute_loss(images, masks)
        else:
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
        lr = self.cfg.optimizer.lr
        weight_decay = self.cfg.optimizer.weight_decay
        param_groups = [
            {"params": self.model.get_decoder_parameters(), "lr": lr, "weight_decay": weight_decay}
        ]

        # Without this group the encoder would receive gradients but never an
        # update, so freeze_encoder: false / unfreeze_stages would silently
        # reproduce the frozen run. Empty whenever the encoder is fully frozen,
        # which leaves the frozen setup byte-identical to before.
        encoder_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
        if encoder_params:
            encoder_lr = self.cfg.optimizer.get("encoder_lr", None)
            encoder_wd = self.cfg.optimizer.get("encoder_weight_decay", None)
            param_groups.append(
                {
                    "params": encoder_params,
                    "lr": lr if encoder_lr is None else encoder_lr,
                    "weight_decay": weight_decay if encoder_wd is None else encoder_wd,
                }
            )

        optimizer = torch.optim.AdamW(params=param_groups)
        total_steps = self.trainer.estimated_stepping_batches

        scheduler = instantiate(
            self.cfg.scheduler,
            optimizer=optimizer,
            total_iters=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
