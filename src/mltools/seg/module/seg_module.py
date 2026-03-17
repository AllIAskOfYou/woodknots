import lightning as L
from hydra.utils import instantiate
from monai.inferers import sliding_window_inference
import torch
from torchmetrics.functional.segmentation import dice_score
from mltools.seg.losses import SegLoss
from mltools.seg.schema import ModuleConfig


def compute_score(outputs, masks):
    preds = outputs > 0.0
    preds = preds.long()
    masks = masks.long()
    score = dice_score(
        preds, masks, average="macro", num_classes=2, input_format="index"
    ).mean()
    return score


class SegModule(L.LightningModule):
    def __init__(self, module_cfg: ModuleConfig):
        super().__init__()
        self.model = instantiate(module_cfg.model)
        # freeze encoder layers
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        self.cfg = module_cfg
        self.loss = SegLoss()
        self.save_hyperparameters(module_cfg)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        images, masks = batch
        outputs = self.model(images)
        loss = self.loss(outputs, masks)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(
            "train_dice",
            compute_score(outputs, masks),
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
                images, roi_size=(res, res), sw_batch_size=4, predictor=self.model
            )
        outputs = outputs.detach()
        masks = masks.detach()

        loss = self.loss(outputs, masks)
        self.log("val_loss", loss, prog_bar=True)

        self.log("val_dice", compute_score(outputs, masks), prog_bar=True)

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params=filter(lambda p: p.requires_grad, self.parameters()),)
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
