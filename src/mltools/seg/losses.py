import torch
import segmentation_models_pytorch as smp


class SegLoss(torch.nn.Module):
    def __init__(
            self,
            start_weight: float = 0.5,
            end_weight: float = 0.5,
            labels_smoothing: float = 0.05,
            smooth_dice_target: bool = True,
        ):
        super().__init__()
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.weight = start_weight
        self.labels_smoothing = labels_smoothing
        # smp's binary Dice pools TP/FP/FN over the whole batch, so a smoothed
        # target contributes labels_smoothing per background pixel. Knots cover
        # ~3% of the pixels, which puts more target mass in the background floor
        # (0.05 x 0.97) than in the knots themselves (0.95 x 0.03) — the Dice
        # term then mostly measures whether the model reproduces that floor.
        # False smooths the BCE target only and keeps the Dice target binary.
        self.smooth_dice_target = smooth_dice_target
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary")

    def update_weight(self, current_iter, total_iters):
        self.weight = self.start_weight + (self.end_weight - self.start_weight) * current_iter / total_iters

    def smooth(self, targets):
        return targets * (1 - 2 * self.labels_smoothing) + self.labels_smoothing

    def forward(self, inputs, targets):
        smoothed = self.smooth(targets)
        bce_loss = self.bce(inputs, smoothed)
        dice_loss = self.dice(inputs, smoothed if self.smooth_dice_target else targets)
        return self.weight * bce_loss + (1 - self.weight) * dice_loss
