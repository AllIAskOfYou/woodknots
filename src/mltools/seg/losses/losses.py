import torch
import segmentation_models_pytorch as smp


class SegLoss(torch.nn.Module):
    def __init__(
            self,
            start_weight: float = 0.5,
            end_weight: float = 0.5,
            labels_smoothing: float = 0.05,
        ):
        super().__init__()
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.weight = start_weight
        self.labels_smoothing = labels_smoothing
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary")

    def update_weight(self, current_iter, total_iters):
        self.weight = self.start_weight + (self.end_weight - self.start_weight) * current_iter / total_iters

    def forward(self, inputs, targets):
        targets = targets * (1 - 2 * self.labels_smoothing) + self.labels_smoothing
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.weight * bce_loss + (1 - self.weight) * dice_loss
