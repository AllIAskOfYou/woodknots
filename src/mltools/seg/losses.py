import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

# Guards the per-instance Dice against a zero denominator, as smp's Dice does.
_EPS = 1e-7


class SegLoss(torch.nn.Module):
    def __init__(
            self,
            start_weight: float = 0.5,
            end_weight: float = 0.5,
            labels_smoothing: float = 0.05,
            smooth_dice_target: bool = True,
            blob_weight: float = 0.0,
            blob_area_exponent: float = 0.5,
            blob_min_area_fraction: float = 5.0e-4,
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
        # Per-instance Dice, added to (not swapped with) the pooled one — see
        # blob_loss below.
        self.blob_weight = blob_weight
        self.blob_area_exponent = blob_area_exponent
        self.blob_min_area_fraction = blob_min_area_fraction
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary")

    def update_weight(self, current_iter, total_iters):
        self.weight = self.start_weight + (self.end_weight - self.start_weight) * current_iter / total_iters

    def smooth(self, targets):
        return targets * (1 - 2 * self.labels_smoothing) + self.labels_smoothing

    def blob_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Area-weighted per-instance Dice, after blob loss (Kofler et al.,
        IPMI 2023), which averages a Dice term over the connected components of
        the ground truth instead of pooling every pixel into one.

        The pooled Dice weighs an instance by its pixel count, and knot areas
        span 117x from the 5th to the 95th percentile, so the large knots
        dominate the gradient — which is what the measured gap between the
        pooled Dice (0.610) and the per-image Dice on knot-containing images
        (0.505) reflects. Each component here instead contributes

            w_i * (1 - Dice_i),   w_i = max(area_i, a_min) ** -blob_area_exponent

        with exponent 0 reproducing the published (unweighted) blob loss, 0.5
        weighting by inverse square root of area, and 1.0 by inverse area.
        Annotation error is roughly constant in pixels, so relative boundary
        error grows like 1/sqrt(area) and inverse-area weighting would put the
        most gradient where the labels are least trustworthy; a_min
        (blob_min_area_fraction of the crop) clamps the weight of the smallest
        components for the same reason.

        Dice_i is computed over the whole image with the pixels of the OTHER
        components excluded from both numerator and denominator, so predicting
        a neighbouring knot neither helps nor hurts instance i. Everything
        outside any component still counts, which is what keeps the term from
        rewarding spurious blobs on its own.

        Components come from the binary target: this term is per-instance, so a
        smoothed background floor would dominate each one even more than it does
        the pooled Dice (see smooth_dice_target). Empty images contribute no
        component and therefore nothing here — the pooled Dice is what
        supervises them, which is why this is an added term rather than a
        replacement.
        """
        if inputs.shape[1] != 1:
            raise NotImplementedError(
                f"blob_loss expects a single foreground channel, got {inputs.shape[1]}"
            )
        probabilities = torch.sigmoid(inputs)[:, 0]
        binary = (targets[:, 0] > 0.5)
        height, width = binary.shape[-2:]
        min_area = max(1.0, self.blob_min_area_fraction * height * width)

        # Components are integer labels of the target, so they carry no
        # gradient and are cheapest to get from OpenCV on the CPU.
        masks = binary.detach().cpu().numpy().astype(np.uint8)

        weighted_loss = probabilities.new_zeros(())
        total_weight = probabilities.new_zeros(())
        for sample in range(masks.shape[0]):
            count, labels = cv2.connectedComponents(masks[sample], connectivity=8)
            if count < 2:  # background only
                continue
            label_map = torch.from_numpy(labels.astype(np.int64)).to(probabilities.device).reshape(-1)
            flat = probabilities[sample].reshape(-1)

            # Sum of predicted probability inside each component (index 0 is
            # background); index_add keeps the gradient, unlike bincount.
            inside = flat.new_zeros(count).index_add(0, label_map, flat)[1:]
            areas = torch.bincount(label_map, minlength=count)[1:].to(flat.dtype)

            # Excluding the other components leaves
            #   sum(p * valid) = sum(p) - (sum over all components - inside_i)
            # so the whole batch of per-component denominators is one expression.
            predicted = flat.sum() - inside.sum() + inside
            dice = 2 * inside / (predicted + areas).clamp_min(_EPS)

            weights = areas.clamp_min(min_area) ** (-self.blob_area_exponent)
            weighted_loss = weighted_loss + (weights * (1.0 - dice)).sum()
            total_weight = total_weight + weights.sum()

        if total_weight == 0:  # no ground-truth instance anywhere in the batch
            return probabilities.new_zeros(())
        return weighted_loss / total_weight

    def forward(self, inputs, targets):
        smoothed = self.smooth(targets)
        bce_loss = self.bce(inputs, smoothed)
        dice_loss = self.dice(inputs, smoothed if self.smooth_dice_target else targets)
        loss = self.weight * bce_loss + (1 - self.weight) * dice_loss
        if self.blob_weight:
            loss = loss + self.blob_weight * self.blob_loss(inputs, targets)
        return loss
