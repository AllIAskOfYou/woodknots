"""Foreground-only Dice and IoU with the counts pooled over the whole
evaluation — the definition the training losses optimise.

The per-case dice/iou of the evaluation are background-inclusive and macro
averaged over cases: each image counts once, the background channel (~98% of
the pixels, scored near 1.0) is averaged in, and an image with no knots at all
can score a perfect 1.0. These pooled metrics are the other convention: the
background channel is dropped, and intersections and cardinalities are summed
across every case before a single division, so one large knot weighs as much
as many small ones and empty images contribute nothing but their false
positives.

Both are computed per class and then averaged over classes, the same way
segmentation_models_pytorch aggregates its Dice loss. With a single benchmark
class (the current dataset) that is just the class's own pooled score.
"""
import math

import numpy as np


class PooledOverlap:
    """Accumulates (prediction, ground truth) label maps per case, then
    compute() -> dice_fg_pooled, iou_fg_pooled, precision_fg_pooled and
    recall_fg_pooled, all pixel-level and pooled the same way. Precision and
    recall split what dice_fg_pooled combines: whether the predicted knot
    pixels are knots, and whether the knot pixels were found.

    Classes that never appear in either the predictions or the ground truth
    are undefined and left out of the class average; if no class ever appears,
    every score is nan. A class with ground truth but no predicted pixel at
    all scores 0 everywhere, precision included."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        # Per foreground class 1..C: |P and G|, |P|, |G| summed over cases.
        self.intersection = np.zeros(num_classes, dtype=np.int64)
        self.pred_sum = np.zeros(num_classes, dtype=np.int64)
        self.gt_sum = np.zeros(num_classes, dtype=np.int64)

    def add(self, pred_label_map: np.ndarray, gt_label_map: np.ndarray) -> None:
        for index in range(self.num_classes):
            pred = pred_label_map == index + 1
            gt = gt_label_map == index + 1
            self.intersection[index] += int(np.count_nonzero(pred & gt))
            self.pred_sum[index] += int(np.count_nonzero(pred))
            self.gt_sum[index] += int(np.count_nonzero(gt))

    def merge(self, other: "PooledOverlap") -> None:
        """Add another accumulator's counts, as if its cases had been added here."""
        self.intersection += other.intersection
        self.pred_sum += other.pred_sum
        self.gt_sum += other.gt_sum

    def compute(self) -> dict[str, float]:
        scores: dict[str, list[float]] = {"dice": [], "iou": [], "precision": [], "recall": []}
        for index in range(self.num_classes):
            intersection = int(self.intersection[index])
            pred_sum = int(self.pred_sum[index])
            gt_sum = int(self.gt_sum[index])
            cardinality = pred_sum + gt_sum
            if cardinality == 0:  # class absent from both sides: undefined
                continue
            scores["dice"].append(2 * intersection / cardinality)
            scores["iou"].append(intersection / (cardinality - intersection))
            # pred_sum == 0 (nothing predicted anywhere) leaves precision
            # undefined; 0.0 keeps it consistent with the recall and dice of
            # that same degenerate run.
            scores["precision"].append(intersection / pred_sum if pred_sum else 0.0)
            scores["recall"].append(intersection / gt_sum if gt_sum else 0.0)
        return {
            f"{name}_fg_pooled": float(np.mean(values)) if values else math.nan
            for name, values in scores.items()
        }
