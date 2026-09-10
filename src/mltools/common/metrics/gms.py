import cv2
import numpy as np
import torch
from typing import Literal
from .metric import Metric


def _compute_instance_gms(preds_alpha, targets_alpha, iou_threshold=0.5):
    """
    Greedy matching using OpenCV for speed and bounding-box optimization.
    Expects numpy arrays.
    """

    # Thresholding (using 0.5 as per your original logic)
    p_mask = (preds_alpha > 0.5).astype(np.uint8)
    t_mask = (targets_alpha > 0.5).astype(np.uint8)

    # 2. Connected Components with Stats
    # stats: [x, y, width, height, area]
    num_p, labels_p, stats_p, _ = cv2.connectedComponentsWithStats(
        p_mask, connectivity=8
    )
    num_t, labels_t, stats_t, _ = cv2.connectedComponentsWithStats(
        t_mask, connectivity=8
    )

    # num_p/t include background (ID 0), so actual object count is num-1
    num_preds = num_p - 1
    num_targets = num_t - 1

    if num_preds == 0:
        return 0, 0, num_targets

    # 3. Calculate Scores (Mean Alpha) for each predicted blob
    pred_info = []
    for i in range(1, num_p):
        # We only look at pixels belonging to this specific label
        mask_i = labels_p == i
        score = preds_alpha[mask_i].mean()
        pred_info.append(
            {
                "id": i,
                "score": score,
                "mask": mask_i,
                "bbox": stats_p[i],  # [x, y, w, h, area]
            }
        )

    # Sort by score descending (Greedy)
    pred_info.sort(key=lambda x: x["score"], reverse=True)

    true_positives = 0
    matched_targets = set()

    # 4. Greedy Matching with Bounding Box Optimization
    for p in pred_info:
        best_iou = 0
        best_t_id = -1

        px, py, pw, ph, _ = p["bbox"]

        for t_id in range(1, num_t):
            if t_id in matched_targets:
                continue

            tx, ty, tw, th, _ = stats_t[t_id]

            # Bounding Box Overlap Check (Fast Pre-filter)
            if not (px < tx + tw and px + pw > tx and py < ty + th and py + ph > ty):
                continue

            # Pixel-level IoU (Only inside the intersection of bounding boxes)
            # This is much faster than checking the whole 256x341 image
            x1, y1 = max(px, tx), max(py, ty)
            x2, y2 = min(px + pw, tx + tw), min(py + ph, th + ty)

            # Crop masks to the overlapping region
            p_crop = p["mask"][y1:y2, x1:x2]
            t_crop = labels_t[y1:y2, x1:x2] == t_id

            inter = np.logical_and(p_crop, t_crop).sum()
            # Union = Area1 + Area2 - Intersection
            union = stats_p[p["id"]][4] + stats_t[t_id][4] - inter

            iou = inter / (union + 1e-7)

            if iou > best_iou:
                best_iou = iou
                best_t_id = t_id

        if best_iou >= iou_threshold:
            true_positives += 1
            matched_targets.add(best_t_id)

    false_positives = num_preds - true_positives
    false_negatives = num_targets - true_positives

    return true_positives, false_positives, false_negatives


def _greedy_matching_score(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    batch_size, num_classes = inputs.shape[:2]
    numerator = np.empty((batch_size, num_classes))
    denominator = np.empty((batch_size, num_classes))
    for i in range(batch_size):
        for j in range(num_classes):
            pred = inputs[i, j]
            target = targets[i, j]
            tp, fp, fn = _compute_instance_gms(pred, target, iou_threshold=0.5)
            numerator[i, j] = 2 * tp
            denominator[i, j] = 2 * tp + fp + fn
    numerator = torch.from_numpy(numerator)
    denominator = torch.from_numpy(denominator)
    return numerator, denominator


class GMSScore(Metric):
    """Instance-level greedy-matching F1 score (connected components + greedy
    IoU matching, ranked by confidence).

    Scores foreground classes only — connected-component instance matching is
    meaningless for background. With background=False (the default) channel 0
    of inputs/targets is taken to be the background class already present in
    the tensor (as mltools.common.evaluate._to_pseudo_logits produces) and is
    dropped; with background=True the tensor is foreground-only and is scored
    as given.
    """

    scores_background_channel = False

    def __init__(
        self,
        average: Literal["micro", "macro", "weighted"] = "macro",
        background: bool = False,
    ):
        super().__init__(soft=True, average=average, background=background)

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor):
        return _greedy_matching_score(inputs, targets)
    

