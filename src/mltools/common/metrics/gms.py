import cv2
import numpy as np
import torch
from typing import Literal
from .metric import Metric, _average_classes


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


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1), end-exclusive; all zero for an empty mask."""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0:
        return 0, 0, 0, 0
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def _match_instances(pred_masks, pred_scores, gt_masks, iou_threshold=0.5):
    """Greedy matching of given instance masks — the rule of
    _compute_instance_gms, applied to instances instead of connected
    components: predictions in descending score order each take the unmatched
    target with the highest IoU, a match if it reaches iou_threshold.
    Instances may overlap. -> (tp, fp, fn)"""
    if len(pred_masks) == 0:
        return 0, 0, len(gt_masks)
    gt_boxes = [_bbox(m) for m in gt_masks]
    gt_areas = [int(m.sum()) for m in gt_masks]
    matched = set()
    tp = 0
    for p in np.argsort(-np.asarray(pred_scores), kind="stable"):
        p_mask = pred_masks[p]
        py0, py1, px0, px1 = _bbox(p_mask)
        p_area = int(p_mask.sum())
        best_iou, best_t = 0.0, -1
        for t, (ty0, ty1, tx0, tx1) in enumerate(gt_boxes):
            if t in matched or not (py0 < ty1 and ty0 < py1 and px0 < tx1 and tx0 < px1):
                continue
            y0, y1, x0, x1 = max(py0, ty0), min(py1, ty1), max(px0, tx0), min(px1, tx1)
            inter = int(np.logical_and(p_mask[y0:y1, x0:x1], gt_masks[t][y0:y1, x0:x1]).sum())
            iou = inter / (p_area + gt_areas[t] - inter + 1e-7)
            if iou > best_iou:
                best_iou, best_t = iou, t
        if best_iou >= iou_threshold:
            tp += 1
            matched.add(best_t)
    return tp, len(pred_masks) - tp, len(gt_masks) - tp


def instance_gms(pred, gt, num_classes: int, iou_threshold: float = 0.5, average="macro") -> float:
    """GMS on instances (mltools.common.instances.Instances): per foreground
    class, F1 = 2TP / (2TP + FP + FN) over matched instances, averaged over
    classes with the reduction GMSScore uses (a class with no instances on
    either side scores 1.0)."""
    classes = range(1, num_classes + 1)
    numerator = np.empty((1, num_classes))
    denominator = np.empty((1, num_classes))
    for i, c in enumerate(classes):
        p, t = pred.labels == c, gt.labels == c
        tp, fp, fn = _match_instances(pred.masks[p], pred.scores[p], gt.masks[t], iou_threshold)
        numerator[0, i] = 2 * tp
        denominator[0, i] = 2 * tp + fp + fn
    support = torch.tensor([[int((gt.labels == c).sum()) for c in classes]])
    score = _average_classes(torch.from_numpy(numerator), torch.from_numpy(denominator), support, average)
    return score.nanmean().item()


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
        super().__init__(average=average, background=background)

    def _preds(self, logits: torch.Tensor) -> torch.Tensor:
        """Probabilities, not the hard decision: components are thresholded
        at 0.5 and scored by their mean probability."""
        return torch.sigmoid(logits)

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor):
        return _greedy_matching_score(inputs, targets)
    

