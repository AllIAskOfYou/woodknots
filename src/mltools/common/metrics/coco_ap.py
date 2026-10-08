"""COCO mask AP (pycocotools' COCOeval, iouType="segm") — the reference
definition of mAP50 and mAP50-95, plus mAP25: the same AP at IoU 0.25, which
credits a detection that finds a knot but outlines it loosely.

AP is a dataset-level metric: all detections of all images are ranked by score
into one precision-recall curve, so it is accumulated over the cases of an
evaluation (add) and computed once (compute), unlike the per-image metrics.
Computed on the stored predictions, i.e. at each method's operating point:
detections below a method's confidence gate (YOLO conf, Mask2Former score) or
outside its thresholded probabilities (semantic methods) are not there, and
the recall they would add counts as precision 0.
"""
import contextlib
import io
import math

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _rle(mask: np.ndarray) -> dict:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


class COCOMaskAP:
    """Accumulates (prediction, ground truth) Instances per image, then
    compute() -> {"mAP25": AP at IoU 0.25, "mAP50": AP at IoU 0.5,
    "mAP50_95": AP averaged over IoU 0.50:0.05:0.95}, over all classes, up to 100 detections per image (the
    COCO defaults). Images without ground truth still count: detections on
    them are false positives."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.images, self.annotations, self.detections = [], [], []

    def add(self, pred, gt) -> None:
        image_id = len(self.images) + 1
        h, w = gt.shape
        self.images.append({"id": image_id, "height": h, "width": w})
        for mask, label in zip(gt.masks, gt.labels):
            rle = _rle(mask)
            self.annotations.append({
                "id": len(self.annotations) + 1,
                "image_id": image_id,
                "category_id": int(label),
                "segmentation": rle,
                "area": float(mask_utils.area(rle)),
                "bbox": mask_utils.toBbox(rle).tolist(),
                "iscrowd": 0,
            })
        for mask, label, score in zip(pred.masks, pred.labels, pred.scores):
            self.detections.append({
                "image_id": image_id,
                "category_id": int(label),
                "segmentation": _rle(mask),
                "score": float(score),
            })

    def merge(self, other: "COCOMaskAP") -> None:
        """Append another accumulator's images, renumbered after this one's, as
        if they had been added here in that order."""
        image_offset, annotation_offset = len(self.images), len(self.annotations)
        self.images += [{**image, "id": image["id"] + image_offset} for image in other.images]
        self.annotations += [
            {**ann, "id": ann["id"] + annotation_offset, "image_id": ann["image_id"] + image_offset}
            for ann in other.annotations
        ]
        self.detections += [
            {**det, "image_id": det["image_id"] + image_offset} for det in other.detections
        ]

    # COCO's ten thresholds plus 0.25. Each threshold is matched independently,
    # so adding one leaves the others' results unchanged.
    IOU_THRESHOLDS = np.array([0.25, *np.linspace(0.5, 0.95, 10)])

    def compute(self) -> dict[str, float]:
        if not self.annotations:
            return {"mAP25": math.nan, "mAP50": math.nan, "mAP50_95": math.nan}
        if not self.detections:
            return {"mAP25": 0.0, "mAP50": 0.0, "mAP50_95": 0.0}
        with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints progress
            coco_gt = COCO()
            coco_gt.dataset = {
                "images": self.images,
                "annotations": self.annotations,
                "categories": [{"id": c} for c in range(1, self.num_classes + 1)],
            }
            coco_gt.createIndex()
            evaluator = COCOeval(coco_gt, coco_gt.loadRes(self.detections), iouType="segm")
            evaluator.params.imgIds = [image["id"] for image in self.images]
            evaluator.params.iouThrs = self.IOU_THRESHOLDS
            evaluator.evaluate()
            evaluator.accumulate()

        # precision: (IoU thresholds, recall points, classes, area ranges, maxDets).
        # Area range "all" and maxDets 100, as COCOeval.summarize's AP rows;
        # entries of -1 (no ground truth for that class) are left out, as there.
        precision = evaluator.eval["precision"][:, :, :, 0, -1]

        def average(rows) -> float:
            values = precision[rows]
            values = values[values > -1]
            return float(values.mean()) if values.size else -1.0

        thresholds = self.IOU_THRESHOLDS
        return {
            "mAP25": average(np.isclose(thresholds, 0.25)),
            "mAP50": average(np.isclose(thresholds, 0.5)),
            "mAP50_95": average(thresholds >= 0.5 - 1e-9),
        }
