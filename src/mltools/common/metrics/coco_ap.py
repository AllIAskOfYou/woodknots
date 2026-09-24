"""COCO mask AP (pycocotools' COCOeval, iouType="segm") — the reference
definition of mAP50 and mAP50-95.

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
    compute() -> {"mAP50": AP at IoU 0.5, "mAP50_95": AP averaged over IoU
    0.50:0.05:0.95}, over all classes, up to 100 detections per image (the
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

    def compute(self) -> dict[str, float]:
        if not self.annotations:
            return {"mAP50": math.nan, "mAP50_95": math.nan}
        if not self.detections:
            return {"mAP50": 0.0, "mAP50_95": 0.0}
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
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        # stats[0]: AP @ IoU 0.50:0.95, stats[1]: AP @ IoU 0.50 (area all, maxDets 100)
        return {"mAP50": float(evaluator.stats[1]), "mAP50_95": float(evaluator.stats[0])}
