"""Instances: the prediction format of every framework.

Each framework writes one predictions/<case>.npz per test image (save_instances):

    masks   (N, H, W) bool     one binary mask per instance, overlaps allowed
    labels  (N,)      int32    class id per instance, 1..C (benchmark classes)
    scores  (N,)      float32  confidence per instance

Instance-native methods (YOLO instance, Mask2Former) store the instances they
predict. Semantic methods store the connected components of their thresholded
probabilities, each scored by its mean probability (probs_to_instances).

Ground-truth instances are never stored: the evaluation rasterises them from
the annotation polygons (load_gt_instances), one instance per annotation. Both
sides' label maps for the pixel metrics are the union of their instances
(instances_to_label_map).
"""
import json
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from pycocotools import mask as mask_utils

# Two floats count as the same annotator confidence (0.2 / 0.6 / 1.0) if they
# differ by less than this; the same tolerance mltools.yolo.dataset uses.
CONFIDENCE_TOLERANCE = 1e-3


class Instances(NamedTuple):
    masks: np.ndarray  # (N, H, W) bool
    labels: np.ndarray  # (N,) int32
    scores: np.ndarray  # (N,) float32

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.masks.shape[1:])

    @property
    def num_instances(self) -> int:
        # Not __len__: NamedTuple's _replace/_make use len() to check the
        # tuple's field count, so overriding it breaks them.
        return len(self.labels)


def make_instances(masks, labels, scores=None, shape=None) -> Instances:
    """Normalise dtypes; shape (H, W) is required when there are no masks."""
    if len(masks) == 0:
        if shape is None:
            raise ValueError("shape is required to build an empty Instances")
        masks = np.zeros((0, *shape), dtype=bool)
    else:
        masks = np.asarray(masks, dtype=bool)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = (
        np.ones(len(labels), dtype=np.float32)
        if scores is None
        else np.asarray(scores, dtype=np.float32).reshape(-1)
    )
    if not (len(masks) == len(labels) == len(scores)):
        raise ValueError(
            f"mismatched lengths: {len(masks)} masks, {len(labels)} labels, {len(scores)} scores"
        )
    return Instances(masks, labels, scores)


def save_instances(path, instances: Instances) -> None:
    np.savez_compressed(
        Path(path), masks=instances.masks, labels=instances.labels, scores=instances.scores
    )


def load_instances(path) -> Instances:
    with np.load(Path(path)) as data:
        return make_instances(
            data["masks"], data["labels"], data["scores"], shape=data["masks"].shape[1:]
        )


def label_map_to_instances(label_map: np.ndarray) -> Instances:
    """Split a (H, W) label map (0 = background, 1..C = classes) into the
    8-connected components of each class, every component — of any size — an
    instance. No size filter: any threshold would have to be chosen from the
    data, which includes the test folds. Scores are 1.0 — a label map carries
    no confidence (probs_to_instances adds them from probabilities)."""
    masks, labels = [], []
    for class_id in np.unique(label_map):
        if class_id == 0:
            continue
        count, components = cv2.connectedComponents(
            (label_map == class_id).astype(np.uint8), connectivity=8
        )
        for component in range(1, count):
            masks.append(components == component)
            labels.append(int(class_id))
    return make_instances(masks, labels, shape=label_map.shape)


def probs_to_instances(probs: np.ndarray) -> Instances:
    """Instances of a semantic prediction. probs: (C, H, W) class
    probabilities. Each pixel takes the most probable of background
    (1 - sum of the class probabilities) and the C classes — for one class
    exactly probs > 0.5 — and every 8-connected component of each class is an
    instance, scored by the mean probability of its class inside it."""
    background = 1.0 - probs.sum(axis=0, keepdims=True)
    label_map = np.concatenate([background, probs], axis=0).argmax(axis=0).astype(np.uint8)
    components = label_map_to_instances(label_map)
    scores = [probs[label - 1][mask].mean() for mask, label in zip(components.masks, components.labels)]
    return components._replace(scores=np.asarray(scores, dtype=np.float32))


def instances_to_label_map(instances: Instances) -> np.ndarray:
    """Collapse instances into a (H, W) uint8 label map: the union of each
    class's instances. Where classes overlap the highest class id wins, the
    same rule mltools.common.masks.mask_to_label_map applies."""
    label_map = np.zeros(instances.shape, dtype=np.uint8)
    for mask, label in sorted(zip(instances.masks, instances.labels), key=lambda ml: ml[1]):
        label_map[mask] = label
    return label_map


def _annotation_confidence(ann: dict) -> float:
    return float(ann.get("attributes", {}).get("confidence", 1.0))


def _annotation_rle(segmentation, h: int, w: int):
    """pycocotools' COCO.annToRLE, which COCO.annToMask (used by every
    framework's label writer) wraps: polygons are rasterised and merged,
    uncompressed RLE is compressed, compressed RLE is used as is."""
    if isinstance(segmentation, list):
        return mask_utils.merge(mask_utils.frPyObjects(segmentation, h, w))
    if isinstance(segmentation["counts"], list):
        return mask_utils.frPyObjects(segmentation, h, w)
    return segmentation


def load_gt_instances(
    annotations_path, class_names: list[str], confidences: list[float], case_ids=None
) -> dict[str, Instances]:
    """Ground-truth instances for every image of a COCO annotation file, keyed
    by case id (image file stem): one instance per annotation whose category
    is in class_names (label = position + 1) and whose confidence attribute is
    one of confidences. Polygons are rasterised with pycocotools at the size
    recorded for the image, exactly as every framework's own label writer
    does, so the union of these masks is the label map those writers produce.
    case_ids limits it to those images (all images when None)."""
    coco = json.loads(Path(annotations_path).read_text())
    category_label = {
        c["id"]: class_names.index(c["name"]) + 1 for c in coco["categories"] if c["name"] in class_names
    }
    wanted = None if case_ids is None else set(case_ids)
    images = {
        im["id"]: im
        for im in coco["images"]
        if wanted is None or Path(im["file_name"]).stem in wanted
    }
    per_image: dict[int, list[dict]] = {img_id: [] for img_id in images}
    for ann in coco["annotations"]:
        if ann["image_id"] not in per_image:
            continue
        conf = _annotation_confidence(ann)
        if ann["category_id"] in category_label and any(
            abs(conf - c) < CONFIDENCE_TOLERANCE for c in confidences
        ):
            per_image[ann["image_id"]].append(ann)

    out = {}
    for img_id, anns in per_image.items():
        im = images[img_id]
        h, w = im["height"], im["width"]
        masks = [mask_utils.decode(_annotation_rle(ann["segmentation"], h, w)).astype(bool) for ann in anns]
        out[Path(im["file_name"]).stem] = make_instances(
            masks, [category_label[a["category_id"]] for a in anns], shape=(h, w)
        )
    return out
