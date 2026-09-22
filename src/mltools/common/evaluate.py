import cv2
import numpy as np
import torch
from pathlib import Path

from mltools.common.instances import (
    instances_to_label_map,
    label_map_to_instances,
    load_gt_instances,
    load_instances,
)
from mltools.common.metrics import Metric, STANDARD_METRICS, instance_gms

_FOREGROUND_LOGIT = 10.0


def _load_label_map(path: Path) -> np.ndarray:
    label_map = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if label_map is None:
        raise RuntimeError(f"Failed to read label map: {path}")
    return label_map


def _to_pseudo_logits(label_map: np.ndarray, num_classes: int) -> torch.Tensor:
    """One-hot decode a (H, W) label map (0=background, 1..C=class) into a
    (1, C+1, H, W) tensor (channel 0 = background, 1..C = each class),
    encoded as pseudo-logits ({0,1} -> {-10, +10}) so Metric.compute()'s
    threshold/sigmoid logic applies unchanged.
    """
    channels = np.stack(
        [(label_map == class_idx) for class_idx in range(num_classes + 1)], axis=0
    )
    logits = np.where(channels, _FOREGROUND_LOGIT, -_FOREGROUND_LOGIT).astype(np.float32)
    return torch.from_numpy(logits).unsqueeze(0)


def check_dir_ready(path: Path, description: str, logger=None) -> None:
    if not path.exists():
        if logger:
            logger.error(f"{description} directory does not exist: {path}")
        raise RuntimeError(f"{description} directory does not exist: {path}")
    if not any(path.iterdir()):
        if logger:
            logger.error(f"{description} directory is empty: {path}")
        raise RuntimeError(f"{description} directory is empty: {path}")


def _to_benchmark_classes(label_map: np.ndarray, num_classes: int) -> np.ndarray:
    """Map a prediction's class ids onto the benchmark classes. With a single
    benchmark class every foreground id is that class; with several, ids must
    already be 1..C in class_names order."""
    if num_classes == 1:
        return (label_map > 0).astype(np.uint8)
    if label_map.max() > num_classes:
        raise RuntimeError(
            f"prediction has class id {label_map.max()}, but the benchmark has {num_classes} classes"
        )
    return label_map


def evaluate_segmentation(
    pred_dir: str,
    annotations_path: str,
    class_names: list[str],
    confidences: list[float],
    metrics: dict[str, Metric] = STANDARD_METRICS,
) -> dict:
    """Score the predicted label maps in pred_dir (<case>.png) against ground
    truth built from the annotations in annotations_path.

    Ground truth is the same for every framework: one instance per annotation
    whose category is in class_names and whose confidence is in confidences
    (mltools.common.instances.load_gt_instances), and the label map that is
    the union of those instances.

    - dice / iou: on the label maps.
    - gms: on instances, the prediction's instances being the 8-connected
      components of its label map. Defined for every method.
    - gms_instances: only for cases whose method also stored its own
      instances (<case>.npz next to the label map, instance-native methods):
      the same metric on those instances. Comparing it with gms shows what
      the native instances change.

    Returns {"metrics": {name: mean_score}, "per_case": {case_id: {name: score}}};
    each metric is averaged over the cases that have it.
    """
    gt_instances = load_gt_instances(annotations_path, class_names, confidences)
    num_classes = len(class_names)

    per_case: dict[str, dict[str, float]] = {}
    for pred_path in sorted(Path(pred_dir).glob("*.png")):
        case_id = pred_path.stem
        if case_id not in gt_instances:
            raise RuntimeError(f"No ground truth in {annotations_path} for prediction: {pred_path.name}")
        gt = gt_instances[case_id]

        pred_map = _to_benchmark_classes(_load_label_map(pred_path), num_classes)
        if pred_map.shape != gt.shape:
            raise RuntimeError(f"{case_id}: prediction shape {pred_map.shape} != ground truth {gt.shape}")

        pred_logits = _to_pseudo_logits(pred_map, num_classes)
        gt_onehot = _to_pseudo_logits(instances_to_label_map(gt), num_classes) > 0
        scores = {name: metric.compute(pred_logits, gt_onehot.float()) for name, metric in metrics.items()}
        scores["gms"] = instance_gms(label_map_to_instances(pred_map), gt, num_classes)

        native_path = pred_path.with_suffix(".npz")
        if native_path.exists():
            native = load_instances(native_path)
            if num_classes == 1:
                native = native._replace(labels=np.ones_like(native.labels))
            scores["gms_instances"] = instance_gms(native, gt, num_classes)

        per_case[case_id] = scores

    names = dict.fromkeys(name for scores in per_case.values() for name in scores)
    mean_metrics = {}
    for name in names:
        values = [scores[name] for scores in per_case.values() if name in scores]
        mean_metrics[name] = sum(values) / len(values)

    return {"metrics": mean_metrics, "per_case": per_case}


def evaluate_run(cfg, pred_dir) -> dict:
    """evaluate_segmentation for one run, with the ground truth and benchmark
    classes taken from the shared dataset config — identical for every
    framework, whatever it was trained on."""
    from mltools.common.setup import get_annotations_path

    annotations_path = get_annotations_path(cfg.dataset)
    if not Path(annotations_path).exists():
        raise RuntimeError(f"Ground-truth annotations not found: {annotations_path} (run setup first)")
    return evaluate_segmentation(
        pred_dir=str(pred_dir),
        annotations_path=annotations_path,
        class_names=list(cfg.dataset.class_names),
        confidences=list(cfg.dataset.eval_confidences),
    )
