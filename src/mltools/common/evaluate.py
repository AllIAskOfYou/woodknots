import numpy as np
import torch
from pathlib import Path

from mltools.common.instances import instances_to_label_map, load_gt_instances, load_instances
from mltools.common.metrics import (
    COCOMaskAP,
    Metric,
    PooledInstanceF1,
    PooledOverlap,
    STANDARD_METRICS,
    instance_gms,
)

_FOREGROUND_LOGIT = 10.0


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


def evaluate_segmentation(
    pred_dir: str,
    annotations_path: str,
    class_names: list[str],
    confidences: list[float],
    metrics: dict[str, Metric] = STANDARD_METRICS,
) -> dict:
    """Score the predicted instances in pred_dir (<case>.npz, see
    mltools.common.instances) against ground truth built from the annotations
    in annotations_path: one instance per annotation whose category is in
    class_names and whose confidence is in confidences.

    - pixel metrics: on the label maps that are the union of each side's
      instances, averaged over cases — dice/iou with the background channel
      averaged in, dice_fg/iou_fg over the foreground classes only (a case
      with no ground truth and no prediction scores 1.0, a false positive on
      such a case scores 0.0);
    - instance_precision_pooled, instance_recall_pooled, instance_f1_pooled:
      greedy instance matching at IoU 0.5 with the counts pooled over all cases
      (dataset-level, so in "metrics" only). instance_f1 in "per_case" is the
      same matching scored per image, where a case with no ground truth and no
      prediction scores 1.0 — kept as a diagnostic;
    - mAP50, mAP50_95: COCO mask AP over all cases at once (dataset-level, so
      in "metrics" only, not "per_case"), at each method's operating point;
    - dice_fg_pooled, iou_fg_pooled, precision_fg_pooled, recall_fg_pooled:
      foreground-only pixel counts pooled over all cases before dividing (the
      definition the losses optimise) — also dataset-level, so likewise not in
      "per_case".

    Returns {"metrics": {name: score}, "per_case": {case_id: {name: score}}};
    per-image metrics are averaged over the cases.
    """
    pred_paths = sorted(Path(pred_dir).glob("*.npz"))
    # Only the evaluated cases (the fold's test split), not the whole dataset.
    gt_instances = load_gt_instances(
        annotations_path, class_names, confidences, case_ids=[p.stem for p in pred_paths]
    )
    predictions = {}
    for pred_path in pred_paths:
        if pred_path.stem not in gt_instances:
            raise RuntimeError(f"No ground truth in {annotations_path} for prediction: {pred_path.name}")
        predictions[pred_path.stem] = load_instances(pred_path)
    return score_instances(predictions, gt_instances, len(class_names), metrics=metrics)


def score_instances(
    predictions: dict,
    gt_instances: dict,
    num_classes: int,
    metrics: dict[str, Metric] = STANDARD_METRICS,
) -> dict:
    """The scoring half of evaluate_segmentation, on instances already in
    memory: {case_id: Instances} for both sides. Split out so callers that vary
    the predictions without rewriting them to disk — a confidence sweep, say —
    score through exactly the same code path as a normal evaluation."""
    per_case: dict[str, dict[str, float]] = {}
    average_precision = COCOMaskAP(num_classes)
    pooled = PooledOverlap(num_classes)
    pooled_instances = PooledInstanceF1(num_classes)
    for case_id, pred in predictions.items():
        gt = gt_instances[case_id]
        if pred.shape != gt.shape:
            raise RuntimeError(f"{case_id}: prediction shape {pred.shape} != ground truth {gt.shape}")
        if pred.num_instances and not (1 <= pred.labels.min() and pred.labels.max() <= num_classes):
            raise RuntimeError(f"{case_id}: instance labels outside the benchmark classes 1..{num_classes}")

        pred_label_map = instances_to_label_map(pred)
        gt_label_map = instances_to_label_map(gt)
        pred_logits = _to_pseudo_logits(pred_label_map, num_classes)
        gt_onehot = _to_pseudo_logits(gt_label_map, num_classes) > 0
        scores = {name: metric.compute(pred_logits, gt_onehot.float()) for name, metric in metrics.items()}
        # Per case, so a knot-free case with no prediction scores 1.0 —
        # a diagnostic, not a headline: instance_*_pooled below is the same
        # matching with the counts pooled over the evaluation instead.
        scores["instance_f1"] = instance_gms(pred, gt, num_classes)
        per_case[case_id] = scores
        average_precision.add(pred, gt)
        pooled.add(pred_label_map, gt_label_map)
        pooled_instances.add(pred, gt)

    names = list(next(iter(per_case.values()))) if per_case else [*metrics, "instance_f1"]
    mean_metrics = {name: sum(scores[name] for scores in per_case.values()) / len(per_case) for name in names}
    mean_metrics |= average_precision.compute()
    mean_metrics |= pooled.compute()
    mean_metrics |= pooled_instances.compute()
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
