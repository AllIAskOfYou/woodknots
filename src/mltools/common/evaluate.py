import cv2
import numpy as np
import torch
from pathlib import Path

from mltools.common.metrics import Metric, STANDARD_METRICS

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


def evaluate_segmentation(
    pred_dir: str,
    gt_dir: str,
    class_names: list[str],
    metrics: dict[str, Metric] = STANDARD_METRICS,
) -> dict:
    """Score predicted label-map PNGs in pred_dir against ground-truth label-map
    PNGs in gt_dir (matched by filename).

    Returns {"metrics": {name: mean_score}, "per_case": {case_id: {name: score}}}.
    """
    pred_paths = sorted(Path(pred_dir).glob("*.png"))
    num_classes = len(class_names)

    per_case: dict[str, dict[str, float]] = {}
    for pred_path in pred_paths:
        case_id = pred_path.stem
        gt_path = Path(gt_dir) / pred_path.name
        if not gt_path.exists():
            raise RuntimeError(f"No matching ground truth for prediction: {pred_path.name}")

        pred_logits = _to_pseudo_logits(_load_label_map(pred_path), num_classes)
        gt = _to_pseudo_logits(_load_label_map(gt_path), num_classes) > 0

        per_case[case_id] = {
            name: metric.compute(pred_logits, gt.float()) for name, metric in metrics.items()
        }

    mean_metrics = {
        name: sum(scores[name] for scores in per_case.values()) / len(per_case)
        for name in metrics
    }

    return {"metrics": mean_metrics, "per_case": per_case}
