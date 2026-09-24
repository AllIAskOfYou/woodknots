from .metric import Metric
from .dice import DiceScore
from .iou import IoUScore
from .gms import GMSScore, instance_gms
from .coco_ap import COCOMaskAP
from .pooled import PooledOverlap

__all__ = [
    "Metric",
    "DiceScore",
    "IoUScore",
    "GMSScore",
    "instance_gms",
    "COCOMaskAP",
    "PooledOverlap",
]

# Pixel-level metrics of the cross-framework evaluation, computed on label maps.
# The instance-level GMS is computed on instances (instance_gms), not from here;
# GMSScore remains for seg's training-time validation metric.
#
# dice/iou average the background channel in (it covers ~97% of the pixels and
# scores near 1.0, so they sit in [~0.5, 1.0]); dice_fg/iou_fg drop it and score
# the knots alone, which is what the training losses optimise. Both are computed
# per case and then averaged over cases — the dataset-pooled counterparts are
# dice_fg_pooled/iou_fg_pooled (PooledOverlap), computed in evaluate.py.
STANDARD_METRICS = {
    "dice": DiceScore(average="macro"),
    "iou": IoUScore(average="macro"),
    "dice_fg": DiceScore(average="macro", score_background=False),
    "iou_fg": IoUScore(average="macro", score_background=False),
}
