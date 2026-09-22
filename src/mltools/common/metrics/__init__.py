from .metric import Metric
from .dice import DiceScore
from .iou import IoUScore
from .gms import GMSScore, instance_gms

__all__ = ["Metric", "DiceScore", "IoUScore", "GMSScore", "instance_gms"]

# Pixel-level metrics of the cross-framework evaluation, computed on label maps.
# The instance-level GMS is computed on instances (instance_gms), not from here;
# GMSScore remains for seg's training-time validation metric.
STANDARD_METRICS = {
    "dice": DiceScore(soft=False, average="macro"),
    "iou": IoUScore(soft=False, average="macro"),
}
