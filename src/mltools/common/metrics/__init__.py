from .metric import Metric
from .dice import DiceScore
from .iou import IoUScore
from .gms import GMSScore

__all__ = ["Metric", "DiceScore", "IoUScore", "GMSScore"]

STANDARD_METRICS = {
    "dice": DiceScore(soft=False, average="macro"),
    "iou": IoUScore(soft=False, average="macro"),
    "gms": GMSScore(average="macro"),
}
