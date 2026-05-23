from .metric import Metric
from .dice import DiceScore
from .iou import IoUScore
from .gms import GMSScore

__all__ = ["Metric", "DiceScore", "IoUScore", "GMSScore"]

STANDARD_METRICS = {
    "dice": DiceScore(soft=False, background=True, average="macro"),
    "dice_nobg": DiceScore(soft=False, background=False, average="macro"),
    "dice_soft": DiceScore(soft=True, background=True, average="macro"),
    "iou": IoUScore(soft=False, background=True, average="macro"),
    "gms": GMSScore(average="macro"),
}
