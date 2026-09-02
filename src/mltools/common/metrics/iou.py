import torch
from .metric import Metric


def _iou_score(inputs: torch.Tensor, targets: torch.Tensor):
    reduce_axis = list(range(2, targets.ndim))
    intersection = inputs * targets
    union = inputs + targets - intersection
    intersection = torch.sum(intersection, dim=reduce_axis)
    union = torch.sum(union, dim=reduce_axis)
    return intersection, union


class IoUScore(Metric):
    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return _iou_score(inputs, targets)
