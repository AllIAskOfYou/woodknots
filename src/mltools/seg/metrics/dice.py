import torch
from .metric import Metric


def _dice_score(inputs: torch.Tensor, targets: torch.Tensor):
    reduce_axis = list(range(2, targets.ndim))
    intersection = torch.sum(inputs * targets, dim=reduce_axis)
    inputs_sum = torch.sum(inputs, dim=reduce_axis)
    targets_sum = torch.sum(targets, dim=reduce_axis)
    return (
        intersection * 2,
        inputs_sum + targets_sum,
    )


class DiceScore(Metric):
    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return _dice_score(inputs, targets)
