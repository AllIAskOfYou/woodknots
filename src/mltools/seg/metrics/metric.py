from abc import ABC, abstractmethod
import torch
from typing import Callable, Literal
from torchmetrics.utilities.compute import _safe_divide
import cv2
import numpy as np


def _average_classes(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    support: torch.Tensor,
    average: Literal["micro", "macro", "weighted"] = "macro",
):
    if average == "micro":
        numerator = torch.sum(numerator, dim=-1)
        denominator = torch.sum(denominator, dim=-1)
        return _safe_divide(numerator, denominator, zero_division=1.0)
    score = _safe_divide(numerator, denominator, zero_division=1.0)
    if average == "macro":
        return score.mean(dim=-1)
    if average == "weighted":
        weights = _safe_divide(
            support, torch.sum(support, dim=-1, keepdim=True), zero_division=0.0
        )
        return torch.sum(score * weights, dim=-1)


class Metric(ABC):

    def __init__(
        self,
        soft: bool = False,
        background: bool = False,
        average: Literal["micro", "macro", "weighted"] = "macro",
    ):
        self.soft = soft
        self.background = background
        self.average = average

    @abstractmethod
    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pass

    def compute(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        inputs = inputs.detach()
        targets = targets.detach()
        if self.soft:
            preds = torch.sigmoid(inputs)
        else:
            preds = inputs > 0.0
            # targets = targets > 0.5
            preds = preds.long()
            targets = targets.long()
        if self.background:
            preds = torch.cat([1 - preds, preds], dim=1)
            targets = torch.cat([1 - targets, targets], dim=1)
        targets_sum = torch.sum(targets, dim=list(range(2, targets.ndim)))
        intersection, union = self.__call__(preds, targets)
        dice = _average_classes(intersection, union, targets_sum, average=self.average)
        return dice.nanmean().item()
