from abc import ABC, abstractmethod
import torch
from typing import Literal


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, zero_division: float = 0.0) -> torch.Tensor:
    """Elementwise numerator/denominator, substituting zero_division wherever
    denominator is 0 (avoids NaN/inf from division by zero). Inlined instead
    of importing torchmetrics.utilities.compute._safe_divide for this one
    helper — that import drags in torchmetrics' entire functional namespace,
    which pulls in torchvision, which pulls in torch._dynamo, which is very
    slow to import and otherwise unused here."""
    fallback = torch.full_like(numerator, zero_division, dtype=torch.float)
    return torch.where(denominator != 0, numerator / denominator, fallback)


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
        average: Literal["micro", "macro", "weighted"] = "macro",
    ):
        self.soft = soft
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
            targets = targets > 0.5
            preds = preds.long()
            targets = targets.long()
        targets_sum = torch.sum(targets, dim=list(range(2, targets.ndim)))
        intersection, union = self.__call__(preds, targets)
        dice = _average_classes(intersection, union, targets_sum, average=self.average)
        return dice.nanmean().item()
