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
    """Channel convention: inputs/targets are (B, C, H, W).

    By default (background=False) the caller supplies every channel that
    should be scored — including an explicit background channel at index 0
    if it wants one, which is what mltools.common.evaluate._to_pseudo_logits
    produces for the cross-framework evaluation path.

    background=True is for callers whose tensors hold foreground channels
    only (e.g. a single-class model emitting (B, 1, H, W)): one background
    channel is synthesised as the complement of the union of all foreground
    channels and prepended, so the class average covers background too.
    """

    # Whether the background channel (index 0) participates in scoring.
    # Instance-level metrics override this — matching connected components
    # against "everything that isn't an object" is meaningless.
    scores_background_channel = True

    def __init__(
        self,
        soft: bool = False,
        average: Literal["micro", "macro", "weighted"] = "macro",
        background: bool = False,
    ):
        self.soft = soft
        self.average = average
        self.background = background

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
        if self.background:
            # Complement of the union of all foreground channels, not of any
            # single one — with C > 1 a per-channel complement would produce
            # C extra channels that each count other classes as background.
            preds = torch.cat([1 - preds.amax(dim=1, keepdim=True), preds], dim=1)
            targets = torch.cat([1 - targets.amax(dim=1, keepdim=True), targets], dim=1)
        if not self.scores_background_channel:
            preds = preds[:, 1:]
            targets = targets[:, 1:]
        targets_sum = torch.sum(targets, dim=list(range(2, targets.ndim)))
        intersection, union = self.__call__(preds, targets)
        dice = _average_classes(intersection, union, targets_sum, average=self.average)
        return dice.nanmean().item()
