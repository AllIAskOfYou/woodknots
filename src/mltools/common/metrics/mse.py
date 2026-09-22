import torch
from .metric import Metric


def _mse_score(inputs: torch.Tensor, targets: torch.Tensor):
    pass


class MSEScore(Metric):
    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return _mse_score(inputs, targets)
