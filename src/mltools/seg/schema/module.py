from dataclasses import dataclass
from typing import Any


@dataclass
class LossConfig:
    weight: float
    labels_smoothing: float

@dataclass
class ModuleConfig:
    resolution: int
    total_epochs: int
    freeze_encoder: bool
    model: Any
    optimizer: Any
    lr_update_interval: str
    scheduler: Any
    loss: LossConfig
