from dataclasses import dataclass
from typing import Any


@dataclass
class ModuleConfig:
    resolution: int
    total_epochs: int
    freeze_encoder: bool
    model: Any
    optimizer: Any
    scheduler: Any
