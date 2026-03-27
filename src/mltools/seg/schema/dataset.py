from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetConfig:
    name: str
    resolution: int
