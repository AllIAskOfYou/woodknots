from dataclasses import dataclass
from typing import Optional

@dataclass
class MaskImageConfig:
    class_name: str
    amount: float

@dataclass
class DatasetConfig:
    name: str
    class_name: str
    resolution: int
    mask_image: Optional[MaskImageConfig] = None
