from typing import Optional
from dataclasses import dataclass


@dataclass
class MaskImageConfig:
    channel_index: int
    steepness: float
    midpoint: float


@dataclass
class PreprocessingConfig:
    mask_image: Optional[MaskImageConfig]
