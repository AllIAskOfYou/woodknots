from typing import Optional
from dataclasses import dataclass


@dataclass
class MaskImageConfig:
    channel_index: int
    amount: float


@dataclass
class PreprocessingConfig:
    mask_image: Optional[MaskImageConfig]
