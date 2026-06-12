from typing import Optional
from dataclasses import dataclass


@dataclass
class MaskImageConfig:
    channel_index: int
    distance: int


@dataclass
class PreprocessingConfig:
    mask_image: Optional[MaskImageConfig]
