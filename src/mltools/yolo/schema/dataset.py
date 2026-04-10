from dataclasses import dataclass
from typing import Literal


@dataclass
class ImageConfig:
    resolution: int


@dataclass
class LabelConfig:
    primary_log: bool
    wood_knot: list[float]
    wood_knot_categorization: Literal["separate", "combined", "cumulative"]


@dataclass
class FoldConfig:
    k: int
    fold: int
    seed: int


@dataclass
class DatasetConfig:
    image_config: ImageConfig
    label_config: LabelConfig
    fold_config: FoldConfig
