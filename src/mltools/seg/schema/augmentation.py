from dataclasses import dataclass


@dataclass
class ElasticConfig:
    alpha: float
    sigma: float


@dataclass
class GridConfig:
    num_steps: int
    distort_limit: float


@dataclass
class DistortionConfig:
    elastic: ElasticConfig
    grid: GridConfig
    p: float
    e2g_ratio: float


@dataclass
class RandomCropConfig:
    scale: list[float]
    rotation: list[int]


@dataclass
class FlipConfig:
    p: float


@dataclass
class ColorJitterConfig:
    brightness: float
    contrast: float
    saturation: float
    hue: float
    p: float


@dataclass
class AugmentationConfig:
    resolution: int
    w2h_ratio: float
    distortion: DistortionConfig
    random_crop: RandomCropConfig
    horizontal_flip: FlipConfig
    vertical_flip: FlipConfig
    color_jitter: ColorJitterConfig
