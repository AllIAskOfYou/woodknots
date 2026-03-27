from dataclasses import dataclass
from .preprocessing import PreprocessingConfig
from .dataset import DatasetConfig
from .augmentation import AugmentationConfig
from .module import ModuleConfig


@dataclass
class ClassGroupConfig:
    value: float
    classes: list[float]


@dataclass
class TrainConfig:
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    augmentation: AugmentationConfig
    module: ModuleConfig
    fold: int
    batch_size: int
    total_epochs: int
    class_name: str
    class_groups: list[ClassGroupConfig]
