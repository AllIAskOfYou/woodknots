from dataclasses import dataclass
from .dataset import DatasetConfig


@dataclass
class TrainConfig:
    dataset: DatasetConfig
    model_path: str
    imgsz: int
    batch: int
    epochs: int
