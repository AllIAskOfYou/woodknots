from dataclasses import dataclass

from .env import EnvConfig
from .dataset import DatasetConfig
from .experiment import ExperimentConfig


@dataclass
class Config:
    framework: str
    env: EnvConfig
    dataset: DatasetConfig
    experiment: ExperimentConfig
    time: str
