from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    name: str
    fold: int
