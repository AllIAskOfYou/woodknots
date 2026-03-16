from dataclasses import dataclass


@dataclass
class DatasetConfig:
    name: str
    class_name: str
    resolution: int
