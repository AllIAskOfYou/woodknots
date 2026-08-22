from dataclasses import dataclass


@dataclass
class EnvConfig:
    project_root: str
    data_root: str
    results_root: str
