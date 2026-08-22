from dataclasses import dataclass


@dataclass
class FoldsConfig:
    file_path: str
    k: int


@dataclass
class DatasetConfig:
    name: str
    hf_repo_id: str
    hf_revision: str
    raw_dir: str
    folds: FoldsConfig
    class_names: list[str]
    resolution: int
