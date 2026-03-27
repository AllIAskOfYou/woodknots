import os
from pathlib import Path
from omegaconf import OmegaConf
import json

from mltools import PROCESSED_DIR
from .coco2custom import COCO2CustomConvertor
from mltools import METADATA_DIR
from mltools.seg.schema import DatasetConfig
from mltools.utils.data import split_k_fold_by_groups


def get_dataset_by_config(cfg: DatasetConfig):
    # 1. Setup Paths (Use Path objects for cleaner code)
    dataset_name = cfg.name
    # Ensure PROCESSED_DIR is absolute if you're using Hydra's runtime CWD
    root_path = Path(PROCESSED_DIR) / dataset_name
    config_path = root_path / "dataset_config.yaml"

    # 2. Check and Load
    if root_path.exists() and config_path.exists():
        existing_cfg = OmegaConf.load(config_path)

        # Compare ONLY the dataset parameters to allow changing training params later
        if existing_cfg != cfg:
            raise ValueError(f"Dataset '{dataset_name}' exists but config differs! ")

        print(f"✅ Loading existing dataset: {dataset_name}")
    else:
        # 3. Generate if missing
        print(f"🚀 Generating dataset: {dataset_name}...")
        root_path.mkdir(parents=True, exist_ok=True)

        convertor = COCO2CustomConvertor(cfg)
        convertor.convert()

        # 4. Save the "Recipe" (The 'dataset' branch only)
        # This is where we save the config into the dataset folder
        OmegaConf.save(config=cfg, f=config_path)

    return root_path


def get_metadata(data_path):
    with open(Path(data_path) / "metadata.json", "r") as f:
        metadata = json.load(f)
    return metadata


def _filter_groups(groups: list[list[str]], valid_ids: set[str]):
    filtered_groups = list()
    for group in groups:
        filtered_group = [id for id in group if id in valid_ids]
        if filtered_group:
            filtered_groups.append(filtered_group)
    return filtered_groups


def get_fold(data_path, k=5, fold=0):
    # load groups from metadata
    with open(Path(METADATA_DIR) / "groups.json", "r") as f:
        groups = json.load(f)

    files = os.listdir(Path(data_path) / "images")
    valid_ids = set(Path(file).stem for file in files)
    groups = _filter_groups(groups, valid_ids)

    train_ids, val_ids = split_k_fold_by_groups(groups, k=k, fold=fold, seed=42)
    print(f"Generated folds, using fold [{fold}].")
    return train_ids, val_ids
