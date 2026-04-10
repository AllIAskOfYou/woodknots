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
