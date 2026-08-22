import json
import numpy as np
import os
from pathlib import Path
from typing import Any


def split_k_fold(data: Any, k: int, fold: int, seed: int = 42):
    generator = np.random.default_rng(seed)
    ids = list(range(len(data)))
    generator.shuffle(ids)
    fold_sizes = np.full(k, len(ids) // k)
    fold_sizes[: len(ids) % k] += 1
    start = np.sum(fold_sizes[:fold])
    end = start + fold_sizes[fold]
    val_indices = ids[start:end]
    train_indices = ids[:start] + ids[end:]
    return train_indices, val_indices


def split_k_fold_by_groups(groups: list[list[str]], k: int, fold: int, seed: int = 42):
    generator = np.random.default_rng(seed)
    group_ids = list(range(len(groups)))
    generator.shuffle(group_ids)
    fold_sizes = np.full(k, len(group_ids) // k)
    fold_sizes[: len(group_ids) % k] += 1
    start = np.sum(fold_sizes[:fold])
    end = start + fold_sizes[fold]
    val_ids = group_ids[start:end]
    train_ids = group_ids[:start] + group_ids[end:]
    val_indices = list()
    train_indices = list()
    for i in val_ids:
        val_indices.extend(groups[i])
    for i in train_ids:
        train_indices.extend(groups[i])
    return train_indices, val_indices


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


def get_fold(
    data_dir: str,
    k: int = 5,
    fold: int = 0,
    valid_ids: set[str] = None,
    seed: int = 42,
):
    data_path = Path(data_dir)
    with open(data_path / "groups.json", "r") as f:
        groups = json.load(f)

    if valid_ids is None and os.path.exists(data_path / "images"):
        files = os.listdir(data_path / "images")
        valid_ids = set(Path(file).stem for file in files)
    elif valid_ids is None:
        raise ValueError("Either data_dir or valid_ids must be provided")
    groups = _filter_groups(groups, valid_ids)

    train_ids, val_ids = split_k_fold_by_groups(groups, k=k, fold=fold, seed=seed)
    return train_ids, val_ids


def get_folds(data_dir: str, k: int = 5, seed: int = 42) -> list[dict[str, list[str]]]:
    data_path = Path(data_dir)
    with open(data_path / "groups.json", "r") as f:
        groups = json.load(f)

    if os.path.exists(data_path / "images"):
        files = os.listdir(data_path / "images")
        valid_ids = set(Path(file).stem for file in files)
    else:
        raise ValueError("Either data_dir or valid_ids must be provided")
    groups = _filter_groups(groups, valid_ids)

    folds = list()
    for fold in range(k):
        train_ids, val_ids = split_k_fold_by_groups(groups, k=k, fold=fold, seed=seed)
        folds.append({"train": train_ids, "val": val_ids})
    return folds
