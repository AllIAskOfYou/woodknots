from typing import Any
import numpy as np


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
