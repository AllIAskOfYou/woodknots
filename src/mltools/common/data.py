import json
import numpy as np
import os
from pathlib import Path
from typing import Any
from ortools.sat.python import cp_model
from functools import cache


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


@cache
def _solve_optimal_partition(group_sizes: tuple[tuple[int, int], ...], k: int):
    """Use an exact CP-SAT (OR-Tools) formulation to partition item sizes
    into k folds with minimal difference in total item count, each fold
    getting exactly n // k groups. group_sizes is a tuple of (index, size)
    pairs so it's hashable (and thus cacheable) — module-level (not nested)
    so the cache actually persists across calls instead of being recreated,
    empty, every time a caller builds a fresh partition."""
    n = len(group_sizes)
    if n % k != 0:
        raise ValueError(
            f"CP-SAT partitioning requires the number of groups ({n}) to divide "
            f"evenly by k ({k}); got remainder {n % k}."
        )
    groups_per_fold = n // k
    sizes = [size for _, size in group_sizes]

    model = cp_model.CpModel()

    # x[i, f] = 1 if group i is assigned to fold f
    x = {(i, f): model.NewBoolVar(f"x_{i}_{f}") for i in range(n) for f in range(k)}

    # Each group assigned to exactly one fold
    for i in range(n):
        model.Add(sum(x[i, f] for f in range(k)) == 1)

    # Each fold gets exactly `groups_per_fold` groups
    for f in range(k):
        model.Add(sum(x[i, f] for i in range(n)) == groups_per_fold)

    # Fold size = sum of group sizes assigned to it
    fold_sizes = []
    for f in range(k):
        s = model.NewIntVar(0, sum(sizes), f"size_{f}")
        model.Add(s == sum(x[i, f] * sizes[i] for i in range(n)))
        fold_sizes.append(s)

    # Minimize the spread (max - min) across fold sizes
    max_size = model.NewIntVar(0, sum(sizes), "max_size")
    min_size = model.NewIntVar(0, sum(sizes), "min_size")
    model.AddMaxEquality(max_size, fold_sizes)
    model.AddMinEquality(min_size, fold_sizes)
    model.Minimize(max_size - min_size)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible solution found")

    partition = [[] for _ in range(k)]
    for i in range(n):
        for f in range(k):
            if solver.Value(x[i, f]) == 1:
                partition[f].append(i)  # store (shuffled) group indices

    print(
        f"Partitioned group sizes: {[[sizes[i] for i in part] for part in partition]} "
        f"(fold totals: {[solver.Value(s) for s in fold_sizes]})"
    )
    return partition


def partition_groups(
    groups: list[list[str]], k: int, seed: int = 42, method: str = "random"
) -> list[list[int]]:
    """Partition group indices into k folds using the given method. Returns
    a list of k lists of group indices (all folds at once, not just one) —
    compute this once and reuse across all k folds rather than recomputing
    per fold, which is what actually makes 'optimal' only solve once instead
    of k times."""

    def split_random():
        generator = np.random.default_rng(seed)
        group_ids = list(range(len(groups)))
        generator.shuffle(group_ids)
        fold_sizes = np.full(k, len(group_ids) // k)
        fold_sizes[: len(group_ids) % k] += 1
        partition = []
        start = 0
        for size in fold_sizes:
            end = start + size
            partition.append(group_ids[start:end])
            start = end
        return partition

    def split_greedy():
        """Greedily assign each group (largest first) to whichever fold
        currently holds the fewest items, so folds end up balanced by total
        item count rather than by number of groups."""
        generator = np.random.default_rng(seed)
        group_ids = list(range(len(groups)))
        generator.shuffle(group_ids)  # reproducible tie-break order for equal-sized groups
        group_ids.sort(key=lambda i: len(groups[i]), reverse=True)

        fold_group_ids = [[] for _ in range(k)]
        fold_sizes = [0] * k
        for i in group_ids:
            target = min(range(k), key=lambda f: fold_sizes[f])
            fold_group_ids[target].append(i)
            fold_sizes[target] += len(groups[i])
        print(f"Greedy fold group ids: {fold_group_ids}")

        return fold_group_ids

    def split_optimal():
        """Use an exact CP-SAT optimization to partition groups into k folds
        with the provably minimal difference in total item count. Groups are
        shuffled first (reproducibly, via seed) so which specific groups land
        in which fold varies across seeds, even though the balance quality
        does not; the partition is computed over the shuffled order and then
        mapped back to original group indices."""
        generator = np.random.default_rng(seed)
        group_ids = list(range(len(groups)))
        generator.shuffle(group_ids)

        group_sizes = tuple(
            (new_idx, len(groups[orig_idx])) for new_idx, orig_idx in enumerate(group_ids)
        )
        raw_partition = _solve_optimal_partition(group_sizes, k)
        return [[group_ids[new_idx] for new_idx in bin_] for bin_ in raw_partition]

    if method == "random":
        return split_random()
    elif method == "greedy":
        return split_greedy()
    elif method == "optimal":
        return split_optimal()
    else:
        raise ValueError(f"Unknown split method: {method!r}. Must be 'random', 'greedy', or 'optimal'.")


def _fold_from_partition(groups: list[list[str]], partition: list[list[int]], fold: int, k: int):
    val_group_ids = partition[fold]
    train_group_ids = [i for f in range(k) if f != fold for i in partition[f]]

    val_indices = list()
    train_indices = list()
    for i in val_group_ids:
        val_indices.extend(groups[i])
    for i in train_group_ids:
        train_indices.extend(groups[i])
    return train_indices, val_indices


def split_k_fold_by_groups(
    groups: list[list[str]], k: int, fold: int, seed: int = 42, method: str = "random"
):
    partition = partition_groups(groups, k=k, seed=seed, method=method)
    return _fold_from_partition(groups, partition, fold, k)


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


def get_folds(data_dir: str, k: int = 5, method: str = "random", seed: int = 42) -> list[dict[str, list[str]]]:
    data_path = Path(data_dir)
    with open(data_path / "groups.json", "r") as f:
        groups = json.load(f)

    if os.path.exists(data_path / "images"):
        files = os.listdir(data_path / "images")
        valid_ids = set(Path(file).stem for file in files)
    else:
        raise ValueError("data_dir must contain an 'images' subdirectory")
    groups = _filter_groups(groups, valid_ids)
    print([len(group) for group in groups])

    partition = partition_groups(groups, k=k, seed=seed, method=method)
    folds = list()
    for fold in range(k):
        train_ids, val_ids = _fold_from_partition(groups, partition, fold, k)
        folds.append({"train": train_ids, "val": val_ids})
    return folds
