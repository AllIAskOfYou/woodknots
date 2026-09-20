import json
from pathlib import Path


def get_split(folds_path: str, fold: int) -> dict[str, list[str]]:
    """{"train", "val", "test"} case ids for split `fold` of a k-fold partition file.

    The folds file lists k entries whose "val" lists are disjoint partitions of
    the dataset, so partition j is folds[j]["val"]. For split i:
      val   = partition i            (checkpoint / model selection)
      test  = partition (i + 1) % k  (final evaluation only)
      train = the remaining k - 2 partitions
    Every case is therefore used for selection exactly once and for testing
    exactly once across the k splits, and never for both within one split."""
    folds = json.loads(Path(folds_path).read_text())
    k = len(folds)
    if k < 3:
        raise ValueError(f"A train/val/test split needs at least 3 folds, {folds_path} has {k}.")
    if not 0 <= fold < k:
        raise ValueError(f"Fold {fold} is out of range for {k} folds in {folds_path}.")

    partitions = [list(entry["val"]) for entry in folds]
    seen: set[str] = set()
    for j, partition in enumerate(partitions):
        overlap = seen.intersection(partition)
        if overlap:
            raise ValueError(
                f"Partition {j} in {folds_path} overlaps earlier partitions: {sorted(overlap)[:5]}"
            )
        seen.update(partition)

    test_fold = (fold + 1) % k
    train = [
        case_id
        for j, partition in enumerate(partitions)
        if j not in (fold, test_fold)
        for case_id in partition
    ]
    return {"train": train, "val": partitions[fold], "test": partitions[test_fold]}
