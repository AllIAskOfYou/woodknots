import json
from pathlib import Path


TRAIN_VAL_TEST = "train_val_test"
TRAIN_TEST = "train_test"
SPLIT_MODES = (TRAIN_VAL_TEST, TRAIN_TEST)


def get_split(folds_path: str, fold: int, mode: str = TRAIN_VAL_TEST) -> dict[str, list[str]]:
    """{"train", "val", "test"} case ids for split `fold` of a k-fold partition file.

    The folds file lists k entries whose "val" lists are disjoint partitions of
    the dataset, so partition j is folds[j]["val"]. For split i:
      test  = partition (i + 1) % k  (final evaluation only), in both modes
      val   = partition i            (checkpoint / model selection)
      train = the remaining k - 2 partitions
    Every case is therefore used for selection exactly once and for testing
    exactly once across the k splits, and never for both within one split.

    mode="train_test" drops the selection partition into training instead:
      train = the k - 1 partitions that are not the test one
      val   = the test partition
    The test partition is deliberately the same one in both modes, so the two
    differ only in how much data is trained on and in whether a checkpoint is
    selected — not in what it is scored against. "val" is then the test set,
    returned so a caller can log per-epoch metrics on it for inspection; it
    must not drive checkpoint selection, which is what makes the mode
    meaningful (see mltools.seg.train)."""
    if mode not in SPLIT_MODES:
        raise ValueError(f"Unknown split mode {mode!r}, expected one of {SPLIT_MODES}.")
    folds = json.loads(Path(folds_path).read_text())
    k = len(folds)
    min_folds = 3 if mode == TRAIN_VAL_TEST else 2
    if k < min_folds:
        raise ValueError(f"A {mode} split needs at least {min_folds} folds, {folds_path} has {k}.")
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
    held_out = (fold, test_fold) if mode == TRAIN_VAL_TEST else (test_fold,)
    train = [
        case_id
        for j, partition in enumerate(partitions)
        if j not in held_out
        for case_id in partition
    ]
    val = partitions[fold] if mode == TRAIN_VAL_TEST else partitions[test_fold]
    return {"train": train, "val": val, "test": partitions[test_fold]}
