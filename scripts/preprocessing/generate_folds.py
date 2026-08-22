import json
import os
from mltools import METADATA_DIR, RAW_DIR
from mltools.common.data import split_k_fold_by_groups


def filter_groups(groups: list[list[str]], valid_ids: set[str]):
    filtered_groups = list()
    for group in groups:
        filtered_group = [id for id in group if id in valid_ids]
        if filtered_group:
            filtered_groups.append(filtered_group)
    return filtered_groups


def main():
    with open(os.path.join(METADATA_DIR, "groups.json"), "r") as f:
        groups = json.load(f)

    files = os.listdir(os.path.join(RAW_DIR, "images"))
    valid_ids = set(".".join(file.split(".")[:-1]) for file in files)
    print(len(valid_ids))
    groups = filter_groups(groups, valid_ids)
    print(sum(len(group) for group in groups))

    splits = list()
    for fold in range(5):
        train_ids, val_ids = split_k_fold_by_groups(groups, k=5, fold=fold)
        splits.append({"train": train_ids, "val": val_ids})

    os.makedirs(RAW_DIR, exist_ok=True)
    print(RAW_DIR)
    with open(os.path.join(RAW_DIR, "splits.json"), "w") as f:
        json.dump(splits, f, indent=4)


if __name__ == "__main__":
    main()
