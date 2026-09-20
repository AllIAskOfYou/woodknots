from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import json
from pathlib import Path
from tqdm import tqdm

from mltools.common.coco_dataset import COCODataset
from mltools.common.logger import make_bar_format
from mltools.common.masks import mask_to_label_map
from mltools.common.splits import get_split
from mltools.common.utils import symlink_relative


def get_framework_root(data_root: str) -> str:
    return str(Path(data_root) / "nnunet_format")


def get_dataset_dir(data_root: str, ds_id: int, ds_name: str) -> str:
    dataset_dir = (
        Path(data_root) / "nnunet_format" / "raw" / f"Dataset{ds_id:03d}_{ds_name}"
    )
    return str(dataset_dir)


def build_dataset_json(
    class_names: list[str],
    num_training: int,
    file_ending: str = ".png",
) -> dict:
    labels = {"background": 0}
    for idx, name in enumerate(class_names):
        labels[name] = idx + 1

    return {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": labels,
        "numTraining": num_training,
        "file_ending": file_ending,
    }


def _process_case(
    coco_ds: COCODataset,
    index: int,
    local_dir: str,
    class_indices: list[int],
    images_tr_dir: Path,
    labels_tr_dir: Path,
) -> None:
    case_id = coco_ds.image_index_to_name(index)

    img = coco_ds.load_image(index)
    mask = coco_ds.build_mask(index, img.shape[:2], binary=True)
    mask_filtered = mask[:, :, class_indices]
    label_map = mask_to_label_map(mask_filtered)

    img_out_path = images_tr_dir / f"{case_id}_0000.png"
    if not img_out_path.exists():
        symlink_relative(img_out_path, Path(local_dir) / "images" / f"{case_id}.png")

    label_out_path = labels_tr_dir / f"{case_id}.png"
    if not label_out_path.exists():
        cv2.imwrite(str(label_out_path), label_map)


def ensure_formatted(
    local_dir: str,
    ds_id: int,
    ds_name: str,
    class_names: list[str],
    folds_path: str,
    data_root: str,
    max_workers: int = 8,
) -> str:
    dataset_dir = Path(get_dataset_dir(data_root, ds_id, ds_name))
    marker = dataset_dir / ".complete"
    if marker.exists():
        return str(dataset_dir)

    coco_ds = COCODataset(
        image_dir=str(Path(local_dir) / "images"),
        annotation_path=str(Path(local_dir) / "annotations.json"),
    )

    images_tr_dir = dataset_dir / "imagesTr"
    labels_tr_dir = dataset_dir / "labelsTr"
    images_tr_dir.mkdir(parents=True, exist_ok=True)
    labels_tr_dir.mkdir(parents=True, exist_ok=True)
    class_indices = [coco_ds.get_channel_by_class_name(name) for name in class_names]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_case,
                coco_ds,
                index,
                local_dir,
                class_indices,
                images_tr_dir,
                labels_tr_dir,
            )
            for index in range(len(coco_ds))
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            bar_format=make_bar_format(desc="Formatting cases"),
        ):
            future.result()  # re-raises any exception from worker threads

    dataset_json = build_dataset_json(
        class_names=class_names,
        num_training=len(coco_ds),
        file_ending=".png",
    )
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    # nnU-Net only knows train/val, so each split's test partition is simply
    # left out of splits_final.json: nnU-Net trains on train and selects
    # checkpoint_best on val, and predict() runs on test afterwards. The
    # planner copies this file into preprocessed/, which is where training
    # reads it from.
    num_folds = len(json.loads(Path(folds_path).read_text()))
    splits = []
    for fold in range(num_folds):
        split = get_split(folds_path, fold)
        splits.append({"train": split["train"], "val": split["val"]})
    with open(dataset_dir / "splits_final.json", "w") as f:
        json.dump(splits, f, indent=2)

    marker.touch()

    return str(dataset_dir)
