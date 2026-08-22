from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from mltools.common.coco_dataset import COCODataset
from mltools.common.logger import make_bar_format


def get_dataset_dir(local_dir: str, ds_hash: str) -> str:
    dataset_dir = Path(local_dir) / "nnunet_format" / f"{ds_hash}"
    return str(dataset_dir)


def mask_to_label_map(
    mask: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Collapse a (H, W, C) soft multi-channel mask into a single-channel
    integer label map, as nnU-Net expects. Background = 0, classes = 1..C.

    Overlapping classes are resolved by picking the channel with the
    highest confidence at each pixel.
    """
    binary = mask >= threshold  # (H, W, C)
    has_any = binary.any(axis=-1)

    # channel index (0-based) with highest confidence, +1 to reserve 0 for background
    label_map = np.zeros(mask.shape[:2], dtype=np.uint8)
    best_channel = np.argmax(mask, axis=-1) + 1
    label_map[has_any] = best_channel[has_any]

    return label_map


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
    mask = coco_ds.build_mask(index, img.shape[:2])
    mask_filtered = mask[:, :, class_indices]
    label_map = mask_to_label_map(mask_filtered)

    img_out_path = images_tr_dir / f"{case_id}_0000.png"
    if not img_out_path.exists():
        img_out_path.symlink_to(Path(local_dir) / "images" / f"{case_id}.png")

    label_out_path = labels_tr_dir / f"{case_id}.png"
    if not label_out_path.exists():
        cv2.imwrite(str(label_out_path), label_map)


def ensure_formatted(
    local_dir: str,
    ds_hash: str,
    class_names: list[str],
    folds_path: str,
    data_dir: str,
    max_workers: int = 8,
) -> str:
    dataset_dir = Path(get_dataset_dir(data_dir, ds_hash))
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

    # create symlink for folds_path to splits_final.json
    splits_final_path = dataset_dir / "splits_final.json"
    splits_final_path.symlink_to(Path(folds_path).resolve())

    marker.touch()

    return str(dataset_dir)
