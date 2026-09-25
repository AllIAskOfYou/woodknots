import os
from pathlib import Path

import albumentations as A
import numpy as np
import tifffile as tiff
import tqdm
from PIL import Image
from torch.utils.data import Dataset

from mltools.common.splits import TRAIN_VAL_TEST, get_split
from mltools.seg.coco2custom import COCO2CustomConvertor

# ensure_formatted bakes every COCO category into a TIFF channel regardless
# of these — folds only affects train.py's fold splitting, class_names is
# never read here at all (channel names come from the data itself via
# COCODataset.get_class_names_ordered()). Excluded from the dataset hash
# (see hash_cfg's exclude_keys) so changing either doesn't spuriously
# invalidate an already-built, byte-identical formatted dataset dir. Used
# identically by setup.py (collect_dataset_variants) and train.py
# (get_dataset_dir) — they must agree or the two would compute different
# hashes for the same dataset.
SETUP_IRRELEVANT_DATASET_KEYS = ["folds", "class_names"]


def get_framework_root(data_root: str) -> str:
    return str(Path(data_root) / "seg_format")


def get_dataset_dir(data_root: str, ds_name: str, ds_hash: str) -> str:
    return str(Path(get_framework_root(data_root)) / f"{ds_name}_{ds_hash}")


def ensure_formatted(local_dir: str, ds_hash: str, ds_name: str, data_root: str) -> str:
    """Fold-independent, marker-file-cached (.complete), same convention as
    nnunet/yolo's ensure_formatted. Wraps COCO2CustomConvertor to write PNG
    images + multi-channel TIFF labels + metadata.json (channel_names) from
    the already-resized local_dir — bakes ALL COCO categories into channels;
    a run's class_name/class_groups just select which channel(s) to train
    on, so they don't affect this dir's identity/hash."""
    dataset_dir = Path(get_dataset_dir(data_root, ds_name, ds_hash))
    marker = dataset_dir / ".complete"
    if marker.exists():
        return str(dataset_dir)

    convertor = COCO2CustomConvertor(
        image_dir=str(Path(local_dir) / "images"),
        annotation_path=str(Path(local_dir) / "annotations.json"),
        output_dir=str(dataset_dir),
    )
    convertor.convert()

    marker.touch()
    return str(dataset_dir)


def write_fold_split(
    dataset_dir: str, folds_path: str, fold: int, mode: str = TRAIN_VAL_TEST
) -> tuple[list[str], list[str], list[str]]:
    """(train_cases, val_cases, test_cases) for split `fold` of the shared
    folds file (same one nnunet/yolo use), see mltools.common.splits.get_split:
    val selects the checkpoint, test is only predicted on for evaluation.
    Under mode="train_test" the selection partition joins training and val is
    the test partition, logged per epoch but never selected on."""
    split = get_split(folds_path, fold, mode=mode)
    return split["train"], split["val"], split["test"]


def load_case(data_path: str, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    image_path = os.path.join(data_path, "images", f"{case_id}.png")
    label_path = os.path.join(data_path, "labels", f"{case_id}.tiff")
    image = np.array(Image.open(image_path))
    label = tiff.imread(label_path).astype(np.float32) / 255.0
    label = np.transpose(label, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    return image, label


class SegDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        channel_index: int,
        cases: list[str] = None,
        preprocessing: A.BasicTransform = None,
        transform: A.BasicTransform = None,
        cache_amount: float = 1.0,
        class_groups: list = None,
        seed: int | None = None,
    ):
        self.data_path = data_path
        self.channel_index = channel_index

        if cases is None:
            self.cases = [
                Path(file).stem
                for file in os.listdir(os.path.join(data_path, "images"))
                if file.endswith(".png")
            ]
        else:
            self.cases = cases
        self.preprocessing = preprocessing
        self.transform = transform
        cache_size = int(len(self.cases) * cache_amount)
        self.cache_size = cache_size
        self.class_groups = class_groups
        # Picks the mosaic partner images; seeded so a run is reproducible.
        self.rng = np.random.default_rng(seed)
        self.data = [
            self.get_case(data_path, case_id)
            for case_id in tqdm.tqdm(self.cases[:cache_size], desc="Caching data...")
        ]

    def get_case(self, data_path, case_id):
        image, label = load_case(data_path, case_id)
        if self.preprocessing is not None:
            preprocessed = self.preprocessing(image=image, mask=label)
            image, label = preprocessed["image"], preprocessed["mask"]
        # index label, keep dims
        label = label[..., self.channel_index : self.channel_index + 1]
        if self.class_groups is not None:
            mapped_label = np.zeros_like(label, dtype=np.float32)
            for group in self.class_groups:
                for class_value in group.classes:
                    mapped_label[label == class_value] = group.value
            label = mapped_label
        return image, label

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        if idx < self.cache_size:
            image, label = self.data[idx]
        else:
            image, label = self.get_case(self.data_path, self.cases[idx])
        if self.transform is not None:
            # select 3 images for mosaic augmentation
            indexes = self.rng.choice(len(self) - 1, size=3, replace=False)
            indexes = [
                i if i < idx else i + 1 for i in indexes
            ]  # avoid selecting the same index
            mosaic_metadata = []
            for i in indexes:
                mosaic_image, mosaic_label = (
                    self.data[i]
                    if i < self.cache_size
                    else self.get_case(self.data_path, self.cases[i])
                )
                mosaic_metadata.append({"image": mosaic_image, "mask": mosaic_label})

            transformed = self.transform(
                image=image, mask=label, mosaic_metadata=mosaic_metadata
            )
            image = transformed["image"]
            label = transformed["mask"]
        return image, label
