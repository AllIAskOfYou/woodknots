import json
import os
import numpy as np
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from mltools.seg.schema.root import ClassGroupConfig
import albumentations as A
import tifffile as tiff


def load_case(data_path, case_id):
    image_path = os.path.join(data_path, "images", f"{case_id}.png")
    label_path = os.path.join(data_path, "labels", f"{case_id}.tiff")
    # Load the image and label using PIL and convert to numpy arrays
    image = np.array(Image.open(image_path))
    # add channel dimension to image
    label = tiff.imread(label_path).astype(np.float32) / 255.0
    label = np.transpose(label, (1, 2, 0))  # (C, H, W) to (H, W, C)
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
        class_groups: list[ClassGroupConfig] = None,
    ):
        self.data_path = data_path
        self.channel_index = channel_index

        # cases
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
        self.rng = np.random.default_rng()
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
