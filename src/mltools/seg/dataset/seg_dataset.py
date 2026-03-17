import json
import os
import numpy as np
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from mltools.seg.schema.root import ClassGroupConfig
import albumentations as A


def load_case(data_path, case_id):
    image_path = os.path.join(data_path, "images", f"{case_id}.png")
    label_path = os.path.join(data_path, "labels", f"{case_id}.png")
    # Load the image and label using PIL and convert to numpy arrays
    image = np.array(Image.open(image_path))
    # add channel dimension to image
    label = (
        np.expand_dims(np.array(Image.open(label_path)), axis=-1).astype(np.float32)
        / 255.0
    )
    return image, label


class SegDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        cases: list[str] = None,
        transform: A.BasicTransform = None,
        cache_amount: float = 1.0,
        class_groups: list[ClassGroupConfig] = None,
    ):
        self.data_path = data_path
        if cases is None:
            self.cases = [
                Path(file).stem
                for file in os.listdir(os.path.join(data_path, "images"))
                if file.endswith(".png")
            ]
        else:
            self.cases = cases
        self.transform = transform
        cache_size = int(len(self.cases) * cache_amount)
        self.cache_size = cache_size
        self.data = [
            load_case(self.data_path, case_id)
            for case_id in tqdm.tqdm(self.cases[:cache_size], desc="Caching data...")
        ]
        self.class_groups = class_groups

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        if idx < self.cache_size:
            image, label = self.data[idx]
        else:
            image, label = load_case(self.data_path, self.cases[idx])
        if self.class_groups is not None:
            for group in self.class_groups:
                for class_value in group.classes:
                    label[label == class_value] = group.value
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]
        return image, label
