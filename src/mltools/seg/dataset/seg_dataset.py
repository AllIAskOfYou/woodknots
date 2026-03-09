import json
import os
import numpy as np
import tqdm
from PIL import Image
from torch.utils.data import Dataset


def load_case(data_path, case_id):
    image_path = os.path.join(data_path, "images", f"{case_id}.png")
    label_path = os.path.join(data_path, "labels", f"{case_id}.png")
    # Load the image and label using PIL and convert to numpy arrays
    image = np.array(Image.open(image_path))
    # add channel dimension to image
    label = np.expand_dims(np.array(Image.open(label_path)), axis=-1).astype(np.float32)
    return image, label


def load_dataset(data_path, fold=0, groups_file=None):
    # Get the list of case IDs for the specified fold
    fold_path = os.path.join(data_path, "splits.json")
    if not os.path.exists(fold_path):
        raise FileNotFoundError(f"Data splits file not found at {fold_path}.")
    print(f"Loading data splits from {fold_path} for fold {fold}")
    with open(fold_path, "r") as f:
        splits = json.load(f)
    train_cases = splits[fold]["train"]
    val_cases = splits[fold]["val"]
    return train_cases, val_cases


class SegDataset(Dataset):
    def __init__(self, data_path, cases, transform=None, cache_size=0):
        self.data_path = data_path
        self.cases = cases
        self.transform = transform
        cache_size = cache_size if cache_size > 0 else len(cases)
        self.cache_size = cache_size
        self.data = [
            load_case(self.data_path, case_id)
            for case_id in tqdm.tqdm(cases[:cache_size], desc="Caching data...")
        ]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        if idx < self.cache_size:
            image, label = self.data[idx]
        else:
            image, label = load_case(self.data_path, self.cases[idx])
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]
        return image, label
