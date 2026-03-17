import os
from cv2 import INTER_AREA
import albumentations as A
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import tqdm

from mltools import RAW_DIR, PROCESSED_DIR
from mltools.seg.dataset import COCODataset
from mltools.seg.schema import DatasetConfig

class COCO2CustomConvertor:
    def __init__(self, cfg: DatasetConfig):
        resize_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=cfg.resolution, interpolation=INTER_AREA),
                A.PadIfNeeded(min_height=cfg.resolution, min_width=cfg.resolution, border_mode=0),
            ]
        )
        self.dataset = COCODataset(
            img_dir=os.path.join(RAW_DIR, "images"),
            ann_file=os.path.join(RAW_DIR, "annotations.json"),
            transforms=resize_transform
        )
        self.dataset_name = cfg.name
        self.output_dir = os.path.join(PROCESSED_DIR, cfg.name)
        self.class_name = cfg.class_name
        self.class_idx = self.dataset.get_channel_by_class_name(self.class_name)

    def save_case(self, image, label, case_id):
        if not np.issubdtype(image.dtype, np.floating) or not np.issubdtype(
            label.dtype, np.floating
        ):
            raise ValueError(
                "Image and label must be in float format (0-1 range) before saving."
            )

        img_path = os.path.join(self.output_dir, "images", f"{case_id}.png")
        mask_path = os.path.join(self.output_dir, "labels", f"{case_id}.png")

        # Convert image and label to uint8 format for saving
        image = (image * 255).astype(np.uint8)
        label = (label * 255).astype(np.uint8)

        Image.fromarray(image).save(img_path)
        Image.fromarray(label).save(mask_path)

    def convert_case(self, idx):
        img, label = self.dataset[idx]
        case_id = self.dataset.image_index_to_name(idx)
        label = label[:, :, self.class_idx]
        self.save_case(img, label, case_id)

    def convert(self):
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels"), exist_ok=True)

        tasks = []
        for idx in tqdm.tqdm(range(len(self.dataset)), desc=f"Preparing tasks..."):
            tasks.append(idx)

        with ThreadPoolExecutor() as executor:
            list(
                tqdm.tqdm(
                    executor.map(self.convert_case, tasks),
                    total=len(tasks),
                    desc=f"Processing dataset...",
                )
            )