import json
import os
from cv2 import INTER_AREA
import albumentations as A
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import tqdm
import tifffile as tiff

from mltools import RAW_DIR, PROCESSED_DIR
from mltools.seg.dataset import COCODataset
from mltools.seg.schema import DatasetConfig

# from mltools.seg.transform import apply_vignette


class COCO2CustomConvertor:
    def __init__(self, cfg: DatasetConfig):
        resize_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=cfg.resolution, interpolation=INTER_AREA),
                A.PadIfNeeded(
                    min_height=cfg.resolution, min_width=cfg.resolution, border_mode=0
                ),
            ]
        )
        self.dataset = COCODataset(
            img_dir=os.path.join(RAW_DIR, "images"),
            ann_file=os.path.join(RAW_DIR, "annotations.json"),
            transforms=resize_transform,
        )
        self.dataset_name = cfg.name
        self.output_dir = os.path.join(PROCESSED_DIR, cfg.name)
        # self.class_name = cfg.class_name
        # self.class_idx = self.dataset.get_channel_by_class_name(self.class_name)
        self.cfg = cfg

    def save_case(self, image, label, case_id):
        if not np.issubdtype(image.dtype, np.floating) or not np.issubdtype(
            label.dtype, np.floating
        ):
            raise ValueError(
                "Image and label must be in float format (0-1 range) before saving."
            )

        img_path = os.path.join(self.output_dir, "images", f"{case_id}.png")
        label_path = os.path.join(self.output_dir, "labels", f"{case_id}.tiff")

        # Convert image and label to uint8 format for saving
        image = (image * 255).astype(np.uint8)
        label = (label * 255).astype(np.uint8)

        Image.fromarray(image).save(img_path)
        # add channel name metadata to label
        metadata = {
            "axes": "YXC",
        }
        tiff.imwrite(label_path, label, photometric="minisblack", metadata=metadata)

    def convert_case(self, idx):
        case_id = self.dataset.image_index_to_name(idx)
        img, label = self.dataset[idx]
        # mask_cfg = self.cfg.mask_image
        # if mask_cfg.amount > 0:
        #     mask_class_idx = self.dataset.get_channel_by_class_name(mask_cfg.class_name)
        #     img = apply_vignette(
        #         img, label[..., mask_class_idx], amount=mask_cfg.amount
        #     )
        # label = label[:, :, self.class_idx]
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

        # save metadata
        metadata = {
            "channel_names": [
                self.dataset.coco.loadCats(cat_id)[0]["name"]
                for cat_id in self.dataset.class_map
            ],
        }
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
