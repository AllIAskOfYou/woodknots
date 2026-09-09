import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tifffile as tiff
from PIL import Image
from tqdm import tqdm

from mltools.common.coco_dataset import COCODataset
from mltools.common.logger import make_bar_format


class COCO2CustomConvertor:
    """Converts an already-resized COCODataset into the custom PNG image +
    multi-page grayscale TIFF label format seg trains on. Resizing itself
    happens upstream (mltools.common.setup.ensure_resized) before this
    convertor ever runs — every COCO category becomes one TIFF page/channel,
    regardless of which one a given experiment later trains on."""

    def __init__(self, image_dir: str, annotation_path: str, output_dir: str):
        self.dataset = COCODataset(image_dir=image_dir, annotation_path=annotation_path)
        self.output_dir = output_dir

    def save_case(self, image: np.ndarray, label: np.ndarray, case_id: str) -> None:
        # image: uint8 (H, W, 3) from COCODataset.load_image.
        # label: float32 (H, W, C) in [0, 1] from COCODataset.build_mask.
        img_path = os.path.join(self.output_dir, "images", f"{case_id}.png")
        label_path = os.path.join(self.output_dir, "labels", f"{case_id}.tiff")

        Image.fromarray(image).save(img_path)

        label_uint8 = (label * 255).astype(np.uint8)
        stack_data = np.moveaxis(label_uint8, -1, 0)  # (H, W, C) -> (C, H, W)
        tiff.imwrite(
            label_path,
            stack_data,
            photometric="minisblack",  # each page is grayscale
            metadata={"axes": "CYX"},
            imagej=True,  # adds scrollbar metadata for viewers
        )

    def convert_case(self, idx: int) -> None:
        case_id = self.dataset.image_index_to_name(idx)
        img, label = self.dataset[idx]
        self.save_case(img, label, case_id)

    def convert(self) -> None:
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels"), exist_ok=True)

        tasks = list(range(len(self.dataset)))
        with ThreadPoolExecutor() as executor:
            list(
                tqdm(
                    executor.map(self.convert_case, tasks),
                    total=len(tasks),
                    bar_format=make_bar_format(desc="Converting dataset"),
                )
            )

        metadata = {"channel_names": self.dataset.get_class_names_ordered()}
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
