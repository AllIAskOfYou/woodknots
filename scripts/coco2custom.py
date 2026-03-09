import os
from cv2 import INTER_AREA
import albumentations as A
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import tqdm

from mltools import RAW_DIR, PROCESSED_DIR
from mltools.seg.dataset import COCODataset


def save_case(image, label, output_dir, case_id):
    if not np.issubdtype(image.dtype, np.floating) or not np.issubdtype(
        label.dtype, np.floating
    ):
        raise ValueError(
            "Image and label must be in float format (0-1 range) before saving."
        )

    img_path = os.path.join(output_dir, "images", f"{case_id}.png")
    mask_path = os.path.join(output_dir, "labels", f"{case_id}.png")

    # Convert image and label to uint8 format for saving
    image = (image * 255).astype(np.uint8)
    label = (label * 255).astype(np.uint8)

    Image.fromarray(image).save(img_path)
    Image.fromarray(label).save(mask_path)


def convert_case(args):
    dataset, class_idx, output_dir, idx = args
    img, label = dataset[idx]
    case_id = dataset.image_index_to_name(idx)
    label = label[:, :, class_idx]
    save_case(img, label, output_dir, case_id)


def mask_image(image, mask):
    pass


def coco2custom(dataset_name: str, class_name: str, res: int):
    output_dir = os.path.join(PROCESSED_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    resize_transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=res, interpolation=INTER_AREA),
            A.PadIfNeeded(min_height=res, min_width=res, border_mode=0),
        ]
    )

    dataset = COCODataset(
        img_dir=os.path.join(RAW_DIR, "images"),
        ann_file=os.path.join(RAW_DIR, "annotations.json"),
        transforms=resize_transform,
        use_confidence=True,
    )

    class_idx = dataset.get_channel_by_class_name(class_name)

    tasks = []
    for idx in tqdm.tqdm(range(len(dataset)), desc=f"Preparing tasks..."):
        tasks.append((dataset, class_idx, output_dir, idx))

    with ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(convert_case, tasks),
                total=len(tasks),
                desc=f"Processing dataset...",
            )
        )
