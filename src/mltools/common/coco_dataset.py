from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import cv2
import io
import json
import numpy as np
import os
from pathlib import Path
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from tqdm import tqdm
from mltools.common.logger import make_bar_format


class COCODataset:
    def __init__(self, image_dir: str, annotation_path: str):
        # disable COCO's internal logging to avoid cluttering output
        with contextlib.redirect_stdout(io.StringIO()):
            self.coco = COCO(annotation_path)
        self.img_dir = image_dir
        self.ids = list(self.coco.imgs.keys())

        # Consistent class mapping
        classes = self.coco.loadCats(self.coco.getCatIds())
        self.class_map = {cls["id"]: idx for idx, cls in enumerate(classes)}

    def image_index_to_name(self, index: int) -> str:
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        return "".join(img_info["file_name"].split(".")[:-1])

    def load_image(self, index: int) -> np.ndarray:
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, img_info["file_name"])

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img  # uint8, unnormalized

    def build_mask(self, index: int, shape: tuple[int, int]) -> np.ndarray:
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        num_classes = len(self.class_map)
        full_mask = np.zeros((shape[0], shape[1], num_classes), dtype=np.float32)

        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in self.class_map:
                continue
            channel = self.class_map[cat_id]
            m = self.coco.annToMask(ann)

            confidence = ann.get("attributes", {}).get("confidence", 1.0)
            m = m.astype(np.float32) * float(confidence)
            full_mask[:, :, channel] = np.maximum(full_mask[:, :, channel], m)

        return full_mask

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        img = self.load_image(index)
        mask = self.build_mask(index, img.shape[:2])
        return img, mask

    def __len__(self) -> int:
        return len(self.ids)

    def get_channel_by_class_name(self, class_name: str) -> int:
        for cat_id, idx in self.class_map.items():
            cat_info = self.coco.loadCats(cat_id)[0]
            if cat_info["name"] == class_name:
                return idx
        raise ValueError(f"Class name '{class_name}' not found in COCO categories.")

    def get_class_names_ordered(self) -> list[str]:
        """Return class names ordered by their channel index (0-based)."""
        idx_to_name = {}
        for cat_id, idx in self.class_map.items():
            cat_info = self.coco.loadCats(cat_id)[0]
            idx_to_name[idx] = cat_info["name"]
        return [idx_to_name[i] for i in sorted(idx_to_name)]


def _process_image(
    dataset: COCODataset,
    index: int,
    img_id: int,
    target_w: int,
    target_h: int,
    images_out_dir: str,
    file_ending: str,
) -> tuple[dict, list[dict]]:
    img_info = dataset.coco.loadImgs(img_id)[0]
    orig_h, orig_w = img_info["height"], img_info["width"]
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    img = dataset.load_image(index)
    resized_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    img_name = Path(img_info["file_name"]).stem
    out_path = os.path.join(images_out_dir, f"{img_name}{file_ending}")
    cv2.imwrite(out_path, cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))

    new_img_info = dict(img_info)
    new_img_info["width"] = target_w
    new_img_info["height"] = target_h
    new_img_info["file_name"] = f"{img_name}{file_ending}"

    new_anns = []
    ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
    for ann in dataset.coco.loadAnns(ann_ids):
        new_ann = dict(ann)
        new_ann["segmentation"] = _resize_segmentation(
            ann["segmentation"], orig_h, orig_w, scale_x, scale_y
        )
        new_ann["bbox"] = _resize_bbox(ann["bbox"], scale_x, scale_y)
        new_ann["area"] = ann["area"] * scale_x * scale_y
        new_anns.append(new_ann)

    return new_img_info, new_anns


def resize_dataset(
    dataset: COCODataset,
    target_size: tuple[int, int],
    output_dir: str,
    ann_filename: str = "annotations.json",
    file_ending: str = ".png",
    max_workers: int = 8,
) -> None:
    """Resize all images + annotations in a COCODataset and write them to output_dir.

    target_size: (width, height)
    """
    images_out_dir = os.path.join(output_dir, "images")
    os.makedirs(images_out_dir, exist_ok=True)

    target_w, target_h = target_size
    coco_data = dataset.coco.dataset

    new_images = []
    new_annotations = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_image,
                dataset,
                index,
                img_id,
                target_w,
                target_h,
                images_out_dir,
                file_ending,
            ): index
            for index, img_id in enumerate(dataset.ids)
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            bar_format=make_bar_format(desc="Resizing images"),
        ):
            new_img_info, new_anns = future.result()
            new_images.append(new_img_info)
            new_annotations.extend(new_anns)

    new_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco_data["categories"],
    }

    with open(os.path.join(output_dir, ann_filename), "w") as f:
        json.dump(new_coco, f)


def _resize_bbox(bbox: list[float], scale_x: float, scale_y: float) -> list[float]:
    x, y, w, h = bbox
    return [x * scale_x, y * scale_y, w * scale_x, h * scale_y]


def _resize_segmentation(
    segmentation, orig_h: int, orig_w: int, scale_x: float, scale_y: float
):
    if isinstance(segmentation, list):
        # polygon format: list of [x1, y1, x2, y2, ...]
        resized = []
        for poly in segmentation:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            resized.append(pts.flatten().tolist())
        return resized

    # RLE format (compressed or uncompressed): resize via mask, re-encode
    rle = segmentation
    if isinstance(rle["counts"], list):
        rle = mask_utils.frPyObjects(rle, orig_h, orig_w)
    m = mask_utils.decode(rle)
    m_resized = cv2.resize(
        m,
        (int(orig_w * scale_x), int(orig_h * scale_y)),
        interpolation=cv2.INTER_NEAREST,
    )
    new_rle = mask_utils.encode(np.asfortranarray(m_resized))
    new_rle["counts"] = new_rle["counts"].decode("utf-8")
    return new_rle
