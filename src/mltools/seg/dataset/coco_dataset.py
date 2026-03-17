import cv2
import numpy as np
import os
from albumentations import BasicTransform
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from typing import Optional


class COCODataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        transforms: Optional[BasicTransform] = None,
    ):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())

        # Consistent class mapping
        classes = self.coco.loadCats(self.coco.getCatIds())
        self.class_map = {cls["id"]: idx for idx, cls in enumerate(classes)}
        self.transforms = transforms

    def image_index_to_name(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        return "".join(img_info["file_name"].split(".")[:-1])

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # 1. Load Image
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, img_info["file_name"])

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Initialize Multi-channel Mask (H, W, C)
        num_classes = len(self.class_map)
        full_mask = np.zeros(
            (img.shape[0], img.shape[1], num_classes), dtype=np.float32
        )

        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id in self.class_map:
                channel = self.class_map[cat_id]
                m = self.coco.annToMask(ann)

                # Use attributes for confidence weighting (Soft Masks)
                confidence = (
                    ann.get("attributes", {}).get("confidance", 1.0)
                )
                m = m.astype(np.float32) * float(confidence)
                full_mask[:, :, channel] = np.maximum(full_mask[:, :, channel], m)

        # 3. Apply Albumentations
        if self.transforms:
            augmented = self.transforms(image=img, mask=full_mask)
            img = augmented["image"]
            full_mask = augmented["mask"]

        img = img.astype(np.float32) / 255.0
        return img, full_mask

    def __len__(self) -> int:
        return len(self.ids)

    def get_channel_by_class_name(self, class_name: str):
        for cat_id, idx in self.class_map.items():
            cat_info = self.coco.loadCats(cat_id)[0]
            if cat_info["name"] == class_name:
                return idx
        raise ValueError(f"Class name '{class_name}' not found in COCO categories.")
