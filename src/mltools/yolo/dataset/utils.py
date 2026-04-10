from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import sys
import cv2
from functools import partial
import hashlib
import json
from omegaconf import OmegaConf
import os
from pathlib import Path
from pycocotools.coco import COCO
import tqdm
import yaml

from mltools import RAW_DIR, PROCESSED_YOLO_DIR
from mltools.utils.data import get_fold
from mltools.yolo.schema import LabelConfig, ImageConfig, FoldConfig, DatasetConfig


CONF_MAP = {0.2: "low", 0.6: "medium", 1.0: "high"}


def load_coco_silent(ann_path):
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            return COCO(ann_path)


def get_config_id(cfg: OmegaConf) -> str:
    config_str = json.dumps(
        OmegaConf.to_container(cfg), sort_keys=True, separators=(",", ":")
    )
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def get_class_names(cfg: LabelConfig):
    names = list()
    if cfg.wood_knot_categorization == "combined":
        names.append("wood_knot")
    else:
        for val in cfg.wood_knot:
            if val not in CONF_MAP:
                raise ValueError(
                    f"Wood knot confidence value {val} is not valid.\n\
                        Must be one of {list(CONF_MAP.keys())}."
                )
            cat_name = CONF_MAP[val]
            names.append(f"wood_knot_{cat_name}")
    if cfg.primary_log:
        names.append("primary_log")
    names_dict = {idx: name for idx, name in enumerate(names)}
    return names_dict


def process_single_image(img_id, coco, raw_dir, images_dir, resolution):
    """Worker function to process one image."""
    try:
        img_info = coco.loadImgs(img_id)[0]
        img_path = raw_dir / "images" / img_info["file_name"]

        img = cv2.imread(str(img_path))
        if img is None:
            return f"Error: Could not read {img_path}"

        # Smallest Max Size logic
        h, w = img.shape[:2]
        scale = resolution / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        dest_path = images_dir / img_info["file_name"]
        cv2.imwrite(str(dest_path), img_resized)
        return None  # Success
    except Exception as e:
        return f"Error processing {img_id}: {str(e)}"


def get_images_by_config(cfg: ImageConfig, coco: COCO):
    base_images_dir = PROCESSED_YOLO_DIR / "images"
    img_cfg_id = get_config_id(cfg)
    images_dir = base_images_dir / img_cfg_id
    print("|_ ", end="")
    if images_dir.exists() and (images_dir / "config.yaml").exists():
        print(f"✅ Using existing images")
        return images_dir

    print(f"🔥 Generating images")
    images_dir.mkdir(parents=True, exist_ok=True)
    img_ids = coco.getImgIds()

    # We use a partial to lock in the constant arguments for the worker
    worker = partial(
        process_single_image,
        coco=coco,
        raw_dir=RAW_DIR,
        images_dir=images_dir,
        resolution=cfg.resolution,
    )

    # Use max available cores
    with ProcessPoolExecutor() as executor:
        # Submit all tasks
        futures = [executor.submit(worker, img_id) for img_id in img_ids]

        # Process with progress bar
        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(img_ids),
            desc="Processing images",
            leave=False,
        ):
            result = future.result()
            if result:
                print(result)

    OmegaConf.save(config=cfg, f=images_dir / "config.yaml")
    return images_dir


def get_yolo_line(ann, img_w, img_h, class_id):
    yolo_rows = []
    for poly in ann["segmentation"]:
        if len(poly) < 6:  # need at least 3 points
            continue

        normalized_coords = []
        for i in range(0, len(poly), 2):
            x = poly[i] / img_w
            y = poly[i + 1] / img_h
            normalized_coords.extend([max(0.0, min(1.0, x)), max(0.0, min(1.0, y))])

        # Format as: <class_id> <x1> <y1> <x2> <y2> ...
        coords_str = " ".join([f"{c:.6f}" for c in normalized_coords])
        yolo_rows.append(f"{class_id} {coords_str}")

    return yolo_rows


def get_labels_by_config(cfg: LabelConfig, coco: COCO):
    base_labels_dir = PROCESSED_YOLO_DIR / "labels"
    label_cfg_id = get_config_id(cfg)
    labels_dir = base_labels_dir / label_cfg_id
    print("|_ ", end="")
    if labels_dir.exists() and (labels_dir / "config.yaml").exists():
        print(f"✅ Using existing labels")
        return labels_dir
    print(f"🔥 Generating labels")
    labels_dir.mkdir(parents=True, exist_ok=True)
    img_ids = coco.getImgIds()
    class_names = get_class_names(cfg)
    names_to_idx = {name: idx for idx, name in class_names.items()}
    wood_knot_cfs = cfg.wood_knot
    for img_id in tqdm.tqdm(img_ids, desc="Processing labels", leave=False):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        h = coco.loadImgs(img_id)[0]["height"]
        w = coco.loadImgs(img_id)[0]["width"]
        annotations = []
        for ann in anns:
            cat_id = ann["category_id"]
            cat_name = coco.loadCats(cat_id)[0]["name"]
            if cat_name == "primary_log" and cfg.primary_log:
                class_idx = names_to_idx["primary_log"]
                annotations.extend(get_yolo_line(ann, w, h, class_idx))
            elif cat_name == "wood_knot":
                cfs = float(ann.get("attributes", {}).get("confidance", 1.0))
                if any(abs(cfs - val) < 1e-3 for val in wood_knot_cfs):
                    if cfg.wood_knot_categorization == "combined":
                        class_idx = names_to_idx["wood_knot"]
                        annotations.extend(get_yolo_line(ann, w, h, class_idx))
                    else:
                        closest_cf = min(wood_knot_cfs, key=lambda x: abs(cfs - x))
                        cat_name = CONF_MAP[closest_cf]
                        class_idx = names_to_idx[f"wood_knot_{cat_name}"]
                        if cfg.wood_knot_categorization == "separate":
                            annotations.extend(get_yolo_line(ann, w, h, class_idx))
                        elif cfg.wood_knot_categorization == "cumulative":
                            for i in range(class_idx + 1):
                                annotations.extend(get_yolo_line(ann, w, h, i))
                        else:
                            raise ValueError(
                                f"Invalid wood knot categorization: {cfg.wood_knot_categorization}"
                            )
        if not annotations:
            continue
        img_info = coco.loadImgs(img_id)[0]
        label_path = labels_dir / f"{Path(img_info['file_name']).stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(annotations))
    OmegaConf.save(config=cfg, f=labels_dir / "config.yaml")
    return labels_dir


def get_fold_by_config(cfg: FoldConfig, coco: COCO):
    base_folds_dir = PROCESSED_YOLO_DIR / "folds"
    fold_cfg_id = get_config_id(cfg)
    fold_dir = base_folds_dir / fold_cfg_id
    print("|_ ", end="")
    if fold_dir.exists() and (fold_dir / "config.yaml").exists():
        print(f"✅ Using existing folds")
        return fold_dir
    print(f"🔥 Generating folds")
    fold_dir.mkdir(parents=True, exist_ok=True)
    img_names = [coco.loadImgs(img_id)[0]["file_name"] for img_id in coco.getImgIds()]
    names_map = {Path(name).stem: name for name in img_names}
    valid_ids = set(names_map.keys())

    train_ids, val_ids = get_fold(
        None, k=cfg.k, fold=cfg.fold, valid_ids=valid_ids, seed=cfg.seed
    )
    with open(fold_dir / "train.txt", "w") as f:
        f.write("\n".join([f"./images/{names_map[img_id]}" for img_id in train_ids]))
    with open(fold_dir / "val.txt", "w") as f:
        f.write("\n".join([f"./images/{names_map[img_id]}" for img_id in val_ids]))
    OmegaConf.save(config=cfg, f=fold_dir / "config.yaml")
    return fold_dir


def get_dataset_by_config(cfg: DatasetConfig):
    base_dataset_dir = PROCESSED_YOLO_DIR / "datasets"

    dataset_cfg_id = get_config_id(cfg)
    dataset_dir = base_dataset_dir / dataset_cfg_id
    if dataset_dir.exists() and (dataset_dir / "config.yaml").exists():
        print(f"✅ Using existing dataset")
        return dataset_dir

    print(f"🚀 Generating dataset:")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    coco = load_coco_silent(RAW_DIR / "annotations.json")

    images_dir = get_images_by_config(cfg.image_config, coco)
    labels_dir = get_labels_by_config(cfg.label_config, coco)
    folds_dir = get_fold_by_config(cfg.fold_config, coco)

    (dataset_dir / "images").symlink_to(images_dir, target_is_directory=True)
    (dataset_dir / "labels").symlink_to(labels_dir, target_is_directory=True)

    # copy train.txt and val.txt to dataset dir for easier access
    (dataset_dir / "train.txt").write_text((folds_dir / "train.txt").read_text())
    (dataset_dir / "val.txt").write_text((folds_dir / "val.txt").read_text())

    data_yaml = {
        "path": str(dataset_dir),
        "train": "train.txt",
        "val": "val.txt",
        "names": get_class_names(cfg.label_config),
    }
    with open(dataset_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    OmegaConf.save(config=cfg, f=dataset_dir / "config.yaml")
    return dataset_dir
