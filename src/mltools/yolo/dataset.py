from pathlib import Path

import yaml
from omegaconf import DictConfig
from tqdm import tqdm

from mltools.common.coco_dataset import COCODataset
from mltools.common.logger import make_bar_format
from mltools.common.splits import get_split
from mltools.common.utils import symlink_relative

CONF_MAP = {0.2: "low", 0.6: "medium", 1.0: "high"}


def get_framework_root(data_root: str) -> str:
    return str(Path(data_root) / "yolo_format")


def get_dataset_dir(data_root: str, ds_name: str, ds_hash: str) -> str:
    return str(Path(get_framework_root(data_root)) / f"{ds_name}_{ds_hash}")


def get_class_names(label_cfg: DictConfig) -> dict[int, str]:
    names = []
    if label_cfg.wood_knot_categorization == "combined":
        names.append("wood_knot")
    else:
        for val in label_cfg.wood_knot_confidences:
            if val not in CONF_MAP:
                raise ValueError(
                    f"Wood knot confidence value {val} is not valid. "
                    f"Must be one of {list(CONF_MAP.keys())}."
                )
            names.append(f"wood_knot_{CONF_MAP[val]}")
    if label_cfg.primary_log:
        names.append("primary_log")
    return {idx: name for idx, name in enumerate(names)}


def classify_annotation(
    cat_name: str,
    confidence: float,
    label_cfg: DictConfig,
    names_to_idx: dict[str, int],
) -> list[int]:
    """Return the class index/indices this annotation belongs to (empty if it
    doesn't match the configured label selection; more than one for
    'cumulative' categorization)."""
    if cat_name == "primary_log":
        if not label_cfg.primary_log:
            return []
        return [names_to_idx["primary_log"]]

    if cat_name != "wood_knot":
        return []

    wood_knot_confidences = label_cfg.wood_knot_confidences
    if not any(abs(confidence - val) < 1e-3 for val in wood_knot_confidences):
        return []

    if label_cfg.wood_knot_categorization == "combined":
        return [names_to_idx["wood_knot"]]

    closest_cf = min(wood_knot_confidences, key=lambda x: abs(confidence - x))
    class_idx = names_to_idx[f"wood_knot_{CONF_MAP[closest_cf]}"]
    if label_cfg.wood_knot_categorization == "separate":
        return [class_idx]
    if label_cfg.wood_knot_categorization == "cumulative":
        return list(range(class_idx + 1))
    raise ValueError(
        f"Invalid wood knot categorization: {label_cfg.wood_knot_categorization}"
    )


def get_yolo_line(ann: dict, img_w: int, img_h: int, class_id: int) -> list[str]:
    yolo_rows = []
    for poly in ann["segmentation"]:
        if len(poly) < 6:  # need at least 3 points
            continue

        normalized_coords = []
        for i in range(0, len(poly), 2):
            x = poly[i] / img_w
            y = poly[i + 1] / img_h
            normalized_coords.extend([max(0.0, min(1.0, x)), max(0.0, min(1.0, y))])

        coords_str = " ".join(f"{c:.6f}" for c in normalized_coords)
        yolo_rows.append(f"{class_id} {coords_str}")
    return yolo_rows


def _process_case(
    coco_ds: COCODataset,
    index: int,
    label_cfg: DictConfig,
    names_to_idx: dict[str, int],
    labels_dir: Path,
) -> None:
    case_id = coco_ds.image_index_to_name(index)
    img_id = coco_ds.ids[index]
    img_info = coco_ds.coco.loadImgs(img_id)[0]
    img_w, img_h = img_info["width"], img_info["height"]

    ann_ids = coco_ds.coco.getAnnIds(imgIds=img_id)
    anns = coco_ds.coco.loadAnns(ann_ids)

    yolo_lines = []
    for ann in anns:
        cat_name = coco_ds.coco.loadCats(ann["category_id"])[0]["name"]
        confidence = float(ann.get("attributes", {}).get("confidence", 1.0))
        class_indices = classify_annotation(cat_name, confidence, label_cfg, names_to_idx)
        for class_idx in class_indices:
            yolo_lines.extend(get_yolo_line(ann, img_w, img_h, class_idx))

    if yolo_lines:
        (labels_dir / f"{case_id}.txt").write_text("\n".join(yolo_lines))


def ensure_formatted(
    local_dir: str,
    ds_hash: str,
    ds_name: str,
    label_cfg: DictConfig,
    data_root: str,
) -> str:
    """Fold-independent: symlinks the resized images and writes YOLO labels/
    for every case. (No ground truth: the evaluation builds it from the
    annotations, identically for every framework.) Marker-file cached, same
    convention as nnunet's ensure_formatted. Directory is named by content
    hash of the dataset config (ds_hash) rather than a manually-assigned
    numeric id — unlike nnU-Net, YOLO's tooling has no requirement for a
    `DatasetXXX_Name`-style numeric id, so there's nothing for one to serve
    beyond bookkeeping that can drift out of sync."""
    dataset_dir = Path(get_dataset_dir(data_root, ds_name, ds_hash))
    marker = dataset_dir / ".complete"
    if marker.exists():
        return str(dataset_dir)

    coco_ds = COCODataset(
        image_dir=str(Path(local_dir) / "images"),
        annotation_path=str(Path(local_dir) / "annotations.json"),
    )

    labels_dir = dataset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    images_link = dataset_dir / "images"
    if not images_link.exists():
        symlink_relative(images_link, Path(local_dir) / "images", target_is_directory=True)

    class_names = get_class_names(label_cfg)
    names_to_idx = {name: idx for idx, name in class_names.items()}

    for index in tqdm(
        range(len(coco_ds)), bar_format=make_bar_format(desc="Formatting cases")
    ):
        _process_case(
            coco_ds, index, label_cfg, names_to_idx, labels_dir
        )

    marker.touch()
    return str(dataset_dir)


def write_fold_split(dataset_dir: str, folds_path: str, fold: int, out_dir: str, class_names: dict[int, str]) -> str:
    """Fold-dependent, cheap, uncached — populates out_dir/{train,val,test}/{images,labels}/
    with symlinks into the shared formatted dataset (no duplication of the
    underlying files), writes out_dir/{train,val,test}.txt, and out_dir/data.yaml.
    Splits come from mltools.common.splits.get_split: Ultralytics selects
    best.pt (and runs early stopping) on val; test is only predicted on.
    Called by both train.py and evaluate.py, so each run gets its own
    fold-specific data.yaml without mutating the shared formatted dataset.

    Each split gets its own images/labels directory tree rather than
    just their own file list. Ultralytics derives its label-cache path purely
    from the labels directory (ignoring which specific files are listed), so
    if both splits pointed at the same shared labels/ dir they'd compute the
    identical cache path — whichever split's dataloader is built second would
    then load the cache the first split had just written, and hitting that
    file mid-write (or a stale one from an earlier crashed run) can look like
    a truncated pickle. Distinct directories give each split its own cache
    path instead.
    """
    split = get_split(folds_path, fold)

    # Resolve to absolute so symlink_relative can compute the hop from each
    # link's own directory, whatever dataset_dir was passed as.
    dataset_dir = Path(dataset_dir).resolve()
    src_images_dir = dataset_dir / "images"
    src_labels_dir = dataset_dir / "labels"
    out_path = Path(out_dir)

    for split_name in ("train", "val", "test"):
        images_dir = out_path / split_name / "images"
        labels_dir = out_path / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        lines = []
        for case_id in split[split_name]:
            image_link = images_dir / f"{case_id}.png"
            if not image_link.exists():
                symlink_relative(image_link, src_images_dir / f"{case_id}.png")
            lines.append(str(image_link))

            src_label = src_labels_dir / f"{case_id}.txt"
            label_link = labels_dir / f"{case_id}.txt"
            if src_label.exists() and not label_link.exists():
                symlink_relative(label_link, src_label)

        (out_path / f"{split_name}.txt").write_text("\n".join(lines))

    data_yaml = {
        "path": str(out_path),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": class_names,
    }
    data_yaml_path = out_path / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    return str(data_yaml_path)
