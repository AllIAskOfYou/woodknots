import hashlib
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path

from mltools.common.coco_dataset import COCODataset, resize_dataset
from mltools.common.utils import hash_cfg


def get_experiment_names(framework_dir: Path) -> list[str]:
    exp_dir = framework_dir / "conf" / "experiment"
    return sorted(f.stem for f in exp_dir.glob("*.yaml"))


def collect_dataset_variants(framework_dir: Path) -> dict:
    """Compose each experiment's config and collect unique dataset definitions."""
    variants = {}  # key: dedup key (e.g. resolution) -> resolved dataset cfg

    config_dir = str((framework_dir / "conf").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for exp_name in get_experiment_names(framework_dir):
            cfg = compose(config_name="config", overrides=[f"experiment={exp_name}"])
            ds_hash = hash_cfg(cfg.dataset)
            if ds_hash not in variants:
                variants[ds_hash] = cfg.dataset

    return variants


def get_resized_dir(data_dir: str, resolution: int) -> str:
    resized_dir = Path(data_dir) / "resized" / f"{resolution}"
    return str(resized_dir)


def ensure_resized(dataset_dir: str, resolution: int) -> str:
    resized_dir = get_resized_dir(dataset_dir, resolution)
    marker = Path(resized_dir) / ".complete"
    if marker.exists():
        return resized_dir
    local_dir = Path(dataset_dir) / "raw"
    images_dir = local_dir / "images"
    annotations_path = local_dir / "annotations.json"

    coco_ds = COCODataset(
        image_dir=str(images_dir), annotation_path=str(annotations_path)
    )
    image_shape = coco_ds.load_image(0).shape
    min_dim = min(image_shape[0], image_shape[1])
    if resolution > min_dim:
        raise ValueError(
            f"Requested resolution {resolution} is larger than the smallest image dimension {min_dim}."
        )
    ratio = resolution / min_dim
    new_shape = (int(image_shape[1] * ratio), int(image_shape[0] * ratio))
    resize_dataset(coco_ds, new_shape, resized_dir)
    marker.touch()
    return resized_dir
