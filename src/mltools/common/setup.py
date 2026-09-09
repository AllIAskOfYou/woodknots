import hashlib
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from pathlib import Path

from mltools.common.coco_dataset import COCODataset, resize_dataset
from mltools.common.utils import hash_cfg


def get_experiment_names(framework_dir: Path) -> list[str]:
    exp_dir = framework_dir / "conf" / "experiment"
    return sorted(f.stem for f in exp_dir.glob("*.yaml"))


def compose_with_hydra_config(overrides: list[str]):
    """hydra.compose() alone does not populate HydraConfig, so any
    ${hydra:...} resolver (e.g. experiment.name: ${hydra:runtime.choices.experiment})
    raises "HydraConfig was not set" the moment it's accessed outside a real
    @hydra.main job. Passing return_hydra_config=True and registering the
    result with HydraConfig fixes that for standalone compose() callers like
    check_training.py / evaluate_all.py. Must be called within an active
    initialize_config_dir(...) context, same as a plain compose() call."""
    cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    return cfg


def collect_dataset_variants(framework_dir: Path, exclude_keys: list[str] | None = None) -> dict:
    """Compose each experiment's config and collect unique dataset definitions.

    exclude_keys is forwarded to hash_cfg — must match whatever a framework's
    own train.py/evaluate.py pass to hash_cfg(cfg.dataset) when resolving
    get_dataset_dir, or the directory this builds and the directory training
    looks up will disagree."""
    variants = {}  # key: dedup key (e.g. resolution) -> resolved dataset cfg

    config_dir = str((framework_dir / "conf").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for exp_name in get_experiment_names(framework_dir):
            cfg = compose(config_name="config", overrides=[f"experiment={exp_name}"])
            ds_hash = hash_cfg(cfg.dataset, exclude_keys=exclude_keys)
            if ds_hash not in variants:
                variants[ds_hash] = cfg.dataset

    return variants


def get_resized_dir(data_dir: str, resolution: int) -> str:
    resized_dir = Path(data_dir).parent / "resized" / f"{resolution}"
    return str(resized_dir)


def ensure_resized(dataset_dir: str, resolution: int) -> str:
    resized_dir = get_resized_dir(dataset_dir, resolution)
    marker = Path(resized_dir) / ".complete"
    if marker.exists():
        return resized_dir
    local_dir = Path(dataset_dir)
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
