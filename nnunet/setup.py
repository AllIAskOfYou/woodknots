import json
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from mltools.common.hf import ensure_dataset
from mltools.common.setup import collect_dataset_variants, ensure_resized
from mltools.nnunet.dataset import ensure_formatted, get_framework_root
from mltools.common.logger import get_logger, indented
from mltools.nnunet.preprocess import preprocess_datasets

logger = get_logger(__name__)


def prepare_datasets(dataset_variants: list[DictConfig]) -> None:
    for ds_idx, ds_hash in enumerate(dataset_variants):
        ds = dataset_variants[ds_hash]
        logger.info(
            f"[{ds_idx + 1}/{len(dataset_variants)}]  Processing dataset variant: {ds.id}"
        )
        with indented():
            dataset_dir = ensure_dataset(
                data_root=ds.data_root,
                repo_id=ds.hf_repo_id,
                revision=ds.hf_revision,
            )
            logger.info(f"Dataset is available at: {dataset_dir}")
        with indented():
            resized_dir = ensure_resized(
                dataset_dir=dataset_dir,
                resolution=ds.resolution,
            )
            logger.info(f"Resized dataset is available at: {resized_dir}")
        with indented():
            formatted_dir = ensure_formatted(
                local_dir=resized_dir,
                ds_id=ds.id,
                ds_name=ds.name,
                class_names=ds.class_names,
                folds_path=ds.folds.file_path,
                data_root=ds.data_root,
            )
            # store cfg into the formatted dir
            cfg_path = Path(formatted_dir) / "dataset_config.yaml"
            OmegaConf.save(ds, cfg_path, resolve=True)
            # store manifest into the formatted dir
            manifest_path = Path(formatted_dir) / "manifest.json"
            manifest = {
                "raw_dir": str(dataset_dir),
                "resized_dir": str(resized_dir),
            }
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=4)
            logger.info(f"nnUNet formatted dataset is available at: {formatted_dir}")


def main():
    framework_dir = Path(__file__).parent
    logger.info(f"Collecting dataset variants from framework dir: {framework_dir}")
    dataset_variants = collect_dataset_variants(framework_dir)
    prepare_datasets(dataset_variants)
    preprocess_datasets(dataset_variants)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
