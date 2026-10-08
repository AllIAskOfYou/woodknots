from omegaconf import DictConfig

from mltools.nnunet.dataset import get_framework_root
from pathlib import Path
from mltools.common.logger import get_logger, indented
from mltools.common.utils import run_command
from mltools.nnunet.utils import build_nnunet_env

logger = get_logger(__name__)

# Extra plans (plans identifier -> planner) planned for every dataset. Their 2d
# configuration reuses the default nnUNetPlans_2d data, so this is planning only.
EXTRA_PLANNERS = {"nnUNetResEncUNetLPlans": "nnUNetPlannerResEncL"}


def preprocess_dataset(cfg: DictConfig) -> None:
    ds_cfg = cfg
    dataset_id = ds_cfg.id
    dataset_name = f"Dataset{dataset_id:03d}_{ds_cfg.name}"

    nnunet_root = Path(get_framework_root(ds_cfg.data_root))
    nnunet_raw_root = nnunet_root / "raw"
    nnunet_preprocessed_root = nnunet_root / "preprocessed"
    nnunet_preprocessed_root.mkdir(parents=True, exist_ok=True)

    env = build_nnunet_env(
        nnunet_raw=str(nnunet_raw_root),
        nnunet_preprocessed=str(nnunet_preprocessed_root),
        nnunet_results="",
    )

    preprocessed_marker = nnunet_preprocessed_root / dataset_name / ".preprocessed"
    if not preprocessed_marker.exists():
        run_command(
            [
                "nnUNetv2_plan_and_preprocess",
                "-d",
                f"{dataset_id:03d}",
                "--verify_dataset_integrity",
            ],
            env=env,
            logger=logger,
        )
        preprocessed_marker.touch()
    else:
        logger.info(f"Dataset {dataset_id} already preprocessed, skipping.")

    for plans_name, planner in EXTRA_PLANNERS.items():
        if (nnunet_preprocessed_root / dataset_name / f"{plans_name}.json").exists():
            logger.info(f"Dataset {dataset_id} already has {plans_name}, skipping.")
            continue
        run_command(
            ["nnUNetv2_plan_experiment", "-d", f"{dataset_id:03d}", "-pl", planner],
            env=env,
            logger=logger,
        )



def preprocess_datasets(dataset_variants: list[DictConfig]) -> None:
    for ds_idx, ds_hash in enumerate(dataset_variants):
        ds = dataset_variants[ds_hash]
        logger.info(
            f"[{ds_idx + 1}/{len(dataset_variants)}]  (nnunet) preprocessing dataset: {ds.id}"
        )
        with indented():
            preprocess_dataset(ds)
