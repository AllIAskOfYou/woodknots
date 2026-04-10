from mltools.yolo.dataset.utils import get_dataset_by_config
from mltools.yolo.schema import TrainConfig
import hydra

if __name__ == "__main__":
    with hydra.initialize(version_base=None, config_path="../configs_yolo"):
        cfg: TrainConfig = hydra.compose(
            config_name="config",
        )

    path = get_dataset_by_config(cfg.dataset)
    print(f"Dataset path: {path}")
