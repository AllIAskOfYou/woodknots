import hydra
from hydra.core.hydra_config import HydraConfig
from ultralytics import YOLO

from mltools.yolo.dataset.utils import get_dataset_by_config
from mltools.yolo.schema import TrainConfig


@hydra.main(version_base="1.3", config_path="../configs_yolo", config_name="config")
def main(cfg: TrainConfig):
    data_path = get_dataset_by_config(cfg.dataset)
    model = YOLO(cfg.model_path)
    exp_path = HydraConfig.get().run.dir
    model.train(
        project=exp_path,
        data=data_path / "data.yaml",
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device="cuda",
        save_json=True,
    )


if __name__ == "__main__":
    main()
