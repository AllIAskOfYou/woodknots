import hydra
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from ultralytics import YOLO

from mltools.yolo.dataset.utils import get_dataset_by_config
from mltools.yolo.schema import TrainConfig
from mltools.seg.metrics import STANDARD_METRICS


# define a callback for logging metrics to TensorBoard
#
class Validator:
    def __init__(self):
        self.inputs = []
        self.targets = []Js 

    def on_val_batch_end(self, trainer):
        self.inputs.append(trainer.validator.preds)
        self.targets.append(trainer.validator.batch["masks"])

    def on_val_end(self, trainer):
        # Triggered after the full validation loop is done
        # Concatenate list of tensors into one large tensor
        full_gt = torch.cat(self.targets, dim=0)

        # Do your custom logic here
        print(f"Full validation tensor shape: {full_gt.shape}")

        # Clear for next epoch
        self.inputs.clear()
        self.targets.clear()


@hydra.main(version_base="1.3", config_path="../configs_yolo", config_name="config")
def main(cfg: TrainConfig):
    data_path = get_dataset_by_config(cfg.dataset)
    model = YOLO(cfg.model_path)
    exp_path = HydraConfig.get().run.dir
    # logger = TensorBoardLogger(save_dir=exp_path, name="logs")
    # validator = Validator()
    # model.add_callback("on_val_batch_end", validator.on_val_batch_end)
    # model.add_callback("on_val_end", validator.on_val_end)
    model.train(
        project=exp_path,
        data=data_path / "data.yaml",
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device="cuda:1",
        save_json=True,
    )


if __name__ == "__main__":
    main()
