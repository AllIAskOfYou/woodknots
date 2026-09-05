import hydra
from omegaconf import DictConfig

from mltools.yolo.train import predict, train, setup_ultralytics


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    setup_ultralytics(cfg)
    train(cfg)
    predict(cfg)


if __name__ == "__main__":
    main()
