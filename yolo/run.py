import hydra
from omegaconf import DictConfig

from mltools.yolo.train import train


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
