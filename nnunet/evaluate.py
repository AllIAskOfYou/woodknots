import hydra
from omegaconf import DictConfig

from mltools.nnunet.evaluate import evaluate


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    evaluate(cfg)


if __name__ == "__main__":
    main()
