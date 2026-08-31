import hydra
from omegaconf import DictConfig

from mltools.nnunet.train import train
from mltools.common.logger import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger.info(
        f"Starting run: framework={cfg.framework}, experiment={cfg.experiment.name}"
    )
    train(cfg)


if __name__ == "__main__":
    main()
