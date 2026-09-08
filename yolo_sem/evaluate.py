import hydra
from omegaconf import DictConfig

from mltools.yolo_sem.evaluate import evaluate
from mltools.common.logger import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    evaluate(cfg)


if __name__ == "__main__":
    logger.info("Starting YOLO-sem evaluation...")
    main()
