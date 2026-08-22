import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def train(cfg: DictConfig):
    logger.info("Starting training using nnuner framework...")
