import hashlib
from omegaconf import DictConfig, OmegaConf
import subprocess
import logging


def hash_string(s: str) -> str:
    """Generate an 8 digit hex hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def hash_cfg(cfg: DictConfig) -> str:
    """Generate an 8 digit hex hash of a DictConfig."""
    yaml = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    return hash_string(yaml)


def run_command(cmd: list[str], env: dict, logger: logging.Logger = None) -> None:
    if logger:
        logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if logger and result.stdout:
        logger.info(result.stdout)
    if result.returncode != 0:
        if logger:
            logger.error(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
