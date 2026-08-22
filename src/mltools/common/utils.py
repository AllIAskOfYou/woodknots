import hashlib
from omegaconf import DictConfig, OmegaConf


def hash_string(s: str) -> str:
    """Generate an 8 digit hex hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def hash_cfg(cfg: DictConfig) -> str:
    """Generate an 8 digit hex hash of a DictConfig."""
    yaml = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    return hash_string(yaml)
