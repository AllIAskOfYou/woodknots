import hashlib
from omegaconf import DictConfig, OmegaConf
import subprocess
import logging


def hash_string(s: str) -> str:
    """Generate an 8 digit hex hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def hash_cfg(cfg: DictConfig, exclude_keys: list[str] | None = None) -> str:
    """Generate an 8 digit hex hash of a DictConfig.

    exclude_keys drops top-level keys before hashing — for a framework whose
    dataset-prep pipeline doesn't consume every field of the shared
    dataset: config group (e.g. folds/class_names, which only affect
    fold-splitting/training, not what gets written to the formatted dataset
    dir), including them in the hash would mean an unrelated change (e.g.
    switching folds files) spuriously invalidates an already-built,
    byte-identical dataset directory. Defaults to hashing everything, so
    existing callers/cached directories are unaffected."""
    # Resolve interpolations first, while cfg still has its full parent
    # config tree for context (e.g. data_root: ${paths.data_root}) — a
    # filtered-then-resolved DictConfig built from a subset of keys would
    # lose that outer context and fail to resolve.
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if exclude_keys:
        resolved = {k: v for k, v in resolved.items() if k not in exclude_keys}
    yaml = OmegaConf.to_yaml(OmegaConf.create(resolved), sort_keys=True)
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
