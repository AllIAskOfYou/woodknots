import hashlib
import os
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import subprocess
import logging


TRAIN_COMPLETE = ".train_complete"
PRED_COMPLETE = ".pred_complete"


def train_complete_marker(run_dir) -> Path:
    """Touched as the very last step of a successful train(), so an
    interrupted run is retrained rather than silently treated as done. Same
    file name in every framework's run dir, which is what scripts/check_runs.py
    reports on."""
    return Path(run_dir) / TRAIN_COMPLETE


def pred_complete_marker(run_dir) -> Path:
    """Touched once every test-case prediction has been written."""
    return Path(run_dir) / PRED_COMPLETE


def symlink_relative(link: Path, target: Path, target_is_directory: bool = False) -> None:
    """Create `link` pointing at `target` by a path relative to the link's own
    directory.

    An absolute target would bake in the project root the link was created
    under, and setup runs inside the enroot container where that root is
    /workspace — such links resolve inside the container but dangle when the
    same files are inspected from the login node. Everything here lives under
    the project root, so a relative path is valid in both."""
    link.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(Path(target).resolve(), link.parent.resolve())
    link.symlink_to(relative, target_is_directory=target_is_directory)


def hash_string(s: str) -> str:
    """Generate an 8 digit hex hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()[:8]


# Dataset keys that only affect the evaluation, never what a formatted dataset
# directory contains — always left out of hash_cfg, so adding or changing them
# doesn't orphan existing formatted datasets.
EVALUATION_ONLY_KEYS = ("eval_confidences",)


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
    exclude = set(exclude_keys or ()) | set(EVALUATION_ONLY_KEYS)
    resolved = {k: v for k, v in resolved.items() if k not in exclude}
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
