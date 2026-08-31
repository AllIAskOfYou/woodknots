from pathlib import Path
from huggingface_hub import snapshot_download
from mltools.common.utils import hash_string


def hash_dataset(data_dir: str, repo_id: str, revision: str) -> str:
    hash_input = f"{data_dir}:{repo_id}:{revision}"
    return hash_string(hash_input)


def ensure_dataset(data_root: str, repo_id: str, revision: str) -> str:
    ds_hash = hash_dataset(data_root, repo_id, revision)
    local_dir = Path(data_root) / "dataset" / ds_hash / "raw"
    marker = local_dir / ".download_complete"
    if marker.exists():
        return local_dir

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
    )
    marker.touch()
    return local_dir.parent
