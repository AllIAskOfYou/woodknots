from huggingface_hub import snapshot_download
import os
from pathlib import Path

from mltools.common.utils import hash_string

from dotenv import load_dotenv  # TODO: move this to a more appropriate place, e.g. main.py
load_dotenv()

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
        token=os.getenv("HF_TOKEN") # TODO: this will not be needed once the datasets are public
    )
    marker.touch()
    return local_dir
