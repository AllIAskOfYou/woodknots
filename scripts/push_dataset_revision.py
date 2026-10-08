"""Upload a file to the WoodKnots dataset repo on the Hub and tag the commit.

The dataset config pins hf_revision, and that pin is hashed into every derived
directory name, so a tagged revision is how a dataset change propagates: bump
the tag in common/conf/dataset/woodknots.yaml and setup rebuilds everything
under a new hash instead of silently mixing versions.

Guards before uploading: the file must be valid COCO JSON, must not contain the
annotation ids named with --absent, and its annotation count must match
--expect-annotations. Nothing is sent if any of those fail.

    uv run scripts/push_dataset_revision.py --dry-run
    uv run scripts/push_dataset_revision.py \\
        --file data/dataset/c04377f6/raw/annotations.json \\
        --tag v1.1 --message "Remove 5 knot annotations lying off the primary log"
"""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

DEFAULT_FILE = "data/dataset/c04377f6/raw/annotations.json"
DEFAULT_REPO = "Elercast/WoodKnots-600"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--path-in-repo", default=None, help="default: the file's name")
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--tag", default="v1.1")
    ap.add_argument("--message", default="Remove 5 knot annotations lying off the primary log")
    ap.add_argument("--absent", type=int, nargs="*", default=[37, 623, 825, 1126, 1127],
                    help="annotation ids that must NOT be in the file")
    ap.add_argument("--expect-annotations", type=int, default=1771)
    ap.add_argument("--dry-run", action="store_true", help="run the checks, send nothing")
    args = ap.parse_args()

    path = Path(args.file)
    data = json.loads(path.read_text())
    count = len(data["annotations"])
    present = sorted({a["id"] for a in data["annotations"]} & set(args.absent))
    print(f"{path}: {count} annotations, {len(data['images'])} images")
    if present:
        print(f"REFUSING: ids that should be absent are still present: {present}")
        return 1
    if count != args.expect_annotations:
        print(f"REFUSING: expected {args.expect_annotations} annotations, found {count}")
        return 1
    print(f"checks passed (none of {args.absent} present)")

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("REFUSING: HF_TOKEN is not set")
        return 1

    path_in_repo = args.path_in_repo or path.name
    if args.dry_run:
        print(f"dry run: would upload {path} -> {args.repo_id}:{path_in_repo}, then tag {args.tag}")
        return 0

    api = HfApi(token=token)
    info = api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.message,
    )
    print(f"uploaded: {info.commit_url if hasattr(info, 'commit_url') else info}")

    api.create_tag(args.repo_id, tag=args.tag, repo_type="dataset", tag_message=args.message)
    print(f"tagged {args.tag}")
    print(f"\nNow set hf_revision: \"{args.tag}\" in common/conf/dataset/woodknots.yaml and re-run setup.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
