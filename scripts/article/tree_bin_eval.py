"""Instance recall and outline quality of the seg pair ensembles per tree (log).

The same analysis as size_bin_eval.py — cached predictions, default and tuned
(leave-one-fold-out) threshold, one matching per image split by group — with
the knots grouped by the tree they belong to: the image groups of the raw
dataset's groups.json, one physical log each. A tree sits in exactly one test
fold, so its values come from that fold (fold mean = that fold's value), still
mean ± sd across the runs. False positives are assigned to the tree of their
image, so knot-free images count towards their tree too.

The split assigns every log's images to one fold, so trees are never seen in
training when they are scored: this shows how unevenly the method generalises
to unseen logs.

    uv run scripts/article/tree_bin_eval.py
    uv run scripts/article/tree_bin_eval.py -o article/tree_bins/seg
"""
import argparse
import json
import sys
from pathlib import Path

from size_bin_eval import PROJECT_ROOT, add_common_args, evaluate

GROUPS = PROJECT_ROOT / "data/dataset/c04377f6/raw/groups.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", default=str(GROUPS), help="list of case-id lists, one per tree")
    add_common_args(ap, "article/tree_bins/seg")
    args = ap.parse_args()

    trees = json.loads(Path(args.groups).read_text())
    tree_of_case = {case: i for i, cases in enumerate(trees) for case in cases}
    data = json.loads(Path(args.annotations).read_text())
    stems = {im["id"]: Path(im["file_name"]).stem for im in data["images"]}
    missing = set(stems.values()) - set(tree_of_case)
    if missing:
        raise SystemExit(f"{len(missing)} images belong to no tree in {args.groups}, e.g. {sorted(missing)[:3]}")

    names = [f"T{i + 1:02d}" for i in range(len(trees))]
    knot_ids = {c["id"] for c in data["categories"] if c["name"] == "wood_knot"}
    with_knots = {stems[a["image_id"]] for a in data["annotations"] if a["category_id"] in knot_ids}
    desc = "trees (groups.json): " + ", ".join(
        f"{n} {len(cases)} img ({sum(c not in with_knots for c in cases)} knot-free, "
        f"{min(cases)}-{max(cases)})"
        for n, cases in zip(names, trees))

    evaluate(
        args, names, lambda ann: tree_of_case[stems[ann["image_id"]]], "by tree (log)", desc,
        fp_group=lambda case, r: tree_of_case[case],
        fp_split_desc="per tree, in the images of that tree", knots_per_fold=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
