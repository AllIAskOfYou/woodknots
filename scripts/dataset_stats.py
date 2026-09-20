#!/usr/bin/env python3
"""Compute dataset statistics from a raw WoodKnots dataset directory.

The directory must contain annotations.json (COCO) and groups.json (a list of
logs, each a list of case ids = image file stems). Standard library only.

    python3 scripts/dataset_stats.py                          # ./ -> ./stats.json
    python3 scripts/dataset_stats.py data_sc/dataset/c04377f6/raw -o stats.json

Image sizes are the annotated width/height, i.e. with EXIF orientation applied
(what cv2.imread returns). Areas use each annotation's COCO "area" field:
  - primary log area as a fraction of the image area (width * height)
  - knot area as a fraction of the primary log area of the same image
"""
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PRIMARY_LOG = "primary_log"
WOOD_KNOT = "wood_knot"
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation between closest ranks (same as numpy's default)."""
    pos = (len(sorted_values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    s = sorted(values)
    return {
        "count": len(s),
        "min": s[0],
        "max": s[-1],
        "mean": statistics.fmean(s),
        "std": statistics.stdev(s) if len(s) > 1 else 0.0,
        "quantiles": {f"p{round(q * 100)}": quantile(s, q) for q in QUANTILES},
    }


def compute_stats(dataset_dir: Path) -> dict:
    coco = json.loads((dataset_dir / "annotations.json").read_text())
    groups = json.loads((dataset_dir / "groups.json").read_text())

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    case_of = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}

    # image resolutions
    resolutions = Counter(f"{im['width']}x{im['height']}" for im in images.values())

    # images per log (log = index in groups.json)
    log_of = {case_id: log for log, cases in enumerate(groups) for case_id in cases}
    images_per_log = Counter(log_of.get(case_of[i]) for i in images)
    unassigned = images_per_log.pop(None, 0)

    # per-image annotations
    log_area = defaultdict(float)
    knots = defaultdict(list)
    for ann in coco["annotations"]:
        name = categories[ann["category_id"]]
        if name == PRIMARY_LOG:
            log_area[ann["image_id"]] += ann["area"]
        elif name == WOOD_KNOT:
            knots[ann["image_id"]].append(ann)

    knots_per_image = Counter(len(knots[i]) for i in images)
    primary_log_fraction = [
        log_area[i] / (im["width"] * im["height"]) for i, im in images.items() if log_area[i] > 0
    ]
    knot_fraction = [
        ann["area"] / log_area[i] for i, anns in knots.items() if log_area[i] > 0 for ann in anns
    ]
    confidences = [
        ann["attributes"]["confidence"]
        for anns in knots.values()
        for ann in anns
        if "confidence" in ann.get("attributes", {})
    ]

    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "num_images": len(images),
        "image_resolutions": {
            res: n for res, n in sorted(resolutions.items(), key=lambda kv: (-kv[1], kv[0]))
        },
        "images_per_log": {
            "num_logs": len(groups),
            "counts": {str(log): images_per_log[log] for log in range(len(groups))},
            "images_not_in_any_log": unassigned,
            "distribution": distribution([images_per_log[log] for log in range(len(groups))]),
        },
        "knots_per_image": {
            "counts": {str(n): c for n, c in sorted(knots_per_image.items())},
            "images_without_knots": knots_per_image.get(0, 0),
            "distribution": distribution([len(knots[i]) for i in images]),
        },
        "total_knots": sum(len(anns) for anns in knots.values()),
        "primary_log_area_fraction_of_image": {
            "images_without_primary_log": sum(1 for i in images if log_area[i] <= 0),
            **distribution(primary_log_fraction),
        },
        "knot_area_fraction_of_primary_log": {
            "knots_without_primary_log": sum(len(anns) for i, anns in knots.items() if log_area[i] <= 0),
            **distribution(knot_fraction),
        },
        "knot_confidence": {
            "counts": {str(v): c for v, c in sorted(Counter(confidences).items())},
            "knots_without_confidence": sum(len(anns) for anns in knots.values()) - len(confidences),
            **distribution(confidences),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "dataset_dir",
        nargs="?",
        default=".",
        help="directory with annotations.json and groups.json (default: .)",
    )
    ap.add_argument("-o", "--output", default="stats.json", help="output JSON file (default: ./stats.json)")
    args = ap.parse_args()

    stats = compute_stats(Path(args.dataset_dir))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Saved dataset statistics to {out}")


if __name__ == "__main__":
    main()
