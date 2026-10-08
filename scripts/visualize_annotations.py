"""Draw ground-truth annotations over dataset images, one PNG per case.

primary_log and wood_knot polygons are drawn as outlines. With no --case, a
random sample of images that contain at least one knot is drawn.

    uv run scripts/visualize_annotations.py
    uv run scripts/visualize_annotations.py --num 20 --width 6 --knot-color "#ff8800"
    uv run scripts/visualize_annotations.py --case 0000 0627
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ANNOTATIONS = "data/dataset/c04377f6/resized/short_side_512/annotations.json"


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise argparse.ArgumentTypeError(f"expected #rrggbb, got {value!r}")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def rings(annotation) -> list[np.ndarray]:
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list):
        return []
    return [
        np.round(np.array(part, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
        for part in segmentation
        if isinstance(part, list) and len(part) >= 6
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default=DEFAULT_ANNOTATIONS,
                    help="COCO annotations.json; images are read from images/ next to it")
    ap.add_argument("--case", nargs="+", default=None, help="case ids (file stems); overrides sampling")
    ap.add_argument("--num", type=int, default=10, help="number of knot images to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-width", type=int, default=3, help="log line width in px")
    ap.add_argument("--knot-width", type=int, default=3, help="knot line width in px")
    ap.add_argument("--log-color", type=hex_to_bgr, default="#00ff00")
    ap.add_argument("--knot-color", type=hex_to_bgr, default="#87cefa")
    ap.add_argument("-o", "--output-dir", default=None,
                    help="default: results/annotation_vis/seed_<seed>, or .../cases with --case")
    args = ap.parse_args()

    annotations_path = Path(args.annotations)
    images_dir = annotations_path.parent / "images"
    data = json.loads(annotations_path.read_text())
    category = {c["id"]: c["name"] for c in data["categories"]}
    images = {Path(im["file_name"]).stem: im for im in data["images"]}

    by_image = defaultdict(list)
    for ann in data["annotations"]:
        by_image[ann["image_id"]].append(ann)

    if args.case:
        missing = [c for c in args.case if c not in images]
        if missing:
            print(f"unknown cases: {missing}")
            return 1
        cases = args.case
    else:
        with_knots = sorted(
            case for case, im in images.items()
            if any(category[a["category_id"]] == "wood_knot" for a in by_image[im["id"]])
        )
        cases = sorted(random.Random(args.seed).sample(with_knots, min(args.num, len(with_knots))))

    colors = {"primary_log": args.log_color, "wood_knot": args.knot_color}
    # One folder per selection, so runs with different seeds don't mix.
    out_dir = Path(args.output_dir or f"results/annotation_vis/{'cases' if args.case else f'seed_{args.seed}'}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        im = images[case]
        image = cv2.imread(str(images_dir / im["file_name"]))
        if image is None:
            print(f"{case}: cannot read {images_dir / im['file_name']}, skipping")
            continue
        # Logs first, so knot outlines stay on top.
        anns = sorted(by_image[im["id"]], key=lambda a: category[a["category_id"]] != "primary_log")
        for ann in anns:
            color = colors.get(category[ann["category_id"]])
            if color is not None:
                width = args.log_width if category[ann["category_id"]] == "primary_log" else args.knot_width
                cv2.polylines(image, rings(ann), True, color, width, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{case}.jpg"), image)

    print(f"Saved {len(cases)} images to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
