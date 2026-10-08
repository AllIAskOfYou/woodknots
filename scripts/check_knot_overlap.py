"""Find wood-knot annotations that do not lie on the primary log.

A knot is part of a log, so its polygon should sit inside the primary_log
polygon of the same image. One that does not is almost always a mis-drawn or
mis-labelled annotation — a knot traced on a background log, on the frame, or
assigned to the wrong image.

Overlap is estimated by sampling: a grid is laid over the knot's bounding box,
the points inside the knot polygon are kept, and the reported fraction is how
many of those also fall inside a primary_log polygon. Sampling rather than exact
polygon clipping keeps this dependency-free (no shapely, no pycocotools), and at
the default grid the fraction is accurate to a few percent — enough to separate
"outside the log" from "on the log".

    uv run scripts/check_knot_overlap.py
    python3 scripts/check_knot_overlap.py --threshold 0.9 --grid 48
    python3 scripts/check_knot_overlap.py -o article/knot_overlap.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_ANNOTATIONS = "data/dataset/c04377f6/resized/short_side_512/annotations.json"


def polygons(annotation) -> list[list[tuple[float, float]]]:
    """COCO polygon segmentation -> list of [(x, y), ...] rings."""
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list):
        return []
    rings = []
    for part in segmentation:
        if isinstance(part, list) and len(part) >= 6:
            rings.append([(part[i], part[i + 1]) for i in range(0, len(part) - 1, 2)])
    return rings


def inside(point, rings) -> bool:
    """Ray casting against a set of rings; a point counts as inside if it is
    inside any of them (parts of one annotation are separate pieces here, not
    holes)."""
    x, y = point
    for ring in rings:
        hit = False
        for i in range(len(ring)):
            x1, y1 = ring[i]
            x2, y2 = ring[(i + 1) % len(ring)]
            if (y1 > y) != (y2 > y):
                cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < cross:
                    hit = not hit
        if hit:
            return True
    return False


def overlap_fraction(knot_rings, log_rings, grid: int) -> tuple[float, int]:
    """(fraction of the knot's sampled area that falls on a log, samples used)."""
    xs = [x for ring in knot_rings for x, _ in ring]
    ys = [y for ring in knot_rings for _, y in ring]
    if not xs:
        return 0.0, 0
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    interior, on_log = 0, 0
    for i in range(grid):
        x = x0 + (x1 - x0) * (i + 0.5) / grid
        for j in range(grid):
            y = y0 + (y1 - y0) * (j + 0.5) / grid
            if inside((x, y), knot_rings):
                interior += 1
                if inside((x, y), log_rings):
                    on_log += 1
    if interior == 0:  # degenerate polygon, fall back to its vertices
        vertices = [p for ring in knot_rings for p in ring]
        on_log = sum(1 for p in vertices if inside(p, log_rings))
        return (on_log / len(vertices) if vertices else 0.0), len(vertices)
    return on_log / interior, interior


def shoelace(ring) -> float:
    area = 0.0
    for i in range(len(ring)):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % len(ring)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="report knots with less than this fraction on a log")
    ap.add_argument("--grid", type=int, default=32, help="samples per bounding-box axis")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.annotations).read_text())
    categories = {c["id"]: c["name"] for c in data["categories"]}
    knot_ids = {i for i, n in categories.items() if n == "wood_knot"}
    log_ids = {i for i, n in categories.items() if n == "primary_log"}
    images = {im["id"]: im for im in data["images"]}

    logs_by_image = defaultdict(list)
    knots_by_image = defaultdict(list)
    for ann in data["annotations"]:
        if ann["category_id"] in log_ids:
            logs_by_image[ann["image_id"]].extend(polygons(ann))
        elif ann["category_id"] in knot_ids:
            knots_by_image[ann["image_id"]].append(ann)

    flagged, checked, no_log = [], 0, []
    for image_id, knots in knots_by_image.items():
        log_rings = logs_by_image.get(image_id, [])
        name = images[image_id]["file_name"]
        if not log_rings:
            no_log.append(name)
            continue
        for ann in knots:
            rings = polygons(ann)
            if not rings:
                continue
            checked += 1
            fraction, samples = overlap_fraction(rings, log_rings, args.grid)
            if fraction < args.threshold:
                flagged.append({
                    "image": name,
                    "case": Path(name).stem,
                    "annotation_id": ann["id"],
                    "on_log_fraction": round(fraction, 4),
                    "area_px": round(sum(shoelace(r) for r in rings), 1),
                    "confidence": ann.get("attributes", {}).get("confidence"),
                    "samples": samples,
                })

    flagged.sort(key=lambda f: (f["on_log_fraction"], -f["area_px"]))
    print(f"checked {checked} knot annotations in {len(knots_by_image)} images "
          f"({args.grid}x{args.grid} sampling grid)")
    if no_log:
        print(f"WARNING: {len(no_log)} images have knots but no primary_log: {no_log[:5]}")
    print(f"\n{len(flagged)} knots with less than {args.threshold:.0%} of their area on a log:")
    if flagged:
        print(f"  {'case':<8}{'ann id':>8}{'on log':>9}{'area px':>10}{'conf':>7}")
        for f in flagged:
            print(f"  {f['case']:<8}{f['annotation_id']:>8}{f['on_log_fraction']:>9.3f}"
                  f"{f['area_px']:>10.0f}{str(f['confidence']):>7}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"annotations": args.annotations, "threshold": args.threshold, "grid": args.grid,
             "checked": checked, "images_without_primary_log": no_log, "flagged": flagged},
            indent=2) + "\n")
        print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
