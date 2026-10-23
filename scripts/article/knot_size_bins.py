#!/usr/bin/env python3
"""Size bins for wood knots with about equal counts.

A knot's size is the radius of the circle with the same area,
r = sqrt(area / pi), in pixels of the resized dataset (shorter side 512, the
images the seg framework trains and is scored on). Areas are the COCO area
field of the 512 annotations. Every knot annotation counts, at every
confidence, since all of them are evaluation ground truth; the 5 test folds
cover every image, so these are exactly the knots the evaluation scores.

The edges are quantiles of r, so each bin holds about 1/n of the knots. They are
also rounded to --round px for reporting, with the counts that rounding gives.
Bin i is [edge_i, edge_i+1): a knot exactly on an edge goes to the larger bin.
Standard library only.

    python3 scripts/article/knot_size_bins.py
    python3 scripts/article/knot_size_bins.py --bins 3 --round 0.5 -o article/knot_size_bins.json
"""
import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data/dataset/c04377f6/resized/short_side_512/annotations.json"
NAMES = {3: ["small", "medium", "large"]}


def bin_index(radius: float, edges: list[float]) -> int:
    """Index of the bin [edges[i-1], edges[i]) holding radius; edges are the inner edges."""
    return sum(radius >= e for e in edges)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    ap.add_argument("--bins", type=int, default=3)
    ap.add_argument("--round", type=float, default=0.5, help="rounding step for the reported edges, px")
    ap.add_argument("--tiny", type=float, default=3.0,
                    help="report knots with an equivalent radius below this, px (likely annotation slips)")
    ap.add_argument("-o", "--output", default=str(PROJECT_ROOT / "article/knot_size_bins.json"))
    args = ap.parse_args()

    data = json.loads(Path(args.annotations).read_text())
    knot_ids = {c["id"] for c in data["categories"] if c["name"] == "wood_knot"}
    knots = [a for a in data["annotations"] if a["category_id"] in knot_ids]
    radii = sorted(math.sqrt(a["area"] / math.pi) for a in knots)
    n = len(radii)
    names = NAMES.get(args.bins, [f"bin_{i}" for i in range(args.bins)])

    # Inner edges at the 1/n, 2/n, ... quantiles (linear interpolation).
    exact = st.quantiles(radii, n=args.bins, method="inclusive")
    rounded = [round(e / args.round) * args.round for e in exact]

    def counts(edges):
        c = [0] * args.bins
        for r in radii:
            c[bin_index(r, edges)] += 1
        return c

    def ranges(edges):
        bounds = [radii[0], *edges, radii[-1]]
        return [(bounds[i], bounds[i + 1]) for i in range(args.bins)]

    lines = [
        f"{n} knots in {len(data['images'])} images ({args.annotations})",
        f"equivalent radius r = sqrt(area / pi), px at short side 512: "
        f"min {radii[0]:.2f}, median {st.median(radii):.2f}, max {radii[-1]:.2f}",
        "",
        f"{'bin':<8}{'r range (exact edges)':>26}{'knots':>8}{'r range (rounded edges)':>28}{'knots':>8}",
    ]
    for name, (lo, hi), c, (rlo, rhi), rc in zip(
        names, ranges(exact), counts(exact), ranges(rounded), counts(rounded)
    ):
        lines.append(f"{name:<8}{f'{lo:.2f} - {hi:.2f}':>26}{c:>8}{f'{rlo:.2f} - {rhi:.2f}':>28}{rc:>8}")
    lines.append(f"\nexact inner edges: {', '.join(f'{e:.3f}' for e in exact)} px; "
                 f"rounded to {args.round}: {', '.join(f'{e:g}' for e in rounded)} px")
    lines.append("equivalent areas of the rounded edges: "
                 + ", ".join(f"{math.pi * e * e:.0f} px²" for e in rounded))
    tiny = sorted(
        ((math.sqrt(a["area"] / math.pi), a) for a in knots if math.sqrt(a["area"] / math.pi) < args.tiny),
        key=lambda t: t[0],
    )
    images = {im["id"]: Path(im["file_name"]).stem for im in data["images"]}
    lines.append(f"\n{len(tiny)} knots below r = {args.tiny:g} px (area < {math.pi * args.tiny ** 2:.0f} px²):")
    lines += [
        f"  case {images[a['image_id']]}  annotation {a['id']}  r {r:.2f} px  area {a['area']:.1f} px²  "
        f"confidence {a.get('attributes', {}).get('confidence')}"
        for r, a in tiny
    ]
    print("\n".join(lines))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "annotations": str(args.annotations),
        "definition": "r = sqrt(area / pi) in px at short side 512; bin i = [edge_i, edge_i+1)",
        "names": names,
        "edges": rounded,
        "edges_exact": exact,
        "counts": counts(rounded),
        "counts_exact": counts(exact),
        "n_knots": n,
        "radius_min": radii[0],
        "radius_max": radii[-1],
        "tiny_radius": args.tiny,
        "tiny": [{"case": images[a["image_id"]], "annotation_id": a["id"], "radius": r, "area": a["area"]}
                 for r, a in tiny],
    }, indent=2) + "\n")
    print(f"\nSaved to {out} (\"edges\" are the rounded ones, used by size_bin_eval.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
