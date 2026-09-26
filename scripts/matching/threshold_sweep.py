"""Sweep the matching merge threshold and integrate the resulting curves.

The merge threshold is a free parameter, so any single-threshold precision /
recall pair is one arbitrary point on a curve. This sweeps it and reports
threshold-free summaries instead: ROC AUC and average precision over all pairs
of identity-annotated knots.

The sweep is exact rather than gridded. The criterion |A and B| / min(|A|,|B|)
> tau is monotone in tau and the candidate pairs do not depend on it, so the
only thresholds that change anything are the candidates' own overlap values,
and the predicted-positive sets at successive values are nested. Descending
through those values therefore traces the whole curve while adding pairs to a
single union-find.

Two caveats the numbers carry:

  * The curve does not reach TPR = 1. Volumes sharing no voxel are never
    candidates, so at tau -> 0 some same-knot pairs are still only reachable
    transitively. tpr_max reports where it stops; roc_auc extends the curve to
    (1, 1) along the chance diagonal, and average_precision is reported both
    over the achieved recall range and over [0, 1] counting unreachable recall
    as zero precision.
  * Pairs from the same image are included as negatives (one image never shows
    a knot twice), even though the matcher never proposes them — they can only
    become false positives transitively, which is the failure mode worth
    exposing.

Run inside the container (needs the position maps, not a GPU):

    uv run scripts/matching/threshold_sweep.py
    uv run scripts/matching/threshold_sweep.py --voxel-size 0.005 -o article/matching_sweep.json
"""
import argparse
import itertools
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from mltools.matching.utils import (
    confusion_matrix,
    construct_voxel_grid,
    correct_annotations,
    parse_coco_annotations,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)

DEFAULTS = {
    "annotations": "data/matching/annotations/instances_default.json",
    "projections": "data/matching/resized_1024_3d_v3_distortion",
    "corrections": "data/matching/annotations/corrections.json",
    "voxel_size": 0.005,
}


def pair_counts(structures: list[list[int]], knot_of: dict[int, int]) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) over all pairs of annotations, counted from the grouping
    instead of enumerating pairs: for each structure, every pair inside it is
    predicted positive, and those sharing a knot identity are the true ones."""
    total = len(knot_of) * (len(knot_of) - 1) // 2
    positives = sum(n * (n - 1) // 2 for n in Counter(knot_of.values()).values())

    predicted = true_positive = 0
    for members in structures:
        members = [m for m in members if m in knot_of]
        predicted += len(members) * (len(members) - 1) // 2
        true_positive += sum(
            n * (n - 1) // 2 for n in Counter(knot_of[m] for m in members).values()
        )
    false_positive = predicted - true_positive
    false_negative = positives - true_positive
    return true_positive, false_positive, false_negative, total - predicted - false_negative


def trapezoid(xs: list[float], ys: list[float]) -> float:
    return sum((xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) / 2 for i in range(len(xs) - 1))


def sweep(grid, knot_of: dict[int, int]) -> list[dict]:
    """One point per distinct overlap value, from the strictest threshold down."""
    overlaps = grid.candidate_overlaps()
    if not overlaps:
        raise RuntimeError("no candidate pairs — check the projections and voxel size")
    by_overlap = sorted(overlaps, key=lambda t: -t[2])
    values = sorted({overlap for _, _, overlap in overlaps}, reverse=True)
    logging.info(f"{len(overlaps)} candidate pairs, {len(values)} distinct overlap values")

    points, pairs, cursor = [], [], 0
    # Strictest first: nothing merged at all, i.e. every pair predicted negative.
    for value in [math.inf, *values]:
        while cursor < len(by_overlap) and by_overlap[cursor][2] >= value:
            i, j, _ = by_overlap[cursor]
            pairs.append((i, j))
            cursor += 1
        tp, fp, fn, tn = pair_counts(grid.structures_from_pairs(pairs), knot_of)
        points.append({
            # The smallest overlap still merged; pass any merge_threshold just
            # below it to reproduce this point (the criterion is a strict >).
            "min_overlap": None if value == math.inf else value,
            "merge_threshold": None if value == math.inf else math.nextafter(value, 0.0),
            "merged_pairs": len(pairs),
            "structures": sum(1 for s in grid.structures_from_pairs(pairs) if len(s) > 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "tpr": tp / (tp + fn) if tp + fn else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        })
    return points


def summarize(points: list[dict]) -> dict:
    roc = sorted({(p["fpr"], p["tpr"]) for p in points})
    fprs = [0.0] + [f for f, _ in roc] + [1.0]
    tprs = [0.0] + [t for _, t in roc] + [1.0]  # chance extension to (1, 1)

    pr = sorted({(p["recall"], p["precision"]) for p in points})
    recalls = [r for r, _ in pr]
    precisions = [p for _, p in pr]
    achieved = trapezoid(recalls, precisions)
    span = recalls[-1] - recalls[0]
    return {
        "roc_auc": trapezoid(fprs, tprs),
        "average_precision_achieved": achieved / span if span else float("nan"),
        "average_precision_full": achieved,  # unreachable recall contributes 0
        "tpr_max": max(p["tpr"] for p in points),
        "fpr_at_tpr_max": max(points, key=lambda p: p["tpr"])["fpr"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default=DEFAULTS["annotations"])
    ap.add_argument("--projections", default=DEFAULTS["projections"])
    ap.add_argument("--corrections", default=DEFAULTS["corrections"])
    ap.add_argument("--voxel-size", type=float, default=DEFAULTS["voxel_size"])
    ap.add_argument("--reference-threshold", type=float, default=0.5,
                    help="threshold whose operating point is printed for reference")
    ap.add_argument("-o", "--output", default=None, help="write the full curve as JSON")
    args = ap.parse_args()

    annotations = parse_coco_annotations(args.annotations)
    if Path(args.corrections).exists():
        correct_annotations(annotations, json.loads(Path(args.corrections).read_text()))
    grid = construct_voxel_grid(annotations, args.projections, args.voxel_size)
    # construct_voxel_grid drops annotations whose image has no projection.
    knot_of = {ann_id: ann["knot_id"] for ann_id, ann in annotations.items()}
    logging.info(f"{len(knot_of)} annotations, {len(set(knot_of.values()))} knot identities")

    points = sweep(grid, knot_of)
    summary = summarize(points)

    # Independent check of the fast pair counting, on one threshold.
    structures = grid.find_structures(merge_threshold=args.reference_threshold)
    expected = confusion_matrix(annotations, structures)
    got = pair_counts(structures, knot_of)
    if tuple(expected) != tuple(got):
        logging.error(f"pair counting disagrees with confusion_matrix: {got} vs {expected}")
        return 1
    logging.info(f"pair counting agrees with confusion_matrix at tau={args.reference_threshold}")

    print(f"\nmatching threshold sweep — voxel_size={args.voxel_size}, "
          f"{len(knot_of)} annotations, {points[-1]['merged_pairs']} candidate pairs")
    header = f"{'min overlap':>12}{'merged':>8}{'tp':>7}{'fp':>7}{'fn':>7}{'precision':>11}{'recall':>8}{'fpr':>9}"
    print(header)
    print("-" * len(header))
    for p in points:
        overlap = "none" if p["min_overlap"] is None else f"{p['min_overlap']:.4f}"
        print(f"{overlap:>12}{p['merged_pairs']:>8}{p['tp']:>7}{p['fp']:>7}{p['fn']:>7}"
              f"{p['precision']:>11.4f}{p['recall']:>8.4f}{p['fpr']:>9.5f}")

    tp, fp, fn, tn = expected
    print(f"\nreference operating point (merge_threshold={args.reference_threshold}): "
          f"precision={tp/(tp+fp) if tp+fp else 1:.4f} recall={tp/(tp+fn) if tp+fn else 0:.4f}")
    print(f"ROC AUC (chance-extended to (1,1)):        {summary['roc_auc']:.4f}")
    print(f"average precision over achieved recall:    {summary['average_precision_achieved']:.4f}")
    print(f"average precision over [0, 1]:             {summary['average_precision_full']:.4f}")
    print(f"maximum reachable recall (tpr_max):        {summary['tpr_max']:.4f} "
          f"at fpr={summary['fpr_at_tpr_max']:.5f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "voxel_size": args.voxel_size,
            "annotations": len(knot_of),
            "knot_identities": len(set(knot_of.values())),
            "summary": summary,
            "curve": points,
        }, indent=2) + "\n")
        print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
