"""Instance recall and outline quality of the seg pair ensembles per annotator
confidence.

The same analysis as size_bin_eval.py — cached predictions, default and tuned
(leave-one-fold-out) threshold, one matching per image split by group — with
the knots grouped by the confidence the annotator gave them (0.2, 0.6, 1.0)
instead of by size. False positives have no confidence, so they are counted
but not split.

Confidence and size are probably related (small or faint knots get a low
confidence), so the summary lists the median knot radius per group: a lower
recall at low confidence may partly be a size effect. size_bin_eval.py gives
the other side.

    uv run scripts/article/confidence_bin_eval.py
    uv run scripts/article/confidence_bin_eval.py -o article/confidence_bins/seg
"""
import argparse
import sys

from size_bin_eval import CONFIDENCE_TOLERANCE, CONFIDENCES, add_common_args, confidence, evaluate

NAMES = {0.2: "low (0.2)", 0.6: "medium (0.6)", 1.0: "high (1.0)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, "article/confidence_bins/seg")
    args = ap.parse_args()

    levels = sorted(CONFIDENCES)

    def group_of(ann: dict) -> int:
        value = confidence(ann)
        return next(i for i, c in enumerate(levels) if abs(value - c) < CONFIDENCE_TOLERANCE)

    evaluate(
        args, [NAMES.get(c, f"{c:g}") for c in levels], group_of, "by annotator confidence",
        "groups: the annotation's confidence attribute (" + ", ".join(f"{c:g}" for c in levels) + ")",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
