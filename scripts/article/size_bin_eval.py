"""Instance recall and outline quality of the seg pair ensembles per knot size.

Uses the cached predictions only: the instances ensemble_pairs.py stored for
each pair ensemble at 512 (pair_<i>/fold_<f>), scored at two operating points:

  - default: every stored instance (threshold 0, the stored operating point);
  - tuned: instances with score >= the threshold the confidence sweep chose for
    that run and fold by leave-one-fold-out (score_sweep.py's *_lofo.csv),
    the same procedure as the comparison table.

Per image, predictions are matched to the annotated knots once, with the
evaluation's own rule (greedy, IoU >= 0.5, match_instance_pairs), and the
matches are then split by the group of their knot — here its size bin
(knot_size_bins.py); confidence_bin_eval.py groups by annotator confidence:

  - recall: found knots / knots, pooled over the fold's images; the groups add
    up to the overall instance_recall_pooled, which is checked;
  - matched Dice / IoU: for every found knot, the Dice (IoU) between it and its
    matched prediction, averaged within the group — how well a found knot is
    outlined, independent of how many are found.

A knot's size is r = sqrt(area / pi) from the 512 annotations' area field, the
value the bins were made from. False positives that touch no knot at all are
counted separately, with their size (r of their pixel area) by the same bins.

Summary convention: per fold, mean over the folds per run, then mean ± sd
across the runs.

    uv run scripts/article/size_bin_eval.py
    uv run scripts/article/size_bin_eval.py -o article/size_bins/seg
"""
import argparse
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from mltools.common.instances import load_gt_instances, load_instances, make_instances
from mltools.common.metrics.gms import match_instance_pairs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = PROJECT_ROOT / "data/dataset/c04377f6/resized/short_side_512/annotations.json"
PRED_TEMPLATE = "results/ensembles/seg/pairs_exp_reference+exp_union/res_512/predictions/{run}/fold_{fold}"
PAIRS_JSON = PROJECT_ROOT / "results/ensembles/seg/pairs_exp_reference+exp_union.json"
CLASS_NAMES = ["wood_knot"]
CONFIDENCES = [0.2, 0.6, 1.0]  # dataset.eval_confidences
CONFIDENCE_TOLERANCE = 1e-3


def knot_annotations(annotations_path: Path) -> dict[str, list[dict]]:
    """{case: [annotation per knot]} in the order load_gt_instances builds its masks."""
    data = json.loads(annotations_path.read_text())
    knot_ids = {c["id"] for c in data["categories"] if c["name"] in CLASS_NAMES}
    stems = {im["id"]: Path(im["file_name"]).stem for im in data["images"]}
    out = {stem: [] for stem in stems.values()}
    for ann in data["annotations"]:
        if ann["category_id"] in knot_ids and any(
            abs(confidence(ann) - c) < CONFIDENCE_TOLERANCE for c in CONFIDENCES
        ):
            out[stems[ann["image_id"]]].append(ann)
    return out


def confidence(ann: dict) -> float:
    return float(ann.get("attributes", {}).get("confidence", 1.0))


def radius(ann: dict) -> float:
    return math.sqrt(ann["area"] / math.pi)


def bin_index(value: float, edges: list[float]) -> int:
    return sum(value >= e for e in edges)


def filter_by_score(pred, threshold: float):
    if threshold <= 0.0 or pred.num_instances == 0:
        return pred
    keep = pred.scores >= threshold
    return make_instances(pred.masks[keep], pred.labels[keep], pred.scores[keep], shape=pred.shape)


def score_fold(predictions, gt, knot_group, threshold, n_groups, fp_group=None) -> dict:
    """Counts for one fold at one threshold. knot_group: {case: [group per knot]};
    fp_group: (case, radius) -> group for false positives, or None to not split them."""
    groups = [{"knots": 0, "found": 0, "dice_sum": 0.0, "iou_sum": 0.0} for _ in range(n_groups)]
    fp = {"count": 0, "pixels": 0, "by_group": [0] * n_groups, "radii": [], "touching_unmatched": 0}
    per_image = {}  # case -> {group: [knots, found, dice_sum, iou_sum]}, for resampling images
    total_pred = 0
    for case, pred in predictions.items():
        pred = filter_by_score(pred, threshold)
        g = gt[case]
        knot = knot_group[case]
        if len(knot) != g.num_instances:
            raise RuntimeError(f"{case}: {len(knot)} annotated knots but {g.num_instances} ground-truth masks")
        total_pred += pred.num_instances
        pairs = match_instance_pairs(pred.masks, pred.scores, g.masks)
        matched_pred = {p for p, _ in pairs}
        image = per_image.setdefault(case, {})
        for t in range(g.num_instances):
            groups[knot[t]]["knots"] += 1
            image.setdefault(knot[t], [0, 0, 0.0, 0.0])[0] += 1
        for p, t in pairs:
            grp = groups[knot[t]]
            inter = int(np.logical_and(pred.masks[p], g.masks[t]).sum())
            p_area, g_area = int(pred.masks[p].sum()), int(g.masks[t].sum())
            dice, iou = 2 * inter / (p_area + g_area), inter / (p_area + g_area - inter)
            grp["found"] += 1
            grp["dice_sum"] += dice
            grp["iou_sum"] += iou
            entry = image[knot[t]]
            entry[1] += 1
            entry[2] += dice
            entry[3] += iou
        any_knot = g.masks.any(axis=0) if g.num_instances else np.zeros(g.shape, dtype=bool)
        for p in range(pred.num_instances):
            if p in matched_pred:
                continue
            if np.logical_and(pred.masks[p], any_knot).any():
                fp["touching_unmatched"] += 1
                continue
            area = int(pred.masks[p].sum())
            r = math.sqrt(area / math.pi)
            fp["count"] += 1
            fp["pixels"] += area
            fp["radii"].append(r)
            if fp_group is not None:
                fp["by_group"][fp_group(case, r)] += 1
    return {"groups": groups, "fp": fp, "predictions": total_pred, "per_image": per_image}


def mean_sd(values) -> str:
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return "-"
    return f"{st.fmean(values):.4f} ± {st.stdev(values):.4f}" if len(values) > 1 else f"{values[0]:.4f}"


def add_common_args(ap: argparse.ArgumentParser, default_output: str) -> None:
    ap.add_argument("--runs", nargs="+", default=[f"pair_{i}" for i in range(5)])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pred-template", default=PRED_TEMPLATE, help="with {run} and {fold}")
    ap.add_argument("--annotations", default=str(ANNOTATIONS))
    ap.add_argument("--lofo", default=str(PROJECT_ROOT / "article/conf_sweep/seg_lofo.csv"))
    ap.add_argument("--selected-on", default="instance_f1_pooled", help="selected_on rows of --lofo to use")
    ap.add_argument("--pairs-json", default=str(PAIRS_JSON),
                    help="stored results, to check the overall recall at the default threshold")
    ap.add_argument("-o", "--output", default=str(PROJECT_ROOT / default_output),
                    help="prefix: writes <prefix>_per_fold.csv, <prefix>_fp.csv, <prefix>_summary.txt")


def evaluate(
    args, names: list[str], group_of: Callable[[dict], int], title: str,
    group_desc: str, fp_group: Callable[[str, float], int] | None = None,
    fp_split_desc: str = "their size by the same bins", knots_per_fold: bool = True,
) -> None:
    """The analysis for any grouping of the knots: group_of(annotation) -> index into names.
    knots_per_fold=False is for groups that each sit in one fold (trees): the
    knots column then gives each group's total, and a group's value is that
    of its own fold (folds without the group have no value and are skipped)."""
    n_groups = len(names)
    annotations = knot_annotations(Path(args.annotations))
    knot_group = {case: [group_of(a) for a in anns] for case, anns in annotations.items()}
    group_radii = [[] for _ in names]
    for anns in annotations.values():
        for a in anns:
            group_radii[group_of(a)].append(radius(a))

    tuned = {}
    for row in csv.DictReader(open(args.lofo, newline="")):
        if row["selected_on"] == args.selected_on:
            tuned[(row["run"], int(row["fold"]))] = (float(row["threshold"]), float(row["instance_recall_pooled"]))
    stored = json.loads(Path(args.pairs_json).read_text())["per_resolution"]["512"]

    results = {}  # (setting, run, fold) -> score_fold output + threshold, reference recall
    for fold in range(args.folds):
        pred_dirs = {run: PROJECT_ROOT / args.pred_template.format(run=run, fold=fold) for run in args.runs}
        cases = sorted(p.stem for p in next(iter(pred_dirs.values())).glob("*.npz"))
        gt = load_gt_instances(args.annotations, CLASS_NAMES, CONFIDENCES, case_ids=cases)
        for run in args.runs:
            predictions = {c: load_instances(pred_dirs[run] / f"{c}.npz") for c in cases}
            for setting in ("default", "tuned"):
                if setting == "default":
                    threshold = 0.0
                    reference = stored[str(fold)][run]["instance_recall_pooled"]
                else:
                    threshold, reference = tuned[(run, fold)]
                out = score_fold(predictions, gt, knot_group, threshold, n_groups, fp_group)
                out.update(threshold=threshold, reference_recall=reference)
                results[(setting, run, fold)] = out
            print(f"fold {fold} {run}: done")

    # Check: the groups' found knots add up to the evaluation's instance recall.
    worst = 0.0
    for out in results.values():
        knots = sum(g["knots"] for g in out["groups"])
        found = sum(g["found"] for g in out["groups"])
        worst = max(worst, abs(found / knots - out["reference_recall"]))
    check = f"overall recall vs the evaluation's instance_recall_pooled: max |difference| {worst:.2e}"

    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{prefix}_per_fold.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["setting", "run", "fold", "threshold", "group", "knots", "found",
                    "recall", "matched_dice", "matched_iou"])
        for (setting, run, fold), out in sorted(results.items()):
            for i, g in enumerate(out["groups"]):
                found = g["found"]
                w.writerow([setting, run, fold, out["threshold"], names[i], g["knots"], found,
                            found / g["knots"] if g["knots"] else "",
                            g["dice_sum"] / found if found else "", g["iou_sum"] / found if found else ""])
    # Per image and group, for resampling images (bootstrap_groups.py). Every
    # test image gets a row, knot-free ones with group "" and zero counts, so a
    # resample draws from all of a fold's images.
    with open(f"{prefix}_per_image.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["setting", "run", "fold", "threshold", "case", "group", "knots", "found", "dice_sum", "iou_sum"])
        for (setting, run, fold), out in sorted(results.items()):
            for case, image in sorted(out["per_image"].items()):
                if not image:
                    w.writerow([setting, run, fold, out["threshold"], case, "", 0, 0, 0.0, 0.0])
                for i, (knots, found, dice, iou) in sorted(image.items()):
                    w.writerow([setting, run, fold, out["threshold"], case, names[i], knots, found, dice, iou])
    with open(f"{prefix}_fp.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["setting", "run", "fold", "threshold", "predictions", "fp_no_knot", "fp_pixels",
                    *((f"fp_{n}" for n in names) if fp_group else ()), "fp_radius_median",
                    "unmatched_touching_knot"])
        for (setting, run, fold), out in sorted(results.items()):
            fp = out["fp"]
            w.writerow([setting, run, fold, out["threshold"], out["predictions"], fp["count"], fp["pixels"],
                        *(fp["by_group"] if fp_group else ()),
                        st.median(fp["radii"]) if fp["radii"] else "", fp["touching_unmatched"]])

    def summarize(setting, value):
        """per fold value -> mean over folds per run -> mean ± sd over runs"""
        per_run = []
        for run in args.runs:
            folds = [value(results[(setting, run, f)]) for f in range(args.folds)]
            folds = [v for v in folds if not math.isnan(v)]
            per_run.append(st.fmean(folds) if folds else math.nan)
        return mean_sd(per_run)

    def ratio(a, b):
        return a / b if b else math.nan

    fold_groups = [results[("default", args.runs[0], f)]["groups"] for f in range(args.folds)]
    lines = [
        f"seg pair ensembles ({len(args.runs)} runs x {args.folds} folds, cached predictions at 512) "
        f"{title}; each value: per fold, mean over folds per run, mean ± sd across runs",
        group_desc,
        ("knots per group and fold: " + "; ".join(
            f"{n} " + "/".join(str(fold_groups[f][i]["knots"]) for f in range(args.folds))
            for i, n in enumerate(names)))
        if knots_per_fold else
        ("test fold per group: " + ", ".join(
            f"{n} {[f for f in range(args.folds) if fold_groups[f][i]['knots']] or '-'}"
            for i, n in enumerate(names))),
        "median knot radius r = sqrt(area / pi) per group, px: " + ", ".join(
            f"{n} {st.median(r):.1f}" for n, r in zip(names, group_radii) if r),
        check,
    ]
    w0 = max(8, *(len(n) + 2 for n in names))
    for setting in ("default", "tuned"):
        thresholds = [results[(setting, r, f)]["threshold"] for r in args.runs for f in range(args.folds)]
        lines += ["", f"=== {setting} threshold "
                  + ("(every stored instance)" if setting == "default"
                     else f"(leave-one-fold-out on {args.selected_on}; chosen "
                          f"{min(thresholds):g}-{max(thresholds):g}, mean {st.fmean(thresholds):.3f})"),
                  f"{'group':<{w0}}{'knots/fold' if knots_per_fold else 'knots':>11}"
                  f"{'recall':>18}{'matched Dice':>18}{'matched IoU':>18}"]
        rows = [(n, lambda o, i=i: [o["groups"][i]]) for i, n in enumerate(names)]
        rows.append(("all", lambda o: o["groups"]))
        for name, pick in rows:
            per_fold_knots = [sum(g["knots"] for g in pick({"groups": k})) for k in fold_groups]
            knots = st.fmean(per_fold_knots) if knots_per_fold else sum(per_fold_knots)
            recall = summarize(setting, lambda o: ratio(sum(g["found"] for g in pick(o)), sum(g["knots"] for g in pick(o))))
            dice = summarize(setting, lambda o: ratio(sum(g["dice_sum"] for g in pick(o)), sum(g["found"] for g in pick(o))))
            iou = summarize(setting, lambda o: ratio(sum(g["iou_sum"] for g in pick(o)), sum(g["found"] for g in pick(o))))
            lines.append(f"{name:<{w0}}{knots:>11.1f}{recall:>18}{dice:>18}{iou:>18}")
        lines.append(
            f"false positives touching no knot, per fold: {summarize(setting, lambda o: o['fp']['count'])}"
            f" (of {summarize(setting, lambda o: o['predictions'])} predictions); "
            f"unmatched but touching a knot: {summarize(setting, lambda o: o['fp']['touching_unmatched'])}"
        )
        if fp_group is not None and knots_per_fold:
            lines.append(f"  {fp_split_desc} (share of them): " + ", ".join(
                f"{n} {summarize(setting, lambda o, i=i: ratio(o['fp']['by_group'][i], o['fp']['count']))}"
                for i, n in enumerate(names)))
        elif fp_group is not None:
            # Each group sits in one fold: its count there, other folds skipped.
            lines.append(f"  {fp_split_desc} (count): " + ", ".join(
                f"{n} {summarize(setting, lambda o, i=i: o['fp']['by_group'][i] if o['groups'][i]['knots'] else math.nan)}"
                for i, n in enumerate(names)))
    summary = "\n".join(lines) + "\n"
    Path(f"{prefix}_summary.txt").write_text(summary)
    print("\n" + summary, end="")
    print(f"\nSaved {prefix}_per_fold.csv, {prefix}_per_image.csv, {prefix}_fp.csv and {prefix}_summary.txt")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bins", default=str(PROJECT_ROOT / "article/knot_size_bins.json"))
    add_common_args(ap, "article/size_bins/seg")
    args = ap.parse_args()

    spec = json.loads(Path(args.bins).read_text())
    edges, names = spec["edges"], spec["names"]
    bounds = [0.0, *edges, math.inf]
    evaluate(
        args, names, lambda a: bin_index(radius(a), edges), "by knot size",
        "bins on r = sqrt(area / pi), px: " + ", ".join(
            f"{n} [{lo:g}, {hi:g})" for n, lo, hi in zip(names, bounds[:-1], bounds[1:])),
        fp_group=lambda case, r: bin_index(r, edges),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
