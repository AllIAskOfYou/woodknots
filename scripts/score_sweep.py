"""Sweep the instance confidence threshold on predictions that already exist.

Every metric but mAP is scored at one operating point: an instance counts if it
is in the stored prediction at all. When a change makes a model detect more at
lower confidence, mAP rises (it integrates over the threshold) while the
fixed-point metrics fall — which is what the blob-loss runs show, gms down 0.06
with mAP50 up 0.03. This sweep separates the two: it drops instances scoring
below a threshold and re-scores, so a model can be compared at its own best
operating point instead of at whichever one prediction happened to use.

No inference is involved. Predictions are loaded once per fold and re-scored
through mltools.common.evaluate.score_instances, the same code path a normal
evaluation uses.

What it cannot do: the stored format is instances (masks, labels, scores), with
no probability map, so raising the threshold removes whole components and never
changes a mask's extent. For semantic methods a component's score is its mean
probability, which is above 0.5 by construction, so thresholds below 0.5 are
no-ops for them; YOLO's instance scores start at its conf gate.

    uv run scripts/score_sweep.py seg exp_reference exp_blob_0
    uv run scripts/score_sweep.py seg exp_blob_0 --metric gms -o article/score_sweep.json
"""
import argparse
import json
import logging
import statistics as st
import sys
from pathlib import Path

from hydra import initialize_config_dir

from mltools.common.evaluate import score_instances
from mltools.common.instances import load_gt_instances, load_instances, make_instances
from mltools.common.setup import compose_with_hydra_config, get_annotations_path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)

DEFAULT_THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

# Short column labels: the real names do not fit side by side, and a header
# that runs its names together is worse than a legend under the table.
SHORT = {
    "dice_fg_pooled": "dice",
    "recall_fg_pooled": "px_rec",
    "instance_precision_pooled": "i_prec",
    "instance_recall_pooled": "i_rec",
    "instance_f1_pooled": "i_F1",
    "instance_f1_knots": "i_F1_kn",
    "dice_fg_knots": "dice_kn",
    "mAP50": "mAP50",
    "instances": "n_inst",
    "clean_empty": "clean",
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def filter_by_score(instances, threshold: float):
    """The prediction as it would have been with a stricter confidence gate."""
    if threshold <= 0.0 or instances.num_instances == 0:
        return instances
    keep = instances.scores >= threshold
    return make_instances(
        instances.masks[keep], instances.labels[keep], instances.scores[keep], shape=instances.shape
    )


def sweep_fold(cfg, pred_dir: Path, thresholds: list[float]) -> dict[float, dict]:
    pred_paths = sorted(pred_dir.glob("*.npz"))
    if not pred_paths:
        raise RuntimeError(f"no predictions in {pred_dir}")
    class_names = list(cfg.dataset.class_names)
    gt = load_gt_instances(
        get_annotations_path(cfg.dataset),
        class_names,
        list(cfg.dataset.eval_confidences),
        case_ids=[p.stem for p in pred_paths],
    )
    predictions = {p.stem: load_instances(p) for p in pred_paths}
    scores = sorted(s for pred in predictions.values() for s in pred.scores.tolist())
    logging.info(
        f"  {len(predictions)} cases, {len(scores)} instances, "
        f"score range [{scores[0]:.3f}, {scores[-1]:.3f}]" if scores else "  no instances"
    )

    # Knot-free cases score a perfect 1.0 on gms and dice_fg for predicting
    # nothing (a class absent from both sides is scored 1.0), so a threshold
    # that merely silences them raises the reported mean without detecting
    # anything. Split the two populations to keep that visible.
    empty_cases = {case for case, instances in gt.items() if instances.num_instances == 0}
    knot_cases = set(predictions) - empty_cases

    out = {}
    for threshold in thresholds:
        filtered = {case: filter_by_score(pred, threshold) for case, pred in predictions.items()}
        result = score_instances(filtered, gt, len(class_names))
        per_case = result["per_case"]
        metrics = result["metrics"]
        metrics["instances"] = sum(p.num_instances for p in filtered.values())
        for name in ("instance_f1", "dice_fg"):
            if knot_cases and name in next(iter(per_case.values())):
                metrics[f"{name}_knots"] = sum(per_case[c][name] for c in knot_cases) / len(knot_cases)
        metrics["clean_empty"] = sum(
            1 for case in empty_cases if filtered[case].num_instances == 0
        )
        metrics["empty_cases"] = len(empty_cases)
        out[threshold] = metrics
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("framework", help="seg, yolo or nnunet")
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    ap.add_argument("--metric", default="instance_f1_pooled", help="metric whose best threshold is highlighted")
    ap.add_argument("--folds", type=int, default=None, help="default: dataset.folds.k")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    config_dir = str(PROJECT_ROOT / args.framework / "conf")
    report = {}
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for experiment in args.experiments:
            probe = compose_with_hydra_config([f"experiment={experiment}", "experiment.fold=0"])
            num_folds = args.folds if args.folds is not None else probe.dataset.folds.k
            per_fold = {}
            for fold in range(num_folds):
                cfg = compose_with_hydra_config(
                    [f"experiment={experiment}", f"experiment.fold={fold}"]
                )
                pred_dir = Path(cfg.paths.run_dir) / "predictions"
                if not pred_dir.exists():
                    logging.warning(f"{experiment} fold {fold}: no {pred_dir}, skipping")
                    continue
                logging.info(f"{experiment} fold {fold}")
                per_fold[fold] = sweep_fold(cfg, pred_dir, args.thresholds)
            if per_fold:
                report[experiment] = per_fold

    if not report:
        print("nothing swept")
        return 1

    counts = {"instances", "clean_empty", "empty_cases"}
    metrics = ["dice_fg_pooled", "recall_fg_pooled", "instance_precision_pooled",
               "instance_recall_pooled", "instance_f1_pooled", "instance_f1_knots",
               "dice_fg_knots", "mAP50", "instances", "clean_empty"]

    def mean_over(per_fold, folds, threshold, metric):
        return st.fmean([per_fold[f][threshold][metric] for f in folds])

    for experiment, per_fold in report.items():
        folds = sorted(per_fold)
        empty = st.fmean([per_fold[f][args.thresholds[0]]["empty_cases"] for f in folds])
        print(f"\n{args.framework}/{experiment} — instance score sweep over {len(folds)} folds "
              f"({empty:.0f} knot-free cases per fold on average)")
        width = max(9, max(len(SHORT.get(m, m)) for m in metrics) + 2)
        header = f"{'thresh':>7}" + "".join(f"{SHORT.get(m, m):>{width}}" for m in metrics)
        print(header)
        print("-" * len(header))
        for threshold in args.thresholds:
            cells = [
                f"{mean_over(per_fold, folds, threshold, m):.1f}" if m in counts
                else f"{mean_over(per_fold, folds, threshold, m):.4f}"
                for m in metrics
            ]
            print(f"{threshold:>7.2f}" + "".join(f"{c:>{width}}" for c in cells))
        print("  " + ",  ".join(f"{SHORT.get(m, m)} = {m}" for m in metrics if SHORT.get(m, m) != m))

        stored = mean_over(per_fold, folds, args.thresholds[0], args.metric)
        # Oracle: one threshold, chosen by looking at the folds it is scored on.
        oracle_t = max(args.thresholds, key=lambda t: mean_over(per_fold, folds, t, args.metric))
        oracle = mean_over(per_fold, folds, oracle_t, args.metric)
        # Leave-one-fold-out: the threshold for fold f is chosen on the other
        # folds only, so the reported value is not tuned on what it scores.
        chosen, values = {}, []
        for f in folds:
            others = [g for g in folds if g != f]
            chosen[f] = max(args.thresholds, key=lambda t: mean_over(per_fold, others, t, args.metric))
            values.append(per_fold[f][chosen[f]][args.metric])
        print(f"  {args.metric} as stored (threshold {args.thresholds[0]:.2f}): {stored:.4f}")
        print(f"  {args.metric} oracle threshold {oracle_t:.2f}: {oracle:.4f} ({oracle - stored:+.4f}) "
              f"— tuned on the test folds, an upper bound")
        print(f"  {args.metric} leave-one-fold-out: {st.fmean(values):.4f} ({st.fmean(values) - stored:+.4f}) "
              f"— thresholds per fold: " + ", ".join(f"{f}:{chosen[f]:.2f}" for f in folds))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"framework": args.framework, "thresholds": args.thresholds,
             "runs": {e: {str(f): {str(t): m for t, m in ts.items()} for f, ts in pf.items()}
                      for e, pf in report.items()}},
            indent=2) + "\n")
        print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
