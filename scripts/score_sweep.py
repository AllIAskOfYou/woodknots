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
    uv run scripts/score_sweep.py nnunet exp_res_1_{0..4} -o article/conf_sweep/nnunet.json

Seed runs (<config>_<i>, scripts/run_variance.py) work as experiments. Runs that
are not experiments — an ensemble, say — are given with --pred-template, whose
{run} and {fold} are filled in for each run and fold; --gt-experiment names the
config whose ground truth (resolution) they are scored against:

    uv run scripts/score_sweep.py seg pair_{0..4} --gt-experiment exp_reference \
        --pred-template "results/ensembles/seg/pairs_exp_reference+exp_union/res_512/predictions/{run}/fold_{fold}"

With -o out.json it also writes out_sweep.csv (one row per run, fold and
threshold) and out_lofo.csv (one row per run, fold and selection metric: the
threshold chosen on that run's other folds and every metric at it on the
held-out fold).
"""
import argparse
import csv
import json
import logging
import statistics as st
import sys
from pathlib import Path

from hydra import initialize_config_dir

from mltools.common.evaluate import case_contribution, combine_contributions
from mltools.common.instances import load_gt_instances, load_instances, make_instances
from mltools.common.setup import (
    compose_with_hydra_config,
    experiment_overrides,
    get_annotations_path,
    get_experiment_names,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)

# 0.0 keeps every stored instance: the operating point the run was evaluated at.
DEFAULT_THRESHOLDS = [0.0, *(round(0.25 + 0.05 * i, 2) for i in range(15)), 0.99]

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

    # A case's prediction only changes when one of its instances crosses the
    # threshold, so its contribution is computed once per set of kept
    # instances and reused: identical to re-scoring every case at every
    # threshold, at a fraction of the cost.
    cache: dict[tuple, dict] = {}
    out = {}
    for threshold in thresholds:
        contributions, kept = {}, 0
        for case, pred in predictions.items():
            keep = tuple(bool(k) for k in (pred.scores >= threshold)) if threshold > 0.0 else None
            key = (case, keep)
            if key not in cache:
                cache[key] = case_contribution(filter_by_score(pred, threshold), gt[case], len(class_names))
            contributions[case] = cache[key]
            kept += pred.num_instances if keep is None else sum(keep)
        result = combine_contributions(contributions, len(class_names))
        per_case = result["per_case"]
        metrics = result["metrics"]
        metrics["instances"] = kept
        for name in ("instance_f1", "dice_fg"):
            if knot_cases and name in next(iter(per_case.values())):
                metrics[f"{name}_knots"] = sum(per_case[c][name] for c in knot_cases) / len(knot_cases)
        metrics["clean_empty"] = sum(1 for case in empty_cases if contributions[case]["pred_empty"])
        metrics["empty_cases"] = len(empty_cases)
        out[threshold] = metrics
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("framework", help="seg, yolo or nnunet")
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    ap.add_argument("--metric", nargs="+", default=["instance_f1_pooled"],
                    help="metric(s) a threshold is selected on (leave-one-fold-out); the first is printed")
    ap.add_argument("--pred-template", default=None,
                    help="prediction dir per run, with {run} and {fold}; the experiments are then run labels")
    ap.add_argument("--gt-experiment", default=None,
                    help="with --pred-template: the config whose ground truth the runs are scored against")
    ap.add_argument("--folds", type=int, default=None, help="default: dataset.folds.k")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    if args.pred_template and not args.gt_experiment:
        raise SystemExit("--pred-template needs --gt-experiment")
    config_dir = str(PROJECT_ROOT / args.framework / "conf")
    config_names = get_experiment_names(PROJECT_ROOT / args.framework)

    def overrides_for(name):
        found = experiment_overrides(name, config_names)
        if found is None:
            raise SystemExit(f"{args.framework}: no experiment config for {name}")
        return found

    report = {}
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for experiment in args.experiments:
            base = overrides_for(args.gt_experiment if args.pred_template else experiment)
            probe = compose_with_hydra_config([*base, "experiment.fold=0"])
            num_folds = args.folds if args.folds is not None else probe.dataset.folds.k
            per_fold = {}
            for fold in range(num_folds):
                cfg = compose_with_hydra_config([*base, f"experiment.fold={fold}"])
                pred_dir = (
                    PROJECT_ROOT / args.pred_template.format(run=experiment, fold=fold)
                    if args.pred_template
                    else Path(cfg.paths.run_dir) / "predictions"
                )
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

        metric = args.metric[0]
        stored = mean_over(per_fold, folds, args.thresholds[0], metric)
        # Oracle: one threshold, chosen by looking at the folds it is scored on.
        oracle_t = max(args.thresholds, key=lambda t: mean_over(per_fold, folds, t, metric))
        oracle = mean_over(per_fold, folds, oracle_t, metric)
        # Leave-one-fold-out: the threshold for fold f is chosen on the other
        # folds only, so the reported value is not tuned on what it scores.
        chosen, values = {}, []
        for f in folds:
            others = [g for g in folds if g != f]
            chosen[f] = max(args.thresholds, key=lambda t: mean_over(per_fold, others, t, metric))
            values.append(per_fold[f][chosen[f]][metric])
        print(f"  {metric} as stored (threshold {args.thresholds[0]:.2f}): {stored:.4f}")
        print(f"  {metric} oracle threshold {oracle_t:.2f}: {oracle:.4f} ({oracle - stored:+.4f}) "
              f"— tuned on the test folds, an upper bound")
        print(f"  {metric} leave-one-fold-out: {st.fmean(values):.4f} ({st.fmean(values) - stored:+.4f}) "
              f"— thresholds per fold: " + ", ".join(f"{f}:{chosen[f]:.2f}" for f in folds))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Leave-one-fold-out per run and selection metric: the threshold for
        # fold f maximises the metric's mean over the run's other folds.
        lofo = {}
        for experiment, per_fold in report.items():
            folds = sorted(per_fold)
            lofo[experiment] = {}
            for metric in args.metric:
                lofo[experiment][metric] = {}
                for f in folds:
                    others = [g for g in folds if g != f]
                    t = max(args.thresholds, key=lambda t: mean_over(per_fold, others, t, metric))
                    lofo[experiment][metric][f] = {"threshold": t, "metrics": per_fold[f][t]}

        out.write_text(json.dumps(
            {"framework": args.framework, "thresholds": args.thresholds, "selection_metrics": args.metric,
             "runs": {e: {str(f): {str(t): m for t, m in ts.items()} for f, ts in pf.items()}
                      for e, pf in report.items()},
             "lofo": {e: {m: {str(f): v for f, v in fs.items()} for m, fs in ms.items()}
                      for e, ms in lofo.items()}},
            indent=2) + "\n")

        metric_names = list(dict.fromkeys(
            m for pf in report.values() for ts in pf.values() for ms in ts.values() for m in ms
        ))
        sweep_csv = out.with_name(out.stem + "_sweep.csv")
        with open(sweep_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["framework", "run", "fold", "threshold", *metric_names])
            for e, pf in report.items():
                for f, ts in sorted(pf.items()):
                    for t, ms in ts.items():
                        writer.writerow([args.framework, e, f, t, *(ms.get(m, "") for m in metric_names)])
        lofo_csv = out.with_name(out.stem + "_lofo.csv")
        with open(lofo_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["framework", "run", "fold", "selected_on", "threshold", *metric_names])
            for e, ms in lofo.items():
                for sel, fs in ms.items():
                    for f, v in sorted(fs.items()):
                        writer.writerow([args.framework, e, f, sel, v["threshold"],
                                         *(v["metrics"].get(m, "") for m in metric_names)])
        print(f"\nSaved to {out}, {sweep_csv} and {lofo_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
