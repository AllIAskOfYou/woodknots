"""Score a YOLO run's last-epoch checkpoint against the one fitness selected.

Everything reported uses train/weights/best.pt, chosen by fitness (box plus
mask mAP50-95). Ultralytics also rewrites train/weights/last.pt every epoch, so
it genuinely holds the final-epoch weights — unlike Lightning's last.ckpt, which
is only written on epochs that also wrote the best checkpoint and therefore ends
up identical to it (see seg/last_vs_best.py).

The gap between the two is what checkpoint selection is worth. It matters more
here than for seg: YOLO's fitness falls roughly a quarter from its peak by the
final epoch, against about 4% for seg, so these runs lean on selection far more
heavily.

No training is involved. The last-epoch weights are re-predicted onto each
fold's test split and scored with the same evaluate_run() the pipeline uses; the
best side is read from the existing eval_results.json, so the comparison is
against exactly the numbers already reported. The run's own predictions/ and
eval_results.json are left untouched; last.pt predictions go to
<run_dir>/last_predictions/.

    uv run yolo/last_vs_best.py
    uv run yolo/last_vs_best.py --experiment exp_freeze_0 -o article/yolo_last_vs_best.json
"""
import argparse
import csv
import json
import logging
import statistics as st
import sys
from pathlib import Path

from hydra import initialize_config_dir
from ultralytics import YOLO

from mltools.common.evaluate import evaluate_run
from mltools.common.setup import compose_with_hydra_config
from mltools.common.utils import hash_cfg
from mltools.yolo.dataset import (
    SETUP_IRRELEVANT_DATASET_KEYS,
    get_class_names,
    get_dataset_dir,
    write_fold_split,
)
from mltools.yolo.train import (
    _split_mode,
    _weights_choice,
    get_imgsz,
    get_mode,
    predict_instances,
    setup_ultralytics,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    stream=sys.stdout,
)

CONFIG_DIR = str((Path(__file__).resolve().parent / "conf").resolve())


def predict_with_weights(cfg, weights: Path, out_dir: Path) -> None:
    """mltools.yolo.train.predict, but from explicit weights into an explicit
    directory and without the .pred_complete bookkeeping."""
    setup_ultralytics(cfg)
    dataset_dir = get_dataset_dir(
        cfg.dataset.data_root,
        cfg.dataset.name,
        hash_cfg(cfg.dataset, exclude_keys=SETUP_IRRELEVANT_DATASET_KEYS),
    )
    write_fold_split(
        dataset_dir,
        cfg.dataset.folds.file_path,
        cfg.experiment.fold,
        cfg.paths.run_dir,
        get_class_names(cfg.dataset.label),
        mode=_split_mode(cfg),
    )
    test_image_paths = [
        Path(line)
        for line in (Path(cfg.paths.run_dir) / "test.txt").read_text().splitlines()
        if line
    ]
    predict_instances(
        YOLO(str(weights)),
        test_image_paths,
        out_dir=out_dir,
        device=cfg.experiment.get("device", "cuda:0"),
        imgsz=get_imgsz(cfg),
        mode=get_mode(cfg),
        conf=cfg.train.conf,
    )


def selected_epoch(run_dir: Path) -> tuple[int | None, int | None]:
    """(epoch of the best fitness, epochs trained) from Ultralytics' own
    results.csv — fitness is the sum of the box and mask mAP50-95 columns, the
    quantity best.pt is selected on."""
    results = run_dir / "train" / "results.csv"
    if not results.exists():
        return None, None
    rows = list(csv.DictReader(open(results)))
    if not rows:
        return None, None

    def fitness(row):
        return sum(
            float(row[k.strip()])
            for k in row
            if k.strip() in ("metrics/mAP50-95(B)", "metrics/mAP50-95(M)")
        )

    best = max(rows, key=fitness)
    return int(float(best[next(k for k in best if k.strip() == "epoch")])), len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--experiment", default="exp_reference")
    ap.add_argument("--folds", type=int, default=None, help="default: dataset.folds.k")
    ap.add_argument(
        "--force", action="store_true", help="re-predict folds already done in this run dir"
    )
    ap.add_argument("-o", "--output", default=None, help="write the per-fold numbers as JSON")
    args = ap.parse_args()

    per_fold: dict[int, dict] = {}
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        probe = compose_with_hydra_config(
            [f"experiment={args.experiment}", "experiment.fold=0"]
        )
        num_folds = args.folds if args.folds is not None else probe.dataset.folds.k

        for fold in range(num_folds):
            cfg = compose_with_hydra_config(
                [f"experiment={args.experiment}", f"experiment.fold={fold}"]
            )
            run_dir = Path(cfg.paths.run_dir)
            # The run stored predictions from train.weights; compare against
            # whichever checkpoint it did NOT use.
            other = "best" if _weights_choice(cfg) == "last" else "last"
            last = run_dir / "train" / "weights" / f"{other}.pt"
            if not last.exists():
                logging.warning(f"fold {fold}: no {last}, skipping")
                continue

            best_results = run_dir / "eval_results.json"
            if not best_results.exists():
                logging.warning(f"fold {fold}: no {best_results}, skipping")
                continue

            out_dir = run_dir / f"{other}_predictions"
            if args.force or not any(out_dir.glob("*.npz")):
                logging.info(f"fold {fold}: predicting from {last}")
                predict_with_weights(cfg, last, out_dir)
            else:
                logging.info(f"fold {fold}: reusing predictions in {out_dir}")

            best_epoch, total_epochs = selected_epoch(run_dir)
            per_fold[fold] = {
                "best": json.loads(best_results.read_text())["metrics"],
                "last": evaluate_run(cfg, out_dir)["metrics"],
                "selected_epoch": best_epoch,
                "total_epochs": total_epochs,
            }
            logging.info(f"fold {fold} last.pt: {per_fold[fold]['last']}")

    if not per_fold:
        print("nothing evaluated")
        return 1

    first = next(iter(per_fold.values()))
    metrics = [m for m in first["best"] if m in first["last"]]
    print(f"\nyolo/{args.experiment}: best (selected) vs last-epoch weights, {len(per_fold)} folds")
    print("  selected epoch per fold: " + ", ".join(
        f"fold {f}: {v['selected_epoch']}/{v['total_epochs']}" for f, v in sorted(per_fold.items())
    ))
    header = f"{'metric':<26}{'best':>10}{'last':>10}{'delta':>10}{'folds last better':>19}"
    print(header)
    print("-" * len(header))
    for metric in metrics:
        best = [per_fold[f]["best"][metric] for f in per_fold]
        last = [per_fold[f]["last"][metric] for f in per_fold]
        deltas = [l_ - b_ for b_, l_ in zip(best, last)]
        better = sum(1 for d in deltas if d > 0)
        print(
            f"{metric:<26}{st.fmean(best):>10.4f}{st.fmean(last):>10.4f}"
            f"{st.fmean(deltas):>+10.4f}{f'{better}/{len(deltas)}':>19}"
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"framework": "yolo", "experiment": args.experiment,
             "per_fold": {str(f): v for f, v in sorted(per_fold.items())}},
            indent=2) + "\n")
        print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
