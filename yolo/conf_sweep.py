"""Sweep the detection-confidence threshold on already-trained YOLO runs.

conf gates whole detections before any mask exists, so it trades missed knots
against false positives on empty images — exactly the knob the Mask2Former
gate analysis swept. Re-predicts each fold's test split from the SAME
best.pt at several conf values and scores each, so no retraining is involved.

Predictions go to <run_dir>/conf_sweep/conf_<value>/ and the run's own
predictions/ (written at train.conf) is left untouched.

    ./scripts/run.sh is not involved — run this inside the container:
    uv run yolo/conf_sweep.py
    uv run yolo/conf_sweep.py --experiment exp_reference_0 --conf 0.05 0.1 0.25 0.4
    uv run yolo/conf_sweep.py -o article/yolo_conf_sweep.json
"""
import argparse
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
from mltools.yolo.dataset import get_class_names, get_dataset_dir, write_fold_split
from mltools.yolo.train import (
    _weights_path,
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

DEFAULT_CONFS = [0.05, 0.1, 0.15, 0.25, 0.4, 0.6]


def sweep_fold(cfg, confs: list[float]) -> dict[float, dict]:
    """{conf: metrics} for one fold's test split."""
    setup_ultralytics(cfg)
    weights_path = _weights_path(cfg)
    if not weights_path.exists():
        raise FileNotFoundError(f"no trained weights at {weights_path}")

    dataset_dir = get_dataset_dir(cfg.dataset.data_root, cfg.dataset.name, hash_cfg(cfg.dataset))
    class_names = get_class_names(cfg.dataset.label)
    write_fold_split(
        dataset_dir, cfg.dataset.folds.file_path, cfg.experiment.fold, cfg.paths.run_dir, class_names
    )
    test_image_paths = [
        Path(line)
        for line in (Path(cfg.paths.run_dir) / "test.txt").read_text().splitlines()
        if line
    ]

    model = YOLO(str(weights_path))
    out = {}
    for conf in confs:
        pred_dir = Path(cfg.paths.run_dir) / "conf_sweep" / f"conf_{conf}"
        predict_instances(
            model,
            test_image_paths,
            out_dir=pred_dir,
            device=cfg.experiment.get("device", "cuda:0"),
            imgsz=get_imgsz(cfg),
            mode=get_mode(cfg),
            conf=conf,
            iou=cfg.train.iou,
        )
        results = evaluate_run(cfg, pred_dir)
        out[conf] = results["metrics"]
        logging.info(f"fold {cfg.experiment.fold} conf={conf}: {results['metrics']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="exp_reference_0")
    ap.add_argument("--conf", nargs="+", type=float, default=DEFAULT_CONFS)
    ap.add_argument("--folds", type=int, default=None, help="default: dataset.folds.k")
    ap.add_argument("-o", "--output", default=None, help="write the per-fold numbers as JSON")
    args = ap.parse_args()

    config_dir = str((Path(__file__).parent / "conf").resolve())
    per_fold = {}
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        probe = compose_with_hydra_config([f"experiment={args.experiment}", "experiment.fold=0"])
        if get_mode(probe) != "instance":
            raise SystemExit(f"{args.experiment} is not in instance mode; conf does nothing in semantic mode.")
        num_folds = args.folds if args.folds is not None else probe.dataset.folds.k
        for fold in range(num_folds):
            cfg = compose_with_hydra_config(
                [f"experiment={args.experiment}", f"experiment.fold={fold}"]
            )
            per_fold[fold] = sweep_fold(cfg, args.conf)

    metrics = list(next(iter(per_fold[0].values())))
    print(f"\n{args.experiment}: {len(per_fold)} folds, conf sweep (train.conf = default 0.25)")
    header = f"{'conf':>6}  " + "  ".join(f"{m:>16}" for m in metrics)
    print(header)
    print("-" * len(header))
    for conf in args.conf:
        cells = []
        for m in metrics:
            vals = [per_fold[f][conf][m] for f in per_fold]
            cells.append(f"{st.fmean(vals):.4f} ±{st.stdev(vals):.4f}" if len(vals) > 1 else f"{vals[0]:.4f}")
        print(f"{conf:>6}  " + "  ".join(c.rjust(16) for c in cells))

    best = max(args.conf, key=lambda c: st.fmean([per_fold[f][c]["dice"] for f in per_fold]))
    print(f"\nbest mean dice at conf={best}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"experiment": args.experiment, "confs": args.conf,
             "per_fold": {str(f): {str(c): m for c, m in v.items()} for f, v in per_fold.items()}},
            indent=2) + "\n")
        print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
