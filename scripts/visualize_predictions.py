"""Render stored predictions over their images, for any framework.

Reads only what an evaluated run already has on disk — predictions/<case>.npz
(the shared instance format), the resized images the annotations belong to, and
those annotations as ground truth — so it needs no model, no GPU and no
inference. That also means it shows exactly what was scored, not a fresh
prediction that might differ.

Each panel draws the ground-truth instances in green and the predicted ones in
red, with the case id and its per-case metrics in the corner. Panels are tiled
into one PNG.

    uv run scripts/visualize_predictions.py nnunet exp_reference
    uv run scripts/visualize_predictions.py seg exp_reference --fold 2 --select best
    uv run scripts/visualize_predictions.py yolo exp_reference --cases 0627 0630
    uv run scripts/visualize_predictions.py nnunet exp_reference \\
        --pred-dir final_predictions -o article/nnunet_final_examples.png

--select picks which cases to show when none are named: worst/best by the
per-case metric (--metric, default dice_fg), most by ground-truth instance
count, or the first ones in order.
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from hydra import initialize_config_dir

from mltools.common.instances import load_gt_instances, load_instances
from mltools.common.setup import compose_with_hydra_config, get_annotations_path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GT_COLOR = (0, 220, 0)       # BGR: ground truth
PRED_COLOR = (0, 0, 255)     # BGR: prediction
PANEL_WIDTH = 640


def overlay(image: np.ndarray, gt, pred, caption: str) -> np.ndarray:
    """Ground truth as filled-outline contours, predictions as contours on top."""
    canvas = image.copy()
    for masks, color, thickness in ((gt.masks, GT_COLOR, 2), (pred.masks, PRED_COLOR, 2)):
        for mask in masks:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, color, thickness)

    scale = PANEL_WIDTH / canvas.shape[1]
    canvas = cv2.resize(canvas, (PANEL_WIDTH, max(1, int(round(canvas.shape[0] * scale)))))
    # Caption on a strip below the image, so nothing is hidden by the text.
    strip = np.zeros((28, canvas.shape[1], 3), dtype=canvas.dtype)
    cv2.putText(strip, caption, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA)
    return np.vstack([canvas, strip])


def tile(panels: list[np.ndarray], columns: int) -> np.ndarray:
    """Grid of panels, padded to a common cell size."""
    height = max(p.shape[0] for p in panels)
    width = max(p.shape[1] for p in panels)
    rows = []
    for start in range(0, len(panels), columns):
        row = []
        for index in range(columns):
            if start + index < len(panels):
                panel = panels[start + index]
                padded = np.zeros((height, width, 3), dtype=panel.dtype)
                padded[: panel.shape[0], : panel.shape[1]] = panel
            else:
                padded = np.zeros((height, width, 3), dtype=panels[0].dtype)
            row.append(padded)
        rows.append(np.hstack(row))
    return np.vstack(rows)


def choose_cases(case_ids, per_case, gt_instances, select, metric, count):
    if select == "first":
        return case_ids[:count]
    if select == "random":
        return random.sample(case_ids, min(count, len(case_ids)))
    if select == "most":
        return sorted(case_ids, key=lambda c: -gt_instances[c].num_instances)[:count]
    scored = [c for c in case_ids if c in per_case and metric in per_case[c]]
    if not scored:
        logging.warning(f"no per-case '{metric}' available, falling back to --select first")
        return case_ids[:count]
    scored.sort(key=lambda c: per_case[c][metric], reverse=(select == "best"))
    return scored[:count]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("framework", help="seg, yolo or nnunet")
    ap.add_argument("experiment")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cases", nargs="+", default=None, help="case ids; overrides --select")
    ap.add_argument("--select", default="worst", choices=["worst", "best", "most", "random", "first"])
    ap.add_argument("--metric", default="dice_fg", help="per-case metric --select worst/best uses")
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--columns", type=int, default=2)
    ap.add_argument("--pred-dir", default="predictions", help="directory inside the run dir")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    config_dir = str(PROJECT_ROOT / args.framework / "conf")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose_with_hydra_config(
            [f"experiment={args.experiment}", f"experiment.fold={args.fold}"]
        )
    run_dir = Path(cfg.paths.run_dir)
    pred_dir = run_dir / args.pred_dir
    pred_paths = sorted(pred_dir.glob("*.npz"))
    if not pred_paths:
        raise SystemExit(f"no predictions in {pred_dir}")

    annotations_path = get_annotations_path(cfg.dataset)
    images_dir = Path(annotations_path).parent / "images"
    case_ids = [p.stem for p in pred_paths]
    gt_instances = load_gt_instances(
        annotations_path, list(cfg.dataset.class_names), list(cfg.dataset.eval_confidences),
        case_ids=case_ids,
    )

    results_path = run_dir / "eval_results.json"
    per_case = json.loads(results_path.read_text()).get("per_case", {}) if results_path.exists() else {}

    chosen = args.cases or choose_cases(
        case_ids, per_case, gt_instances, args.select, args.metric, args.num
    )
    logging.info(f"{args.framework}/{args.experiment} fold {args.fold}: {len(chosen)} cases from {pred_dir}")

    panels = []
    for case_id in chosen:
        image = cv2.imread(str(images_dir / f"{case_id}.png"))
        if image is None:
            logging.warning(f"{case_id}: no image in {images_dir}, skipping")
            continue
        pred = load_instances(pred_dir / f"{case_id}.npz")
        gt = gt_instances[case_id]
        scores = per_case.get(case_id, {})
        caption = f"{case_id}  gt={gt.num_instances} pred={pred.num_instances}"
        if args.metric in scores:
            caption += f"  {args.metric}={scores[args.metric]:.3f}"
        panels.append(overlay(image, gt, pred, caption))

    if not panels:
        raise SystemExit("nothing to draw")

    out = Path(args.output) if args.output else run_dir / f"{args.pred_dir}_vis.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), tile(panels, args.columns))
    print(f"green = ground truth, red = prediction")
    print(f"Saved {len(panels)} panels to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
