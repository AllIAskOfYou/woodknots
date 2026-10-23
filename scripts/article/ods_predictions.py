"""Qualitative out-of-distribution check: the seg pair ensemble on data/ods.

The ensemble is the one the evaluation scores: exp_reference_<run> (512) and
exp_union_<run> (1024) of one split, their knot probabilities averaged 1:1 at
512 (seg/ensemble_pairs.py). New images are prepared like the training ones:
read with OpenCV as RGB and resized bilinearly so the shorter side equals each
model's resolution; each model predicts with its own sliding window, the 1024
map is resized to the 512 image, and the average becomes instances as in the
evaluation (p > 0.5, 8-connected components scored by mean probability).

Output per image, drawn at --draw-short-side, outlines only:
  blue (#0072B2)            annotated knots
  orange (#D55E00), solid   predicted knots kept at the tuned threshold
  orange (#D55E00), dotted  predicted knots below it (kept only at the default)
and a table of knots, predictions and matches (IoU >= 0.5, the evaluation's
rule) at both thresholds.

The tuned threshold defaults to the one the confidence sweep chose for that
run and split (article/conf_sweep/seg_lofo.csv).

    uv run scripts/article/ods_predictions.py
    uv run scripts/article/ods_predictions.py --run 0 --fold 0 --threshold 0.6 -o article/ods
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from hydra import initialize_config_dir

from mltools.common.coco_dataset import short_side_size
from mltools.common.instances import make_instances, probs_to_instances
from mltools.common.metrics.gms import match_instance_pairs
from mltools.common.setup import compose_with_hydra_config, experiment_overrides, get_experiment_names
from mltools.common.splits import TRAIN_TEST
from mltools.seg.seg_module import SegModule
from mltools.seg.train import FINAL_CHECKPOINT_NAME, _checkpoint_dir, _split_mode
from mltools.seg.transforms import INFERENCE_TRANSFORM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEG_DIR = PROJECT_ROOT / "seg"
def hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


# Green annotations, red predictions kept at the tuned threshold, yellow below it.
GT_COLOR, PRED_COLOR, PRED_COLOR_F = hex_to_bgr("#00DC00"), hex_to_bgr("#FF0000"), hex_to_bgr("#FFDC00")


def checkpoint(cfg) -> Path:
    """The checkpoint mltools.seg.train.predict uses for this run."""
    if _split_mode(cfg) == TRAIN_TEST:
        return _checkpoint_dir(cfg) / FINAL_CHECKPOINT_NAME
    best = sorted(_checkpoint_dir(cfg).glob("best-model-*.ckpt"))
    if len(best) != 1:
        raise RuntimeError(f"expected one best checkpoint in {_checkpoint_dir(cfg)}, found {len(best)}")
    return best[0]


def find_image(images_dir: Path, file_name: str) -> Path:
    """The annotations say 0.jpg, the files are 0.JPG: match case-insensitively."""
    exact = images_dir / file_name
    if exact.exists():
        return exact
    for path in images_dir.iterdir():
        if path.name.lower() == file_name.lower():
            return path
    raise FileNotFoundError(f"{file_name} not in {images_dir}")


def predict(module, image_rgb: np.ndarray, resolution: int, device) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    tw, th = short_side_size(w, h, resolution)
    resized = cv2.resize(image_rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
    tensor = INFERENCE_TRANSFORM(image=resized)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        return module.forward(tensor)[0, 0].float().cpu().numpy()  # (th, tw) knot probability


ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_box(box, degrees: int, width: int, height: int):
    """A bounding box (x, y, w, h) in an image of width x height, moved with the
    image's clockwise rotation by degrees (the cv2.rotate convention)."""
    x, y, bw, bh = box
    if degrees == 90:
        return height - (y + bh), x, bh, bw
    if degrees == 180:
        return width - (x + bw), height - (y + bh), bw, bh
    if degrees == 270:
        return y, width - (x + bw), bh, bw
    return box


def gt_masks(annotations: list[dict], scale: float, shape: tuple[int, int]) -> np.ndarray:
    masks = []
    for ann in annotations:
        mask = np.zeros(shape, dtype=np.uint8)
        for poly in ann["segmentation"]:
            points = np.round(np.asarray(poly, dtype=np.float64).reshape(-1, 2) * scale).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
        masks.append(mask.astype(bool))
    return np.stack(masks) if masks else np.zeros((0, *shape), dtype=bool)


# Helvetica-like sans-serif for the scores (Liberation Sans: Arial metrics,
# freely redistributable), kept in the repository so the container has it.
DEFAULT_FONT = Path(__file__).resolve().parent / "fonts" / "LiberationSans-Regular.ttf"


def render_text(text: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """The text as a tight, antialiased coverage mask (uint8, H x W)."""
    x0, y0, x1, y1 = font.getbbox(text)
    image = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(image).text((-x0, -y0), text, font=font, fill=255)
    return np.asarray(image)


def paste_text_with_edge(image_bgr, text_mask, top_left, color, edge=2) -> None:
    """Paint text_mask at top_left (x, y) in color, with a black edge that
    follows every glyph: the mask widened by edge px, painted black first.
    One rendering of the text, so the edge cannot be misaligned with it."""
    mask = cv2.copyMakeBorder(text_mask, edge, edge, edge, edge, cv2.BORDER_CONSTANT, value=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge + 1, 2 * edge + 1))
    halo = cv2.dilate(mask, kernel)
    x0, y0 = top_left[0] - edge, top_left[1] - edge
    x1, y1 = x0 + mask.shape[1], y0 + mask.shape[0]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(image_bgr.shape[1], x1), min(image_bgr.shape[0], y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    sub = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
    region = image_bgr[cy0:cy1, cx0:cx1].astype(np.float32)
    a = (halo[sub].astype(np.float32) / 255.0)[..., None]
    region = region * (1 - a)
    a = (mask[sub].astype(np.float32) / 255.0)[..., None]
    region = region * (1 - a) + np.asarray(color, dtype=np.float32) * a
    image_bgr[cy0:cy1, cx0:cx1] = np.clip(region, 0, 255).astype(np.uint8)


def draw_outlines(image_bgr, masks, color, thickness, occupied) -> list:
    """Outlines of masks with a black edge; marks them in occupied (bool, H x W)
    so labels avoid them. Returns each mask's bounding box (x, y, w, h)."""
    edge = max(thickness + 1, int(round(1.5 * thickness)))  # cv2 needs an int width
    boxes = []
    for mask in masks:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image_bgr, contours, -1, (0, 0, 0), edge, cv2.LINE_AA)
        cv2.drawContours(image_bgr, contours, -1, color, thickness, cv2.LINE_AA)
        marked = np.zeros(occupied.shape, dtype=np.uint8)
        cv2.drawContours(marked, contours, -1, 1, edge + 2)
        occupied |= marked.astype(bool)
        boxes.append(cv2.boundingRect(max(contours, key=cv2.contourArea)) if contours else None)
    return boxes


def place_label(image_bgr, text, box, color, font, occupied, edge=2) -> None:
    """Draw text next to box (x, y, w, h): the first candidate position that
    lies inside the image and covers no outline or earlier label, else the
    in-image candidate covering the least. The label's area is then marked
    in occupied."""
    text_mask = render_text(text, font)
    th, tw = text_mask.shape
    gap = 3
    height, width = occupied.shape
    bx, by, bw, bh = box

    def rect(top_left):  # the area the label and its edge cover
        return top_left[0] - edge, top_left[1] - edge, top_left[0] + tw + edge, top_left[1] + th + edge

    def clamp_x(x):  # keep a label placed above or below inside the image
        return min(max(x, edge), width - tw - edge)

    above, below = by - gap - edge - th, by + bh + gap + edge
    right, left = bx + bw + gap + edge, bx - gap - edge - tw
    candidates = [
        (clamp_x(bx), above), (clamp_x(bx + bw - tw), above),   # above, left / right aligned
        (clamp_x(bx), below), (clamp_x(bx + bw - tw), below),   # below
        (right, by), (left, by), (right, by + bh - th), (left, by + bh - th),  # beside
    ]
    best, best_cover = None, None
    for top_left in candidates:
        x0, y0, x1, y1 = rect(top_left)
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            continue
        cover = int(occupied[y0:y1, x0:x1].sum())
        if best is None or cover < best_cover:
            best, best_cover = top_left, cover
        if cover == 0:
            break
    if best is None:  # no candidate fits: pin it inside the image next to the box
        best = (clamp_x(bx), min(max(above, edge), height - th - edge))
    paste_text_with_edge(image_bgr, text_mask, best, color, edge)
    x0, y0, x1, y1 = rect(best)
    occupied[max(0, y0):y1, max(0, x0):x1] = True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ods", default=str(PROJECT_ROOT / "data/ods"))
    ap.add_argument("--base", default="exp_reference")
    ap.add_argument("--partner", default="exp_union")
    ap.add_argument("--run", type=int, default=0, help="seed index of both members")
    ap.add_argument("--fold", type=int, default=0, help="split whose models are used (split i tests on partition i+1)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="tuned threshold; default: the sweep's choice for pair_<run>, this fold")
    ap.add_argument("--lofo", default=str(PROJECT_ROOT / "article/conf_sweep/seg_lofo.csv"))
    ap.add_argument("--selected-on", default="instance_f1_pooled")
    ap.add_argument("--rotate", type=int, default=90, choices=sorted(ROTATIONS),
                    help="rotate the drawing clockwise by this many degrees before the scores are added")
    ap.add_argument("--draw-short-side", type=int, default=1024, help="shorter side of the output images")
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--font", default=str(DEFAULT_FONT), help="TrueType/OpenType font for the scores")
    ap.add_argument("--font-size", type=int, default=None, help="px; default 8 x --thickness")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("-o", "--output", default=str(PROJECT_ROOT / "article/ods"))
    args = ap.parse_args()

    if args.threshold is None:
        rows = [r for r in csv.DictReader(open(args.lofo, newline=""))
                if r["run"] == f"pair_{args.run}" and int(r["fold"]) == args.fold
                and r["selected_on"] == args.selected_on]
        if not rows:
            raise SystemExit(f"no tuned threshold for pair_{args.run} fold {args.fold} in {args.lofo}")
        args.threshold = float(rows[0]["threshold"])

    ods = Path(args.ods)
    coco = json.loads((ods / "annotations.json").read_text())
    knot_ids = {c["id"] for c in coco["categories"] if c["name"] == "wood_knot"}
    knots_by_image = {im["id"]: [] for im in coco["images"]}
    for ann in coco["annotations"]:
        if ann["category_id"] in knot_ids:
            knots_by_image[ann["image_id"]].append(ann)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    names = [f"{args.base}_{args.run}", f"{args.partner}_{args.run}"]
    config_names = get_experiment_names(SEG_DIR)
    members = []
    with initialize_config_dir(config_dir=str(SEG_DIR / "conf"), version_base=None):
        for name in names:
            overrides = experiment_overrides(name, config_names)
            if overrides is None:
                raise SystemExit(f"no seg experiment config for {name}")
            cfg = compose_with_hydra_config([*overrides, f"experiment.fold={args.fold}"])
            ckpt = checkpoint(cfg)
            module = SegModule.load_from_checkpoint(str(ckpt), weights_only=False)
            if hasattr(module.model, "predict_instances"):
                raise SystemExit(f"{name}: instance-native models have no probability map to average")
            module.eval().to(device)
            members.append((name, int(cfg.dataset.resolution), module))
            print(f"{name}: {ckpt} (resolution {cfg.dataset.resolution})")
    base_resolution = members[0][1]

    font = ImageFont.truetype(args.font, args.font_size or 8 * args.thickness)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = [f"{'image':<10}{'knots':>6}{'pred@0':>8}{'found@0':>9}{'pred@t':>8}{'found@t':>9}"]
    totals = np.zeros(5, dtype=int)
    for im in coco["images"]:
        path = find_image(ods / "images", im["file_name"])
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        bw, bh = short_side_size(w, h, base_resolution)

        probs = []
        for name, resolution, module in members:
            p = predict(module, image, resolution, device)
            if p.shape != (bh, bw):
                p = cv2.resize(p, (bw, bh), interpolation=cv2.INTER_LINEAR)
            probs.append(p)
        fused = np.mean(probs, axis=0)
        pred = probs_to_instances(fused[None])
        kept = pred.scores >= args.threshold
        tuned = make_instances(pred.masks[kept], pred.labels[kept], pred.scores[kept], shape=pred.shape)

        # Ground truth rasterised at the scored (base) resolution.
        gt = gt_masks(knots_by_image[im["id"]], bw / w, (bh, bw))
        found_default = len(match_instance_pairs(pred.masks, pred.scores, gt))
        found_tuned = len(match_instance_pairs(tuned.masks, tuned.scores, gt))
        row = [len(gt), pred.num_instances, found_default, tuned.num_instances, found_tuned]
        totals += row
        table.append(f"{path.stem:<10}" + "".join(f"{v:>{n}}" for v, n in zip(row, (6, 8, 9, 8, 9))))

        # Drawing, at --draw-short-side.
        dw, dh = short_side_size(w, h, args.draw_short_side)
        canvas = cv2.cvtColor(cv2.resize(image, (dw, dh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        up = lambda masks: np.stack([
            cv2.resize(m.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks
        ]) if len(masks) else masks
        # Every outline first, then the scores, so no label is drawn over an outline.
        occupied = np.zeros(canvas.shape[:2], dtype=bool)
        draw_outlines(canvas, up(gt), GT_COLOR, args.thickness, occupied)
        dropped_boxes = draw_outlines(canvas, up(pred.masks[~kept]), PRED_COLOR_F, args.thickness, occupied)
        kept_boxes = draw_outlines(canvas, up(pred.masks[kept]), PRED_COLOR, args.thickness, occupied)
        # Rotate the drawing (and what the labels must avoid), then add the
        # scores upright in the rotated frame.
        if ROTATIONS[args.rotate] is not None:
            ch, cw = canvas.shape[:2]
            canvas = cv2.rotate(canvas, ROTATIONS[args.rotate])
            occupied = cv2.rotate(occupied.astype(np.uint8), ROTATIONS[args.rotate]).astype(bool)
            turn = lambda boxes: [None if b is None else rotate_box(b, args.rotate, cw, ch) for b in boxes]
            kept_boxes, dropped_boxes = turn(kept_boxes), turn(dropped_boxes)
        for boxes, scores, color in ((kept_boxes, pred.scores[kept], PRED_COLOR),
                                     (dropped_boxes, pred.scores[~kept], PRED_COLOR_F)):
            for box, score in zip(boxes, scores):
                if box is not None:
                    place_label(canvas, f"{score:.2f}", box, color, font, occupied)
        cv2.imwrite(str(out_dir / f"{path.stem}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

    table.append(f"{'all':<10}" + "".join(f"{v:>{n}}" for v, n in zip(totals, (6, 8, 9, 8, 9))))
    report = "\n".join([
        f"seg pair ensemble {names[0]} + {names[1]}, split {args.fold}, on {args.ods}",
        f"default threshold 0 (every instance), tuned threshold t = {args.threshold:g}; "
        f"found = matched at IoU >= 0.5 at {base_resolution}",
        "outlines: green annotation, red prediction kept at t, yellow prediction below t; score above each",
        "",
        *table,
    ]) + "\n"
    (out_dir / "summary.txt").write_text(report)
    print("\n" + report, end="")
    print(f"\nSaved {len(coco['images'])} images and summary.txt to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
