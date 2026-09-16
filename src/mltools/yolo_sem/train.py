import shutil
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from ultralytics import YOLO

from mltools.common.logger import get_logger
from mltools.common.utils import hash_cfg
from mltools.yolo.dataset import get_class_names, get_dataset_dir, write_fold_split
from mltools.yolo.train import (
    _best_fitness,
    get_imgsz,
    _resolve_model_path,
    _train_complete_marker,
    _weights_path,
)

logger = get_logger(__name__)


def _background_id(model: YOLO, num_classes: int) -> int:
    """YOLO26-sem appends an implicit background class after the declared
    ones for multi-class datasets (raw ids 0..num_classes-1 are real classes,
    num_classes is background); for a single-class dataset the raw output is
    documented to already be background=0/class=1, matching this project's
    label-map convention with no remap needed. Looked up from the loaded
    model's own names rather than hardcoded, since this is inferred from
    Ultralytics' docs, not yet verified against a live model/result — check
    this against a real run before trusting it silently."""
    names = getattr(model, "names", None) or {}
    for idx, name in names.items():
        if str(name).lower() == "background":
            return int(idx)
    return 0 if num_classes == 1 else num_classes


def predict_label_maps(
    model: YOLO, image_paths: list[Path], num_classes: int, out_dir: Path, device: str, imgsz: int
) -> None:
    """Run inference over image_paths and write each result's dense
    per-pixel class map directly as a label map (background=0, classes
    1..C) to out_dir/<case>.png. Unlike instance segmentation there are no
    per-object masks to rasterize/order by confidence — the model already
    outputs one class id per pixel."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bg_id = _background_id(model, num_classes)
    results = model.predict(
        source=[str(p) for p in image_paths], device=device, imgsz=imgsz, verbose=False
    )
    for path, result in zip(image_paths, results):
        raw = result.semantic_mask.data.cpu().numpy().astype(np.int32)
        label_map = raw if bg_id == 0 else np.where(raw == bg_id, 0, raw + 1)
        cv2.imwrite(str(out_dir / f"{path.stem}.png"), label_map.astype(np.uint8))


def train(cfg: DictConfig) -> None:
    weights_path = _weights_path(cfg)
    marker = _train_complete_marker(cfg)
    if marker.exists():
        logger.info(
            f"[{cfg.experiment.name} | fold {cfg.experiment.fold}] "
            f"Training already completed (found {marker}), skipping."
        )
        return

    dataset_dir = get_dataset_dir(cfg.dataset.data_root, cfg.dataset.name, hash_cfg(cfg.dataset))
    class_names = get_class_names(cfg.dataset.label)
    data_yaml = write_fold_split(
        dataset_dir, cfg.dataset.folds.file_path, cfg.experiment.fold, cfg.paths.run_dir, class_names
    )

    device = cfg.experiment.get("device", "cuda:0")
    tag = f"[{cfg.experiment.name} | fold {cfg.experiment.fold}]"

    # Phase 1: main training run, mosaic on throughout — close_mosaic=0 disables
    # Ultralytics' automatic mosaic-off tail, since patience-based early stopping
    # could otherwise end training before that tail is ever reached.
    logger.info(f"{tag} Starting training...")
    model = YOLO(_resolve_model_path(cfg))
    aug = cfg.train.augmentation
    # copy_paste/copy_paste_mode are intentionally not forwarded here (unlike
    # yolo/train.py) — they paste discrete object instances into images,
    # which isn't a meaningful operation without per-object instances.
    shared_kwargs = dict(
        project=str(cfg.paths.run_dir),
        exist_ok=True,
        data=data_yaml,
        imgsz=get_imgsz(cfg),
        batch=cfg.experiment.batch,
        device=device,
        save_json=True,
        hsv_h=aug.hsv_h,
        hsv_s=aug.hsv_s,
        hsv_v=aug.hsv_v,
        flipud=aug.flipud,
        fliplr=aug.fliplr,
        degrees=aug.degrees,
        translate=aug.translate,
        scale=aug.scale,
        freeze=cfg.train.freeze,
    )
    model.train(
        name="train",
        epochs=cfg.train.epochs,
        patience=cfg.train.patience,
        mosaic=aug.mosaic,
        close_mosaic=0,
        **shared_kwargs,
    )
    phase1_fitness = _best_fitness(model)
    logger.info(f"{tag} Training complete, best fitness={phase1_fitness:.4f}, weights at {weights_path}")

    # Phase 2: short fixed fine-tune without mosaic, starting from phase 1's
    # best weights, guaranteeing a mosaic-free polish happens regardless of
    # when (or whether) phase 1's patience triggered early stopping. No
    # patience here — it's a fixed, short budget by design.
    logger.info(f"{tag} Fine-tuning for {cfg.train.finetune_epochs} epochs without mosaic...")
    finetune_model = YOLO(str(weights_path))
    finetune_model.train(
        name="finetune",
        epochs=cfg.train.finetune_epochs,
        mosaic=0,
        close_mosaic=0,
        **shared_kwargs,
    )
    phase2_fitness = _best_fitness(finetune_model)
    logger.info(f"{tag} Fine-tuning complete, best fitness={phase2_fitness:.4f}")
    finetune_best = Path(cfg.paths.run_dir) / "finetune" / "weights" / "best.pt"

    if phase1_fitness is not None and phase2_fitness is not None and phase2_fitness > phase1_fitness:
        logger.info(
            f"{tag} Fine-tuned weights improved fitness "
            f"({phase2_fitness:.4f} > {phase1_fitness:.4f}), replacing {weights_path}."
        )
        shutil.copy(finetune_best, weights_path)
    else:
        logger.info(
            f"{tag} Fine-tuning did not improve fitness "
            f"(phase1={phase1_fitness}, phase2={phase2_fitness}), keeping phase-1 weights."
        )

    marker.touch()


def predict(cfg: DictConfig) -> None:
    weights_path = _weights_path(cfg)
    marker = _train_complete_marker(cfg)
    if not marker.exists():
        raise RuntimeError(
            f"Cannot generate predictions: training has not completed (no {marker}). "
            f"Run training first."
        )

    pred_dir = Path(cfg.paths.run_dir) / "predictions"
    if pred_dir.exists() and any(pred_dir.iterdir()):
        logger.info(
            f"[{cfg.experiment.name} | fold {cfg.experiment.fold}] "
            f"Predictions already generated (found {pred_dir}), skipping."
        )
        return

    dataset_dir = get_dataset_dir(cfg.dataset.data_root, cfg.dataset.name, hash_cfg(cfg.dataset))
    class_names = get_class_names(cfg.dataset.label)
    # Ultralytics' own post-training validation only computes aggregate metrics/plots,
    # not per-case rasterized predictions like nnU-Net saves automatically — generate
    # them explicitly so evaluate.py can stay pure scoring.
    write_fold_split(
        dataset_dir, cfg.dataset.folds.file_path, cfg.experiment.fold, cfg.paths.run_dir, class_names
    )
    val_image_paths = [
        Path(line)
        for line in (Path(cfg.paths.run_dir) / "val.txt").read_text().splitlines()
        if line
    ]

    device = cfg.experiment.get("device", "cuda:0")
    logger.info(
        f"[{cfg.experiment.name} | fold {cfg.experiment.fold}] Generating validation predictions..."
    )
    predict_label_maps(
        YOLO(str(weights_path)),
        val_image_paths,
        num_classes=len(class_names),
        out_dir=pred_dir,
        device=device,
        imgsz=get_imgsz(cfg),
    )
