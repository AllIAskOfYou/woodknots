import shutil
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from ultralytics import YOLO, settings


from mltools.common.logger import get_logger
from mltools.common.utils import hash_cfg
from mltools.yolo.dataset import get_class_names, get_dataset_dir, write_fold_split

logger = get_logger(__name__)

def setup_ultralytics(cfg: DictConfig) -> None:
    ultralytics_root = Path(cfg.paths.project_root) / ".ultralytics"
    ultralytics_root.mkdir(parents=True, exist_ok=True)
    settings.update({
        "runs_dir": str(Path(cfg.paths.run_dir) / "runs"),
        "weights_dir": str(ultralytics_root / "weights"),
        "datasets_dir": str(ultralytics_root / "datasets"),
    })


def _resolve_model_path(cfg: DictConfig) -> str:
    """settings['weights_dir'] only affects Ultralytics' lookup of an
    already-cached asset, not where a fresh download lands — a bare name
    like 'yolo26m-seg.pt' downloads relative to CWD regardless of that
    setting. Resolve explicitly against weights_dir so first-time downloads
    land there instead."""
    model_path = Path(cfg.experiment.model_path)
    if model_path.is_absolute() or model_path.exists():
        return str(model_path)
    weights_dir = Path(settings["weights_dir"])
    weights_dir.mkdir(parents=True, exist_ok=True)
    return str(weights_dir / model_path)


def _weights_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.run_dir) / "train" / "weights" / "best.pt"


def _train_complete_marker(cfg: DictConfig) -> Path:
    """best.pt gets written after the first improving epoch, not just at the
    end — so its mere existence doesn't mean training actually finished (an
    interrupted run leaves a partial best.pt behind too). This marker is
    touched only as the very last step of a successful train() call, so an
    interrupted run is correctly detected as incomplete and retrained from
    scratch rather than silently treated as done."""
    return Path(cfg.paths.run_dir) / "train" / ".complete"


def _best_fitness(model: YOLO) -> float | None:
    return getattr(getattr(model, "trainer", None), "best_fitness", None)


def predict_label_maps(
    model: YOLO, image_paths: list[Path], num_classes: int, out_dir: Path, device: str
) -> None:
    """Run inference over image_paths and rasterize each result's predicted
    instance masks into a single-channel label map (background=0, classes
    1..C), writing out_dir/<case>.png. Lower-confidence detections are
    painted first so higher-confidence ones win on overlap."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = model.predict(
        source=[str(p) for p in image_paths], device=device, verbose=False
    )
    for path, result in zip(image_paths, results):
        h, w = result.orig_shape
        label_map = np.zeros((h, w), dtype=np.uint8)
        if result.masks is not None:
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            masks = result.masks.data.cpu().numpy()
            order = np.argsort(confs)  # ascending -> highest confidence painted last
            for i in order:
                mask_resized = cv2.resize(
                    masks[i], (w, h), interpolation=cv2.INTER_NEAREST
                )
                label_map[mask_resized > 0.5] = classes[i] + 1
        cv2.imwrite(str(out_dir / f"{path.stem}.png"), label_map)


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
    shared_kwargs = dict(
        project=str(cfg.paths.run_dir),
        exist_ok=True,
        data=data_yaml,
        imgsz=cfg.dataset.resolution,
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
        copy_paste=aug.copy_paste,
        copy_paste_mode=aug.copy_paste_mode,
        freeze=cfg.train.freeze,
    )
    model.train(
        name="train",
        epochs=cfg.train.epochs,
        patience=cfg.train.patience,
        mosaic=aug.mosaic,
        close_mosaic=0,
        **shared_kwargs
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
        **shared_kwargs
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
    )
