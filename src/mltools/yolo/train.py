import shutil
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from ultralytics import YOLO, settings


from mltools.common.logger import get_logger
from mltools.common.instances import Instances, make_instances, save_instances
from mltools.common.utils import hash_cfg, pred_complete_marker, train_complete_marker
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


# Ultralytics resizes an image's LONG side to imgsz, while dataset.resolution
# is every image's SHORT side (see mltools.common.setup.ensure_resized). 1.5x
# makes the long side of the dataset's most common 3:2 images (and their
# portrait versions) exactly imgsz, so YOLO sees them at the same scale as
# seg/nnU-Net; 4:3 images are slightly upscaled and 16:9 / ~2.2:1 ones
# downscaled (x0.84 / x0.68).
IMGSZ_PER_RESOLUTION = 1.5


def get_imgsz(cfg: DictConfig) -> int:
    return int(round(cfg.dataset.resolution * IMGSZ_PER_RESOLUTION / 32) * 32)


# train.mode: which task the YOLO checkpoint in experiment.model_path solves.
# "instance" = *-seg.pt, per-object masks; "semantic" = *-sem.pt, one dense
# class map. Everything else (dataset, splits, two-phase training, scoring) is
# identical, so the mode only changes how a prediction becomes a label map and
# whether copy-paste augmentation applies.
INSTANCE = "instance"
SEMANTIC = "semantic"
MODES = (INSTANCE, SEMANTIC)


def get_mode(cfg: DictConfig) -> str:
    mode = cfg.train.get("mode", INSTANCE)
    if mode not in MODES:
        raise ValueError(f"Unknown train.mode {mode!r}, expected one of {MODES}.")
    return mode


def _weights_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.run_dir) / "train" / "weights" / "best.pt"


def _train_complete_marker(cfg: DictConfig) -> Path:
    # best.pt gets written after the first improving epoch, not just at the
    # end, so its mere existence doesn't mean training actually finished.
    return train_complete_marker(cfg.paths.run_dir)


def _pred_complete_marker(cfg: DictConfig) -> Path:
    return pred_complete_marker(cfg.paths.run_dir)


def _best_fitness(model: YOLO) -> float | None:
    return getattr(getattr(model, "trainer", None), "best_fitness", None)


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


def _instance_label_map(result) -> np.ndarray:
    """Rasterize the result's predicted instance masks into a single-channel
    label map (background=0, classes 1..C). Lower-confidence detections are
    painted first so higher-confidence ones win on overlap."""
    h, w = result.orig_shape
    label_map = np.zeros((h, w), dtype=np.uint8)
    if result.masks is not None:
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        masks = result.masks.data.cpu().numpy()
        order = np.argsort(confs)  # ascending -> highest confidence painted last
        for i in order:
            label_map[masks[i] > 0.5] = classes[i] + 1
    return label_map


def _native_instances(result) -> Instances:
    """The model's own instances, unchanged: one mask per detection that
    passed the confidence gate, at the original image size (retina_masks),
    label = class index + 1, score = detection confidence. Unlike the label
    map, overlapping detections stay separate."""
    if result.masks is None:
        return make_instances([], [], shape=result.orig_shape)
    return make_instances(
        result.masks.data.cpu().numpy() > 0.5,
        result.boxes.cls.cpu().numpy().astype(int) + 1,
        result.boxes.conf.cpu().numpy(),
    )


def _semantic_label_map(result, bg_id: int) -> np.ndarray:
    """The dense per-pixel class map the model already outputs, remapped to
    the label-map convention (background=0, classes 1..C). No per-object masks
    to rasterize or order by confidence here."""
    raw = result.semantic_mask.data.cpu().numpy().astype(np.int32)
    label_map = raw if bg_id == 0 else np.where(raw == bg_id, 0, raw + 1)
    return label_map.astype(np.uint8)


def predict_label_maps(
    model: YOLO,
    image_paths: list[Path],
    num_classes: int,
    out_dir: Path,
    device: str,
    imgsz: int,
    mode: str,
    conf: float,
    iou: float,
) -> None:
    """Run inference over image_paths and write each prediction as a label map
    to out_dir/<case>.png, per train.mode (see get_mode). Instance mode also
    writes the detected instances themselves to out_dir/<case>.npz
    (mltools.common.instances), which the evaluation scores separately."""
    out_dir.mkdir(parents=True, exist_ok=True)
    semantic = mode == SEMANTIC
    # retina_masks=True: instance masks come back at the original image size
    # with the letterbox padding already removed. Without it they're at the
    # padded inference size, and resizing those straight to (w, h)
    # shifts/stretches them whenever an image's aspect ratio needs padding to
    # a stride multiple. Semantic results are mapped back by Ultralytics
    # itself, so the flag doesn't apply there.
    results = model.predict(
        source=[str(p) for p in image_paths],
        device=device,
        imgsz=imgsz,
        verbose=False,
        # conf/iou gate detections, so they do nothing in semantic mode.
        **({} if semantic else {"retina_masks": True, "conf": conf, "iou": iou}),
    )
    bg_id = _background_id(model, num_classes) if semantic else 0
    for path, result in zip(image_paths, results):
        label_map = (
            _semantic_label_map(result, bg_id) if semantic else _instance_label_map(result)
        )
        cv2.imwrite(str(out_dir / f"{path.stem}.png"), label_map)
        if not semantic:
            save_instances(out_dir / f"{path.stem}.npz", _native_instances(result))


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
        seed=cfg.seed,
        deterministic=cfg.train.deterministic,
    )
    if get_mode(cfg) == INSTANCE:
        # copy-paste pastes discrete object instances into images, which isn't
        # a meaningful operation without per-object instances.
        shared_kwargs.update(
            copy_paste=aug.copy_paste,
            copy_paste_mode=aug.copy_paste_mode,
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

    pred_marker = _pred_complete_marker(cfg)
    if pred_marker.exists():
        logger.info(
            f"[{cfg.experiment.name} | fold {cfg.experiment.fold}] "
            f"Predictions already generated (found {pred_marker}), skipping."
        )
        return

    pred_dir = Path(cfg.paths.run_dir) / "predictions"

    dataset_dir = get_dataset_dir(cfg.dataset.data_root, cfg.dataset.name, hash_cfg(cfg.dataset))
    class_names = get_class_names(cfg.dataset.label)
    # Ultralytics' own post-training validation only computes aggregate metrics/plots,
    # not per-case rasterized predictions like nnU-Net saves automatically — generate
    # them explicitly so evaluate.py can stay pure scoring.
    write_fold_split(
        dataset_dir, cfg.dataset.folds.file_path, cfg.experiment.fold, cfg.paths.run_dir, class_names
    )
    # Predict on the held-out test split only; val already selected best.pt.
    test_image_paths = [
        Path(line)
        for line in (Path(cfg.paths.run_dir) / "test.txt").read_text().splitlines()
        if line
    ]

    device = cfg.experiment.get("device", "cuda:0")
    logger.info(
        f"[{cfg.experiment.name} | fold {cfg.experiment.fold}] Generating test predictions..."
    )
    predict_label_maps(
        YOLO(str(weights_path)),
        test_image_paths,
        num_classes=len(class_names),
        out_dir=pred_dir,
        device=device,
        imgsz=get_imgsz(cfg),
        mode=get_mode(cfg),
        conf=cfg.train.conf,
        iou=cfg.train.iou,
    )

    pred_marker.touch()
