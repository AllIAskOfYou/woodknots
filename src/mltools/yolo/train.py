import shutil
from pathlib import Path

import torch.nn.functional as F
from omegaconf import DictConfig
from ultralytics import YOLO, settings
from ultralytics.models.yolo.semantic import SemanticSegmentationPredictor
from ultralytics.utils import ops


from mltools.common.logger import get_logger
from mltools.common.instances import Instances, make_instances, probs_to_instances, save_instances
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
# identical, so the mode only changes how predictions become instances and
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


def _native_instances(result) -> Instances:
    """The model's own instances, unchanged: one mask per detection that
    passed the confidence gate, at the original image size (retina_masks),
    label = class index + 1, score = detection confidence. Overlapping
    detections stay separate."""
    # shape is passed even when there are masks: Ultralytics can return a
    # masks object holding zero masks (every detection's mask was empty and
    # filtered out), which leaves nothing to take the image size from.
    if result.masks is None:
        return make_instances([], [], shape=result.orig_shape)
    return make_instances(
        result.masks.data.cpu().numpy() > 0.5,
        result.boxes.cls.cpu().numpy().astype(int) + 1,
        result.boxes.conf.cpu().numpy(),
        shape=result.orig_shape,
    )


class _ProbabilitySemanticPredictor(SemanticSegmentationPredictor):
    """Ultralytics' semantic predictor, additionally keeping the foreground
    probability map (result.semantic_probs, (1, H, W) at the original image
    size) that it otherwise reduces to a class map. Single-class models only:
    the evaluation requires YOLO's classes to be the benchmark's."""

    def postprocess(self, preds, img, orig_imgs):
        results = super().postprocess(preds, img, orig_imgs)
        logits = preds[0] if isinstance(preds, (tuple, list)) else preds
        for result, pred in zip(results, logits):
            if pred.ndim != 3 or pred.shape[0] != 1:
                raise NotImplementedError(
                    f"probabilities need single-class logits, got {tuple(pred.shape)} "
                    f"(a multi-class model, or an export with an in-graph argmax)"
                )
            pred = pred[None].float()
            if pred.shape[2:] != img.shape[2:]:
                pred = F.interpolate(pred, img.shape[2:], mode="bilinear")
            result.semantic_probs = ops.scale_masks(pred, result.orig_shape)[0].sigmoid()
        return results


def predict_instances(
    model: YOLO,
    image_paths: list[Path],
    out_dir: Path,
    device: str,
    imgsz: int,
    mode: str,
    conf: float,
    iou: float,
) -> None:
    """Run inference over image_paths and write each prediction to
    out_dir/<case>.npz as instances (mltools.common.instances): in instance
    mode the detected instances; in semantic mode the connected components of
    the probability map, scored by mean probability."""
    out_dir.mkdir(parents=True, exist_ok=True)
    semantic = mode == SEMANTIC
    # retina_masks=True: instance masks at the original image size with the
    # letterbox padding removed. conf/iou gate detections, so they do nothing
    # in semantic mode, whose predictor maps the probabilities back itself.
    results = model.predict(
        source=[str(p) for p in image_paths],
        predictor=_ProbabilitySemanticPredictor if semantic else None,
        device=device,
        imgsz=imgsz,
        verbose=False,
        **({} if semantic else {"retina_masks": True, "conf": conf, "iou": iou}),
    )
    for path, result in zip(image_paths, results):
        instances = (
            probs_to_instances(result.semantic_probs.cpu().numpy())
            if semantic
            else _native_instances(result)
        )
        save_instances(out_dir / f"{path.stem}.npz", instances)


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
    predict_instances(
        YOLO(str(weights_path)),
        test_image_paths,
        out_dir=pred_dir,
        device=device,
        imgsz=get_imgsz(cfg),
        mode=get_mode(cfg),
        conf=cfg.train.conf,
        iou=cfg.train.iou,
    )

    pred_marker.touch()
