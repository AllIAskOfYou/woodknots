from pathlib import Path

import torch.nn.functional as F
from omegaconf import DictConfig
from ultralytics import YOLO, settings
from ultralytics.models.yolo.semantic import SemanticSegmentationPredictor
from ultralytics.utils import ops


from mltools.common.logger import get_logger
from mltools.common.instances import Instances, make_instances, probs_to_instances, save_instances
from mltools.common.splits import TRAIN_TEST, TRAIN_VAL_TEST
from mltools.common.utils import hash_cfg, pred_complete_marker, train_complete_marker
from mltools.yolo.dataset import (
    SETUP_IRRELEVANT_DATASET_KEYS,
    get_class_names,
    get_dataset_dir,
    write_fold_split,
)

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
# class map. Everything else (dataset, splits, training, scoring) is
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


# train.weights: which checkpoint predict() loads. Ultralytics writes both —
# best.pt by fitness (box plus mask mAP50-95) and last.pt rewritten every epoch,
# so last.pt really is the final model, unlike Lightning's last.ckpt.
LAST, BEST = "last", "best"
WEIGHTS_CHOICES = (LAST, BEST)


def _split_mode(cfg: DictConfig) -> str:
    """train.split_mode: train_val_test withholds a partition that selects the
    checkpoint; train_test trains on it too and keeps the final epoch. Under
    train_test the val loader Ultralytics insists on is fed the TEST partition,
    logged for inspection only — which is why train.weights must not be 'best'
    there, and why patience must not be able to fire."""
    mode = cfg.train.get("split_mode", TRAIN_VAL_TEST)
    if mode not in (TRAIN_VAL_TEST, TRAIN_TEST):
        raise ValueError(
            f"Unknown train.split_mode {mode!r}, expected one of {(TRAIN_VAL_TEST, TRAIN_TEST)}."
        )
    return mode


def _weights_choice(cfg: DictConfig) -> str:
    choice = cfg.train.get("weights", LAST)
    if choice not in WEIGHTS_CHOICES:
        raise ValueError(f"Unknown train.weights {choice!r}, expected one of {WEIGHTS_CHOICES}.")
    if _split_mode(cfg) == TRAIN_TEST and choice == BEST:
        raise ValueError(
            "train.weights='best' with train.split_mode='train_test' selects a checkpoint "
            "on the test partition, which is what that split mode exists to avoid. Use "
            "weights='last', or split_mode='train_val_test'."
        )
    return choice


def _weights_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.run_dir) / "train" / "weights" / f"{_weights_choice(cfg)}.pt"


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
) -> None:
    """Run inference over image_paths and write each prediction to
    out_dir/<case>.npz as instances (mltools.common.instances): in instance
    mode the detected instances; in semantic mode the connected components of
    the probability map, scored by mean probability."""
    out_dir.mkdir(parents=True, exist_ok=True)
    semantic = mode == SEMANTIC
    # retina_masks=True: instance masks at the original image size with the
    # letterbox padding removed. conf gates detections, so it does nothing in
    # semantic mode, whose predictor maps the probabilities back itself.
    results = model.predict(
        source=[str(p) for p in image_paths],
        predictor=_ProbabilitySemanticPredictor if semantic else None,
        device=device,
        imgsz=imgsz,
        verbose=False,
        **({} if semantic else {"retina_masks": True, "conf": conf}),
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

    split_mode = _split_mode(cfg)
    if split_mode == TRAIN_TEST and cfg.train.patience < cfg.train.epochs:
        # Early stopping reads the val loader, which in this mode holds the test
        # partition — so a patience that can fire chooses the stopping epoch on
        # the test data even though no checkpoint is selected.
        raise ValueError(
            f"train.split_mode='train_test' needs patience >= epochs so early stopping "
            f"cannot fire on the test partition; got patience={cfg.train.patience}, "
            f"epochs={cfg.train.epochs}."
        )

    dataset_dir = get_dataset_dir(
        cfg.dataset.data_root,
        cfg.dataset.name,
        hash_cfg(cfg.dataset, exclude_keys=SETUP_IRRELEVANT_DATASET_KEYS),
    )
    class_names = get_class_names(cfg.dataset.label)
    data_yaml = write_fold_split(
        dataset_dir,
        cfg.dataset.folds.file_path,
        cfg.experiment.fold,
        cfg.paths.run_dir,
        class_names,
        mode=split_mode,
    )

    device = cfg.experiment.get("device", "cuda:0")
    tag = f"[{cfg.experiment.name} | fold {cfg.experiment.fold}]"

    logger.info(f"{tag} Starting training...")
    model = YOLO(_resolve_model_path(cfg))
    aug = cfg.train.augmentation
    train_kwargs = dict(
        name="train",
        project=str(cfg.paths.run_dir),
        exist_ok=True,
        data=data_yaml,
        imgsz=get_imgsz(cfg),
        batch=cfg.experiment.batch,
        device=device,
        save_json=True,
        epochs=cfg.train.epochs,
        patience=cfg.train.patience,
        hsv_h=aug.hsv_h,
        hsv_s=aug.hsv_s,
        hsv_v=aug.hsv_v,
        flipud=aug.flipud,
        fliplr=aug.fliplr,
        degrees=aug.degrees,
        translate=aug.translate,
        scale=aug.scale,
        mosaic=aug.mosaic,
        # Ultralytics turns mosaic off for the last close_mosaic epochs. With
        # patience-based early stopping a run can end before that tail is
        # reached, so whether it happens at all depends on when training stops.
        close_mosaic=aug.close_mosaic,
        freeze=cfg.train.freeze,
        seed=cfg.seed,
        deterministic=cfg.train.deterministic,
    )
    if get_mode(cfg) == INSTANCE:
        # copy-paste pastes discrete object instances into images, which isn't
        # a meaningful operation without per-object instances.
        train_kwargs.update(
            copy_paste=aug.copy_paste,
            copy_paste_mode=aug.copy_paste_mode,
        )
    model.train(**train_kwargs)
    logger.info(
        f"{tag} Training complete, best fitness={_best_fitness(model):.4f}, weights at {weights_path}"
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

    dataset_dir = get_dataset_dir(
        cfg.dataset.data_root,
        cfg.dataset.name,
        hash_cfg(cfg.dataset, exclude_keys=SETUP_IRRELEVANT_DATASET_KEYS),
    )
    class_names = get_class_names(cfg.dataset.label)
    # Ultralytics' own post-training validation only computes aggregate metrics/plots,
    # not per-case rasterized predictions like nnU-Net saves automatically — generate
    # them explicitly so evaluate.py can stay pure scoring.
    write_fold_split(
        dataset_dir,
        cfg.dataset.folds.file_path,
        cfg.experiment.fold,
        cfg.paths.run_dir,
        class_names,
        mode=_split_mode(cfg),
    )
    # Predict on the held-out test split only, from the checkpoint train.weights
    # names (last.pt by default, i.e. no selection).
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
    )

    pred_marker.touch()
