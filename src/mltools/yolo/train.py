from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from ultralytics import YOLO

from mltools.common.logger import get_logger
from mltools.yolo.dataset import get_class_names, get_dataset_dir, write_fold_split

logger = get_logger(__name__)


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
    dataset_dir = get_dataset_dir(cfg.dataset.data_root, cfg.dataset.id, cfg.dataset.name)
    class_names = get_class_names(cfg.dataset.label)
    data_yaml = write_fold_split(
        dataset_dir, cfg.dataset.folds.file_path, cfg.experiment.fold, cfg.paths.run_dir, class_names
    )

    weights_path = Path(cfg.paths.run_dir) / "train" / "weights" / "best.pt"
    if weights_path.exists():
        logger.info(
            f"[{cfg.experiment.name} | fold {cfg.experiment.fold}] "
            f"Training already completed (found {weights_path}), skipping."
        )
        return

    device = cfg.experiment.get("device", "cuda:0")
    model = YOLO(cfg.experiment.model_path)
    logger.info(
        f"[{cfg.experiment.name} | fold {cfg.experiment.fold}] Starting training..."
    )
    model.train(
        project=str(cfg.paths.run_dir),
        name="train",
        exist_ok=True,
        data=data_yaml,
        epochs=cfg.experiment.epochs,
        imgsz=cfg.dataset.resolution,
        batch=cfg.experiment.batch,
        device=device,
        save_json=True,
        # Augmentation (helps prevent overfitting on texture)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.2,
        scale=0.9,
        mosaic=1.0,
        close_mosaic=20,
    )

    # Ultralytics' own post-training validation only computes aggregate metrics/plots,
    # not per-case rasterized predictions like nnU-Net saves automatically — generate
    # them explicitly so evaluate.py can stay pure scoring.
    val_image_paths = [
        Path(line)
        for line in (Path(cfg.paths.run_dir) / "val.txt").read_text().splitlines()
        if line
    ]
    logger.info(
        f"[{cfg.experiment.name} | fold {cfg.experiment.fold}] Generating validation predictions..."
    )
    predict_label_maps(
        YOLO(str(weights_path)),
        val_image_paths,
        num_classes=len(class_names),
        out_dir=Path(cfg.paths.run_dir) / "predictions",
        device=device,
    )
