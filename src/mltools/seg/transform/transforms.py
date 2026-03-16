import albumentations as A
from omegaconf import DictConfig
from mltools.seg.schema import AugmentationConfig
import numpy as np

INFERENCE_TRANSFORM = A.Compose(
    [
        A.Normalize(),
        A.ToTensorV2(transpose_mask=True),
    ]
)


def rotate_if_portrait(data, **kwargs):
    h, w = data.shape[:2]
    if h > w:
        # Rotate 90 degrees to make it landscape
        data = np.rot90(data)
    return data


force_landscape = A.Lambda(
    name="force_landscape", image=rotate_if_portrait, mask=rotate_if_portrait
)


def get_train_transform(cfg: AugmentationConfig):
    return A.Compose(
        [
            force_landscape,
            A.ColorJitter(**cfg.color_jitter),
            A.OneOf(
                [
                    A.ElasticTransform(**cfg.distortion.elastic, p=1.0),
                    A.GridDistortion(**cfg.distortion.grid, p=cfg.distortion.e2g_ratio),
                ],
                p=cfg.distortion.p,
            ),
            A.RandomSizedCrop(
                min_max_height=(
                    int(cfg.resolution * cfg.random_crop.scale[0]),
                    int(cfg.resolution * cfg.random_crop.scale[1]),
                ),
                w2h_ratio=cfg.w2h_ratio,
                size=(cfg.resolution, int(cfg.resolution * cfg.w2h_ratio)),
                p=1.0,
            ),
            A.SafeRotate(limit=cfg.random_crop.rotation, p=1.0),
            A.HorizontalFlip(**cfg.horizontal_flip),
            A.VerticalFlip(**cfg.vertical_flip),
            A.Normalize(),
            A.ToTensorV2(transpose_mask=True),
        ]
    )
