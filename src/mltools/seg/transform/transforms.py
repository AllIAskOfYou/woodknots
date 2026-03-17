import albumentations as A
import cv2
from mltools.seg.schema import AugmentationConfig
import numpy as np
from omegaconf import DictConfig
from scipy.stats import norm

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

def get_vignette(labels, amount=0.5):
    labels = labels.astype(np.float32)
    mask = (labels > 0).astype(np.uint8) * 255
    if amount <= 0:
        return np.ones_like(mask, dtype=np.float32)
    if amount >= 1:
        return (mask > 0).astype(np.float32)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.float32)

    # mask should be uint8 (0 and 255)
    dist, labels_indices = cv2.distanceTransformWithLabels(~mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    # return labels_indices, dist

    # The kernel size should be proportional to image resolution
    res = min(labels.shape[0:2])
    blur_size = int(res * 0.05) | 1 # Must be odd
    vignette_map = cv2.GaussianBlur(dist, (blur_size, blur_size), 0)

    # apply bell curve for falloff control
    scale = res * 0.5
    vignette_map = norm.cdf(vignette_map, loc=0, scale=(scale*(1-amount)))
    vignette_map = (1 - vignette_map)
    cv2.normalize(vignette_map, vignette_map, 0, 1.0, cv2.NORM_MINMAX)
    # use labels_indices to index labels
    vignette_map = vignette_map[:, :, np.newaxis]
    print(labels.shape)
    flat_labels = labels[labels > 0].flatten()
    # add dim to the end
    labels_map = flat_labels[labels_indices - 1][..., np.newaxis]
    print(vignette_map.shape, labels_map.shape)
    vignette_map = vignette_map * labels_map
    return vignette_map

def apply_vignette(image, labels, amount=0.5):
    """
    Applies a contour-aware vignette.
    amount: 0.0 (no effect) to 1.0 (full mask falloff)
    """
    vignette_map = get_vignette(labels, amount)

    # apply to image
    # ensure image is float for multiplication
    result = image * vignette_map
    
    return result


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
            A.Lambda(
                name="mask_vignette", image=lambda x, **kwargs: x, mask=lambda mask, **kwargs: get_vignette(mask, amount=cfg.mask_vignette_amount)
            ),
            A.ToTensorV2(transpose_mask=True),
        ]
    )
