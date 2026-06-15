import albumentations as A
import cv2
from mltools.seg.schema import PreprocessingConfig
from mltools.seg.schema import AugmentationConfig
import numpy as np
from scipy.stats import norm


def rotate_if_portrait(data, **kwargs):
    h, w = data.shape[:2]
    if h > w:
        # Rotate 90 degrees to make it landscape
        data = np.rot90(data)
    return data


class ForceLandscape(A.DualTransform):
    def __init__(self, p=1.0):
        super().__init__(True, p)

    def apply(self, img, **kwargs):
        return rotate_if_portrait(img)

    def apply_to_mask(self, mask, **kwargs):
        return rotate_if_portrait(mask)


def get_vignette(labels, steepness=5, midpoint=0.1, invert=False):
    binary_mask = (labels > 0).astype(np.uint8)
    if not invert:
        binary_mask = 1 - binary_mask
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=np.float32)
    if steepness <= 0:
        return labels

    # normalize distance to [0, 1] inside each component based on label_indices
    num_labels, labels_map = cv2.connectedComponents(binary_mask)
    vignette_map = np.zeros(shape=labels.shape[:2], dtype=np.float32)
    for i in range(1, num_labels):  # Skip background
        component_mask = (labels_map == i).astype(np.uint8)
        dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            continue
        # blur based on component size
        bb = cv2.boundingRect(component_mask)
        res = int(min(bb[2], bb[3]))
        blur_size = int(res * 0.5) | 1  # Must be odd
        blur_size = np.clip(blur_size, 3, 51)  # Clamp to reasonable range
        dist = cv2.GaussianBlur(dist, (blur_size, blur_size), 0)
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        dist = 1 / (1 + np.exp(-steepness * (dist - midpoint)))
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        dist = np.clip(dist, 0, 1)
        vignette_map += dist
    if not invert:
        vignette_map = 1 - vignette_map
    return vignette_map[..., np.newaxis]


class ImageMaskTransform(A.DualTransform):
    def __init__(self, distance, channel_index, p=1.0):
        super().__init__(True, p)
        self.distance = distance
        self.channel_index = channel_index

    def apply(self, img, **kwargs):
        mask = kwargs.get("image_mask", None)
        if mask is None:
            raise ValueError("Mask is required for ImageMaskTransform")
        mask_channel = mask[..., self.channel_index : self.channel_index + 1]
        binary_mask = (~(mask_channel > 0)).astype(np.uint8)
        dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        alpha = dist < self.distance
        # blur the alpha mask to create a smooth transition
        res = int(self.distance * 2) | 1  # Must be odd
        alpha = cv2.GaussianBlur(alpha.astype(np.float32), (res, res), 0)
        alpha = alpha[..., np.newaxis]
        return (img.astype(np.float32) * alpha).astype(img.dtype)

    def apply_to_mask(self, mask, **kwargs):
        return mask

    def get_params_dependent_on_data(self, params, data):
        # this is the key part
        return {"image_mask": data["mask"]}


class MaskVignetteTransform(A.DualTransform):
    def __init__(self, steepness, midpoint, p=1.0):
        super().__init__(True, p)
        self.steepness = steepness
        self.midpoint = midpoint

    def apply(self, img, **kwargs):
        return img

    def apply_to_mask(self, mask, **kwargs):
        if self.steepness <= 0:
            return mask
        vignette_map = get_vignette(mask, self.steepness, self.midpoint, invert=True)
        res = int(min(mask.shape[1], mask.shape[0]) * 0.1) | 1
        mask_blurred = cv2.GaussianBlur(mask.astype(np.float32), (res, res), 0)
        return mask_blurred[..., np.newaxis] * vignette_map


def get_preprocessing_transform(cfg: PreprocessingConfig):
    transforms = []
    if cfg.mask_image is not None and cfg.mask_image.distance > 0:
        transforms.append(
            ImageMaskTransform(
                distance=cfg.mask_image.distance,
                channel_index=cfg.mask_image.channel_index,
            )
        )
    return A.Compose(transforms)


def get_train_transform(cfg: AugmentationConfig):
    # Common size parameters
    target_size = (cfg.resolution, int(cfg.resolution * cfg.w2h_ratio))

    # Common crop parameters
    crop_params = {
        "min_max_height": (
            int(cfg.resolution * cfg.random_crop.scale[0]),
            int(cfg.resolution * cfg.random_crop.scale[1]),
        ),
        "w2h_ratio": cfg.w2h_ratio,
        "size": target_size,
        "p": 1.0,
    }

    # Common distortion transforms
    distortion_transforms = A.OneOf(
        [
            A.ElasticTransform(**cfg.distortion.elastic, p=1.0),
            A.GridDistortion(
                **cfg.distortion.grid,
                p=cfg.distortion.e2g_ratio,
            ),
        ],
        p=cfg.distortion.p,
    )

    # Mosaic branch
    mosaic_branch = A.Compose(
        [
            A.Mosaic(
                grid_yx=(2, 2),
                target_size=target_size,
                cell_shape=target_size,
                center_range=(0.4, 0.6),
                metadata_key="mosaic_metadata",
                p=1.0,
            ),
            A.ColorJitter(**cfg.color_jitter),
            distortion_transforms,
            A.Rotate(limit=(-25, 25), p=1.0),
            A.RandomSizedCrop(**crop_params),
        ],
        p=cfg.mosaic_p,
    )

    # Non-mosaic branch
    non_mosaic_branch = A.Compose(
        [
            A.ColorJitter(**cfg.color_jitter),
            distortion_transforms,
            A.RandomSizedCrop(**crop_params),
            A.SafeRotate(limit=cfg.random_crop.rotation, p=1.0),
        ],
        p=1 - cfg.mosaic_p,
    )

    return A.Compose(
        [
            A.OneOf([mosaic_branch, non_mosaic_branch], p=1.0),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                fill=0,
                p=0.4,
            ),
            A.HorizontalFlip(**cfg.horizontal_flip),
            A.VerticalFlip(**cfg.vertical_flip),
            A.Normalize(),
            MaskVignetteTransform(
                steepness=cfg.mask_vignette.steepness,
                midpoint=cfg.mask_vignette.midpoint,
                p=1.0,
            ),
            A.ToTensorV2(transpose_mask=True),
        ]
    )


def get_val_transform(cfg: AugmentationConfig):
    return A.Compose(
        [
            A.Normalize(),
            MaskVignetteTransform(
                steepness=cfg.mask_vignette.steepness,
                midpoint=cfg.mask_vignette.midpoint,
                p=1.0,
            ),
            A.ToTensorV2(transpose_mask=True),
        ]
    )


INFERENCE_TRANSFORM = A.Compose(
    [
        A.Normalize(),
        A.ToTensorV2(transpose_mask=True),
    ]
)
