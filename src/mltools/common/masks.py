import numpy as np


def mask_to_label_map(
    mask: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Collapse a (H, W, C) multi-channel mask into a single-channel
    integer label map. Background = 0, classes = 1..C.

    Overlapping classes are resolved by picking the highest channel index.
    """
    binary = mask >= threshold  # (H, W, C)
    has_any = binary.any(axis=-1)

    # np.argmax keeps the first max on a tie; reverse the channel axis so it
    # keeps the last (highest-index) one instead, then map the index back.
    num_classes = mask.shape[-1]
    label_map = np.zeros(mask.shape[:2], dtype=np.uint8)
    best_channel = num_classes - np.argmax(mask[..., ::-1], axis=-1)
    label_map[has_any] = best_channel[has_any]

    return label_map
