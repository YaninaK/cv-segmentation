import logging

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

logger = logging.getLogger(__name__)

__all__ = ["transform_data"]


def data_transform(sample: dict) -> dict:
    """
    Apply horizontal and vertical flips with 50% probability
    """
    image = sample["image"]
    mask = sample["mask"]

    if np.random.random() > 0.5:
        image = np.fliplr(image)  # Horizontal flip
        mask = np.fliplr(mask)

    if np.random.random() > 0.5:
        image = np.flipud(image)  # Vertical flip
        mask = np.flipud(mask)

    # Label refinement
    mask = binary_fill_holes(mask)
    mask = apply_dilation(mask)

    sample = {"image": image, "mask": mask}

    return sample


def apply_dilation(mask: np.array) -> np.array:
    """
    Structuring Element & Morphological Operations
    """
    kernel_size = (2, 2)
    kernel_shape = cv2.MORPH_RECT

    # Create the adaptive structuring element
    adaptive_kernel = cv2.getStructuringElement(kernel_shape, kernel_size)

    # Apply morphological operations : dilation
    mask = cv2.dilate(np.ones((36, 36)) * mask, adaptive_kernel)

    return mask
