
"""Image preprocessing for cell segmentation."""

import cv2
import numpy as np


def apply_clahe(image, clip_limit=2.0, tile_size=8):
    """Apply CLAHE to a grayscale microscopy image."""

    if image.ndim != 2:
        raise ValueError("Expected a 2D grayscale image")

    # Convert image to uint8 while preserving its intensity range.
    img = image.astype(np.float32)
    low, high = img.min(), img.max()

    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)

    img = (img - low) / (high - low)
    img = (img * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )

    return clahe.apply(img)
