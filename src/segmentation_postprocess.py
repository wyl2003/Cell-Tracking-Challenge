
"""Post-processing for Cellpose instance segmentation."""

import numpy as np


def filter_border_cells(mask: np.ndarray, margin: int = 0):
    """
    Remove cell instances whose masks are too close to the image border.

    Parameters
    ----------
    mask : np.ndarray
        2D instance segmentation mask.
        Background = 0, individual cells = positive integers.

    margin : int
        Border margin in pixels.
        margin=0 removes only cells touching the image boundary.

    Returns
    -------
    filtered : np.ndarray
        Filtered instance mask.
    removed_ids : list[int]
        Labels of removed cell instances.
    """
    if mask.ndim != 2:
        raise ValueError("Expected a 2D instance mask")

    if margin < 0:
        raise ValueError("margin must be non-negative")

    h, w = mask.shape
    filtered = mask.copy()

    # Collect labels inside the border region.
    m = margin + 1

    border_ids = np.unique(
        np.concatenate([
            mask[:m, :].ravel(),
            mask[-m:, :].ravel(),
            mask[:, :m].ravel(),
            mask[:, -m:].ravel(),
        ])
    )

    removed_ids = [
        int(label) for label in border_ids if label != 0
    ]

    for label in removed_ids:
        filtered[filtered == label] = 0

    return filtered, removed_ids


from scipy import ndimage as ndi


def refine_mask_boundary(mask: np.ndarray, operation="erosion", iterations=1):
    """
    Refine instance mask boundaries using morphology.

    Each cell is processed independently.

    Parameters
    ----------
    mask : np.ndarray
        2D instance segmentation mask.
    operation : str
        "erosion" or "dilation".
    iterations : int
        Number of morphological iterations.

    Returns
    -------
    np.ndarray
        Refined instance mask.
    """
    if mask.ndim != 2:
        raise ValueError("Expected a 2D instance mask")

    if operation not in ("erosion", "dilation"):
        raise ValueError("Invalid operation")

    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    result = np.zeros_like(mask)
    structure = ndi.generate_binary_structure(2, 1)

    for cell_id in np.unique(mask):
        if cell_id == 0:
            continue

        region = mask == cell_id

        if operation == "erosion":
            refined = ndi.binary_erosion(
                region,
                structure=structure,
                iterations=iterations
            )
        else:
            refined = ndi.binary_dilation(
                region,
                structure=structure,
                iterations=iterations
            )

            # Do not overwrite other original cells.
            refined &= (mask == 0) | region

        # Preserve all original instance IDs.
        # For dilation, first ID gets priority in overlaps.
        result[refined & (result == 0)] = cell_id

    return result


def merge_multiscale_masks(
    baseline: np.ndarray,
    small_scale: np.ndarray,
    overlap_threshold: float = 0.10,
    min_area: int = 200,
) -> np.ndarray:
    """
    Add new instances from small-scale predictions while
    preserving all original baseline instances.

    overlap_threshold:
        Maximum allowed fraction of a small-scale instance
        overlapping the baseline foreground.
    """
    if baseline.shape != small_scale.shape:
        raise ValueError("Mask shapes must match")

    result = baseline.copy()
    occupied = baseline > 0

    next_id = int(baseline.max()) + 1

    for cell_id in np.unique(small_scale):
        if cell_id == 0:
            continue

        candidate = small_scale == cell_id
        area = int(candidate.sum())

        if area < min_area:
            continue

        overlap = np.count_nonzero(candidate & occupied)
        overlap_ratio = overlap / area

        if overlap_ratio > overlap_threshold:
            continue

        # Never overwrite baseline cells.
        new_region = candidate & ~occupied

        if np.count_nonzero(new_region) < min_area:
            continue

        result[new_region] = next_id
        next_id += 1

    return result
