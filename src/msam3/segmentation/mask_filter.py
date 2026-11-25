"""Conservative per-image filters applied before 2D labels vote on 3D points."""

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter


def interior_labels(labels, confidence, margin, min_confidence, no_erosion=()):
    """Return only confident pixels that are safely inside one semantic mask.

    Unknown (255) pixels, uncertain pixels, image borders, and pixels near a label transition
    cannot vote. This prevents one noisy mask edge from painting a visible point cloud.
    Class indices in ``no_erosion`` vote up to their edge: erosion would erase them entirely.
    """
    labels = np.asarray(labels, dtype=np.uint8)
    confidence = np.asarray(confidence, dtype=np.float32)
    if labels.ndim != 2 or confidence.shape != labels.shape:
        raise ValueError("labels and confidence must be equally sized two-dimensional arrays")
    accepted = (labels != 255) & np.isfinite(confidence) & (confidence >= min_confidence)
    exempt = accepted & np.isin(labels, list(no_erosion)) if no_erosion else None
    if margin:
        # A vote needs the whole square neighbourhood to be accepted and to have
        # the same class. Separable min/max filters avoid sixteen full-image copies
        # for the usual two-pixel margin.
        size = 2 * margin + 1
        accepted = minimum_filter(accepted, size=size, mode="constant", cval=0)
        lowest = minimum_filter(labels, size=size, mode="constant", cval=255)
        highest = maximum_filter(labels, size=size, mode="constant", cval=255)
        accepted &= (lowest == labels) & (highest == labels)
    if exempt is not None:
        accepted |= exempt
    return np.where(accepted, labels, 255).astype(np.uint8, copy=False)
