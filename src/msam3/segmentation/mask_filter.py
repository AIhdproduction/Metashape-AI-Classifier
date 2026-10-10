"""Conservative per-image filters applied before 2D labels vote on 3D points."""

import numpy as np


def interior_labels(labels, confidence, margin, min_confidence):
    """Return only confident pixels that are safely inside one semantic mask.

    Unknown (255) pixels, uncertain pixels, image borders, and pixels near a label transition
    cannot vote. This prevents one noisy SAM mask edge from painting a visible point cloud.
    """
    labels = np.asarray(labels, dtype=np.uint8)
    confidence = np.asarray(confidence, dtype=np.float32)
    if labels.ndim != 2 or confidence.shape != labels.shape:
        raise ValueError("labels and confidence must be equally sized two-dimensional arrays")
    accepted = (labels != 255) & np.isfinite(confidence) & (confidence >= min_confidence)
    for _ in range(margin):
        eroded = accepted.copy()
        for dy, dx in (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ):
            neighbour_labels = np.full_like(labels, 255)
            neighbour_accepted = np.zeros_like(accepted)
            target_y = slice(max(0, dy), labels.shape[0] + min(0, dy))
            target_x = slice(max(0, dx), labels.shape[1] + min(0, dx))
            source_y = slice(max(0, -dy), labels.shape[0] - max(0, dy))
            source_x = slice(max(0, -dx), labels.shape[1] - max(0, dx))
            neighbour_labels[target_y, target_x] = labels[source_y, source_x]
            neighbour_accepted[target_y, target_x] = accepted[source_y, source_x]
            eroded &= neighbour_accepted & (labels == neighbour_labels)
        accepted = eroded
    return np.where(accepted, labels, 255).astype(np.uint8, copy=False)
