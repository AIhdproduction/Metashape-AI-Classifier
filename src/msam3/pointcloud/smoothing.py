"""Optional local majority smoothing, preserving undecided points and high-share labels."""

import numpy as np
from scipy.spatial import cKDTree

from msam3.config import SmoothingConfig


def smooth(points, labels, shares, config: SmoothingConfig, unknown: int, batch_size: int):
    result = labels.copy()
    if not config.enabled:
        return result
    tree = cKDTree(points)
    candidates = np.flatnonzero((shares < config.max_share_to_change) & (labels != unknown))
    for start in range(0, len(candidates), batch_size):
        ids = candidates[start : start + batch_size]
        _, neighbours = tree.query(points[ids], k=min(config.k, len(points)), workers=-1)
        neighbours = np.asarray(neighbours).reshape(len(ids), -1)
        for index, local in zip(ids, neighbours, strict=True):
            values = labels[local]
            values = values[values != unknown]
            if len(values):
                counts = np.bincount(values, minlength=256)
                winner = counts.argmax()
                if counts[winner] > len(values) / 2:
                    result[index] = winner
    return result
