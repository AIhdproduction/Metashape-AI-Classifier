import numpy as np
import pytest

from msam3.segmentation.mask_filter import interior_labels


def test_only_confident_mask_interiors_can_vote():
    labels = np.full((9, 9), 2, dtype=np.uint8)
    labels[3:6, 3:6] = 3
    confidence = np.ones((9, 9), dtype=np.float32)
    confidence[1, 1] = 0.4

    filtered = interior_labels(labels, confidence, margin=1, min_confidence=0.5)

    assert filtered[4, 4] == 3  # Stable vegetation-mask interior.
    assert filtered[3, 3] == 255  # Class boundary cannot vote.
    assert filtered[2, 4] == 255  # Road pixel adjacent to vegetation cannot vote.
    assert filtered[1, 1] == 255  # Low-confidence pixel cannot vote.
    assert filtered[0, 4] == 255  # Image edge cannot vote.
    assert filtered[7, 7] == 2  # Stable road-mask interior remains usable.


def test_interior_labels_rejects_incompatible_arrays():
    with pytest.raises(ValueError, match="equally sized"):
        interior_labels(np.zeros((2, 2), np.uint8), np.zeros((2, 3), np.float32), 1, 0.5)
