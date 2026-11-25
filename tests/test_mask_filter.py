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


@pytest.mark.parametrize("margin", [0, 1, 2, 4])
def test_interior_matches_direct_square_check(margin):
    rng = np.random.default_rng(31)
    labels = rng.choice([0, 1, 2, 255], size=(19, 23)).astype(np.uint8)
    confidence = rng.random((19, 23)).astype(np.float32)
    confidence[4, 6] = np.nan
    expected = np.full(labels.shape, 255, dtype=np.uint8)
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label = labels[y, x]
            if label == 255 or not np.isfinite(confidence[y, x]) or confidence[y, x] < 0.3:
                continue
            if label == 2:
                expected[y, x] = label
                continue
            if (
                y < margin
                or x < margin
                or y + margin >= len(labels)
                or x + margin >= labels.shape[1]
            ):
                continue
            area = labels[y - margin : y + margin + 1, x - margin : x + margin + 1]
            scores = confidence[y - margin : y + margin + 1, x - margin : x + margin + 1]
            if np.all(area == label) and np.all(np.isfinite(scores) & (scores >= 0.3)):
                expected[y, x] = label
    np.testing.assert_array_equal(
        interior_labels(labels, confidence, margin, 0.3, no_erosion=(2,)), expected
    )
