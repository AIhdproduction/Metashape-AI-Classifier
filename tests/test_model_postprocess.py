import numpy as np
import pytest

from msam3 import backends, user_classes
from msam3.config import SmoothingConfig, load_classes, load_settings
from msam3.pointcloud.postprocess import apply_postprocess
from msam3.pointcloud.smoothing import smooth
from msam3.pointcloud.voting import Votes, apply_ground_assist, apply_vegetation_assist

FLURSEG = backends.MODELS["flurseg"]


@pytest.fixture
def flurseg():
    classes = FLURSEG.classes()
    settings = FLURSEG.apply_profile(load_settings(), classes)
    return settings, classes


def noisy_road(seed=4):
    """A 3 cm road with a wide and a one-point-wide marking, a raised kerb and a cover."""
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.arange(-1.5, 1.5, 0.03), np.arange(-1.2, 1.2, 0.03))
    columns = np.rint((xx.ravel() + 1.5) / 0.03).astype(int)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    points[:, :2] += 0.0137 + rng.uniform(-0.005, 0.005, (len(points), 2))
    points[:, 2] += rng.normal(0, 0.004, len(points))
    labels = np.full(len(points), 71, np.uint8)
    labels[(columns >= 60) & (columns < 64)] = 82
    labels[columns == 22] = 82
    kerb = (columns >= 94) & (columns < 98)
    labels[kerb] = 74
    points[kerb, 2] += 0.03
    labels[(points[:, 0] + 0.31) ** 2 + (points[:, 1] - 0.13) ** 2 < 0.24**2] = 75
    thin = np.isin(labels, (82, 74, 75))
    shares = rng.uniform(0.71, 0.99, len(points))
    # The attackable case: thin classes win their vote only narrowly.
    shares[thin] = rng.uniform(0.401, 0.449, thin.sum())
    holes = ~thin & (rng.random(len(points)) < 0.03)
    labels_with_holes = labels.copy()
    labels_with_holes[holes] = 1
    shares[holes] = rng.uniform(0.01, 0.1, holes.sum())
    return points.astype(np.float32), labels, labels_with_holes, shares.astype(np.float32)


def test_flurseg_postfilter_keeps_thin_classes_in_a_noisy_road(flurseg, tmp_path):
    settings, classes = flurseg
    points, truth, labels, shares = noisy_road()

    result = apply_postprocess(
        points, labels, shares, classes, settings.postprocess, tmp_path, batch_size=1999
    )

    for code in (82, 74, 75):
        assert np.mean(result[truth == code] == code) > 0.97, code
    filled = (truth == 71) & (labels == 1)
    assert np.mean(result[filled] == 71) > 0.8  # Road repair still closes the real holes.


def test_the_protection_is_what_keeps_the_thin_line(flurseg, tmp_path):
    settings, classes = flurseg
    points, truth, labels, shares = noisy_road()
    unprotected = settings.postprocess.model_copy(deep=True)
    unprotected.protected_classes = []
    unprotected.cluster.min_same_overrides = {}

    result = apply_postprocess(points, labels, shares, classes, unprotected, tmp_path, 1999)

    assert np.mean(result[truth == 82] == 82) < 0.95


def test_surface_replacement_never_takes_a_protected_class(flurseg, tmp_path):
    settings, classes = flurseg
    config = settings.postprocess.model_copy(deep=True)
    config.road.surface_replace_classes = ["road_marking", "curb", "utility_cover"]
    points, truth, labels, shares = noisy_road()
    result = apply_postprocess(points, labels, shares, classes, config, tmp_path, 1999)
    assert np.mean(result[truth == 82] == 82) > 0.97


def test_smoothing_keeps_thin_classes(flurseg):
    settings, _ = flurseg
    points, truth, _, shares = noisy_road()
    config = SmoothingConfig(enabled=True, k=15)
    erased = smooth(points, truth, shares, config, 1, 1999)
    kept = smooth(points, truth, shares, config, 1, 1999, {82, 74, 75})
    thin = np.isin(truth, (82, 74, 75))
    assert np.any(erased[thin] != truth[thin])
    np.testing.assert_array_equal(kept[thin], truth[thin])
    assert settings.smoothing.protected_classes == ["curb", "road_marking", "utility_cover"]


@pytest.mark.parametrize("terrain", [71, 82, 75, 74])
def test_vegetation_height_uses_each_flurseg_terrain_code(flurseg, tmp_path, terrain):
    settings, classes = flurseg
    rng = np.random.default_rng(terrain)
    xx, yy = np.meshgrid(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1))
    ground = np.column_stack((xx.ravel(), yy.ravel(), rng.normal(0, 0.005, xx.size)))
    ground[:, :2] += rng.uniform(0.011, 0.029, (len(ground), 2))
    plants = np.array([[0.137, 0.111, 0.137], [-0.223, 0.173, 0.913]])
    points = np.vstack((ground, plants)).astype(np.float32)
    labels = np.array([terrain] * len(ground) + [5, 3], np.uint8)
    config = settings.postprocess.model_copy(deep=True)
    # Two single plants: only the height split is under test.
    config.cluster.enabled = False
    for rule in config.class_rules.values():
        rule.min_same_neighbors = 0
    config.road.enabled = False
    shares = rng.uniform(0.71, 0.99, len(points))

    result = apply_postprocess(points, labels, shares, classes, config, tmp_path, 999)

    assert result[-2:].tolist() == [3, 5]


def test_flurseg_assists(flurseg, tmp_path):
    settings, classes = flurseg
    votes = Votes(3, classes, settings.voting, tmp_path)
    votes.weights[0, [1, 2, 4, 7]] = [0.11, 0.31, 0.33, 0.37]
    votes.views[0] = 1
    labels, shares = votes.decide()
    labels, shares, _ = apply_ground_assist(
        labels, shares, votes, np.array([True, True, False]), settings.ground_assist
    )
    assert labels.tolist() == [71, 2, 1]
    labels, _, _ = apply_vegetation_assist(
        labels, shares, np.array([2, 2, 5]), classes, settings.vegetation_assist
    )
    assert labels.tolist() == [71, 2, 5]


@pytest.mark.parametrize("disabled", [item.name for item in load_classes().classes] + ["many"])
def test_disabled_sam_classes_never_crash_the_postfilter(tmp_path, disabled):
    standard = load_classes()
    names = [disabled]
    if disabled == "many":
        names = ["ground", "road", "sidewalk", "low_vegetation", "tree", "building"]
    classes = user_classes.apply(standard, user_classes.UserClasses(disabled=names))
    settings = load_settings()
    settings.segmentation.backend = "sam3"
    settings = backends.MODELS["sam3"].apply_profile(settings, classes)
    rng = np.random.default_rng(12)
    points = rng.uniform(-1, 1, (1200, 3)).astype(np.float32)
    points[:, 2] *= 0.17
    codes = [item.code for item in classes.classes] + [1]
    labels = rng.choice(codes, len(points)).astype(np.uint8)
    shares = rng.uniform(0.2, 0.9, len(points)).astype(np.float32)

    result = apply_postprocess(points, labels, shares, classes, settings.postprocess, tmp_path, 500)

    assert set(result) <= set(codes)


def test_own_class_gets_small_object_cleanup(tmp_path):
    standard = load_classes()
    edits = user_classes.save_class(
        standard, user_classes.UserClasses(), "manhole", ["manhole cover"], "#123456"
    )
    classes = user_classes.apply(standard, edits)
    settings = load_settings()
    settings.segmentation.backend = "sam3"
    settings = backends.MODELS["sam3"].apply_profile(settings, classes)
    assert settings.postprocess.class_rules["manhole"] == backends.CUSTOM_CLASS_RULE
    assert "manhole" not in load_settings().postprocess.class_rules
    rng = np.random.default_rng(2)
    cover = rng.uniform(-0.15, 0.15, (90, 3))
    cover[:, 2] *= 0.03
    points = np.vstack((cover, [[4.013, 4.027, 1.019]])).astype(np.float32)
    labels = np.full(len(points), classes.classes[-1].code, np.uint8)
    shares = rng.uniform(0.5, 0.9, len(points)).astype(np.float32)

    result = apply_postprocess(points, labels, shares, classes, settings.postprocess, tmp_path, 50)

    assert np.all(result[:-1] == classes.classes[-1].code) and result[-1] == 1
