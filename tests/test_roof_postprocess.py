"""Ground on a supported roof can be removed without taking roads or terrain with it."""

import numpy as np
import pytest

from msam3.backends import MODELS
from msam3.config import load_settings
from msam3.pointcloud.postprocess import _repair_roof_ground, apply_postprocess


def scene(slope):
    """Known roof surrounding a Ground patch, plus ground, a terrace and a below-roof road."""
    grid = np.arange(-2.0, 2.01, 0.15)
    roof = np.array([[x, y, 4.0 + slope * x] for x in grid for y in grid], np.float32)
    roof_normal = np.array([-slope, 0, 1], np.float32)
    roof_normal /= np.linalg.norm(roof_normal)
    hole = (np.abs(roof[:, 0]) < 0.65) & (np.abs(roof[:, 1]) < 0.65)
    nearby_ground = np.array(
        [
            [x, y, z]
            for x in np.arange(2.3, 3.8, 0.15)
            for y in np.arange(-1.0, 1.0, 0.15)
            for z in (4.0 + slope * x,)
        ],
        np.float32,
    )
    outside = np.arange(-5.0, 5.1, 0.25)
    terrain = np.array(
        [[x, y, 0.0] for x in outside for y in outside if abs(x) > 2.5 or abs(y) > 2.5],
        np.float32,
    )
    road = np.array(
        [
            [x, y, 3.9 + slope * x]
            for x in np.arange(-0.4, 0.4, 0.15)
            for y in np.arange(-0.4, 0.4, 0.15)
        ],
        np.float32,
    )
    points = np.vstack((roof, nearby_ground, terrain, road))
    normals = np.tile(roof_normal, (len(points), 1))
    terrain_start = len(roof) + len(nearby_ground)
    normals[terrain_start : terrain_start + len(terrain)] = (0, 0, 1)
    return points, normals, hole, len(roof), len(road)


@pytest.mark.parametrize("model", ["sam3", "flurseg"])
@pytest.mark.parametrize("slope", [0.0, 0.75])
@pytest.mark.parametrize("tilt", [0, 30])
def test_ground_inside_flat_or_pitched_roof_only(model, slope, tilt):
    backend = MODELS[model]
    classes = backend.classes()
    cfg = backend.apply_profile(load_settings(), classes).postprocess
    points, normals, hole, roof_count, road_count = scene(slope)
    labels = np.full(len(points), 2, np.uint8)
    labels[:roof_count] = np.where(hole, 2, 6)
    labels[-road_count:] = 71 if model == "flurseg" else 11
    before = labels.copy()
    yaw = np.deg2rad(tilt)
    rotation = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]], np.float32
    )

    count = _repair_roof_ground(
        points @ rotation.T, labels, normals @ rotation.T, classes, cfg, 256, rotation[:, 2]
    )

    assert count == hole.sum()
    np.testing.assert_array_equal(labels[:roof_count], 6)
    # This includes adjacent Ground at roof elevation, the lower terrain, and every road point.
    np.testing.assert_array_equal(labels[roof_count:], before[roof_count:])


def test_ground_stays_ground_when_same_plane_has_no_lower_terrain():
    backend = MODELS["sam3"]
    classes = backend.classes()
    cfg = backend.apply_profile(load_settings(), classes).postprocess
    points, normals, hole, roof_count, _ = scene(0.0)
    labels = np.where(hole, 2, 6).astype(np.uint8)
    before = labels.copy()

    count = _repair_roof_ground(
        points[:roof_count], labels, normals[:roof_count], classes, cfg, 256
    )

    assert count == 0
    np.testing.assert_array_equal(labels, before)


def test_roof_filter_at_end_changes_only_ground(tmp_path):
    backend = MODELS["flurseg"]
    classes = backend.classes()
    cfg = backend.apply_profile(load_settings(), classes).postprocess
    points, normals, hole, roof_count, road_count = scene(0.0)
    labels = np.full(len(points), 2, np.uint8)
    labels[:roof_count] = np.where(hole, 2, 6)
    points[-road_count:, 0] += 6.0
    points[-road_count:, 2] = 0.0
    labels[-road_count:] = 71
    shares = np.ones(len(points), np.float32)
    baseline = cfg.model_copy(deep=True)
    baseline.building.enabled = False

    before = apply_postprocess(
        points, labels, shares, classes, baseline, tmp_path / "before", 512, normals=normals
    )
    after = apply_postprocess(
        points, labels, shares, classes, cfg, tmp_path / "after", 512, normals=normals
    )

    changed = before != after
    assert np.any(changed[:roof_count] & hole)
    assert np.all(before[changed] == 2)
    assert np.all(after[changed] == 6)
    np.testing.assert_array_equal(after[roof_count:], before[roof_count:])
    np.testing.assert_array_equal(after[-road_count:], 71)
