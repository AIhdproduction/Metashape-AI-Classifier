"""Per-class geometry review: a label moves only where the target's local surface carries it."""

import numpy as np
import pytest
from pydantic import ValidationError
from pyproj import CRS

from msam3 import backends
from msam3.config import GeometryReviewRule, load_settings
from msam3.pointcloud.postprocess import (
    _fit_height_planes,
    _review_geometry_by_class,
    apply_postprocess,
    level_rotation,
)
from msam3.pointcloud.spatial import metric_up

FLURSEG = backends.MODELS["flurseg"]
BUILDING, OTHER, CONSTRUCTION, LOW_VEGETATION, CURB, GROUND = 6, 64, 67, 3, 74, 2


@pytest.fixture
def flurseg():
    classes = FLURSEG.classes()
    return FLURSEG.apply_profile(load_settings(), classes), classes


def grid(axis_a, axis_b, fixed, value, step=0.1, size=2.0):
    """A plane of points on a square grid; ``fixed`` is the constant axis."""
    a, b = np.meshgrid(np.arange(0, size, step), np.arange(0, size, step))
    columns = {fixed: np.full(a.size, value), axis_a: a.ravel(), axis_b: b.ravel()}
    return np.column_stack([columns[0], columns[1], columns[2]]).astype(np.float32)


def scene(candidates, candidate_normals, candidate_labels):
    """A 2 x 2 m roof at z=3 and a 2 x 2 m facade at x=0, both labelled building."""
    roof = grid(0, 1, 2, 3.0)
    wall = grid(1, 2, 0, 0.0)
    points = np.vstack((roof, wall, np.asarray(candidates, dtype=np.float32)))
    normals = np.vstack(
        (
            np.tile([0, 0, 1], (len(roof), 1)),
            np.tile([1, 0, 0], (len(wall), 1)),
            np.asarray(candidate_normals),
        )
    ).astype(np.float32)
    labels = np.array([BUILDING] * (len(roof) + len(wall)) + list(candidate_labels), dtype=np.uint8)
    return points, normals, labels


def review(settings, classes, points, labels, normals, up=None):
    labels = labels.copy()
    changes = _review_geometry_by_class(
        points, labels, normals, classes, settings.postprocess, 1000, up
    )
    return labels, changes


def test_unknown_fills_a_hole_but_never_grows_past_the_edge(flurseg):
    settings, classes = flurseg
    candidates = [[0.0, 1.05, 1.05], [0.0, 2.05, 1.0], [1.05, 1.05, 3.0], [2.05, 1.0, 3.0]]
    normals = [[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]]
    points, normals, labels = scene(candidates, normals, [1, 1, 1, 1])
    result, _ = review(settings, classes, points, labels, normals)
    # Inside facade and roof: filled. 5 cm past their edges, on the same plane: left alone.
    assert result[-4:].tolist() == [BUILDING, 1, BUILDING, 1]


def test_flurseg_sources_count_only_on_their_own_surface(flurseg):
    settings, classes = flurseg
    candidates = [
        [0.0, 1.05, 1.05],  # other on the facade -> building
        [1.05, 1.05, 3.0],  # flat other on the roof -> stays other
        [1.15, 1.15, 3.0],  # construction on the roof -> building
        [0.0, 1.15, 1.15],  # construction on the facade (scaffolding) -> stays
        [1.05, 0.55, 3.0],  # low vegetation is never a source
    ]
    normals = [[1, 0, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1]]
    labels_in = [OTHER, OTHER, CONSTRUCTION, CONSTRUCTION, LOW_VEGETATION]
    points, normals, labels = scene(candidates, normals, labels_in)
    result, changes = review(settings, classes, points, labels, normals)
    assert result[-5:].tolist() == [BUILDING, OTHER, BUILDING, CONSTRUCTION, LOW_VEGETATION]
    assert changes == {"building": {OTHER: 1, CONSTRUCTION: 1}}


def test_flurseg_profile_never_turns_low_vegetation_into_building(flurseg):
    settings, _ = flurseg
    for rule in settings.postprocess.geometry_rules.values():
        assert "low_vegetation" not in rule.source_classes


def test_candidate_beside_a_protected_kerb_keeps_its_label(flurseg):
    settings, classes = flurseg
    candidates = [[1.05, 1.05, 3.0], [1.15, 1.15, 3.0]]
    points, normals, labels = scene(candidates, [[0, 0, 1], [0, 0, 1]], [CONSTRUCTION, CURB])
    result, _ = review(settings, classes, points, labels, normals)
    assert result[-2:].tolist() == [CONSTRUCTION, CURB]


def test_normal_must_agree_with_the_plane(flurseg):
    settings, classes = flurseg
    # On the roof plane, but its own normal is horizontal: a different surface touches here.
    points, normals, labels = scene([[1.15, 1.15, 3.0]], [[1, 0, 0]], [CONSTRUCTION])
    result, _ = review(settings, classes, points, labels, normals)
    assert result[-1] == CONSTRUCTION


def test_roof_and_facade_follow_the_true_vertical_in_a_tilted_frame(flurseg):
    settings, classes = flurseg
    candidates = [[0.0, 1.05, 1.05], [1.05, 1.05, 3.0], [1.15, 1.15, 3.0]]
    normals_in = [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
    points, normals, labels = scene(candidates, normals_in, [OTHER, OTHER, CONSTRUCTION])
    expected, _ = review(settings, classes, points, labels, normals)
    # Metashape's internal frame was tilted by 31.7 degrees in the test project; at 60 degrees
    # a facade normal would even look horizontal if z were taken as up.
    angle = np.radians(60)
    tilt = np.array(
        [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    )
    up = tilt @ [0, 0, 1]
    tilted, _ = review(
        settings,
        classes,
        (points @ tilt.T).astype(np.float32),
        labels,
        (normals @ tilt.T).astype(np.float32),
        up,
    )
    assert tilted.tolist() == expected.tolist()
    wrong, _ = review(
        settings,
        classes,
        (points @ tilt.T).astype(np.float32),
        labels,
        (normals @ tilt.T).astype(np.float32),
    )
    assert wrong.tolist() != expected.tolist()


def test_two_targets_choose_the_closer_plane(flurseg):
    settings, classes = flurseg
    config = settings.postprocess.model_copy(deep=True)
    config.geometry_rules = {
        "building": GeometryReviewRule(include_unknown=True, surfaces=["horizontal"]),
        "ground": GeometryReviewRule(include_unknown=True, surfaces=["horizontal"]),
    }
    roof = grid(0, 1, 2, 3.0)
    ground = grid(0, 1, 2, 2.92)  # 8 cm below: both planes lie within the unknown limit.
    candidate = np.array([[1.05, 1.05, 2.99]], dtype=np.float32)
    points = np.vstack((roof, ground, candidate))
    normals = np.tile(np.array([0, 0, 1], np.float32), (len(points), 1))
    labels = np.array([BUILDING] * len(roof) + [GROUND] * len(ground) + [1], np.uint8)
    _review_geometry_by_class(points, labels, normals, classes, config, 1000)
    assert labels[-1] == BUILDING


def test_geometry_rule_validation():
    with pytest.raises(ValidationError, match="listed twice"):
        GeometryReviewRule(surfaces=["vertical", "vertical"])
    with pytest.raises(ValidationError, match="exceeds neighbors"):
        GeometryReviewRule(neighbors=8, min_support_neighbors=9)
    assert GeometryReviewRule().unknown_min_coverage_sectors == 7


def test_level_rotation_and_metric_up():
    up = np.array([-0.0013, 0.5253, 0.8509])
    rotation = level_rotation(up)
    assert np.allclose(rotation @ (up / np.linalg.norm(up)), [0, 0, 1])
    assert np.allclose(rotation @ rotation.T, np.eye(3))
    # Local or projected CRS: world z is up, seen through the chunk's linear transform.
    transform = np.eye(4)
    transform[:3, :3] = level_rotation([0, 1, 1]) * 2.0  # Internal (0,1,1) maps to world z.
    assert np.allclose(metric_up(transform, np.zeros(3), None), np.array([0, 1, 1]) / 2**0.5)
    # Geocentric export: the ellipsoid normal at the cloud (here 47 N, 7.5 E).
    latitude, longitude = np.radians(47.0), np.radians(7.5)
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    n = a / np.sqrt(1 - e2 * np.sin(latitude) ** 2)
    centre = np.array(
        [
            n * np.cos(latitude) * np.cos(longitude),
            n * np.cos(latitude) * np.sin(longitude),
            n * (1 - e2) * np.sin(latitude),
        ]
    )
    transform = np.eye(4)
    transform[:3, 3] = centre
    wkt = CRS.from_epsg(4978).to_wkt("WKT1_GDAL")  # As Metashape writes it.
    expected = [
        np.cos(latitude) * np.cos(longitude),
        np.cos(latitude) * np.sin(longitude),
        np.sin(latitude),
    ]
    assert np.allclose(metric_up(transform, np.zeros(3), wkt), expected, atol=1e-9)


def test_levelled_postfilter_matches_an_untilted_cloud(flurseg, tmp_path):
    settings, classes = flurseg
    rng = np.random.default_rng(3)
    road = grid(0, 1, 2, 0.0, step=0.08, size=3.0)
    points = (road + rng.normal(0, 0.003, road.shape)).astype(np.float32)
    labels = np.full(len(points), 71, np.uint8)  # FlurSeg paved_surface, level in reality.
    normals = np.tile(np.array([0, 0, 1], np.float32), (len(points), 1))
    shares = np.full(len(points), 0.9, np.float32)
    angle = np.radians(50)  # Steeper than the 40 degree road limit.
    tilt = np.array(
        [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    )

    def run(level, frame):
        config = settings.postprocess.model_copy(deep=True)
        config.level_to_vertical = level
        return apply_postprocess(
            (points @ frame.T).astype(np.float32),
            labels,
            shares,
            classes,
            config,
            tmp_path,
            1000,
            normals=(normals @ frame.T).astype(np.float32),
            up=frame @ [0, 0, 1],
        )

    reference = run(False, np.eye(3))
    assert np.mean(reference == 71) > 0.99
    # The internal frame alone must not turn a level road into a steep one.
    assert np.array_equal(run(True, tilt), reference)
    assert np.mean(run(False, tilt) == 71) < 0.5


def test_closed_form_height_planes_match_least_squares():
    rng = np.random.default_rng(1)
    xy = rng.uniform(-1, 1, (200, 8, 2)) + rng.uniform(-400, 400, (200, 1, 2))
    z = 0.1 * xy[..., 0] - 0.05 * xy[..., 1] + 3 + rng.normal(0, 0.01, (200, 8))
    valid = rng.random((200, 8)) > 0.2
    valid[:, :3] = True
    _, coefficients = _fit_height_planes(xy, z, valid)
    for row in range(200):
        design = np.column_stack((xy[row][valid[row]], np.ones(valid[row].sum())))
        expected = np.linalg.lstsq(design, z[row][valid[row]], rcond=None)[0]
        height = design @ coefficients[row]
        assert np.allclose(height, design @ expected, atol=1e-6)
    # Collinear neighbours keep the former pseudo-inverse result, unchanged.
    line = np.stack([np.arange(8) * 0.25 + 300, np.full(8, 200.0)], -1)[None]
    flat = np.ones((1, 8), bool)
    heights = 0.02 * line[..., 0] + 1
    design = np.concatenate((line, np.ones((1, 8, 1))), axis=2)
    expected = np.linalg.pinv(design[0].T @ design[0]) @ design[0].T @ heights[0]
    assert np.allclose(_fit_height_planes(line, heights, flat)[1][0], expected)


def site(rng, slope_degrees=0.0, walls=True):
    """A hillside of terrain with two box buildings, in a level frame (z up)."""
    x, y = rng.uniform(-30, 30, (2, 120_000))
    z = np.tan(np.radians(slope_degrees)) * x + rng.normal(0, 0.01, x.size)
    parts = [np.column_stack((x, y, z))]
    if walls:
        for cx, cy in ((-10, 5), (12, -8)):
            base = np.tan(np.radians(slope_degrees)) * cx
            height = rng.uniform(0, 8, 20_000) + base
            side = rng.uniform(-4, 4, 20_000)
            for face in range(4):
                ox, oy = [(4, 0), (-4, 0), (0, 4), (0, -4)][face]
                if ox:
                    parts.append(np.column_stack((np.full(20_000, cx + ox), cy + side, height)))
                else:
                    parts.append(np.column_stack((cx + side, np.full(20_000, cy + oy), height)))
    return np.vstack(parts)


def tilt_matrix(degrees):
    """A rotation that tilts the level frame's z axis by ``degrees``."""
    radians = np.radians(degrees)
    return level_rotation([0, np.sin(radians), np.cos(radians)])


@pytest.mark.parametrize(
    ("slope", "walls", "limit"),
    [(0, False, 1.0), (0, True, 1.0), (20, True, 1.0)],
)
def test_vertical_comes_from_the_cloud_in_any_frame(slope, walls, limit):
    from msam3.pointcloud.spatial import estimate_up

    rng = np.random.default_rng(5)
    level = site(rng, slope, walls)
    cameras = np.array([[0, 0, 80.0], [20, 10, 80.0], [-20, -10, 80.0]])
    for tilt in (0, 31.7, 75, 150):
        rotation = tilt_matrix(tilt)
        points = (level @ rotation.T).astype(np.float32)
        true_up = rotation @ [0, 0, 1]
        moved_cameras = cameras @ rotation.T
        up, info = estimate_up(points, moved_cameras)
        angle = np.degrees(np.arccos(np.clip(up @ true_up, -1, 1)))
        # Also upside down (150 degrees): the cameras above the scene give the sign.
        assert angle < limit, (tilt, angle, info)
        assert info["method"] == ("walls" if walls else "surfaces")


def test_walls_overrule_a_hillside():
    from msam3.pointcloud.spatial import estimate_up

    rng = np.random.default_rng(6)
    points = site(rng, 20, walls=True).astype(np.float32)
    up, info = estimate_up(points, np.array([[0, 0, 80.0]]))
    assert np.degrees(np.arccos(up[2])) < 1.0 and info["surface_to_final_degrees"] > 5


def test_cloud_without_surfaces_is_reported():
    from msam3.pointcloud.spatial import estimate_up

    noise = np.random.default_rng(7).uniform(-5, 5, (50_000, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="planar"):
        estimate_up(noise, None)


def test_pipeline_vertical_falls_back_without_planes_or_crs():
    from types import SimpleNamespace

    from msam3.pipeline import vertical

    noise = np.random.default_rng(8).uniform(-5, 5, (20_000, 3)).astype(np.float32)
    cloud = SimpleNamespace(origin_internal=np.zeros(3), metres_per_internal=1.0)
    camera = SimpleNamespace(transform=np.eye(4))
    broken = {"label": "x", "export_crs_wkt": "not a CRS"}
    up, info = vertical(noise, [camera], cloud, np.eye(4), broken)
    assert info["source"] == "frame_z" and up.tolist() == [0, 0, 1]
    wkt = CRS.from_epsg(2056).to_wkt("WKT1_GDAL")
    up, info = vertical(noise, [camera], cloud, np.eye(4), {"label": "x", "export_crs_wkt": wkt})
    assert info["source"] == "crs" and np.allclose(up, [0, 0, 1])


def meadow(rng):
    """A 20 x 20 m sloping meadow (grass top +-1 cm) with a 0.6 m bush, all low vegetation."""
    x, y = rng.uniform(0, 20, (2, 60_000))
    z = 0.1 * x + rng.normal(0, 0.01, x.size)
    bush = (x - 10) ** 2 + (y - 10) ** 2 < 1.0
    z[bush] += rng.uniform(0.3, 0.6, bush.sum())
    return np.column_stack((x, y, z)).astype(np.float32), bush


def test_meadow_on_the_terrain_model_becomes_ground_but_the_bush_stays(flurseg):
    from msam3.pointcloud.postprocess import _restore_dtm_ground

    settings, classes = flurseg
    rng = np.random.default_rng(9)
    points, bush = meadow(rng)
    labels = np.full(len(points), LOW_VEGETATION, np.uint8)
    labels[:50] = CURB  # Protected classes and other classes are never touched.
    for tilt in (0, 31.7):
        rotation = tilt_matrix(tilt)
        result = labels.copy()
        report = _restore_dtm_ground(
            (points @ rotation.T).astype(np.float32),
            result,
            classes,
            settings.postprocess,
            10_000,
            rotation @ [0, 0, 1],
        )
        open_meadow = ~bush & (labels == LOW_VEGETATION)
        assert np.mean(result[open_meadow] == GROUND) > 0.9, tilt
        assert np.mean(result[bush] == LOW_VEGETATION) > 0.95, tilt
        assert (result[:50] == CURB).all() and report["nodes"] > 300


def test_terrain_model_ground_can_be_switched_off(flurseg):
    from msam3.pointcloud.postprocess import _restore_dtm_ground

    settings, classes = flurseg
    config = settings.postprocess.model_copy(deep=True)
    config.dtm_ground.enabled = False
    points, _ = meadow(np.random.default_rng(9))
    labels = np.full(len(points), LOW_VEGETATION, np.uint8)
    assert _restore_dtm_ground(points, labels, classes, config, 10_000)["nodes"] == 0
    assert (labels == LOW_VEGETATION).all()


def test_compact_cover_check_keeps_manholes_and_moves_long_or_raised_groups(flurseg):
    from scipy.spatial import cKDTree

    from msam3.pointcloud.postprocess import _check_compact_classes

    settings, classes = flurseg
    road = grid(0, 1, 2, 0.0, step=0.05, size=8.0)
    labels = np.full(len(road), 71, np.uint8)
    x, y = road[:, 0], road[:, 1]
    manhole = (x - 2) ** 2 + (y - 2) ** 2 < 0.3**2
    channel = (x > 4) & (x < 7.5) & (y > 5) & (y < 5.6)
    labels[manhole | channel] = 75
    box = grid(0, 2, 1, 2.0, step=0.05, size=1.0) + np.float32([5, 0, 0])  # 1 m high wall
    points = np.vstack((road, box))
    labels = np.r_[labels, np.full(len(box), 75, np.uint8)]
    fragment = np.flatnonzero((np.abs(x - 6.5) < 0.06) & (np.abs(y - 2) < 0.06))
    labels[fragment] = 75
    for tilt in (0, 31.7):
        rotation = tilt_matrix(tilt)
        moved = (points @ rotation.T).astype(np.float32)
        result = labels.copy()
        report = _check_compact_classes(
            cKDTree(moved), moved, result, classes, settings.postprocess, rotation @ [0, 0, 1]
        )
        assert (result[: len(road)][manhole] == 75).all(), tilt
        assert (result[: len(road)][channel] == 71).all(), tilt
        assert (result[fragment] == 71).all(), tilt
        assert not (result[len(road) :] == 75).any(), tilt
        reasons = report["utility_cover"]["rejected_groups"]
        assert reasons == {"not_flat": 1, "too_long": 1, "too_small": 1}, tilt


def test_sam3_has_protected_cover_and_marking_classes_with_flurseg_codes():
    from msam3.config import load_classes

    settings = load_settings()
    names = {item.name: item.code for item in load_classes().classes}
    assert names["utility_cover"] == 75 and names["road_marking"] == 82
    for name in ("utility_cover", "road_marking"):
        assert name in settings.postprocess.protected_classes
        assert name in settings.visibility.no_erosion_classes
    assert "utility_cover" in settings.postprocess.protected_classes
    assert "utility_cover" in settings.visibility.no_erosion_classes
    assert "utility_cover" in settings.postprocess.compact_classes


def test_both_models_use_one_colour_per_las_code():
    from msam3.config import load_classes

    sam = {item.code: item.color.upper() for item in load_classes().classes}
    flurseg = {item.code: item.color.upper() for item in FLURSEG.classes().classes}
    shared = set(sam) & set(flurseg)
    assert {2, 3, 5, 6, 67, 74, 75, 82, 91} <= shared
    assert {code: sam[code] for code in shared} == {code: flurseg[code] for code in shared}
    assert load_classes().unknown_color == FLURSEG.classes().unknown_color
