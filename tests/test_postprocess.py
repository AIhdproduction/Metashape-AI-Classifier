import numpy as np

from msam3.config import Classes, PostprocessConfig
from msam3.pointcloud.postprocess import apply_postprocess


def classes():
    return Classes.model_validate(
        {
            "classes": [
                {"name": "ground", "code": 2, "prompts": ["ground"]},
                {"name": "road", "code": 11, "prompts": ["road"]},
                {"name": "low_vegetation", "code": 3, "prompts": ["low vegetation"]},
                {"name": "building", "code": 6, "prompts": ["building"]},
            ]
        }
    )


def settings():
    return PostprocessConfig.model_validate(
        {
            "neighborhood_radius_m": 0.45,
            "neighbors": 8,
            "npi_preserve_confidence": 0.9,
            "class_rules": {
                "road": {
                    "min_same_neighbors": 2,
                    "fill_neighbors": 3,
                    "fill_share": 0.6,
                    "max_replaced_confidence": 0.6,
                },
                "ground": {
                    "min_same_neighbors": 2,
                    "fill_neighbors": 3,
                    "fill_share": 0.6,
                    "max_replaced_confidence": 0.6,
                },
            },
            "building": {"horizontal_radius_m": 0.4, "min_vertical_drop_m": 1.0},
            "road": {
                "max_slope_degrees": 40,
                "plane_neighbors": 8,
                "surface_radius_m": 1.5,
                "surface_neighbors": 8,
                "min_surface_road_neighbors": 4,
                "min_surface_coverage_sectors": 6,
                "max_surface_offset_m": 0.08,
                "surface_replace_classes": ["low_vegetation"],
            },
        }
    )


def test_postprocess_removes_isolated_points_fills_road_holes_and_repairs_roofs(tmp_path):
    road = np.array(
        [[x, y, 0.0] for x in (-0.1, 0.0, 0.1) for y in (-0.1, 0.0, 0.1)], dtype=np.float32
    )
    roof = np.array(
        [[1.0, 0.0, 2.0], [0.8, 0.0, 0.0], [1.2, 0.0, 0.0], [1.0, -0.2, 0.0], [1.0, 0.2, 0.0]],
        dtype=np.float32,
    )
    points = np.vstack((road, roof, [[4.0, 4.0, 0.0]])).astype(np.float32)
    labels = np.array([11, 11, 11, 11, 1, 11, 11, 11, 11, 2, 6, 6, 6, 6, 2], dtype=np.uint8)
    shares = np.array([0.7] * 4 + [0.0] + [0.7] * 4 + [0.8] + [0.7] * 4 + [0.2])

    filtered = apply_postprocess(
        points, labels, shares, classes(), settings(), tmp_path, batch_size=32
    )

    assert filtered[4] == 11  # Low-confidence hole surrounded by horizontal road.
    assert filtered[9] == 6  # Ground point is clearly above four building support points.
    assert filtered[-1] == 1  # Isolated low-confidence ground point is removed by NPI.
    assert (tmp_path / "post_sam_filter" / "postprocess_report.json").is_file()


def test_postprocess_rejects_vertical_road_surfaces(tmp_path):
    points = np.array(
        [[2.0, x, z] for x in (-0.1, 0.0, 0.1) for z in (-0.1, 0.0, 0.1)], dtype=np.float32
    )
    labels = np.full(len(points), 11, dtype=np.uint8)
    shares = np.full(len(points), 0.9, dtype=np.float32)

    filtered = apply_postprocess(
        points, labels, shares, classes(), settings(), tmp_path, batch_size=32
    )

    assert np.all(filtered == 1)


def test_postprocess_replaces_vegetation_enclosed_by_a_flat_road(tmp_path):
    road = np.array(
        [[x, y, 0.0] for x in (-0.1, 0.0, 0.1) for y in (-0.1, 0.0, 0.1) if (x, y) != (0.0, 0.0)],
        dtype=np.float32,
    )
    enclosed = np.array([[0.0, 0.0, 0.02]], dtype=np.float32)
    roadside = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    points = np.vstack((road, enclosed, roadside))
    labels = np.array([11] * len(road) + [3, 3], dtype=np.uint8)
    shares = np.full(len(points), 0.99, dtype=np.float32)

    filtered = apply_postprocess(
        points, labels, shares, classes(), settings(), tmp_path, batch_size=32
    )

    assert filtered[-2] == 11  # High-confidence false vegetation inside the road is corrected.
    assert filtered[-1] == 3  # Vegetation beside the road lacks all-around road support.
