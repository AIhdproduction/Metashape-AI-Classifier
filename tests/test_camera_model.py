import numpy as np
import pytest
import torch
from scipy.optimize import least_squares

from msam3.camera_model import Camera, project_numpy, project_torch, similarity_scale


def camera():
    return Camera(
        1,
        800,
        600,
        np.eye(4),
        {
            "f": 500,
            "cx": 3,
            "cy": -2,
            "k1": 0.01,
            "k2": -0.005,
            "k3": 0.001,
            "k4": -0.0001,
            "p1": 0.002,
            "p2": 0.003,
            "p3": 0.3,
            "p4": 0.2,
            "b1": 1.2,
            "b2": 0.7,
        },
    )


def test_numpy_torch_distortion_and_numeric_inverse():
    cam = camera()
    points = np.random.default_rng(5).uniform(-2, 2, (100, 3))
    points[:, 2] += 10
    uv, depth, valid = project_numpy(points, cam)
    actual, actual_depth, actual_valid = project_torch(torch.tensor(points), cam)
    np.testing.assert_allclose(actual.numpy(), uv, atol=1e-10)
    np.testing.assert_allclose(actual_depth.numpy(), depth)
    np.testing.assert_array_equal(actual_valid.numpy(), valid)
    for pixel, original in zip(uv[:10], points[:10], strict=True):

        def residual(xy, original=original, pixel=pixel):
            return project_numpy(np.array([[*xy, original[2]]]), cam)[0][0] - pixel

        fitted = least_squares(residual, [0.0, 0.0], xtol=1e-12, gtol=1e-12, ftol=1e-12)
        np.testing.assert_allclose(fitted.x, original[:2], atol=1e-8)


def test_metashape_231_reference_and_orientation():
    # Independently measured via Calibration.project in the local Metashape 2.3.1 probe.
    cam = Camera(
        1,
        800,
        600,
        np.eye(4),
        {"f": 500, "p1": 0.002, "p2": 0.003, "p3": 0.3, "p4": 0.2},
        orientation=6,
    )
    uv, _, _ = project_numpy(np.array([[1.0, 2.0, 10.0]]), cam)
    np.testing.assert_allclose(uv[0], [450.132015, 400.2386425], atol=1e-10)


def test_behind_outside_and_metric_origin():
    cam = camera()
    _, _, valid = project_numpy(np.array([[0, 0, -1], [0, 0, 0], [100, 0, 1]]), cam)
    assert not valid.any()
    origin, scale = np.array([10.0, 20.0, 5.0]), 3.0
    points = np.array([[1.0, 2.0, 10.0]])
    relative = (points - origin) * scale
    actual = project_numpy(relative, cam, cam.relative_matrix(origin, scale))[0]
    np.testing.assert_allclose(actual, project_numpy(points, cam)[0])
    transform = np.diag([3.0, 3.0, 3.0, 1.0])
    assert similarity_scale(transform) == 3
    transform[0, 0] = 2
    with pytest.raises(ValueError, match="uniform"):
        similarity_scale(transform)
