"""Metashape frame projection; raw pixels, X right, Y down, Z forward."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class Camera:
    key: int
    width: int
    height: int
    transform: np.ndarray
    calibration: dict[str, float]
    label: str = ""
    image_path: str = ""
    orientation: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Camera":
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        fields["transform"] = np.asarray(data["transform"], dtype=np.float64)
        fields["extra"] = {k: v for k, v in data.items() if k not in cls.__dataclass_fields__}
        camera = cls(**fields)
        if camera.transform.shape != (4, 4) or not np.isfinite(camera.transform).all():
            raise ValueError(f"Camera {camera.key}: invalid transform")
        if camera.calibration["f"] <= 0 or min(camera.width, camera.height) <= 0:
            raise ValueError(f"Camera {camera.key}: invalid calibration/dimensions")
        return camera

    def relative_matrix(self, origin: np.ndarray, metres_per_internal: float) -> np.ndarray:
        """Map local metric coordinates to camera coordinates, avoiding large float32 offsets."""
        inverse = np.linalg.inv(self.transform)
        matrix = inverse.copy()
        matrix[:3, :3] /= metres_per_internal
        matrix[:3, 3] = inverse[:3, :3] @ origin + inverse[:3, 3]
        return matrix


def _distort(x, y, calibration):
    k = {
        name: calibration.get(name, 0.0)
        for name in ("k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4", "b1", "b2", "cx", "cy")
    }
    r2 = x * x + y * y
    radial = 1 + r2 * (k["k1"] + r2 * (k["k2"] + r2 * (k["k3"] + r2 * k["k4"])))
    tang = 1 + k["p3"] * r2 + k["p4"] * r2 * r2
    xd = x * radial + (k["p1"] * (r2 + 2 * x * x) + 2 * k["p2"] * x * y) * tang
    yd = y * radial + (k["p2"] * (r2 + 2 * y * y) + 2 * k["p1"] * x * y) * tang
    return xd, yd, k


def project_numpy(points: np.ndarray, camera: Camera, matrix: np.ndarray | None = None):
    matrix = np.linalg.inv(camera.transform) if matrix is None else matrix
    xyz = np.asarray(points, dtype=np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
    z = xyz[:, 2]
    safe_z = np.where(z > 0, z, 1.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        xd, yd, k = _distort(xyz[:, 0] / safe_z, xyz[:, 1] / safe_z, camera.calibration)
        u = camera.width / 2 + k["cx"] + xd * (camera.calibration["f"] + k["b1"]) + yd * k["b2"]
        v = camera.height / 2 + k["cy"] + yd * camera.calibration["f"]
    uv = np.column_stack((u, v))
    valid = (z > 0) & np.isfinite(uv).all(axis=1)
    valid &= (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    return uv, z, valid


def project_torch(points: torch.Tensor, camera: Camera, matrix: np.ndarray | None = None):
    matrix = np.linalg.inv(camera.transform) if matrix is None else matrix
    transform = torch.as_tensor(matrix, dtype=points.dtype, device=points.device)
    xyz = points @ transform[:3, :3].T + transform[:3, 3]
    z = xyz[:, 2]
    safe_z = torch.where(z > 0, z, torch.ones_like(z))
    xd, yd, k = _distort(xyz[:, 0] / safe_z, xyz[:, 1] / safe_z, camera.calibration)
    u = camera.width / 2 + k["cx"] + xd * (camera.calibration["f"] + k["b1"]) + yd * k["b2"]
    v = camera.height / 2 + k["cy"] + yd * camera.calibration["f"]
    uv = torch.stack((u, v), dim=1)
    valid = (z > 0) & torch.isfinite(uv).all(dim=1)
    valid &= (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    return uv, z, valid


def similarity_scale(transform: np.ndarray) -> float:
    """Reject shears/nonuniform scale: every threshold is specified in metres."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Invalid chunk transform")
    if not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("Chunk transform is not affine")
    singular = np.linalg.svd(matrix[:3, :3], compute_uv=False)
    if singular.min() <= 0 or not np.allclose(singular, singular[0], rtol=1e-6):
        raise ValueError("Chunk transform must have a uniform, nonzero metric scale")
    return float(singular.mean())
