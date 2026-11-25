"""Streaming coordinates, disk-partitioned voxels, conservative blocks and propagation."""

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import laspy
import numpy as np
from scipy.spatial import cKDTree

from msam3.camera_model import Camera, project_numpy, similarity_scale


@dataclass
class MetricCloud:
    points: np.memmap
    origin_internal: np.ndarray
    metres_per_internal: float


def metric_cloud(
    path: Path,
    transform: np.ndarray,
    directory: Path,
    batch_size: int,
    metres_per_world_unit: float = 1.0,
) -> MetricCloud:
    scale = similarity_scale(transform) * metres_per_world_unit
    inverse = np.linalg.inv(transform)
    with laspy.open(path) as reader:
        count = reader.header.point_count
        if count == 0:
            raise ValueError("Exported dense cloud is empty")
        center = (reader.header.mins + reader.header.maxs) / 2
        origin = inverse[:3, :3] @ center + inverse[:3, 3]
        points = np.memmap(
            directory / "metric_points.dat", mode="w+", dtype="float32", shape=(count, 3)
        )
        offset = 0
        for block in reader.chunk_iterator(batch_size):
            xyz = np.column_stack((block.x, block.y, block.z)).astype(np.float64)
            internal = xyz @ inverse[:3, :3].T + inverse[:3, 3]
            relative = (internal - origin) * scale
            if not np.isfinite(relative).all():
                raise ValueError("LAS contains non-finite coordinates")
            points[offset : offset + len(block)] = relative
            offset += len(block)
        points.flush()
    return MetricCloud(points, origin, scale)


def _fibonacci_hemisphere(count: int) -> np.ndarray:
    """Evenly spread unit vectors with z >= 0 (axes: +u and -u are the same direction)."""
    index = np.arange(count) + 0.5
    z = 1 - index / count
    radius = np.sqrt(1 - z * z)
    angle = np.pi * (1 + 5**0.5) * index
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))


def _axial_mean(normals: np.ndarray, axis: np.ndarray, window_degrees: float) -> np.ndarray:
    """Mean direction of the normals within the window around ``axis``, signs aligned."""
    cosine = normals @ axis
    inside = np.abs(cosine) >= np.cos(np.radians(window_degrees))
    if not np.any(inside):
        return axis
    mean = (normals[inside] * np.sign(cosine[inside])[:, None]).sum(axis=0)
    return mean / np.linalg.norm(mean)


def _plane_normals(points, query_count, support_count, radius, rng):
    """PCA normals of planar neighbourhoods, from the coordinates alone."""
    support = points[rng.choice(len(points), min(support_count, len(points)), replace=False)]
    queries = support[rng.choice(len(support), min(query_count, len(support)), replace=False)]
    distances, indices = cKDTree(support).query(
        queries, k=16, distance_upper_bound=radius, workers=-1
    )
    valid = np.isfinite(distances)
    count = valid.sum(axis=1)
    usable = count >= 8
    neighbours = support[np.minimum(indices[usable], len(support) - 1)].astype(np.float64)
    weights = valid[usable][:, :, None]
    mean = (neighbours * weights).sum(axis=1) / count[usable, None]
    centred = (neighbours - mean[:, None, :]) * weights
    eigenvalues, vectors = np.linalg.eigh(np.einsum("bki,bkj->bij", centred, centred))
    total = np.maximum(eigenvalues.sum(axis=1), 1e-12)
    # Flat and extended: little spread across the plane, real spread inside it.
    planar = (eigenvalues[:, 0] / total < 0.01) & (eigenvalues[:, 1] / total > 0.1)
    return vectors[planar, :, 0]


def estimate_up(points, camera_centres=None, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Estimate the true vertical from the point cloud itself, in the frame of ``points``.

    Metashape's internal axes are arbitrary and a project's CRS may be local or missing, so the
    cloud decides. First the most common surface normal (terrain, roads, roofs). Walls are then
    the stronger evidence, because they are vertical also on a hillside: when the cloud has walls
    in at least two directions, the vertical is the direction perpendicular to all of them.
    Cameras look down on the scene and fix the sign.
    """
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype=np.float32)
    info = {"source": "point_cloud"}
    normals = _plane_normals(points, 200_000, 2_000_000, 0.5, rng)
    info["planar_normals"] = int(len(normals))
    if len(normals) < 100:
        raise ValueError("Too few planar surfaces in the point cloud to find the vertical")
    sample = normals[rng.choice(len(normals), min(50_000, len(normals)), replace=False)]
    candidates = _fibonacci_hemisphere(2000)
    threshold = np.cos(np.radians(10))
    scores = np.zeros(len(candidates), dtype=np.int64)
    for start in range(0, len(candidates), 250):
        block = candidates[start : start + 250]
        scores[start : start + 250] = (np.abs(sample @ block.T) >= threshold).sum(axis=0)
    up = candidates[int(np.argmax(scores))]
    for _ in range(5):
        up = _axial_mean(normals, up, 10)
    info["method"] = "surfaces"
    surface_up = up
    # The first wall search is wide, so that walls on a hillside of up to 40 degrees are found
    # although the surface estimate leans with the hill; later passes narrow down to 15.
    for band in (40, 15, 15, 15, 15):
        walls = normals[np.abs(normals @ up) < np.sin(np.radians(band))]
        if len(walls) < max(500, 0.03 * len(normals)):
            break
        values, vectors = np.linalg.eigh(walls.T @ walls / len(walls))
        # Walls of one direction only leave the vertical free to turn about their normal.
        if values[1] < 0.1 * values[2]:
            break
        candidate = vectors[:, 0] * np.sign(vectors[:, 0] @ up)
        if np.degrees(np.arccos(np.clip(candidate @ surface_up, -1, 1))) > 40:
            break
        up = candidate
        info["method"] = "walls"
        info["wall_normals"] = int(len(walls))
    centres = None if camera_centres is None else np.asarray(camera_centres, dtype=np.float64)
    if centres is not None and len(centres):
        height = (centres - points.mean(axis=0)) @ up
        if np.median(height) < 0:
            up = -up
        info["sign"] = "cameras"
    else:
        if up[2] < 0:
            up = -up
        info["sign"] = "internal_z"
    info["up"] = [float(value) for value in up]
    info["tilt_from_frame_z_degrees"] = float(np.degrees(np.arccos(np.clip(abs(up[2]), 0, 1))))
    info["surface_to_final_degrees"] = float(
        np.degrees(np.arccos(np.clip(abs(surface_up @ up), 0, 1)))
    )
    return up, info


def metric_up(
    transform: np.ndarray, origin_internal: np.ndarray, crs_wkt: str | None
) -> np.ndarray:
    """Vertical of the export CRS in the metric frame; only a cross-check of :func:`estimate_up`.

    Metashape's internal axes are arbitrary. Georeferenced chunks are exported in geocentric
    coordinates, whose vertical is the ellipsoid normal at the cloud; a local or projected CRS
    has its z axis up.
    """
    from pyproj import CRS

    linear = np.asarray(transform, dtype=np.float64)[:3, :3]
    up_world = np.array([0.0, 0.0, 1.0])
    crs = CRS.from_wkt(crs_wkt) if crs_wkt else None
    if crs is not None and crs.is_geocentric:
        centre = linear @ np.asarray(origin_internal, dtype=np.float64) + transform[:3, 3]
        # The ellipsoid normal is the gradient of x²/a² + y²/a² + z²/b²; a few hundred metres
        # above the surface it differs from the exact foot-point normal by microradians.
        a = crs.ellipsoid.semi_major_metre
        b = crs.ellipsoid.semi_minor_metre
        up_world = centre / np.array([a * a, a * a, b * b])
    # Directions map with the inverse linear part; the metric scale does not turn them.
    up = np.linalg.solve(linear, up_world)
    return up / np.linalg.norm(up)


def representative_class_mask(source: Path, indices, code: int, batch_size: int) -> np.ndarray:
    """Read a LAS classification only for sorted voxel representatives."""
    indices = np.asarray(indices)
    if len(indices) and np.any(np.diff(indices) < 0):
        raise ValueError("Representative indices must be sorted")
    mask = np.zeros(len(indices), dtype=bool)
    offset = 0
    with laspy.open(source) as reader:
        for block in reader.chunk_iterator(batch_size):
            left, right = np.searchsorted(indices, [offset, offset + len(block)])
            if left != right:
                local = indices[left:right] - offset
                mask[left:right] = np.asarray(block.classification)[local] == code
            offset += len(block)
    return mask


def representative_attributes(
    source: Path,
    indices,
    batch_size: int,
    *,
    include_rgb: bool = False,
    include_normals: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Read source classifications, RGB and normals in one LAS scan."""
    indices = np.asarray(indices)
    if len(indices) and np.any(np.diff(indices) < 0):
        raise ValueError("Representative indices must be sorted")
    classifications = np.empty(len(indices), dtype=np.uint8)
    offset = 0
    with laspy.open(source) as reader:
        available = set(reader.header.point_format.dimension_names)
        has_rgb = include_rgb and {"red", "green", "blue"}.issubset(available)
        normal_names = next(
            (
                names
                for names in (
                    ("normal x", "normal y", "normal z"),
                    ("normal_x", "normal_y", "normal_z"),
                    ("NormalX", "NormalY", "NormalZ"),
                    ("nx", "ny", "nz"),
                )
                if set(names).issubset(available)
            ),
            None,
        )
        colors = np.empty((len(indices), 3), dtype=np.uint16) if has_rgb else None
        normals = (
            np.empty((len(indices), 3), dtype=np.float32)
            if include_normals and normal_names is not None
            else None
        )
        for block in reader.chunk_iterator(batch_size):
            left, right = np.searchsorted(indices, [offset, offset + len(block)])
            if left != right:
                local = indices[left:right] - offset
                classifications[left:right] = np.asarray(block.classification)[local]
                if colors is not None:
                    colors[left:right, 0] = np.asarray(block.red)[local]
                    colors[left:right, 1] = np.asarray(block.green)[local]
                    colors[left:right, 2] = np.asarray(block.blue)[local]
                if normals is not None:
                    for axis, name in enumerate(normal_names):
                        normals[left:right, axis] = np.asarray(block[name])[local]
            offset += len(block)
    return classifications, colors, normals


def representative_classifications(source: Path, indices, batch_size: int) -> np.ndarray:
    """Read original Metashape classification values for sorted voting representatives."""
    return representative_attributes(source, indices, batch_size)[0]


def voxel_representatives(
    points, size: float, directory: Path, batch_size: int = 500_000, buckets: int = 256
) -> np.ndarray:
    """Keep the first original point per voxel, independent of I/O batch boundaries."""
    dtype = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8"), ("index", "<i8")])
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"bucket-{index}.dat" for index in range(buckets)]
    for path in paths:
        path.write_bytes(b"")
    try:
        for start in range(0, len(points), batch_size):
            coordinates = np.floor(
                np.asarray(points[start : start + batch_size], dtype=np.float64) / size
            ).astype(np.int64)
            records = np.empty(len(coordinates), dtype=dtype)
            for axis, name in enumerate(("x", "y", "z")):
                records[name] = coordinates[:, axis]
            records["index"] = np.arange(start, start + len(coordinates))
            # Integer overflow is intentional, giving a reproducible spatial hash.
            hashed = (
                (coordinates[:, 0] * 73856093)
                ^ (coordinates[:, 1] * 19349663)
                ^ (coordinates[:, 2] * 83492791)
            ) % buckets
            order = np.argsort(hashed, kind="stable")
            split = np.r_[0, np.flatnonzero(np.diff(hashed[order])) + 1, len(order)]
            for left, right in zip(split[:-1], split[1:], strict=True):
                with paths[int(hashed[order[left]])].open("ab") as stream:
                    records[order[left:right]].tofile(stream)
        selected = []
        for path in paths:
            records = np.fromfile(path, dtype=dtype)
            if len(records):
                _, first = np.unique(records[["x", "y", "z"]], return_index=True)
                selected.append(records["index"][first])
        return np.sort(np.concatenate(selected))
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


@dataclass
class Blocks:
    order: np.ndarray
    offsets: np.ndarray
    corners: np.ndarray

    def batches(self, camera: Camera, matrix: np.ndarray, batch_size: int):
        flat = self.corners.reshape(-1, 3)
        uv, z, _ = project_numpy(flat, camera, matrix)
        z = z.reshape(-1, 8)
        keep = z.max(axis=1) > 0
        # Distorted projection can turn between corners. Cull only behind-camera blocks
        # in that case; rejecting a block merely because no corner is inside is incorrect.
        distorted = any(
            camera.calibration.get(k, 0) for k in ("k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4")
        )
        if not distorted:
            uv = uv.reshape(-1, 8, 2)
            intersects = (
                (uv[:, :, 0].max(axis=1) >= 0)
                & (uv[:, :, 0].min(axis=1) < camera.width)
                & (uv[:, :, 1].max(axis=1) >= 0)
                & (uv[:, :, 1].min(axis=1) < camera.height)
            )
            keep &= (z.min(axis=1) <= 0) | intersects
        pending, length = [], 0
        for index in np.flatnonzero(keep):
            left, right = self.offsets[index : index + 2]
            for start in range(left, right, batch_size):
                ids = self.order[start : min(right, start + batch_size)]
                pending.append(ids)
                length += len(ids)
                if length >= batch_size:
                    yield np.concatenate(pending)
                    pending, length = [], 0
        if pending:
            yield np.concatenate(pending)


def spatial_blocks(points: np.ndarray, size: float) -> Blocks:
    keys = np.floor(points.astype(np.float64) / size).astype(np.int64)
    order = np.lexsort(keys.T[::-1])
    sorted_keys = keys[order]
    offsets = np.r_[
        0, np.flatnonzero(np.any(np.diff(sorted_keys, axis=0), axis=1)) + 1, len(points)
    ]
    lower = sorted_keys[offsets[:-1]].astype(np.float64) * size
    corners = lower[:, None, :] + np.array(list(product((0, size), repeat=3)))[None, :, :]
    return Blocks(order, offsets, corners)


def propagate(
    points, representatives: np.ndarray, codes: np.ndarray, output: Path, batch_size: int
) -> np.memmap:
    tree = cKDTree(representatives)
    labels = np.memmap(output, mode="w+", dtype="uint8", shape=(len(points),))
    for start in range(0, len(points), batch_size):
        _, indices = tree.query(points[start : start + batch_size], workers=-1)
        labels[start : start + len(indices)] = codes[indices]
    labels.flush()
    return labels
