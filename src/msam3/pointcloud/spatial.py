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


def representative_classifications(source: Path, indices, batch_size: int) -> np.ndarray:
    """Read original Metashape classification values for sorted voting representatives."""
    indices = np.asarray(indices)
    if len(indices) and np.any(np.diff(indices) < 0):
        raise ValueError("Representative indices must be sorted")
    classifications = np.empty(len(indices), dtype=np.uint8)
    offset = 0
    with laspy.open(source) as reader:
        for block in reader.chunk_iterator(batch_size):
            left, right = np.searchsorted(indices, [offset, offset + len(block)])
            if left != right:
                local = indices[left:right] - offset
                classifications[left:right] = np.asarray(block.classification)[local]
            offset += len(block)
    return classifications


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
