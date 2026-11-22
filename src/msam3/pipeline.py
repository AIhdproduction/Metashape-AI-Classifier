"""External stage 2: validated images -> visibility -> votes -> original-resolution LAS."""

import logging
import time
from dataclasses import replace
from pathlib import Path

import laspy
import numpy as np
import torch
from PIL import Image

from msam3 import backends
from msam3.cache import LabelCache, atomic_json
from msam3.camera_model import project_torch
from msam3.config import Classes, Settings
from msam3.pointcloud.las_io import write_classified
from msam3.pointcloud.postprocess import apply_postprocess
from msam3.pointcloud.smoothing import smooth
from msam3.pointcloud.spatial import (
    metric_cloud,
    propagate,
    representative_class_mask,
    representative_classifications,
    spatial_blocks,
    voxel_representatives,
)
from msam3.pointcloud.visibility import ZBuffer
from msam3.pointcloud.voting import Votes, apply_ground_assist, apply_vegetation_assist
from msam3.report import write_report
from msam3.segmentation.mask_filter import interior_labels
from msam3.validation import image_path, load_raw_image, read_package, validate_package

LOG = logging.getLogger(__name__)


def segmentation_identity(settings: Settings, classes: Classes) -> str:
    return backends.selected(settings).segmentation_identity(settings, classes)


def representative_normals(source: Path, indices, transform, batch_size):
    with laspy.open(source) as reader:
        available = set(reader.header.point_format.dimension_names)
        names = next(
            (
                names
                for names in (
                    ("normal_x", "normal_y", "normal_z"),
                    ("NormalX", "NormalY", "NormalZ"),
                    ("nx", "ny", "nz"),
                )
                if set(names).issubset(available)
            ),
            None,
        )
        if names is None:
            raise ValueError(
                "Incidence weighting requested but LAS has no normal vector "
                "extra dimensions. Disable voting.use_incidence_weight."
            )
        normals = np.empty((len(indices), 3), dtype=np.float32)
        offset = 0
        for points in reader.chunk_iterator(batch_size):
            left, right = np.searchsorted(indices, [offset, offset + len(points)])
            ids = indices[left:right] - offset
            normals[left:right] = np.column_stack([np.asarray(points[name])[ids] for name in names])
            offset += len(points)
    # Normals are covectors: world/export normals -> internal normals via A^T.
    normals = normals @ transform[:3, :3]
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 0) or not np.isfinite(normals).all():
        raise ValueError("Incidence weighting requires finite, nonzero normals")
    return normals / lengths[:, None]


def process_chunk(
    directory: Path,
    settings: Settings,
    classes: Classes,
    model_hash: str,
    backend_factory=None,
    voting_only: bool = False,
    validation: dict | None = None,
    device: str | None = None,
    model: dict | None = None,
) -> dict:
    started = time.perf_counter()
    segmenter = backends.selected(settings)
    model = model or segmenter.describe()
    # The class-dependent part of the settings follows the selected model.
    settings = segmenter.apply_profile(settings, classes)
    code_of = {item.name: item.code for item in classes.classes}
    no_erosion = [
        index
        for index, item in enumerate(classes.classes)
        if item.name in settings.visibility.no_erosion_classes
    ]
    validation = validation or validate_package(directory, settings)
    metadata, cameras, _ = read_package(directory)
    device = device or settings.gpu.device
    transform = np.array(metadata.get("export_transform", metadata["transform"]), dtype=np.float64)
    cloud = metric_cloud(
        directory / "dense.las",
        transform,
        directory,
        settings.voting.batch_points,
        metadata.get("metres_per_world_unit", 1.0),
    )
    indices = voxel_representatives(
        cloud.points,
        settings.voting.voxel_size_m,
        directory / "voxels",
        settings.voting.batch_points,
        settings.voting.voxel_buckets,
    )
    points = np.asarray(cloud.points[indices])
    ground_hint = representative_class_mask(
        directory / "dense.las", indices, 2, settings.voting.batch_points
    )
    metashape_classes = representative_classifications(
        directory / "dense.las", indices, settings.voting.batch_points
    )
    LOG.info(
        "Chunk %s: %d original points -> %d voting points",
        metadata["label"],
        len(cloud.points),
        len(points),
    )
    blocks = spatial_blocks(points, settings.voting.block_size_m)
    votes = Votes(len(points), classes, settings.voting, directory)
    normals = (
        representative_normals(
            directory / "dense.las", indices, transform, settings.voting.batch_points
        )
        if settings.voting.use_incidence_weight
        else None
    )
    # One folder per model: switching back reuses its images, and the identity still decides.
    cache = LabelCache(
        directory / "cache" / segmenter.key, segmentation_identity(settings, classes), model_hash
    )
    preparation = time.perf_counter() - started
    segmentation_seconds, voting_seconds, images_used = 0.0, 0.0, 0
    backend = None
    for camera_index, camera in enumerate(cameras, start=1):
        # A cropped dense cloud often leaves some registered cameras looking only at
        # discarded areas.  Skip them before loading an image or invoking the model.
        # Blocks use conservative frustum culling, so a useful camera is never
        # rejected merely because its projected block corners are imperfect.
        matrix = camera.relative_matrix(cloud.origin_internal, cloud.metres_per_internal)
        if next(blocks.batches(camera, matrix, settings.voting.batch_points), None) is None:
            LOG.info(
                "Chunk %s: camera %d/%d (%s) skipped; no remaining cloud points in view",
                metadata["label"],
                camera_index,
                len(cameras),
                camera.label,
            )
            continue
        image_start = time.perf_counter()
        size = (
            max(1, round(camera.width * settings.cache.scale)),
            max(1, round(camera.height * settings.cache.scale)),
        )
        identity = cache.identity(camera, image_path(camera, settings), size)
        cached = cache.load(str(camera.key), identity)
        if cached is None:
            if voting_only:
                raise RuntimeError(
                    f"Missing/stale segmentation cache for camera {camera.label}. "
                    "Rerun stage2 without --voting-only."
                )
            if backend is None:
                if backend_factory is None:
                    raise RuntimeError("A segmentation backend is required for uncached images")
                backend = backend_factory()
            image = load_raw_image(camera, settings)
            try:
                labels, confidence = segmenter.segment(
                    backend, image, classes, settings, directory / "tile_scores.dat"
                )
                cached = cache.save(str(camera.key), labels, confidence, identity)
            finally:
                image.close()
            if hasattr(backend, "release_image"):
                backend.release_image()
        else:
            LOG.info(
                "Chunk %s: camera %d/%d (%s) reused from cache",
                metadata["label"],
                camera_index,
                len(cameras),
                camera.label,
            )
        labels, confidence = cached
        if np.any((labels != 255) & (labels >= len(classes.classes))):
            raise ValueError("Cached image contains out-of-range class indices")
        vote_labels = interior_labels(
            labels,
            confidence,
            settings.visibility.mask_interior_px,
            settings.visibility.min_mask_confidence,
            no_erosion,
        )
        segmentation_seconds += time.perf_counter() - image_start
        vote_start = time.perf_counter()
        zbuffer = None
        depth_map = None
        if settings.visibility.mode == "zbuffer":
            zbuffer = ZBuffer(camera.width, camera.height, settings.visibility, device)
            for ids in blocks.batches(camera, matrix, settings.voting.batch_points):
                xyz = torch.as_tensor(points[ids], device=device)
                uv, z, valid = project_torch(xyz, camera, matrix)
                zbuffer.add(uv[valid], z[valid] * cloud.metres_per_internal)
        else:
            info = camera.extra.get("depth_map")
            if info is None:
                raise ValueError("depth_maps mode requires maps exported by stage1 in this mode")
            with Image.open(info["path"]) as image:
                depth_map = torch.as_tensor(np.array(image, dtype=np.float32), device=device)
            if depth_map.shape != (info["height"], info["width"]):
                raise ValueError("Invalid calibrated depth-map dimensions")
            depth_camera = replace(
                camera, width=info["width"], height=info["height"], calibration=info["calibration"]
            )
        for ids in blocks.batches(camera, matrix, settings.voting.batch_points):
            xyz = torch.as_tensor(points[ids], device=device)
            uv, z, valid = project_torch(xyz, camera, matrix)
            local_ids = torch.nonzero(valid).flatten()
            if not len(local_ids):
                continue
            uv, z = uv[valid], z[valid] * cloud.metres_per_internal
            if zbuffer is not None:
                visible = zbuffer.visible(uv, z)
            else:
                depth_uv, _, depth_valid = project_torch(xyz[valid], depth_camera, matrix)
                x = depth_uv[:, 0].nan_to_num().long().clamp(0, depth_camera.width - 1)
                y = depth_uv[:, 1].nan_to_num().long().clamp(0, depth_camera.height - 1)
                reference = depth_map[y, x] * cloud.metres_per_internal
                visible = depth_valid & torch.isfinite(reference) & (reference > 0)
                visible &= (
                    z
                    <= reference * (1 + settings.visibility.rel_tolerance)
                    + settings.visibility.abs_tolerance_m
                )
            visible_ids = ids[local_ids[visible].cpu().numpy()]
            pixels = uv[visible].cpu().numpy()
            x = np.clip((pixels[:, 0] * size[0] / camera.width).astype(int), 0, size[0] - 1)
            y = np.clip((pixels[:, 1] * size[1] / camera.height).astype(int), 0, size[1] - 1)
            weight = 1.0
            if normals is not None:
                center = (
                    camera.transform[:3, 3] - cloud.origin_internal
                ) * cloud.metres_per_internal
                rays = center - points[visible_ids]
                rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
                weight = np.abs(np.sum(rays * normals[visible_ids], axis=1))
            votes.add(visible_ids, vote_labels[y, x], confidence[y, x], weight)
        images_used += 1
        voting_seconds += time.perf_counter() - vote_start
        LOG.info(
            "Chunk %s: camera %d/%d (%s) complete; %d images used",
            metadata["label"],
            camera_index,
            len(cameras),
            camera.label,
            images_used,
        )
        del zbuffer, depth_map
    LOG.info("Chunk %s: all cameras processed; post-filter and LAS follow", metadata["label"])
    result, shares = votes.decide()
    result, shares, ground_assist = apply_ground_assist(
        result, shares, votes, ground_hint, settings.ground_assist
    )
    result, shares, vegetation_assist = apply_vegetation_assist(
        result, shares, metashape_classes, classes, settings.vegetation_assist
    )
    result = smooth(
        points,
        result,
        shares,
        settings.smoothing,
        classes.unknown_code,
        settings.voting.batch_points,
        {code_of[name] for name in settings.smoothing.protected_classes if name in code_of},
    )
    result = apply_postprocess(
        points,
        result,
        shares,
        classes,
        settings.postprocess,
        directory,
        settings.voting.batch_points,
    )
    output_start = time.perf_counter()
    all_codes = propagate(
        cloud.points, points, result, directory / "all_codes.dat", settings.voting.batch_points
    )
    output = directory / "classified_internal.las"
    counts = write_classified(
        directory / "dense.las",
        output,
        all_codes,
        classes,
        settings.voting.batch_points,
        model["title"],
    )
    timings = {
        "preparation": preparation,
        "segmentation": segmentation_seconds,
        "voting": voting_seconds,
        "output": time.perf_counter() - output_start,
        "stage2": time.perf_counter() - started,
    }
    report = write_report(
        directory / "report.json",
        counts,
        classes,
        images_used,
        votes.views,
        validation,
        timings,
        len(points),
        ground_assist,
        vegetation_assist,
        model,
    )
    atomic_json(
        directory / "stage2_done.json",
        {
            "output": str(output),
            "report": report,
            "model_identity": segmenter.run_identity(settings, classes, model_hash),
        },
    )
    return {
        "directory": str(directory),
        "classified_las": str(output),
        "counts": {str(i): int(v) for i, v in enumerate(counts) if v},
        "report": report,
    }
