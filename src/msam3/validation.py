"""Independent geometry gates; all computation here is CPU-only."""

import json
import logging
from pathlib import Path

import laspy
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from msam3.camera_model import Camera, project_numpy, similarity_scale
from msam3.config import Settings

LOG = logging.getLogger(__name__)


def image_path(camera: Camera, settings: Settings) -> Path:
    name = camera.image_path.replace("\\", "/")
    for old, new in sorted(settings.paths.image_path_remap.items(), key=lambda item: -len(item[0])):
        old = old.replace("\\", "/")
        if name.casefold().startswith(old.casefold()):
            name = new.replace("\\", "/") + name[len(old) :]
            break
    path = Path(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Image not found: {path}. Set paths.image_path_remap if images moved."
        )
    return path


def load_raw_image(camera: Camera, settings: Settings) -> Image.Image:
    with Image.open(image_path(camera, settings)) as source:
        if source.size != (camera.width, camera.height):
            raise ValueError(
                f"Camera {camera.key}: image dimensions {source.size} differ from "
                f"calibration {(camera.width, camera.height)}; no automatic rotation"
            )
        # Deliberately do not call ImageOps.exif_transpose: Metashape projects to raw pixels.
        return source.convert("RGB")


def read_package(directory: Path):
    metadata = json.loads((directory / "chunk.json").read_text(encoding="utf-8"))
    cameras = [
        Camera.from_dict(data)
        for data in json.loads((directory / "cameras.json").read_text(encoding="utf-8"))
    ]
    samples = json.loads((directory / "validation_samples.json").read_text(encoding="utf-8"))
    if not cameras:
        raise ValueError("Stage 1 package contains no cameras")
    return metadata, cameras, samples


def validate_package(directory: Path, settings: Settings) -> dict:
    metadata, cameras, samples = read_package(directory)
    transform = np.array(metadata.get("export_transform", metadata["transform"]), np.float64)
    similarity_scale(transform)
    unit = metadata.get("metres_per_world_unit", 1.0)
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError("Invalid CRS metric conversion")
    internal = np.array([sample["internal"] for sample in samples["samples"]], np.float64)
    if internal.ndim != 2 or internal.shape[1] != 3 or not len(internal):
        raise ValueError("Geometry validation requires nonempty 3D tie-point samples")
    by_key = {camera.key: camera for camera in cameras}
    errors = []
    for projection in samples["projections"]:
        uv, depth, _ = project_numpy(
            internal[[projection["sample"]]], by_key[projection["camera_key"]]
        )
        error = float(np.linalg.norm(uv[0] - projection["pixel"]))
        if depth[0] <= 0 or not np.isfinite(error):
            raise ValueError("Validation sample is behind the camera or non-finite")
        errors.append(error)
    if not errors:
        raise ValueError("No ground-truth Metashape projections; cannot validate geometry")
    maximum, mean = float(np.max(errors)), float(np.mean(errors))
    LOG.info("Projection validation: mean %.6f px, max %.6f px", mean, maximum)
    if maximum > settings.validation.max_reprojection_px:
        raise ValueError(
            f"Projection validation failed: max {maximum:.4f} px exceeds "
            f"{settings.validation.max_reprojection_px}. Check calibration, "
            "rolling shutter and image orientation; segmentation has not started."
        )
    world = internal @ transform[:3, :3].T + transform[:3, 3]
    expected_world = np.array(
        [sample.get("export_world", sample["world"]) for sample in samples["samples"]]
    )
    if not np.allclose(world, expected_world, atol=0.001 / unit, rtol=0):
        raise ValueError("Export transform does not reproduce Metashape validation coordinates")
    minimum_distances = np.full(len(world), np.inf)
    with laspy.open(directory / "dense.las") as reader:
        if reader.header.point_count != metadata["point_count"]:
            raise ValueError("Exported LAS count differs from original dense-cloud count")
        tolerance = reader.header.scales * 2
        inside = (
            (world >= reader.header.mins - tolerance) & (world <= reader.header.maxs + tolerance)
        ).all(axis=1)
        if not inside.all():
            raise ValueError(
                f"Validation failed: {int((~inside).sum())} tie points lie outside "
                "the exported dense-cloud bounding box. Check CRS and cloud extent."
            )
        for block in reader.chunk_iterator(settings.voting.batch_points):
            xyz = np.column_stack((block.x, block.y, block.z))
            distances, _ = cKDTree(xyz).query(world, workers=-1)
            np.minimum(minimum_distances, distances * unit, out=minimum_distances)
    median = float(np.median(minimum_distances))
    LOG.info("Tie-to-dense median distance: %.6f m", median)
    if median > settings.validation.max_tie_to_dense_m:
        raise ValueError(
            f"Validation failed: tie-to-dense median {median:.4f} m exceeds "
            f"{settings.validation.max_tie_to_dense_m} m. Check scale/CRS and cloud."
        )
    for camera in cameras:
        with Image.open(image_path(camera, settings)) as image:
            if image.size != (camera.width, camera.height):
                raise ValueError(f"Image/calibration dimension mismatch for {camera.label}")
    if settings.debug.save_overlays:
        target = directory / "overlays"
        target.mkdir(exist_ok=True)
        for camera in cameras[:: settings.debug.overlay_every_n]:
            image = load_raw_image(camera, settings)
            draw = ImageDraw.Draw(image)
            uv, _, valid = project_numpy(internal, camera)
            for u, v in uv[valid]:
                draw.ellipse((u - 4, v - 4, u + 4, v + 4), outline="red", width=2)
            image.save(target / f"{camera.key}.png")
            image.close()
    return {
        "passed": True,
        "mean_reprojection_px": mean,
        "max_reprojection_px": maximum,
        "median_tie_to_dense_m": median,
        "projection_count": len(errors),
        "tie_point_count": len(internal),
    }
