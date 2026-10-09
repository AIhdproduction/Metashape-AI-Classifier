"""Export immutable dense-cloud geometry and independent camera validation samples."""

import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path

import Metashape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    calibration_values,
    camera_record,
    export_cloud,
    matrix_values,
    open_document,
    run_guarded,
    source_signature,
    write_json,
    write_legends,
)

LOG = logging.getLogger("msam3.stage1")


def samples_for(chunk, cameras, export_transform, export_crs):
    points = []
    generator = random.Random(0)
    valid_count = 0
    if chunk.tie_points is not None:
        for point in chunk.tie_points.points:
            if not point.valid:
                continue
            coordinate = point.coord
            divisor = coordinate[3] if len(coordinate) == 4 else 1.0
            if divisor == 0:
                raise RuntimeError("Valid tie point has an invalid homogeneous coordinate")
            coordinate = Metashape.Vector([coordinate[i] / divisor for i in range(3)])
            valid_count += 1
            if len(points) < 500:
                points.append(coordinate)
            else:
                slot = generator.randrange(valid_count)
                if slot < 500:
                    points[slot] = coordinate
    if not points:
        raise RuntimeError("Validation requires valid tie points; none remain in " + chunk.label)
    records = []
    for point in points:
        world = chunk.transform.matrix.mulp(point)
        if chunk.crs.geoccs is not None:
            exported = Metashape.CoordinateSystem.transform(world, chunk.crs.geoccs, export_crs)
        else:
            exported = world
        affine = export_transform.mulp(point)
        if (exported - affine).norm() > 0.001:
            raise RuntimeError("Non-affine datum conversion exceeds 1 mm; export in WGS 84 first")
        records.append(
            {"internal": list(point), "world": list(world), "export_world": list(exported)}
        )
    positions = sorted(
        {
            round(i * (len(cameras) - 1) / max(1, min(20, len(cameras)) - 1))
            for i in range(min(20, len(cameras)))
        }
    )
    projections = []
    for index in positions:
        camera = cameras[index]
        calibration = camera.calibration or camera.sensor.calibration
        for sample, point in enumerate(points):
            pixel = camera.project(point)
            if pixel is not None and 0 <= pixel.x < calibration.width:
                if 0 <= pixel.y < calibration.height:
                    projections.append(
                        {"camera_key": camera.key, "sample": sample, "pixel": list(pixel)}
                    )
    if not projections:
        raise RuntimeError("No in-image validation projections in chunk " + chunk.label)
    return {"samples": records, "projections": projections}


def export_chunk(chunk, job):
    skipped = Counter()
    cameras = []
    for camera in chunk.cameras:
        if not camera.enabled:
            skipped["disabled"] += 1
        elif camera.transform is None:
            skipped["unaligned"] += 1
        elif camera.sensor.type != Metashape.Sensor.Type.Frame:
            kind = str(camera.sensor.type).lower()
            reason = (
                "fisheye" if "fisheye" in kind else "spherical" if "spherical" in kind else "other"
            )
            skipped[reason] += 1
        else:
            cameras.append(camera)
            if camera.sensor.rolling_shutter != Metashape.Shutter.Model.Disabled:
                LOG.warning(
                    "Camera %s has rolling shutter compensation; the projection gate "
                    "must pass before frame-model processing",
                    camera.label,
                )
    LOG.info("Chunk %s: %d frame cameras; skipped %s", chunk.label, len(cameras), dict(skipped))
    if not cameras or chunk.point_cloud is None:
        LOG.warning(
            "Skipping chunk %s: aligned frame cameras and a dense cloud are required", chunk.label
        )
        return None
    if chunk.crs is None:
        raise RuntimeError("Chunk CRS is missing: " + chunk.label)
    directory = Path(job["work_dir"]) / ("chunk-" + str(chunk.key))
    directory.mkdir(parents=True, exist_ok=True)
    transform = chunk.transform.matrix
    export_transform = transform
    unit = 1.0
    if chunk.crs.geoccs is not None:
        export_crs = Metashape.CoordinateSystem("EPSG::4978")
        datum = Metashape.CoordinateSystem.datumTransform(chunk.crs.geoccs, export_crs)
        export_transform = datum * transform
    else:
        export_crs = chunk.crs
        units = re.findall(r'UNIT\["[^"]+",\s*([\d.eE+-]+)', chunk.crs.wkt)
        if not units:
            raise RuntimeError("Cannot establish local CRS units in metres: " + chunk.crs.wkt)
        unit = float(units[-1])
    records = [camera_record(camera, job["project"]) for camera in cameras]
    package = {
        "key": chunk.key,
        "label": chunk.label,
        "point_cloud_key": chunk.point_cloud.key,
        "point_count": chunk.point_cloud.point_count,
        "transform": matrix_values(transform),
        "export_transform": matrix_values(export_transform),
        "crs_wkt": chunk.crs.wkt,
        "export_crs_wkt": export_crs.wkt,
        "metres_per_world_unit": unit,
        "metashape_version": Metashape.app.version,
        "skipped_cameras": dict(skipped),
        "source_signature": source_signature(chunk, records),
    }
    validation = samples_for(chunk, cameras, export_transform, export_crs)
    export_cloud(chunk, directory / "dense.las", export_crs, chunk.point_cloud.key)
    write_legends(directory, job["classes"])
    if job["settings"]["visibility"]["mode"] == "depth_maps":
        if chunk.depth_maps is None:
            raise RuntimeError("depth_maps mode requires retained Metashape depth maps")
        for camera, record in zip(cameras, records, strict=False):
            if camera not in chunk.depth_maps.keys():
                raise RuntimeError("No depth map for camera " + camera.label)
            depth = chunk.depth_maps[camera]
            calibration = depth.calibration
            path = directory / ("depth-" + str(camera.key) + ".tif")
            depth.image().save(str(path))
            record["depth_map"] = {
                "path": str(path),
                "width": calibration.width,
                "height": calibration.height,
                "calibration": calibration_values(calibration),
            }
    write_json(directory / "chunk.json", package)
    write_json(directory / "cameras.json", records)
    write_json(directory / "validation_samples.json", validation)
    return {
        "key": chunk.key,
        "label": chunk.label,
        "directory": str(directory),
        "camera_count": len(cameras),
        "point_count": chunk.point_cloud.point_count,
    }


def main(job):
    document = open_document(job["project"])
    selected = job["settings"]["metashape"]["chunks"]
    if selected:
        missing = set(selected) - {c.label for c in document.chunks}
        if missing:
            raise RuntimeError("Requested chunk labels do not exist: " + ", ".join(sorted(missing)))
    chunks = []
    for chunk in document.chunks:
        if (chunk.label in selected) if selected else chunk.enabled:
            result = export_chunk(chunk, job)
            if result is not None:
                chunks.append(result)
    if not chunks:
        raise RuntimeError("No eligible chunks: need aligned frame cameras and chunk.point_cloud")
    # Deliberately do not save: stage 1 must not change the project.
    write_json(
        Path(job["work_dir"]) / "stage1_done.json",
        {"schema": 1, "chunks": chunks, "stage1_identity": job["stage1_identity"]},
    )


if __name__ == "__main__":
    run_guarded(main)
