"""Export immutable dense-cloud geometry and independent camera validation samples."""

import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path

import Metashape

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metashape_scripts.common import (
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


def stage1_identity(job):
    """Mirror the driver's cache identity after an in-place ground classification save."""
    project = Path(job["project"])
    paths = [project, project.with_suffix(".files") / "project.zip"]
    project_state = {
        str(path): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in paths
        if path.is_file()
    }
    value = {
        "schema": 1,
        "project": project_state,
        "chunks": job["settings"]["metashape"]["chunks"],
        "depth_maps": job["settings"]["visibility"]["mode"] == "depth_maps",
        "ground_assist": job["settings"]["ground_assist"],
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def point_class_counts(point_cloud):
    return {str(int(code)): int(count) for code, count in point_cloud.point_count_by_class.items()}


# Both models and older versions, including the copies that Metashape numbers.
CLASSIFIED_LABEL = re.compile(r"^(SAM ?3|FlurSeg) classified\b", re.IGNORECASE)


def is_msam3_import(point_cloud):
    """A classified result of any model or version. It must never become the next input."""
    keys = getattr(point_cloud, "meta", {}).keys()
    if "msam3.import_id" in keys or "msam3.source_point_cloud_key" in keys:
        return True
    label = getattr(point_cloud, "label", None)
    return isinstance(label, str) and CLASSIFIED_LABEL.match(label.strip()) is not None


def source_point_cloud(chunk):
    """Return a non-empty original cloud, even if it is not currently selected."""

    def usable(point_cloud):
        return (
            point_cloud is not None
            and not is_msam3_import(point_cloud)
            and point_cloud.point_count > 0
        )

    active = getattr(chunk, "point_cloud", None)
    point_clouds = list(getattr(chunk, "point_clouds", []))
    if usable(active):
        return active
    if active is not None:
        source_key = (
            active.meta["msam3.source_point_cloud_key"]
            if "msam3.source_point_cloud_key" in active.meta.keys()
            else None
        )
        matched = next(
            (
                point_cloud
                for point_cloud in point_clouds
                if str(point_cloud.key) == str(source_key)
            ),
            None,
        )
        if usable(matched):
            return matched
    candidates = [point_cloud for point_cloud in point_clouds if usable(point_cloud)]
    if not candidates:
        return None
    # Prefer the complete source if a project retains multiple unclassified cloud assets.
    return max(candidates, key=lambda point_cloud: point_cloud.point_count)


def classify_ground_support(chunk, job, progress=None):
    """Classify only unclassified source points and expose ground as a hint for the model."""
    config = job["settings"]["ground_assist"]
    point_cloud = chunk.point_cloud
    before = point_class_counts(point_cloud)
    result = {
        "enabled": bool(config["enabled"]),
        "classified_before_export": False,
        "ground_before": before.get("2", 0),
        "ground_after": before.get("2", 0),
        "ground_added": 0,
    }
    if not config["enabled"] or not config["classify_before_export"]:
        return result
    source_classes = {
        "created": Metashape.PointClass.Created,
        "unclassified": Metashape.PointClass.Unclassified,
    }
    for source in config["source_classes"]:
        point_cloud.classifyGroundPoints(
            max_angle=config["max_angle_degrees"],
            max_distance=config["max_distance_m"],
            max_terrain_slope=config["max_terrain_slope_degrees"],
            cell_size=config["cell_size_m"],
            erosion_radius=config["erosion_radius_m"],
            source_class=source_classes[source],
            keep_existing=True,
            **({"progress": progress} if progress is not None else {}),
        )
    after = point_class_counts(point_cloud)
    result["classified_before_export"] = True
    result["ground_after"] = after.get("2", 0)
    result["ground_added"] = result["ground_after"] - result["ground_before"]
    LOG.info(
        "Chunk %s: Metashape ground support %d -> %d points",
        chunk.label,
        result["ground_before"],
        result["ground_after"],
    )
    return result


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


def export_chunk(chunk, job, progress=None):
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
    if not cameras:
        LOG.warning("Skipping chunk %s: no aligned frame cameras", chunk.label)
        return None
    point_cloud = source_point_cloud(chunk)
    if point_cloud is None:
        raise RuntimeError(
            "Chunk "
            + chunk.label
            + " has no usable source point cloud. Run Workflow > Build Point Cloud in Metashape, "
            "save the project, then start the Classifier again."
        )
    if chunk.point_cloud is not point_cloud:
        chunk.point_cloud = point_cloud
        LOG.info("Chunk %s: selected source point cloud %s", chunk.label, point_cloud.label)
    if chunk.crs is None:
        raise RuntimeError("Chunk CRS is missing: " + chunk.label)
    ground_assist = classify_ground_support(chunk, job, progress)
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
        "metashape_ground_assist": ground_assist,
    }
    validation = samples_for(chunk, cameras, export_transform, export_crs)
    export_cloud(
        chunk, directory / "dense.las", export_crs, chunk.point_cloud.key, progress=progress
    )
    write_legends(directory, job["classes"], job.get("model", {}).get("title"))
    if job["settings"]["visibility"]["mode"] == "depth_maps":
        if chunk.depth_maps is None:
            raise RuntimeError("depth_maps mode requires retained Metashape depth maps")
        # No strict=: Python 3.9 (inferred for Metashape 2.2) lacks it; lengths are equal anyway.
        for camera, record in zip(cameras, records):
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


def main(job, document=None, progress=None):
    if document is None:
        document = open_document(job["project"])
    selected = job["settings"]["metashape"]["chunks"]
    if selected:
        missing = set(selected) - {c.label for c in document.chunks}
        if missing:
            raise RuntimeError("Requested chunk labels do not exist: " + ", ".join(sorted(missing)))
    chunks = []
    ground_classified = False
    for chunk in document.chunks:
        if (chunk.label in selected) if selected else chunk.enabled:
            result = export_chunk(chunk, job, progress=progress)
            if result is not None:
                chunks.append(result)
                metadata = json.loads(
                    (Path(result["directory"]) / "chunk.json").read_text(encoding="utf-8")
                )
                ground_classified |= metadata["metashape_ground_assist"]["classified_before_export"]
    if not chunks:
        raise RuntimeError("No eligible chunks: need aligned frame cameras and chunk.point_cloud")
    if ground_classified:
        document.save()
        job["stage1_identity"] = stage1_identity(job)
        LOG.info("Saved Metashape ground classification before segmentation")
    write_json(
        Path(job["work_dir"]) / "stage1_done.json",
        {"schema": 1, "chunks": chunks, "stage1_identity": job["stage1_identity"]},
    )


if __name__ == "__main__":
    run_guarded(main)
