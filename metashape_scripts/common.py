"""Shared embedded-Python helpers. Standard library and Metashape only."""

import csv
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import Metashape

LOG = logging.getLogger("msam3.metashape")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_job():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) != 2:
        raise RuntimeError("Expected exactly one script argument: the absolute job.json path")
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    check_metashape()
    LOG.info("Metashape %s; job %s", Metashape.app.version, sys.argv[1])
    return job


def check_metashape():
    version = tuple(int(p) for p in Metashape.app.version.split(".")[:2])
    if version < (2, 0):
        raise RuntimeError("Metashape Professional 2.x is required; version 1.x is unsupported")
    if version < (2, 2):
        raise RuntimeError("Metashape 2.2+ is required for verified point-cloud import options")
    if version[0] != 2:
        raise RuntimeError(
            "This adapter supports Metashape 2.x; newer major APIs need verification"
        )
    if not Metashape.app.activated:
        raise RuntimeError(
            "No active Metashape Professional license. Make a floating license "
            "available or activate your installation, then retry. See the log."
        )


def open_document(project):
    path = Path(project)
    locks = [path.with_suffix(".files") / "lock", Path(str(path) + ".lock")]
    if any(lock.exists() for lock in locks):
        raise RuntimeError("Project is locked. Close it in the Metashape GUI before running msam3.")
    document = Metashape.Document()
    try:
        document.open(str(path), read_only=False, ignore_lock=False)
    except RuntimeError as error:
        raise RuntimeError(
            "Cannot open project for writing. Close it in Metashape and check its lock, "
            "permissions and Professional license: " + str(error)
        ) from error
    return document


def matrix_values(matrix):
    return [[float(matrix[row, col]) for col in range(4)] for row in range(4)]


def calibration_values(calibration):
    names = ("f", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2", "b1", "b2")
    values = {name: float(getattr(calibration, name)) for name in names}
    for name in ("p3", "p4"):
        values[name] = float(getattr(calibration, name, 0.0))
    return values


def camera_record(camera, project):
    calibration = camera.calibration or camera.sensor.calibration
    if calibration is None:
        raise RuntimeError("Missing adjusted calibration for camera " + camera.label)
    if camera.photo is None or not camera.photo.path:
        raise RuntimeError("Missing source photo for camera " + camera.label)
    path = Path(camera.photo.path)
    if not path.is_absolute():
        path = Path(project).parent / path
    return {
        "key": camera.key,
        "label": camera.label,
        "image_path": str(path.resolve()),
        "width": calibration.width,
        "height": calibration.height,
        "transform": matrix_values(camera.transform),
        "calibration": calibration_values(calibration),
        "orientation": camera.orientation,
    }


def source_signature(chunk, cameras):
    value = {
        "point_cloud_key": chunk.point_cloud.key,
        "point_count": chunk.point_cloud.point_count,
        "transform": matrix_values(chunk.transform.matrix),
        "crs": chunk.crs.wkt if chunk.crs else None,
        "cameras": cameras,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def safe_name(label):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")[:80] or "chunk"


def export_cloud(chunk, path, crs, key, classes=None, progress=None):
    parameters = dict(
        path=str(path),
        format=Metashape.PointCloudFormatLAS,
        source_data=Metashape.PointCloudData,
        point_clouds=[key],
        crs=crs,
        save_point_color=True,
        save_point_classification=True,
        save_point_confidence=True,
        save_point_normal=True,
        colors_rgb_8bit=False,
        binary=True,
        clip_to_boundary=False,
        clip_to_region=False,
    )
    if classes is not None:
        parameters["classes"] = classes
    if progress is not None:
        parameters["progress"] = progress
    chunk.exportPointCloud(**parameters)
    if not Path(path).is_file():
        raise RuntimeError("Metashape did not create the LAS export: " + str(path))


def write_legends(directory, classes):
    rows = [
        {
            "name": "Unclassified",
            "code": classes["unknown_code"],
            "color": classes.get("unknown_color", "#808080"),
            "prompts": [],
        }
    ]
    rows.extend(classes["classes"])
    write_json(Path(directory) / "class_legend.json", rows)
    with (Path(directory) / "class_legend.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "code", "color", "prompts"])
        writer.writeheader()
        writer.writerows(
            dict(
                row,
                color=row.get("color", classes.get("unknown_color", "#808080")),
                prompts=" | ".join(row["prompts"]),
            )
            for row in rows
        )


def run_guarded(function):
    try:
        function(read_job())
    except Exception:
        LOG.exception("Metashape stage failed")
        raise SystemExit(1) from None
