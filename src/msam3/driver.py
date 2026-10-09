"""Job manifests, stage orchestration, project locks, resume, and final LAS assembly."""

import json
import logging
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

import laspy
import numpy as np

from msam3.cache import atomic_json, file_hash, json_hash
from msam3.config import Classes, Settings
from msam3.metashape_runner import run_metashape
from msam3.pipeline import process_chunk
from msam3.pointcloud.las_io import merge_class_exports
from msam3.segmentation.sam3_backend import SAM3Backend, require_checkpoint
from msam3.validation import validate_package

LOG = logging.getLogger(__name__)


def project_state(project: Path) -> dict:
    paths = [project, project.with_suffix(".files") / "project.zip"]
    return {
        str(path): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in paths
        if path.is_file()
    }


def check_project(project: Path) -> None:
    if not project.is_file() or project.suffix.lower() not in (".psx", ".psz"):
        raise FileNotFoundError(f"Saved Metashape .psx/.psz project not found: {project}")
    for lock in (project.with_suffix(".files") / "lock", Path(str(project) + ".lock")):
        if lock.exists():
            raise RuntimeError(f"Project is locked ({lock}). Close it in Metashape before running.")


@contextmanager
def job_lock(path: Path):
    """OS lock is released on process exit, including crashes; the file may remain."""
    stream = path.open("a+b")
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise RuntimeError(
            "Another msam3 process is using this project's work directory"
        ) from error
    try:
        yield
    finally:
        stream.close()


def job_directory(project: Path, settings: Settings) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", project.stem)
    suffix = json_hash(str(project.resolve()).casefold())[:12]
    return settings.paths.work_dir / f"{safe}-{suffix}"


def execute(
    action: str, project: Path, settings: Settings, classes: Classes, voting_only: bool = False
) -> dict:
    # The missing-model instruction takes precedence over starting any external application.
    if action in ("run", "stage2"):
        require_checkpoint(settings.paths.model)
    project = project.resolve()
    check_project(project)
    work = job_directory(project, settings)
    work.mkdir(parents=True, exist_ok=True)
    output = settings.paths.output_dir / work.name
    handler = logging.FileHandler(work / "msam3.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        with job_lock(work / "msam3.lock"):
            return _execute(action, project, work, output, settings, classes, voting_only)
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def _execute(action, project, work, output, settings, classes, voting_only):
    started = time.perf_counter()
    identity = json_hash(
        {
            "schema": 1,
            "project": project_state(project),
            "chunks": settings.metashape.chunks,
            "depth_maps": settings.visibility.mode == "depth_maps",
        }
    )
    job = {
        "schema": 1,
        "project": str(project),
        "work_dir": str(work),
        "output_dir": str(output),
        "settings": settings.model_dump(mode="json"),
        "classes": classes.model_dump(),
        "stage1_identity": identity,
    }
    job_path = work / "job.json"
    atomic_json(job_path, job)
    stage1_marker = work / "stage1_done.json"
    stage1 = (
        json.loads(stage1_marker.read_text(encoding="utf-8")) if stage1_marker.is_file() else None
    )
    reusable = stage1 is not None and stage1.get("stage1_identity") == identity
    if reusable:
        reusable = all(
            all(
                (Path(chunk["directory"]) / name).is_file()
                for name in ("dense.las", "chunk.json", "cameras.json", "validation_samples.json")
            )
            for chunk in stage1["chunks"]
        )
    stage1_seconds = 0.0
    if not reusable:
        if action in ("stage2", "stage3"):
            raise RuntimeError(
                "Stage 1 export is missing or stale. Run msam3 stage1 --project ... first."
            )
        start = time.perf_counter()
        stage1 = run_metashape(1, job_path, work, settings.metashape)
        stage1_seconds = time.perf_counter() - start
    else:
        LOG.info("Reusing verified stage 1 package in %s", work)
    if action == "stage1":
        return stage1
    if action in ("run", "validate", "stage2"):
        # Validate ALL chunks before checking CUDA or constructing the model.
        validation = {
            chunk["key"]: validate_package(Path(chunk["directory"]), settings)
            for chunk in stage1["chunks"]
        }
        atomic_json(work / "validation.json", validation)
        if action == "validate":
            return validation
        model_hash = file_hash(settings.paths.model)
        LOG.info("Checkpoint SHA-256: %s", model_hash)
        backend = None

        def backend_factory():
            nonlocal backend
            if backend is None:
                backend = SAM3Backend(settings.paths.model, settings.gpu, settings.sam)
            return backend

        inputs = []
        for chunk in stage1["chunks"]:
            result = process_chunk(
                Path(chunk["directory"]),
                settings,
                classes,
                model_hash,
                backend_factory,
                voting_only,
                validation[chunk["key"]],
            )
            result["report"]["timing_seconds"]["stage1"] = stage1_seconds
            result["classes_hash"] = json_hash(classes.model_dump())
            result["stage1_identity"] = identity
            result["las_sha256"] = file_hash(Path(result["classified_las"]))
            atomic_json(Path(chunk["directory"]) / "stage2_result.json", result)
            inputs.append(result)
        if action == "stage2":
            return {"chunks": inputs}
    else:
        inputs = []
        for chunk in stage1["chunks"]:
            path = Path(chunk["directory"]) / "stage2_result.json"
            if not path.is_file():
                raise RuntimeError("Stage 2 result is missing; run msam3 stage2 first")
            inputs.append(json.loads(path.read_text(encoding="utf-8")))
    for result in inputs:
        if result["classes_hash"] != json_hash(classes.model_dump()):
            raise ValueError("Classes changed since stage 2. Rerun stage2 before importing.")
        if result["stage1_identity"] != identity:
            raise ValueError("Stage 2 results belong to an older stage 1 export; rerun stage2.")
        if file_hash(Path(result["classified_las"])) != result["las_sha256"]:
            raise ValueError("Classified LAS changed since stage 2; rerun stage2")
        metadata = json.loads(
            (Path(result["directory"]) / "chunk.json").read_text(encoding="utf-8")
        )
        result["import_id"] = json_hash(
            {"las": result["las_sha256"], "source": metadata["source_signature"]}
        )
        with laspy.open(result["classified_las"]) as reader:
            # Explicit millimetre precision prevents Metashape's coarse auto quantization.
            result["precision_m"] = min(0.001, float(np.min(reader.header.scales)))
    job["stage3_inputs"] = inputs
    atomic_json(job_path, job)
    start = time.perf_counter()
    stage3 = run_metashape(3, job_path, work, settings.metashape)
    # Persist our own project save before final LAS assembly, so disk-full recovery can resume.
    job["stage1_identity"] = json_hash(
        {
            "schema": 1,
            "project": project_state(project),
            "chunks": settings.metashape.chunks,
            "depth_maps": settings.visibility.mode == "depth_maps",
        }
    )
    stage1["stage1_identity"] = job["stage1_identity"]
    atomic_json(stage1_marker, stage1)
    for entry in inputs:
        entry["stage1_identity"] = job["stage1_identity"]
        atomic_json(Path(entry["directory"]) / "stage2_result.json", entry)
    atomic_json(job_path, job)
    for result in stage3["chunks"]:
        counts = merge_class_exports(
            result["parts"],
            Path(result["output"]),
            classes,
            result["crs_wkt"],
            settings.voting.batch_points,
        )
        report = next(item["report"] for item in inputs if item["directory"] == result["directory"])
        if int(counts.sum()) != report["point_count"]:
            raise RuntimeError("Final LAS point count differs from classified source")
        report["timing_seconds"]["stage3"] = time.perf_counter() - start
        report["timing_seconds"]["driver_total"] = time.perf_counter() - started
        report["output_las"] = result["output"]
        report["metashape_asset_key"] = result["asset_key"]
        atomic_json(Path(result["directory"]) / "report.json", report)
        atomic_json(Path(result["output"]).with_suffix(".report.json"), report)
    atomic_json(
        output / "report.json",
        {"chunks": stage3["chunks"], "timing_seconds": time.perf_counter() - started},
    )
    atomic_json(work / "run_done.json", stage3)
    LOG.info("Complete. Classified project saved; final LAS files: %s", output)
    return stage3
