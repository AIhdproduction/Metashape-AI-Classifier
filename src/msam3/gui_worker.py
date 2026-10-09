"""CUDA-only worker used by the Metashape GUI launcher.

This module never imports Metashape and never opens the project. The running Metashape GUI
remains the only process that exports from and saves to the project.
"""

import argparse
import json
import logging
from pathlib import Path

from msam3.cache import atomic_json, json_hash
from msam3.config import Classes, Settings, load_classes, load_settings
from msam3.driver import _execute, job_directory, job_lock, prepare_stage3_inputs, project_state
from msam3.pointcloud.las_io import merge_class_exports
from msam3.segmentation.sam3_backend import require_checkpoint

LOG = logging.getLogger(__name__)


def prepare(session: Path) -> None:
    project = Path(json.loads((session / "session.json").read_text(encoding="utf-8"))["project"])
    settings = load_settings()
    classes = load_classes()
    require_checkpoint(settings.paths.model)
    work = job_directory(project, settings)
    export = project.parent / "export"
    work.mkdir(parents=True, exist_ok=True)
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
        # The one-click Metashape workflow keeps its delivered LAS next to the project.
        # Work data remains in the repository and never pollutes the project directory.
        "output_dir": str(export),
        "final_las_name": "punktwolke_class.las",
        "settings": settings.model_dump(mode="json"),
        "classes": classes.model_dump(),
        "stage1_identity": identity,
    }
    atomic_json(session / "job.json", job)


def stage2(session: Path) -> None:
    job_path = session / "job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    project = Path(job["project"])
    work = Path(job["work_dir"])
    settings = Settings.model_validate(job["settings"])
    classes = Classes.model_validate(job["classes"])
    with job_lock(work / "msam3.lock"):
        result = _execute(
            "stage2", project, work, Path(job["output_dir"]), settings, classes, False
        )
    job["stage3_inputs"] = prepare_stage3_inputs(result["chunks"], job["stage1_identity"], classes)
    atomic_json(job_path, job)


def finalize(session: Path) -> None:
    """Assemble Metashape's class partitions into the lossless final LAS in the venv."""
    job = json.loads((session / "job.json").read_text(encoding="utf-8"))
    completed = json.loads((Path(job["work_dir"]) / "stage3_done.json").read_text(encoding="utf-8"))
    classes = Classes.model_validate(job["classes"])
    inputs = {item["directory"]: item for item in job["stage3_inputs"]}
    for result in completed["chunks"]:
        source = inputs[result["directory"]]
        counts = merge_class_exports(
            result["parts"],
            Path(result["output"]),
            classes,
            result["crs_wkt"],
            job["settings"]["voting"]["batch_points"],
        )
        if int(counts.sum()) != sum(int(value) for value in source["counts"].values()):
            raise RuntimeError("Final LAS point count differs from the imported point cloud")
    atomic_json(
        session / "finalize_done.json",
        {"outputs": [item["output"] for item in completed["chunks"]]},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "stage2", "finalize"))
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        {"prepare": prepare, "stage2": stage2, "finalize": finalize}[args.action](
            args.session.resolve()
        )
        return 0
    except Exception:
        LOG.exception("GUI worker failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
