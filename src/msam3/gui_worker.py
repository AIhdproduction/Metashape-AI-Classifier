"""CUDA-only worker used by the Metashape GUI launcher.

This module never imports Metashape and never opens the project. The running Metashape GUI
remains the only process that exports from and saves to the project.
"""

import argparse
import json
import logging
from pathlib import Path

from msam3 import backends
from msam3.cache import atomic_json
from msam3.config import Classes, Settings, load_settings, merge_settings
from msam3.driver import (
    _execute,
    clean_work_directory,
    export_identity,
    job_directory,
    job_lock,
    prepare_stage3_inputs,
)
from msam3.pointcloud.las_io import describe_output, legend, merge_class_exports

LOG = logging.getLogger(__name__)


def session_options(session: Path) -> dict:
    choices = session / "options.json"
    return json.loads(choices.read_text(encoding="utf-8")) if choices.is_file() else {}


def settings_for_session(session: Path) -> Settings:
    settings = load_settings()
    override = session_options(session).get("settings_override", {})
    if not isinstance(override, dict):
        raise ValueError("Die Einstellungen aus dem Fenster sind ungültig.")
    # The app's choices apply to this run only; default.yml is never written.
    return Settings.model_validate(merge_settings(settings.model_dump(mode="json"), override))


def prepare(session: Path) -> None:
    project = Path(json.loads((session / "session.json").read_text(encoding="utf-8"))["project"])
    settings = settings_for_session(session)
    segmenter = backends.selected(settings)
    # Only the selected model: missing SAM weights never block FlurSeg, and the reverse.
    segmenter.require_files(settings)
    segmenter.require_runtime()
    # The app snapshots the classes at Start, so later edits cannot change a running job.
    classes = segmenter.classes(snapshot=session_options(session).get("classes"))
    # In this short-lived process, so that a PC without CUDA fails before the long export.
    segmenter.require_gpu(settings)
    work = job_directory(project, settings)
    # Reuse the project's export folder so a rerun updates the delivered LAS and legends
    # in place. The final writer uses an atomic replace for same-name files.
    export = project.parent / "export"
    export.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    identity = export_identity(project, settings)
    job = {
        "schema": 1,
        "project": str(project),
        "work_dir": str(work),
        # The one-click Metashape workflow keeps its delivered LAS next to the project.
        # Work data remains in the repository and never pollutes the project directory.
        "output_dir": str(export),
        "final_las_name": "punktwolke_class.las",
        "settings": segmenter.job_settings(settings),
        "model": segmenter.describe(),
        "classes": classes.model_dump(),
        "stage1_identity": identity,
    }
    atomic_json(session / "job.json", job)
    LOG.info("Prepared %s with %d classes", segmenter.title, len(classes.classes))


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
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["stage3_inputs"] = prepare_stage3_inputs(
        result["chunks"], job["stage1_identity"], classes, result["model_identity"]
    )
    atomic_json(job_path, job)


def finalize(session: Path) -> None:
    """Assemble Metashape's class partitions into the lossless final LAS in the venv."""
    job = json.loads((session / "job.json").read_text(encoding="utf-8"))
    completed = json.loads((Path(job["work_dir"]) / "stage3_done.json").read_text(encoding="utf-8"))
    classes = Classes.model_validate(job["classes"])
    inputs = {item["directory"]: item for item in job["stage3_inputs"]}
    title = job.get("model", {}).get("title")
    totals = dict.fromkeys(range(256), 0)
    files = []
    for result in completed["chunks"]:
        source = inputs[result["directory"]]
        counts = merge_class_exports(
            result["parts"],
            Path(result["output"]),
            classes,
            result["crs_wkt"],
            job["settings"]["voting"]["batch_points"],
            title,
        )
        for code, count in enumerate(counts):
            totals[code] += int(count)
        if int(counts.sum()) != sum(int(value) for value in source["counts"].values()):
            raise RuntimeError("Final LAS point count differs from the imported point cloud")
        files.append(describe_output(Path(result["output"]), result["parts"]))
        LOG.info(
            "Final classified LAS (LAS %s, point format %d, all class codes preserved): %s",
            files[-1]["las_version"],
            files[-1]["point_format"],
            files[-1]["path"],
        )
    cleanup = None
    if not job["settings"]["paths"].get("keep_work_data", False):
        # Metashape has saved the project and the final LAS is verified, so nothing needs
        # the export, depth maps or image caches any more.
        work = Path(job["work_dir"])
        try:
            with job_lock(work / "msam3.lock"):
                cleanup = clean_work_directory(work, [Path(item["path"]) for item in files])
        except RuntimeError as error:
            LOG.warning("Work data kept: %s", error)
    atomic_json(
        session / "finalize_done.json",
        {
            "work_cleanup": cleanup,
            "outputs": [item["path"] for item in files],
            "files": files,
            "model": job.get("model", {}),
            "label": job["settings"]["metashape"]["imported_label"],
            "point_count": sum(totals.values()),
            "classes": [
                {
                    "name": row["name"],
                    "code": row["code"],
                    "color": row["color"],
                    "count": totals[row["code"]],
                }
                for row in legend(classes)
            ],
        },
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
