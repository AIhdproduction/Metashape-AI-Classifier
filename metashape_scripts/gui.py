"""Direct Metashape launcher without custom Qt widgets or a second Metashape process."""

import json
import os
import subprocess
import traceback
import uuid
from pathlib import Path

import Metashape

from metashape_scripts import stage1_export, stage3_import
from metashape_scripts.common import check_metashape, write_json

ROOT = Path(__file__).resolve().parent.parent
_running = False


def run_worker(action, session):
    """Run CUDA work in the installed venv and retain a concise error log."""
    log_path = session / (action + ".log")
    environment = os.environ.copy()
    # The embedded interpreter must not influence the independent venv interpreter.
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [
                str(ROOT / ".venv/Scripts/python.exe"),
                "-u",
                "-m",
                "msam3.gui_worker",
                action,
                "--session",
                str(session),
            ],
            cwd=str(ROOT),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if action == "stage2":
            subprocess.Popen(
                [
                    str(ROOT / ".venv/Scripts/python.exe"),
                    "-m",
                    "msam3.gui_progress",
                    "--session",
                    str(session),
                    "--pid",
                    str(process.pid),
                ],
                cwd=str(ROOT),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        returncode = process.wait()
    write_json(session / (action + "_status.json"), {"returncode": returncode})
    if returncode:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        raise RuntimeError("\n".join(lines[-18:]) or "Der Hintergrundprozess ist fehlgeschlagen.")


def select_classified_asset(document, result):
    current_key = document.chunk.key if document.chunk is not None else None
    entry = next(
        (item for item in result["chunks"] if item["chunk_key"] == current_key), result["chunks"][0]
    )
    chunk = document.findChunk(entry["chunk_key"])
    asset = next(item for item in chunk.point_clouds if item.key == entry["asset_key"])
    chunk.point_cloud = asset
    document.chunk = chunk
    Metashape.app.model_view.view_mode = Metashape.PointCloudData
    Metashape.app.model_view.point_cloud_view_mode = (
        Metashape.Application.ModelView.PointCloudViewClassification
    )
    Metashape.app.update()


def start():
    global _running
    if _running:
        return
    _running = True
    try:
        check_metashape()
        document = Metashape.app.document
        if not document.path or not Path(document.path).is_file():
            raise RuntimeError("Bitte zuerst das Projekt als .psx oder .psz speichern.")
        if document.read_only:
            raise RuntimeError("Das Projekt ist schreibgeschuetzt. Bitte schreibbar oeffnen.")
        if not (ROOT / ".venv/Scripts/python.exe").is_file():
            raise RuntimeError("Bitte zuerst install.bat im Repository ausfuehren.")
        if not document.chunks:
            raise RuntimeError("Das Projekt enthaelt keine Chunks.")
        document.save()
        session = ROOT / "work" / "gui-sessions" / uuid.uuid4().hex
        session.mkdir(parents=True)
        write_json(session / "session.json", {"project": str(Path(document.path).resolve())})

        print("SAM 3: Konfiguration wird vorbereitet ...")
        run_worker("prepare", session)
        job = json.loads((session / "job.json").read_text(encoding="utf-8"))

        print("SAM 3: Punktwolke und Kameras werden aus Metashape exportiert ...")
        stage1_export.main(job, document=document)
        Metashape.app.update()

        print(
            "SAM 3: Geometriepruefung und Bildklassifikation laufen. Bitte Metashape offen lassen."
        )
        run_worker("stage2", session)

        print("SAM 3: Klassifizierte Punktwolke wird importiert ...")
        job = json.loads((session / "job.json").read_text(encoding="utf-8"))
        defaults = {chunk.key: chunk.point_cloud for chunk in document.chunks}
        try:
            stage3_import.main(job, document=document)
        except Exception:
            # The project was saved before the run. Restore its displayed source asset if an
            # import validation fails before saving, so an incomplete asset is not selected.
            for chunk in document.chunks:
                if chunk.key in defaults:
                    chunk.point_cloud = defaults[chunk.key]
            Metashape.app.update()
            raise
        result = json.loads(
            (Path(job["work_dir"]) / "stage3_done.json").read_text(encoding="utf-8")
        )
        select_classified_asset(document, result)
        Metashape.app.messageBox(
            "SAM 3 ist fertig. Die neue Punktwolke 'SAM3 classified' ist nach Klassen dargestellt."
        )
    except Exception as error:
        traceback.print_exc()
        Metashape.app.messageBox(
            "SAM 3 wurde beendet, ohne das Projekt zu speichern:\n" + str(error)
        )
    finally:
        _running = False
