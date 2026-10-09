"""Direct Metashape launcher using an external progress window."""

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


class UserCancelled(RuntimeError):
    """The operator requested cancellation from the independent progress window."""


def worker_environment():
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def launch_progress(session):
    """Open the independent, topmost status window before any lengthy Metashape work."""
    progress_python = ROOT / ".venv/Scripts/pythonw.exe"
    if not progress_python.is_file():
        progress_python = ROOT / ".venv/Scripts/python.exe"
    subprocess.Popen(
        [str(progress_python), "-m", "msam3.gui_progress", "--session", str(session)],
        cwd=str(ROOT),
        env=worker_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def run_worker(action, session):
    """Run CUDA work in the installed venv and retain a concise error log."""
    if (session / "cancel_requested.json").is_file():
        raise UserCancelled("Vom Benutzer abgebrochen.")
    log_path = session / (action + ".log")
    worker_path = session / "worker.json"
    environment = worker_environment()
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
        write_json(worker_path, {"pid": process.pid, "action": action})
        try:
            # Covers the brief race where the operator clicks cancel while this worker starts.
            if (session / "cancel_requested.json").is_file():
                process.terminate()
            returncode = process.wait()
        finally:
            worker_path.unlink(missing_ok=True)
    write_json(session / (action + "_status.json"), {"returncode": returncode})
    if returncode:
        if (session / "cancel_requested.json").is_file():
            raise UserCancelled("Vom Benutzer abgebrochen.")
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
    session = None
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
        write_json(session / "launcher_status.json", {"phase": "prepare"})
        launch_progress(session)

        print("SAM 3: Konfiguration wird vorbereitet ...")
        run_worker("prepare", session)
        job = json.loads((session / "job.json").read_text(encoding="utf-8"))

        write_json(session / "launcher_status.json", {"phase": "export"})
        print("SAM 3: Punktwolke und Kameras werden aus Metashape exportiert ...")
        stage1_export.main(job, document=document)
        Metashape.app.update()

        write_json(session / "launcher_status.json", {"phase": "classify"})
        print(
            "SAM 3: Geometriepruefung und Bildklassifikation laufen. Bitte Metashape offen lassen."
        )
        run_worker("stage2", session)

        write_json(session / "launcher_status.json", {"phase": "import"})
        print("SAM 3: Klassifizierte Punktwolke wird importiert ...")
        job = json.loads((session / "job.json").read_text(encoding="utf-8"))
        defaults = {chunk.key: chunk.point_cloud for chunk in document.chunks}
        try:
            stage3_import.main(job, document=document)
        except Exception:
            for chunk in document.chunks:
                if chunk.key in defaults:
                    chunk.point_cloud = defaults[chunk.key]
            Metashape.app.update()
            raise
        result = json.loads(
            (Path(job["work_dir"]) / "stage3_done.json").read_text(encoding="utf-8")
        )
        select_classified_asset(document, result)
        write_json(session / "launcher_status.json", {"phase": "complete"})
        Metashape.app.messageBox(
            "SAM 3 ist fertig. Die neue Punktwolke 'SAM3 classified' ist nach Klassen dargestellt."
        )
    except UserCancelled:
        if session is not None:
            write_json(session / "launcher_status.json", {"phase": "cancelled"})
        Metashape.app.messageBox(
            "SAM 3 wurde abgebrochen. Es wurde keine klassifizierte Punktwolke importiert."
        )
    except Exception as error:
        if session is not None:
            write_json(session / "launcher_status.json", {"phase": "error", "message": str(error)})
        traceback.print_exc()
        Metashape.app.messageBox(
            "SAM 3 wurde beendet, ohne das Projekt zu speichern:\n" + str(error)
        )
    finally:
        _running = False
