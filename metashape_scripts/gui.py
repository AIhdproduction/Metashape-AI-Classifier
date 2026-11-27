"""Direct Metashape launcher. Options, progress and result live in the app of the venv."""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

import Metashape

from metashape_scripts import stage1_export, stage3_import
from metashape_scripts.common import check_metashape, write_json

ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "Metashape AI Classifier"
INSTALL_HINT = (
    "Bitte install.bat im Repository erneut ausführen (nach jedem git pull) und danach das "
    "Skript nochmals über Werkzeuge > Skript ausführen (Tools > Run Script) starten."
)
POLL_SECONDS = 0.25
CHECK_SECONDS = 180
STARTUP_SECONDS = 180  # Until the app's server answers, not until the user presses Start.
HEARTBEAT_SECONDS = 120  # Without a sign of life from an app that was up.
_running = False


def say(text, stream=None):
    """Print to Metashape's console. A code page without umlauts must never abort a run."""
    stream = stream or sys.stdout
    try:
        print(text, file=stream)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"), file=stream)


class UserCancelled(RuntimeError):
    """The operator requested cancellation from the app."""


def worker_environment():
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    # The logs are read as UTF-8, and camera labels may contain umlauts.
    environment["PYTHONIOENCODING"] = "utf-8"
    # The models are local files. Nothing is downloaded or reported at run time.
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return environment


def log_tail(path, count=12):
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-count:])


def final_las_message(session):
    """Name the verified LAS files; a native Metashape LAS/LAZ export would truncate codes."""
    try:
        result = json.loads((session / "finalize_done.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    files = result.get("files") or [{"path": path} for path in result.get("outputs", [])]
    if not files:
        return ""
    lines = ["", "", "Fertige LAS-Datei mit allen Klassencodes:"]
    for item in files:
        details = ""
        if "las_version" in item:
            details = " (LAS {}, Punktformat {})".format(item["las_version"], item["point_format"])
        lines.append(item["path"] + details)
    lines.append(
        "Diese Datei verwenden: Ein eigener LAS/LAZ-Export aus Metashape (LAS 1.2) "
        "kürzt Klassencodes über 31."
    )
    return "\n".join(lines)


def stop_process(process):
    """End a child of this launcher. The venv's launcher exe takes its Python process along."""
    try:
        process.kill()
        process.wait(timeout=10)
        return True
    except (OSError, subprocess.SubprocessError) as error:
        say(APP_NAME + ": Ein Hintergrundprozess lässt sich nicht beenden: " + str(error))
        return False


def check_installation(session):
    """After git pull an old .venv lacks packages: say so instead of failing without a window."""
    log_path = session / "ui.log"
    process = None
    try:
        with log_path.open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(
                [str(ROOT / ".venv/Scripts/python.exe"), "-m", "msam3.install_check"],
                cwd=str(ROOT),
                env=worker_environment(),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            returncode = process.wait(timeout=CHECK_SECONDS)
    except subprocess.TimeoutExpired as error:
        stop_process(process)
        raise RuntimeError(
            "Die Prüfung der Installation hat nach "
            + str(CHECK_SECONDS)
            + " Sekunden nicht geantwortet und wurde beendet. "
            + INSTALL_HINT
            + "\n\n"
            + log_tail(log_path)
            + "\nProtokoll: "
            + str(log_path)
        ) from error
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "Die Installation lässt sich nicht prüfen ("
            + str(error)
            + "). "
            + INSTALL_HINT
            + "\nProtokoll: "
            + str(log_path)
        ) from error
    if returncode:
        raise RuntimeError(
            "Die Installation in .venv ist nicht aktuell. "
            + INSTALL_HINT
            + "\n\n"
            + log_tail(log_path)
            + "\nProtokoll: "
            + str(log_path)
        )


def app_failure(reason, log_path):
    return RuntimeError(
        "Das Fenster von "
        + APP_NAME
        + " "
        + reason
        + " "
        + INSTALL_HINT
        + "\n\n"
        + log_tail(log_path)
        + "\nProtokoll: "
        + str(log_path)
    )


def modified(path):
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def launch_options(session):
    """Open the app and return as soon as Start wrote options.json; the app keeps running.

    Returns False when the user cancelled. Raises when the app ended without a choice, did not
    come up in time (no ui.json), or came up and then stopped touching ui_alive. An app that is
    alive may wait for the user as long as it likes.
    """
    app_python = ROOT / ".venv/Scripts/pythonw.exe"
    if not app_python.is_file():
        app_python = ROOT / ".venv/Scripts/python.exe"
    log_path = session / "ui.log"
    with log_path.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [str(app_python), "-m", "msam3.gui_options", "--session", str(session)],
            cwd=str(ROOT),
            env=worker_environment(),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    failure = None
    app_up = False
    beat = None
    silent = 0.0
    last = time.monotonic()
    while True:
        # Read the exit state first: the app may write its choice just before it ends.
        ended = process.poll() is not None
        if (session / "options.json").is_file():
            return True
        if (session / "options_cancelled.json").is_file():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                stop_process(process)
            return False
        if failure:
            raise app_failure(failure, log_path)
        if ended:
            raise app_failure(
                "wurde unerwartet beendet oder konnte nicht geöffnet werden.", log_path
            )
        now = time.monotonic()
        # Capped, so that a sleeping PC does not count as a silent app.
        silent += min(now - last, 2.0)
        last = now
        if not app_up and (session / "ui.json").is_file():
            app_up, silent = True, 0.0
        if app_up:
            current = modified(session / "ui_alive")
            if current is not None and current != beat:
                beat, silent = current, 0.0
            if silent >= HEARTBEAT_SECONDS:
                failure = "reagiert nicht mehr und wurde beendet."
        elif silent >= STARTUP_SECONDS:
            failure = "ist nicht rechtzeitig gestartet und wurde beendet."
        if failure:
            stop_process(process)
            continue  # A choice that was written just now still counts.
        time.sleep(POLL_SECONDS)


def selected_title(session):
    """The model's name for messages: from job.json, or from the app's choice before that."""
    for name in ("job.json", "options.json"):
        try:
            value = json.loads((session / name).read_text(encoding="utf-8"))
            return value["model"]["title"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return APP_NAME


def remove_worker_file(path):
    for _ in range(40):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:  # The app reads the file at this moment.
            time.sleep(0.05)


def enabled_image_sizes(document):
    """Read enabled camera dimensions without loading all photos into Metashape memory."""
    sizes = []
    for chunk in document.chunks:
        if not getattr(chunk, "enabled", True):
            continue
        for camera in chunk.cameras:
            if not getattr(camera, "enabled", True) or getattr(camera, "photo", None) is None:
                continue
            calibration = getattr(getattr(camera, "sensor", None), "calibration", None)
            width = int(getattr(calibration, "width", 0) or 0)
            height = int(getattr(calibration, "height", 0) or 0)
            if width > 0 and height > 0:
                sizes.append([width, height])
    return sizes


def raise_if_cancelled(session):
    if (session / "cancel_requested.json").is_file():
        raise UserCancelled("Vom Benutzer abgebrochen.")


def run_worker(action, session, cancellable=True):
    """Run CUDA work in the installed venv and retain a concise error log."""
    if cancellable:
        raise_if_cancelled(session)
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
        returncode = None
        try:
            write_json(worker_path, {"pid": process.pid, "action": action})
            # Covers the brief race where the operator clicks cancel while this worker starts.
            if cancellable and (session / "cancel_requested.json").is_file():
                process.terminate()
            returncode = process.wait()
        finally:
            # An interrupted script must not leave the GPU busy. If the worker cannot be ended,
            # worker.json stays, so that the app can offer Stop.
            if returncode is not None or stop_process(process):
                remove_worker_file(worker_path)
    write_json(session / (action + "_status.json"), {"returncode": returncode})
    cancelled = cancellable and (session / "cancel_requested.json").is_file()
    # A cancel that arrives as the classification ends must still prevent the import.
    if cancelled and (returncode or action == "stage2"):
        raise UserCancelled("Vom Benutzer abgebrochen.")
    if returncode:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        raise RuntimeError("\n".join(lines[-18:]) or "Der Hintergrundprozess ist fehlgeschlagen.")


def select_classified_asset(document, result, display_mode):
    current_key = document.chunk.key if document.chunk is not None else None
    entry = next(
        (item for item in result["chunks"] if item["chunk_key"] == current_key), result["chunks"][0]
    )
    chunk = document.findChunk(entry["chunk_key"])
    asset = next(item for item in chunk.point_clouds if item.key == entry["asset_key"])
    chunk.point_cloud = asset
    document.chunk = chunk
    Metashape.app.model_view.view_mode = Metashape.PointCloudData
    if display_mode == "image_colors":
        Metashape.app.model_view.point_cloud_view_mode = (
            Metashape.Application.ModelView.PointCloudViewColor
        )
    else:
        Metashape.app.model_view.point_cloud_view_mode = (
            Metashape.Application.ModelView.PointCloudViewClassification
        )
    Metashape.app.update()


def restore_source_point_clouds(document):
    """Select the original photo cloud even when Metashape has no active cloud selected."""
    for chunk in document.chunks:
        selected = getattr(chunk, "point_cloud", None)
        source = stage1_export.source_point_cloud(chunk)
        if source is None:
            say(APP_NAME + ": Keine verwendbare Fotopunktwolke in " + chunk.label + " gefunden.")
            continue
        if selected is not source:
            chunk.point_cloud = source
            say(
                APP_NAME
                + ": Die ursprüngliche Fotopunktwolke wird als Eingabe verwendet: "
                + source.label
            )


def start():
    global _running
    if _running:
        return
    _running = True
    session = None
    title = APP_NAME
    phase = None
    imported = False
    timeline = []

    def set_phase(name, **values):
        nonlocal phase
        timeline.append({"phase": name, "at": time.time()})
        write_json(
            session / "launcher_status.json", dict(values, phase=name, timeline=list(timeline))
        )
        phase = name

    try:
        check_metashape()
        document = Metashape.app.document
        if not document.path or not Path(document.path).is_file():
            raise RuntimeError("Bitte zuerst das Projekt als .psx oder .psz speichern.")
        if document.read_only:
            raise RuntimeError("Das Projekt ist schreibgeschützt. Bitte schreibbar öffnen.")
        if not document.chunks:
            raise RuntimeError("Das Projekt enthält keine Chunks.")
        # Explicit chunk selections are applied later, during export.
        if not any(
            stage1_export.source_point_cloud(chunk) is not None for chunk in document.chunks
        ):
            raise RuntimeError(
                "Das Projekt enthält keine fertig berechnete Quellpunktwolke. "
                "Bitte zuerst in Metashape «Workflow > Build Point Cloud» "
                "(Punktwolke erstellen) ausführen und das Projekt speichern. "
                "Danach den Classifier erneut starten."
            )
        if not (ROOT / ".venv/Scripts/python.exe").is_file():
            raise RuntimeError("Bitte zuerst install.bat im Repository ausführen.")
        session = ROOT / "work" / "gui-sessions" / uuid.uuid4().hex
        session.mkdir(parents=True)
        check_installation(session)
        restore_source_point_clouds(document)
        document.save()
        write_json(
            session / "session.json",
            {
                "project": str(Path(document.path).resolve()),
                "image_count": sum(
                    1
                    for chunk in document.chunks
                    for camera in chunk.cameras
                    if getattr(chunk, "enabled", True)
                    and getattr(camera, "enabled", True)
                    and getattr(camera, "photo", None) is not None
                ),
                "image_sizes": enabled_image_sizes(document),
                "created_at": time.time(),
                # The app notices when Metashape is gone and does not wait for it forever.
                "launcher_pid": os.getpid(),
            },
        )
        if not launch_options(session):
            say(APP_NAME + ": Start im Fenster abgebrochen.")
            shutil.rmtree(session, ignore_errors=True)
            phase = "cancelled"
            return
        title = selected_title(session)
        set_phase("prepare")

        say(title + ": Konfiguration wird vorbereitet ...")
        run_worker("prepare", session)
        job = json.loads((session / "job.json").read_text(encoding="utf-8"))
        title = selected_title(session)

        raise_if_cancelled(session)
        set_phase("export")
        say(title + ": Punktwolke und Kameras werden aus Metashape exportiert ...")
        # Again: the app may have read the phase "prepare" a moment before it wrote the marker.
        raise_if_cancelled(session)
        stage1_export.main(job, document=document)
        write_json(session / "job.json", job)
        Metashape.app.update()

        set_phase("classify")
        say(
            title
            + ": Geometrieprüfung und Bildklassifikation laufen. Bitte Metashape offen lassen."
        )
        run_worker("stage2", session)

        set_phase("import")
        say(title + ": Klassifizierte Punktwolke wird importiert ...")
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
        imported = True
        result = json.loads(
            (Path(job["work_dir"]) / "stage3_done.json").read_text(encoding="utf-8")
        )
        run_worker("finalize", session, cancellable=False)
        settings = job["settings"]["metashape"]
        select_classified_asset(document, result, settings["display_mode"])
        set_phase("complete")
        message = title + " ist fertig. Die neue Punktwolke '" + settings["imported_label"] + "'"
        if settings["display_mode"] == "image_colors":
            message += " zeigt die Originalfarben aus den Fotos."
        else:
            message += " zeigt die Klassen."
        message += final_las_message(session)
        say(message)
        Metashape.app.messageBox(message)
    except UserCancelled:
        if session is not None:
            set_phase("cancelled")
        Metashape.app.messageBox(
            title + " wurde abgebrochen. Es wurde keine klassifizierte Punktwolke importiert."
        )
    except Exception as error:
        # Also when writing the status is what failed, the user must still see the message.
        say(traceback.format_exc(), sys.stderr)
        try:
            if session is not None:
                set_phase("error", message=str(error))
        except OSError:
            say(traceback.format_exc(), sys.stderr)
        if imported:
            reason = (
                " konnte den Abschluss nicht beenden. Die klassifizierte Punktwolke ist bereits "
                "importiert und gespeichert:\n"
            )
        elif any(entry["phase"] == "import" for entry in timeline):
            # Metashape may already show the new cloud, but nothing of it was saved.
            reason = " wurde beim Import beendet, ohne das Projekt zu speichern:\n"
        else:
            reason = " wurde beendet, ohne eine Klassifikation zu importieren:\n"
        Metashape.app.messageBox(title + reason + str(error))
    finally:
        _running = False
        if session is not None and phase not in ("complete", "cancelled", "error"):
            # The script itself was interrupted; the app must not wait for it.
            try:
                set_phase("error", message="Das Skript in Metashape wurde unterbrochen.")
            except OSError:
                say(traceback.format_exc(), sys.stderr)
