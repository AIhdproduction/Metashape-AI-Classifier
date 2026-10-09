"""Responsive in-Metashape launcher for the CUDA-only SAM 3 worker."""

import json
import re
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import Metashape
from PySide2 import QtCore, QtWidgets

from metashape_scripts import stage1_export, stage3_import
from metashape_scripts.common import check_metashape, write_json

ROOT = Path(__file__).resolve().parent.parent
IMAGE_LINE = re.compile(
    r"^(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ INFO .* image "
    r"(?P<done>\d+)/(?P<total>\d+) \((?P<name>[^)]+)\) complete$"
)
_dialog = None


def duration(seconds):
    seconds = max(0, int(seconds))
    return "{:02d}:{:02d}:{:02d}".format(seconds // 3600, seconds // 60 % 60, seconds % 60)


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


class ClassificationDialog(QtWidgets.QDialog):
    """Owns only the GUI; all Metashape API calls remain on Metashape's main thread."""

    def __init__(self, document, session):
        super().__init__(QtWidgets.QApplication.activeWindow())
        self.document = document
        self.session = session
        self.job = None
        self.process = None
        self.log_stream = None
        self.action = None
        self.started = datetime.now()
        self.image_times = []
        self.completed = False
        self.setWindowTitle("SAM 3 – Metashape-Klassifikation")
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setMinimumWidth(660)
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("SAM 3 klassifiziert die Punktwolke")
        title.setStyleSheet("font-weight: bold; font-size: 14px")
        layout.addWidget(title)
        self.status = QtWidgets.QLabel("Vorbereitung wird gestartet …")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 0)
        layout.addWidget(self.bar)
        self.detail = QtWidgets.QLabel("Metashape bleibt während der Bildklassifikation bedienbar.")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(100)
        self.log.setMinimumHeight(155)
        layout.addWidget(self.log)
        self.button = QtWidgets.QPushButton("Abbrechen")
        self.button.clicked.connect(self.cancel_or_close)
        layout.addWidget(self.button)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh_progress)

    def start(self):
        self.show()
        QtCore.QTimer.singleShot(0, self.prepare)

    def prepare(self):
        self.status.setText("1/3: Einstellungen und Modell werden geprüft …")
        self.launch_worker("prepare")

    def launch_worker(self, action):
        self.action = action
        log_path = self.session / (action + ".log")
        self.log_stream = log_path.open("w", encoding="utf-8")
        self.process = QtCore.QProcess(self)
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.remove("PYTHONHOME")
        environment.remove("PYTHONPATH")
        environment.insert("PYTHONNOUSERSITE", "1")
        environment.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.worker_finished)
        self.process.errorOccurred.connect(self.worker_error)
        self.process.start(
            str(ROOT / ".venv/Scripts/python.exe"),
            ["-u", "-m", "msam3.gui_worker", action, "--session", str(self.session)],
        )
        if action == "stage2":
            self.image_times = []
            self.bar.setRange(0, 0)
            self.status.setText("2/3: SAM 3 verarbeitet die Kamerabilder …")
            self.detail.setText("Bitte die Klassifikation nicht parallel erneut starten.")
            self.timer.start()

    def read_output(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not text:
            return
        self.log_stream.write(text)
        self.log_stream.flush()
        self.log.appendPlainText(text.rstrip())
        self.record_images(text)

    def record_images(self, text):
        for line in text.splitlines():
            match = IMAGE_LINE.match(line)
            if match:
                self.image_times.append(match)

    def refresh_progress(self):
        if not self.image_times:
            elapsed = (datetime.now() - self.started).total_seconds()
            self.detail.setText(
                "Laufzeit {} · Modell und Punktwolke werden vorbereitet.".format(duration(elapsed))
            )
            return
        match = self.image_times[-1]
        done, total = int(match["done"]), int(match["total"])
        if self.bar.maximum() != total:
            self.bar.setRange(0, total)
        self.bar.setValue(done)
        elapsed = (datetime.now() - self.started).total_seconds()
        detail = "Bild {} von {} fertig: {} · {:.1%} · Laufzeit {}".format(
            done, total, match["name"], done / total, duration(elapsed)
        )
        if len(self.image_times) > 1:
            first = datetime.strptime(self.image_times[0]["time"], "%Y-%m-%d %H:%M:%S")
            last = datetime.strptime(match["time"], "%Y-%m-%d %H:%M:%S")
            average = (last - first).total_seconds() / (len(self.image_times) - 1)
            detail += " · Restzeit etwa {}".format(duration((total - done) * average))
        self.detail.setText(detail)

    def worker_error(self, _error):
        if self.process is not None and self.process.state() == QtCore.QProcess.NotRunning:
            self.fail("Der SAM-3-Hintergrundprozess konnte nicht gestartet werden.")

    def worker_finished(self, exit_code, _exit_status):
        if self.completed:
            return
        self.timer.stop()
        self.read_output()
        action = self.action
        if self.log_stream is not None:
            self.log_stream.close()
            self.log_stream = None
        self.process = None
        if exit_code:
            self.fail("{} wurde beendet. Details stehen im Protokoll unten.".format(action))
            return
        if action == "prepare":
            try:
                self.job = json.loads((self.session / "job.json").read_text(encoding="utf-8"))
                self.run_stage1()
            except Exception as error:
                self.fail(str(error))
        elif action == "stage2":
            try:
                self.run_stage3()
            except Exception as error:
                self.fail(str(error))

    def native_progress(self, value):
        self.bar.setRange(0, 100)
        self.bar.setValue(max(0, min(100, round(value))))
        # Native Metashape exports/imports call this callback repeatedly. Process only paint
        # events; no user input is accepted while a native document operation is active.
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
        return True

    def run_stage1(self):
        self.status.setText("1/3: Punktwolke und Kameras werden exportiert …")
        self.detail.setText("Die native Metashape-Operation zeigt ihren Fortschritt hier.")
        stage1_export.main(self.job, document=self.document, progress=self.native_progress)
        Metashape.app.update()
        self.launch_worker("stage2")

    def run_stage3(self):
        self.status.setText("3/3: Klassifizierte Punktwolke wird importiert …")
        self.detail.setText("Die native Metashape-Operation zeigt ihren Fortschritt hier.")
        defaults = {chunk.key: chunk.point_cloud for chunk in self.document.chunks}
        try:
            self.job = json.loads((self.session / "job.json").read_text(encoding="utf-8"))
            stage3_import.main(self.job, document=self.document, progress=self.native_progress)
        except Exception:
            for chunk in self.document.chunks:
                if chunk.key in defaults:
                    chunk.point_cloud = defaults[chunk.key]
            Metashape.app.update()
            raise
        result = json.loads(
            (Path(self.job["work_dir"]) / "stage3_done.json").read_text(encoding="utf-8")
        )
        select_classified_asset(self.document, result)
        self.completed = True
        self.bar.setRange(0, 1)
        self.bar.setValue(1)
        self.status.setText("Fertig: 'SAM3 classified' ist nach Klassen dargestellt.")
        self.detail.setText("Die Punktwolke und der LAS-Export wurden erstellt.")
        self.button.setText("Schließen")

    def fail(self, message):
        self.completed = True
        self.timer.stop()
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.status.setText("Abgebrochen: " + message)
        self.detail.setText("Keine unvollständige Klassifikation wurde gespeichert.")
        self.button.setText("Schließen")

    def cancel_or_close(self):
        if self.completed:
            self.close()
            return
        if self.process is not None and self.process.state() != QtCore.QProcess.NotRunning:
            self.status.setText("Abbruch wird ausgeführt …")
            self.button.setEnabled(False)
            self.process.terminate()
            if not self.process.waitForFinished(5000):
                self.process.kill()
            return
        self.fail("Vom Benutzer abgebrochen.")

    def closeEvent(self, event):
        if not self.completed:
            event.ignore()
            self.cancel_or_close()
            return
        event.accept()


def start():
    global _dialog
    if _dialog is not None:
        _dialog.raise_()
        _dialog.activateWindow()
        return
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
        _dialog = ClassificationDialog(document, session)
        _dialog.destroyed.connect(lambda *_: clear_dialog())
        _dialog.start()
    except Exception as error:
        traceback.print_exc()
        Metashape.app.messageBox("SAM 3 konnte nicht gestartet werden:\n" + str(error))


def clear_dialog():
    global _dialog
    _dialog = None
