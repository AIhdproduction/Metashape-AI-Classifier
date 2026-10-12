"""Independent Windows progress window for a long-running Metashape GUI classification."""

import argparse
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from tkinter import StringVar, Tk, ttk

import psutil

from msam3.cache import atomic_json

IMAGE_LINE = re.compile(
    r"^(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ INFO .* (?:image|camera) "
    r"(?P<done>\d+)/(?P<total>\d+) \((?P<name>[^)]+)\) complete(?:; \d+ images used)?$"
)
PHASES = {
    "prepare": "1/4: Einstellungen und Modell werden geprüft",
    "export": "2/4: Metashape exportiert Punktwolke und Kameras",
    "classify": "3/4: SAM 3 klassifiziert die Kamerabilder",
    "import": "4/4: Metashape importiert die klassifizierte Punktwolke",
}


def image_progress(path: Path):
    if not path.is_file():
        return 0, 0, "Vorbereitung", None
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = IMAGE_LINE.match(line)
        if match:
            records.append(match)
    if not records:
        return 0, 0, "SAM 3 Modell und Punktwolke werden vorbereitet", None
    current = records[-1]
    average = None
    if len(records) > 1:
        first = datetime.strptime(records[0]["time"], "%Y-%m-%d %H:%M:%S")
        last = datetime.strptime(current["time"], "%Y-%m-%d %H:%M:%S")
        average = (last - first).total_seconds() / (len(records) - 1)
    return int(current["done"]), int(current["total"]), current["name"], average


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class ProgressWindow:
    def __init__(self, session: Path, worker_pid: int | None):
        self.session = session
        self.worker_pid = worker_pid
        self.started = time.time()
        self.root = Tk()
        self.root.title("SAM 3 – Metashape-Klassifikation")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self.close_window)
        frame = ttk.Frame(self.root, padding=18)
        frame.grid()
        ttk.Label(
            frame,
            text="SAM 3 klassifiziert die Metashape-Punktwolke",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.status = StringVar(value="Vorbereitung läuft …")
        ttk.Label(frame, textvariable=self.status, wraplength=520).grid(
            row=1, column=0, pady=(12, 4), sticky="w"
        )
        self.bar = ttk.Progressbar(frame, length=520, mode="indeterminate")
        self.bar.grid(row=2, column=0, pady=4, sticky="ew")
        self.bar.start(12)
        self.detail = StringVar(value="Bitte Metashape geöffnet lassen.")
        ttk.Label(frame, textvariable=self.detail).grid(row=3, column=0, pady=(8, 0), sticky="w")
        self.button = ttk.Button(frame, text="Abbrechen", command=self.request_cancel)
        self.button.grid(row=4, column=0, pady=(14, 0), sticky="e")
        self.root.after(250, self.refresh)

    def close_window(self):
        shutil.rmtree(self.session, ignore_errors=True)
        self.root.destroy()

    def show_terminal(self, title, detail, success=False):
        self.bar.stop()
        self.bar.configure(mode="determinate", maximum=1, value=int(success))
        self.status.set(title)
        self.detail.set(detail)
        self.button.configure(text="Schließen", state="normal", command=self.close_window)

    def request_cancel(self):
        atomic_json(self.session / "cancel_requested.json", {"requested_at": time.time()})
        self.button.configure(state="disabled", text="Abbruch wird ausgeführt …")
        worker_path = self.session / "worker.json"
        if not worker_path.is_file():
            return
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        try:
            process = psutil.Process(int(worker["pid"]))
            processes = process.children(recursive=True) + [process]
            for item in processes:
                item.terminate()
            _, alive = psutil.wait_procs(processes, timeout=4)
            for item in alive:
                item.kill()
        except (psutil.Error, OSError, ValueError, KeyError):
            # The worker may already have exited; its launcher will observe the marker.
            pass

    def refresh(self):
        status_path = self.session / "launcher_status.json"
        status = {}
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
        phase = status.get("phase")
        elapsed = time.time() - self.started
        if phase == "error":
            self.show_terminal(
                "Die Klassifikation wurde mit einem Fehler beendet.",
                status.get("message") or "Details stehen im Metashape-Fehlerfenster.",
            )
            return
        if phase == "cancelled":
            self.show_terminal(
                "Die Klassifikation wurde abgebrochen.",
                "Es wurde keine klassifizierte Punktwolke importiert.",
            )
            return
        if phase == "complete":
            self.show_terminal(
                "Fertig: Die klassifizierte Punktwolke wurde importiert.",
                "Dieses Fenster schließt gleich automatisch.",
                success=True,
            )
            self.root.after(3500, self.close_window)
            return
        if phase in {"prepare", "export", "import"}:
            self.button.configure(state="disabled", text="Abbrechen nur während Berechnung")
            if phase == "prepare":
                self.button.configure(state="normal", text="Abbrechen")
            self.status.set(PHASES[phase] + " …")
            self.detail.set(f"Laufzeit {duration(elapsed)} · Bitte Metashape geöffnet lassen.")
            self.root.after(1000, self.refresh)
            return

        self.button.configure(state="normal", text="Abbrechen")
        done, total, name, average = image_progress(self.session / "stage2.log")
        if total:
            if self.bar["mode"] != "determinate":
                self.bar.stop()
                self.bar.configure(mode="determinate", maximum=total)
            self.bar["value"] = done
            self.status.set(f"Bild {done} von {total} fertig: {name}")
            message = f"{done / total:.1%} abgeschlossen · Laufzeit {duration(elapsed)}"
            if average is not None:
                message += f" · geschätzt noch {duration((total - done) * average)}"
            self.detail.set(message)
        else:
            self.status.set(name + " …")
            self.detail.set(f"Laufzeit {duration(elapsed)} · Bitte Metashape geöffnet lassen.")

        result = self.session / "stage2_status.json"
        if result.is_file():
            worker_status = json.loads(result.read_text(encoding="utf-8"))
            if worker_status["returncode"]:
                self.status.set("Die Bildklassifikation wurde mit einem Fehler beendet.")
                self.detail.set(
                    "Details stehen im Metashape-Fehlerfenster und im Arbeitsprotokoll."
                )
                return
        if self.worker_pid is not None and not psutil.pid_exists(self.worker_pid):
            self.status.set("Der Hintergrundprozess wurde beendet.")
            self.detail.set("Metashape zeigt das Ergebnis oder den Fehler an.")
            self.root.after(3500, self.root.destroy)
            return
        self.root.after(1000, self.refresh)

    def run(self):
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    ProgressWindow(args.session.resolve(), args.pid).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
