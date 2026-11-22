"""Progress, result and cancellation of one Metashape session, read from its files.

The window itself is in msam3.app. ``python -m msam3.gui_progress --session <dir>`` opens it
again for a session that is running or finished.
"""

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import psutil

from msam3.cache import atomic_json

LOG = logging.getLogger(__name__)
# Matches today's "camera 3/9 (label) complete; 3 images used" and the older "image 3/9" lines.
IMAGE_LINE = re.compile(
    r"^(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ INFO (?:.* )?(?:image|camera) "
    r"(?P<done>\d+)/(?P<total>\d+) \((?P<name>.*)\) complete(?:; \d+ images used)?$"
)
SKIPPED_LINE = re.compile(r" (?:image|camera) (?P<done>\d+)/(?P<total>\d+) \(.*\) skipped;")
CACHED_LINE = re.compile(r" camera \d+/\d+ \(.*\) reused from cache$")
PLAN_LINE = re.compile(r" Segmentation plan: \d+ chunks, (?P<total>\d+) cameras$")
STEPS = ("prepare", "export", "classify", "import")
FINAL = ("complete", "cancelled", "error")
CANCELLABLE = ("prepare", "classify")
LOG_FILES = (
    "ui.log",
    "prepare.log",
    "stage2.log",
    "finalize.log",
)


def read_json(path: Path, default=None):
    """Read a file that another process may be replacing right now."""
    for _ in range(4):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except (OSError, ValueError):
            time.sleep(0.02)
    return default


@dataclass(frozen=True)
class ImageProgress:
    done: int = 0
    total: int = 0
    name: str = ""
    seconds_per_image: float | None = None
    finishing: bool = False

    @property
    def percent(self) -> int | None:
        # 100 is reserved for the finished step; the post-filter follows the last image.
        return min(99, int(self.done / self.total * 100)) if self.total else None

    @property
    def remaining_seconds(self) -> float | None:
        if self.seconds_per_image is None or not self.total or self.finishing:
            return None
        return max(0, self.total - self.done) * self.seconds_per_image


def image_progress(path: Path) -> ImageProgress:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ImageProgress()
    completed = []
    planned = chunk_total = previous_total = last_done = 0
    finishing = False
    for line in lines:
        if plan := PLAN_LINE.search(line):
            planned = int(plan["total"])
            continue
        if "all cameras processed" in line:
            finishing = True
            continue
        event = IMAGE_LINE.match(line)
        if event:
            completed.append(event)
        else:
            event = SKIPPED_LINE.search(line)
        if event:
            done = int(event["done"])
            if done <= last_done:
                previous_total += chunk_total  # The camera counter restarts with every chunk.
            last_done, chunk_total = done, int(event["total"])
            finishing = False
    average = None
    if len(completed) > 1:
        first, last = (
            datetime.strptime(event["time"], "%Y-%m-%d %H:%M:%S")
            for event in (completed[0], completed[-1])
        )
        average = (last - first).total_seconds() / (len(completed) - 1)
    total = max(planned, previous_total + chunk_total)
    done = previous_total + last_done
    return ImageProgress(
        done,
        total,
        completed[-1]["name"] if completed else "",
        average,
        finishing and done >= total,
    )


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


@dataclass(frozen=True)
class Step:
    key: str
    state: str = "waiting"  # waiting | active | done | error | cancelled
    seconds: float | None = None
    percent: int | None = None


@dataclass(frozen=True)
class RunView:
    state: str  # ready | running | done | error | cancelled
    phase: str = ""
    steps: tuple[Step, ...] = tuple(Step(key) for key in STEPS)
    total_seconds: float | None = None
    images: ImageProgress = field(default_factory=ImageProgress)
    message: str = ""
    interrupted: bool = False
    cancel_requested: bool = False
    result: dict = field(default_factory=dict)
    # A verified worker that still runs although nobody waits for it.
    worker_running: bool = False

    @property
    def can_cancel(self) -> bool:
        return self.state == "running" and self.phase in CANCELLABLE and not self.cancel_requested

    @property
    def final(self) -> bool:
        return self.state in ("done", "error", "cancelled")


def read_run(session: Path, launcher_alive: bool = True, now: float | None = None) -> RunView:
    """Derive the display state from the files of the process protocol."""
    now = time.time() if now is None else now
    status = read_json(session / "launcher_status.json", {})
    phase = status.get("phase", "")
    started = (session / "options.json").is_file()
    if not started and phase not in FINAL:
        if (session / "options_cancelled.json").is_file():
            return RunView("cancelled", "cancelled")
        if not launcher_alive:
            return RunView("error", "error", interrupted=True)
        return RunView("ready")

    timeline = [entry for entry in status.get("timeline", []) if entry.get("phase")]
    if not timeline and phase in STEPS:
        timeline = [{"phase": phase, "at": now}]
    interrupted = phase not in FINAL and not launcher_alive
    if phase == "complete":
        state = "done"
    elif phase == "cancelled":
        state = "cancelled"
    elif phase == "error" or interrupted:
        state = "error"
    else:
        state, phase = "running", phase or "prepare"

    working = [entry["phase"] for entry in timeline if entry["phase"] in STEPS]
    current = working[-1] if working else "prepare"
    reached = STEPS.index(current) if current in STEPS else 0
    images = image_progress(session / "stage2.log")
    steps = []
    for index, key in enumerate(STEPS):
        if state == "done" or index < reached:
            step_state = "done"
        elif index > reached:
            step_state = "waiting"
        elif state == "running":
            step_state = "active"
        else:
            step_state = state if started else "waiting"
        seconds = None
        for position, entry in enumerate(timeline):
            if entry["phase"] != key:
                continue
            if position + 1 < len(timeline):
                seconds = timeline[position + 1]["at"] - entry["at"]
            elif state == "running":
                seconds = now - entry["at"]
        percent = 100 if step_state == "done" else None
        if step_state != "done" and key == "classify" and index == reached:
            percent = images.percent
        steps.append(Step(key, step_state, seconds, percent))
    total = None
    if timeline:
        if status.get("phase") in FINAL:
            end = timeline[-1]["at"]
        else:
            end = now if state == "running" else None
        total = end - timeline[0]["at"] if end is not None else None
    return RunView(
        state,
        phase,
        tuple(steps),
        total,
        images,
        str(status.get("message", "")),
        interrupted,
        (session / "cancel_requested.json").is_file(),
        read_json(session / "finalize_done.json", {}) if state == "done" else {},
        state == "error" and worker_process(session) is not None,
    )


def read_log(session: Path, lines: int = 5000) -> str:
    """The session's logs in the order of the workflow, shortened to the end."""
    collected = []
    for name in LOG_FILES:
        try:
            content = (session / name).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        if content:
            collected += [f"[{name}]", *content]
    return "\n".join(collected[-lines:])


def request_cancel(session: Path) -> bool:
    """Write the marker and stop the worker. Export and import cannot be cancelled."""
    phase = read_json(session / "launcher_status.json", {}).get("phase", "prepare")
    if phase not in CANCELLABLE:
        return False
    atomic_json(session / "cancel_requested.json", {"requested_at": time.time()})
    if process := worker_process(session):
        # The worker may already have exited; its launcher observes the marker either way.
        terminate_tree(process)
    return True


def worker_process(session: Path) -> psutil.Process | None:
    """The running, cancellable worker of exactly this session, or None."""
    worker = read_json(session / "worker.json")
    if not worker or worker.get("action") == "finalize":
        return None
    try:
        process = psutil.Process(int(worker["pid"]))
        command = process.cmdline()
    except (psutil.Error, OSError, ValueError, KeyError, TypeError):
        return None
    # A finished worker's PID may already belong to some other program or session.
    if "msam3.gui_worker" in command and any(session.name in part for part in command):
        return process
    return None


def terminate_tree(process: psutil.Process) -> None:
    try:
        processes = [*process.children(recursive=True), process]
        for item in processes:
            try:
                item.terminate()
            except psutil.NoSuchProcess:
                continue
        _, alive = psutil.wait_procs(processes, timeout=4)
        for item in alive:
            item.kill()
        psutil.wait_procs(alive, timeout=4)
    except (psutil.Error, OSError) as error:
        LOG.info("The worker could not be stopped directly (%s)", error)


def stop_worker(session: Path) -> bool:
    """Stop the worker that a dead launcher left behind. False when none is running any more."""
    process = worker_process(session)
    if process is None:
        return False
    atomic_json(session / "cancel_requested.json", {"requested_at": time.time()})
    terminate_tree(process)
    return True


def recent_sessions(root: Path, limit: int = 8) -> list[dict]:
    """The latest started session per project, newest first."""
    entries = []
    for path in root.glob("*/options.json"):
        session = read_json(path.parent / "session.json")
        options = read_json(path)
        if not session or not session.get("project") or options is None:
            continue
        status = read_json(path.parent / "launcher_status.json", {})
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue
        entries.append(
            {
                "session": path.parent,
                "project": session["project"],
                # Sessions from before the model choice were always SAM 3.
                "model": options.get("model", {}).get("title", "SAM 3"),
                "phase": status.get("phase", ""),
                "at": session.get("created_at", modified),
            }
        )
    entries.sort(key=lambda entry: entry["at"], reverse=True)
    latest = {}
    for entry in entries:
        latest.setdefault(str(entry["project"]).casefold(), entry)
    return list(latest.values())[:limit]


def directory_size(path: Path) -> int:
    total, pending = 0, [path]
    while pending:
        try:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    else:
                        total += entry.stat(follow_symlinks=False).st_size
        except OSError:  # Work files come and go while the job runs.
            continue
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--pid", type=int, help="Ignored; kept for older callers")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    from msam3.app.server import serve

    return serve(args.session.resolve(), open_browser=not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
