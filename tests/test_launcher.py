"""The launcher with the real embedded scripts, the real worker logic and a fake Metashape."""

import ast
import importlib
import io
import json
import logging
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import fake_metashape
import laspy
import numpy as np
import psutil
import pytest
from synthetic import FakeFlurSeg, FakeSam, write_flurseg_files

from msam3 import backends, gui_progress, gui_worker
from msam3.app.controller import Controller
from msam3.config import load_settings

ROOT = Path(__file__).parents[1]


class Simulation:
    def __init__(self, root, monkeypatch):
        self.root, self.backend = root, "flurseg"
        self.install_code = 0
        self.ui = "start"  # start | cancel | crash | silent (the test plays the app)
        self.install_hangs = False
        self.gpu_checks = []
        self.killed = []
        self.on_sleep = None
        self.now = 5000.0
        self.cancel_during = None
        self.fail_during = None
        self.phases, self.modules = [], []
        (root / "project").mkdir()
        self.metashape = fake_metashape.build(root / "project")
        self.document = self.metashape.app.document

        settings = load_settings()
        settings.paths.flurseg_model = write_flurseg_files(root / "models" / "flurseg")
        settings.paths.model = root / "models" / "sam3.pt"
        settings.paths.model.write_bytes(b"sam")
        settings.paths.work_dir = root / "work"
        settings.gpu.device = "cpu"
        settings.voting.voxel_buckets = 8
        self.settings = settings
        monkeypatch.setattr(gui_worker, "load_settings", lambda: settings.model_copy(deep=True))
        monkeypatch.setattr(backends.Model, "require_runtime", lambda self: None)
        for key, model in backends.MODELS.items():
            monkeypatch.setattr(
                model, "require_gpu", lambda settings, key=key: self.gpu_checks.append(key)
            )
        monkeypatch.setattr(backends.MODELS["flurseg"], "create", lambda settings: FakeFlurSeg())
        monkeypatch.setattr(backends.MODELS["sam3"], "create", lambda settings: FakeSam())

        monkeypatch.setitem(sys.modules, "Metashape", self.metashape)
        names = [
            "metashape_scripts.common",
            "metashape_scripts.stage1_export",
            "metashape_scripts.stage3_import",
            "metashape_scripts.gui",
        ]
        monkeypatch.syspath_prepend(str(ROOT))
        for name in names:
            monkeypatch.delitem(sys.modules, name, raising=False)
        self.gui = [importlib.import_module(name) for name in names][-1]
        venv = root / "repository" / ".venv" / "Scripts"
        venv.mkdir(parents=True)
        (venv / "python.exe").write_bytes(b"")
        monkeypatch.setattr(self.gui, "ROOT", root / "repository")
        monkeypatch.setattr(self.gui, "POLL_SECONDS", 1.0)  # On the fake clock.
        # Only inside the launcher, not for the whole test process.
        fake = types.SimpleNamespace(**{**vars(self.gui.subprocess), "Popen": self.popen})
        monkeypatch.setattr(self.gui, "subprocess", fake)
        clock = {"monotonic": lambda: self.now, "sleep": self.sleep}
        fake_time = types.SimpleNamespace(**{**vars(self.gui.time), **clock})
        monkeypatch.setattr(self.gui, "time", fake_time)

    def sleep(self, seconds):
        self.now += seconds
        assert self.now < 5000.0 + 40_000, "the launcher waits forever"
        if self.on_sleep:
            self.on_sleep(self.now - 5000.0)

    def popen(self, arguments, **options):
        return Process(self, [str(item) for item in arguments], options)

    def start(self, backend="flurseg"):
        self.backend = backend
        self.gui.start()
        return self.metashape.app.messages[-1] if self.metashape.app.messages else ""

    @property
    def sessions(self):
        folder = self.root / "repository" / "work" / "gui-sessions"
        return sorted(folder.glob("*"), key=lambda path: path.stat().st_mtime_ns)


class Process:
    pid = 2**22 + 7  # No real process.

    def __init__(self, simulation, arguments, options):
        self.simulation, self.arguments, self.output = simulation, arguments, options["stdout"]
        self.module = arguments[arguments.index("-m") + 1]
        simulation.modules.append(self.module)
        assert options["env"]["PYTHONIOENCODING"] == "utf-8"
        assert options["env"]["HF_HUB_OFFLINE"] == "1"
        self.code = None
        if self.module == "msam3.gui_options":
            self.open_app(Path(arguments[arguments.index("--session") + 1]))

    def kill(self):
        self.simulation.killed.append(self.module)
        self.code = -9

    def open_app(self, session):
        simulation = self.simulation
        simulation.session = session
        if simulation.ui == "silent":
            self.output.write("app: last line before it hung\n")
            return
        if simulation.ui == "crash":
            self.output.write("ModuleNotFoundError: No module named 'nicegui'\n")
            self.code = 1
            return
        controller = Controller(session, settings=simulation.settings)
        if simulation.ui == "cancel":
            controller.cancel_before_start()
            self.code = 1
        else:
            controller.select_model(simulation.backend)
            controller.start()

    def poll(self):
        return self.code

    def terminate(self):
        self.code = 1

    def wait(self, timeout=None):
        if self.code is not None:
            return self.code
        if self.module == "msam3.install_check":
            if self.simulation.install_hangs:
                raise subprocess.TimeoutExpired("install_check", timeout)
            self.code = self.simulation.install_code
            if self.code:
                self.output.write("Die Installation ist veraltet: nicegui fehlt\n")
        elif self.module == "msam3.gui_worker" and self.code is None:
            self.code = self.work()
        return self.code or 0

    def work(self):
        simulation = self.simulation
        position = self.arguments.index("msam3.gui_worker") + 1
        action = self.arguments[position]
        session = Path(self.arguments[self.arguments.index("--session") + 1])
        status = json.loads((session / "launcher_status.json").read_text(encoding="utf-8"))
        simulation.phases.append((action, status["phase"]))
        if simulation.fail_during == action:
            self.output.write("ERROR synthetic worker failure\n")
            return 1
        handler = logging.StreamHandler(self.output)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger = logging.getLogger()
        level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            getattr(gui_worker, action)(session)
        finally:
            handler.flush()
            logger.removeHandler(handler)
            logger.setLevel(level)
        if simulation.cancel_during == action:
            # The user's click lands just as the worker finishes successfully.
            (session / "cancel_requested.json").write_text("{}", encoding="utf-8")
        return 0


@pytest.fixture
def simulation(tmp_path, monkeypatch):
    return Simulation(tmp_path, monkeypatch)


def test_full_run_with_each_model_in_one_project(simulation):
    chunk = simulation.document.chunks[0]
    source = chunk.point_cloud
    output = simulation.root / "project" / "export" / "punktwolke_class.las"
    seen = set()

    for number, backend in enumerate(("flurseg", "sam3", "flurseg"), start=1):
        model = backends.MODELS[backend]
        message = simulation.start(backend)

        assert f"{model.title} ist fertig" in message, message
        assert model.asset_label in message
        assert len(chunk.imports) in (number, number - 1)
        assert chunk.point_cloud.label == model.asset_label
        assert chunk.point_cloud.meta["msam3.source_point_cloud_key"] == str(source.key)
        assert simulation.phases[-3:] == [
            ("prepare", "prepare"),
            ("stage2", "classify"),
            ("finalize", "import"),
        ]
        session = next(path for path in simulation.sessions if path not in seen)
        seen.add(session)
        run = gui_progress.read_run(session)
        assert run.state == "done" and all(step.state == "done" for step in run.steps)
        assert run.result["label"] == model.asset_label
        assert run.result["point_count"] == source.point_count
        assert run.result["outputs"] == [str(output)]
        delivered = run.result["files"][0]
        assert delivered["path"] == str(output)
        assert (delivered["las_version"], delivered["point_format"]) == ("1.4", 7)
        # The completion message names the exact delivered file.
        assert str(output) in message and "LAS 1.4, Punktformat 7" in message
        progress = gui_progress.image_progress(session / "stage2.log")
        assert (progress.done, progress.total) == (3, 3)
        assert "(DJI (2) Süd)" in gui_progress.read_log(session)

        las = laspy.read(output)
        codes = set(np.unique(np.asarray(las.classification)))
        assert len(las) == source.point_count and str(las.header.version) == "1.4"
        assert (71 if backend == "flurseg" else 11) in codes
        # Metashape's class parts carry codes >= 32 in the five-bit field; no bit may leak.
        assert not np.any(las.key_point) and not np.any(las.withheld)
        assert not np.any(las.synthetic)
        legend = json.loads((output.parent / "class_legend.json").read_text(encoding="utf-8"))
        assert legend[0]["model"] == model.title
        # A classified cloud of either model never becomes the next input.
        assert chunk.full_exports[-1] == source.key
        # Only the classified LAS is kept; the work data of the run is gone.
        metadata = json.loads((session / "job.json").read_text(encoding="utf-8"))
        assert sorted(p.name for p in Path(metadata["work_dir"]).iterdir()) == ["msam3.lock"]
        assert run.result["work_cleanup"]["deleted"]
        # The user is not told; the deletion does not concern them.
        assert "Zwischendaten" not in message
    assert simulation.modules.count("msam3.gui_options") == 3


def test_stale_install_is_reported_in_german(simulation):
    simulation.install_code = 3
    message = simulation.start()
    assert "install.bat" in message and "erneut ausführen" in message
    assert "nicegui fehlt" in message and "ui.log" in message
    assert "msam3.gui_options" not in simulation.modules
    assert simulation.document.saves == 0


def test_crashed_app_is_not_silent_and_does_not_hang(simulation):
    simulation.ui = "crash"
    message = simulation.start()
    assert "unerwartet beendet" in message and "install.bat" in message
    assert "No module named 'nicegui'" in message
    assert "msam3.gui_worker" not in simulation.modules


def test_cancel_before_start(simulation):
    simulation.ui = "cancel"
    assert simulation.start() == ""
    assert "msam3.gui_worker" not in simulation.modules
    assert not simulation.sessions  # As before, such a session leaves nothing behind.


def test_cancel_as_classification_ends_prevents_the_import(simulation):
    simulation.cancel_during = "stage2"
    message = simulation.start()
    assert "FlurSeg wurde abgebrochen" in message
    assert not simulation.document.chunks[0].imports
    assert gui_progress.read_run(simulation.sessions[-1]).state == "cancelled"


def test_cancel_during_preparation(simulation):
    simulation.cancel_during = "prepare"
    assert "abgebrochen" in simulation.start("sam3")
    assert not simulation.document.chunks[0].imports


def test_worker_failure_is_shown_and_a_second_run_works(simulation):
    simulation.fail_during = "stage2"
    message = simulation.start()
    assert "synthetic worker failure" in message
    run = gui_progress.read_run(simulation.sessions[-1])
    assert run.state == "error" and "synthetic" in run.message
    simulation.fail_during = None
    assert "ist fertig" in simulation.start()


def test_failure_during_import_says_that_nothing_was_saved(simulation, monkeypatch):
    stage3 = sys.modules["metashape_scripts.stage3_import"]

    def fail(job, document=None, progress=None):
        raise RuntimeError("count mismatch")

    monkeypatch.setattr(stage3, "main", fail)
    message = simulation.start()
    assert "beim Import" in message and "ohne das Projekt zu speichern" in message


def test_failure_after_import_says_that_it_is_imported(simulation):
    simulation.fail_during = "finalize"
    assert "bereits importiert und gespeichert" in simulation.start()


def test_every_classified_asset_is_excluded_as_input(simulation):
    stage1 = sys.modules["metashape_scripts.stage1_export"]
    cloud = fake_metashape.PointCloud
    codes, xyz = np.zeros(4, np.uint8), np.zeros((4, 3))
    photo = cloud(1, "Dense Cloud", xyz[:2], codes[:2])
    for label in ("FlurSeg classified", "SAM3 classified", "SAM 3 classified", "SAM3 classified 2"):
        previous = cloud(2, label, xyz, codes)  # Older version: no meta at all.
        chunk = type("Chunk", (), {"point_cloud": previous, "point_clouds": [previous, photo]})
        assert stage1.source_point_cloud(chunk) is photo, label
    renamed = cloud(3, "Mein Resultat", xyz, codes)
    renamed.meta["msam3.import_id"] = "earlier"
    chunk = type("Chunk", (), {"point_cloud": renamed, "point_clouds": [renamed, photo]})
    assert stage1.source_point_cloud(chunk) is photo
    empty = cloud(4, "Empty Point Cloud", xyz[:0], codes[:0])
    chunk.point_cloud = empty
    chunk.point_clouds.append(empty)
    assert stage1.source_point_cloud(chunk) is photo


@pytest.mark.parametrize("kind", ["missing", "empty", "classified_only"])
def test_missing_source_cloud_shows_build_instructions_before_start(simulation, kind):
    chunk = simulation.document.chunks[0]
    if kind == "missing":
        chunk.point_clouds = []
        chunk.point_cloud = None
    elif kind == "empty":
        chunk.point_cloud.codes = chunk.point_cloud.codes[:0]
    else:
        chunk.point_cloud.meta["msam3.import_id"] = "earlier"

    message = simulation.start()

    assert "keine fertig berechnete Quellpunktwolke" in message
    assert "Workflow > Build Point Cloud" in message
    assert "das Projekt speichern" in message
    assert not simulation.modules
    assert not simulation.sessions


def test_hanging_install_check_is_ended(simulation):
    simulation.install_hangs = True
    message = simulation.start()
    assert simulation.killed == ["msam3.install_check"]
    assert "180 Sekunden" in message and "install.bat" in message and "ui.log" in message
    assert "msam3.gui_options" not in simulation.modules


def test_app_that_never_comes_up_is_ended_at_the_startup_deadline(simulation):
    simulation.ui = "silent"
    message = simulation.start()
    assert simulation.killed == ["msam3.gui_options"]
    assert 180 <= simulation.now - 5000.0 < 182
    assert "nicht rechtzeitig gestartet" in message and "last line before it hung" in message
    assert "ui.log" in message and "install.bat" in message
    assert gui_progress.read_run(simulation.sessions[-1]).state == "error"


def play_app(simulation, beats_until, start_at=None):
    """The app comes up after 20 s and shows signs of life until ``beats_until``."""
    simulation.ui = "silent"
    state = {"beat": 0}

    def on_sleep(seconds):
        session = simulation.session
        if seconds < 20:
            return
        if not (session / "ui.json").exists():
            (session / "ui.json").write_text("{}", encoding="utf-8")
        if seconds < beats_until and seconds >= state["beat"] + 10:
            state["beat"] = seconds
            alive = session / "ui_alive"
            alive.touch()
            stamp = 1_700_000_000_000_000_000 + int(seconds * 1e9)
            os.utime(alive, ns=(stamp, stamp))
        if start_at is not None and seconds >= start_at:
            if not (session / "options.json").exists():
                Controller(session, settings=simulation.settings).start()

    simulation.on_sleep = on_sleep


def test_user_may_take_an_hour_once_the_app_is_up(simulation):
    play_app(simulation, beats_until=99_999, start_at=3600)
    assert "ist fertig" in simulation.start()
    assert not simulation.killed


def test_app_whose_monitor_died_is_ended(simulation):
    play_app(simulation, beats_until=600)
    message = simulation.start()
    assert simulation.killed == ["msam3.gui_options"]
    assert 690 <= simulation.now - 5000.0 < 725
    assert "reagiert nicht mehr" in message


def test_sleep_of_the_pc_is_no_silent_app(simulation):
    play_app(simulation, beats_until=99_999, start_at=4000)
    play = simulation.on_sleep

    def on_sleep(seconds):
        if 100 <= seconds < 101:
            simulation.now += 3600
            return
        play(seconds)

    simulation.on_sleep = on_sleep
    assert "ist fertig" in simulation.start()
    assert not simulation.killed


def test_start_at_the_deadline_still_counts(simulation, monkeypatch):
    simulation.ui = "silent"

    def kill(process):
        Controller(simulation.session, settings=simulation.settings).start()
        simulation.killed.append(process.module)

    monkeypatch.setattr(Process, "kill", kill)
    monkeypatch.setattr(simulation.gui, "STARTUP_SECONDS", 3)
    original = Process.wait

    def wait(process, timeout=None):
        return 0 if process.module == "msam3.gui_options" else original(process, timeout)

    monkeypatch.setattr(Process, "wait", wait)
    assert "ist fertig" in simulation.start()
    assert simulation.killed == ["msam3.gui_options"]


def exports(simulation, monkeypatch):
    stage1 = sys.modules["metashape_scripts.stage1_export"]
    calls, original = [], stage1.main

    def main(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(stage1, "main", main)
    return calls


def test_cancel_as_preparation_ends_does_not_export(simulation, monkeypatch):
    calls = exports(simulation, monkeypatch)
    simulation.cancel_during = "prepare"
    assert "abgebrochen" in simulation.start()
    assert not calls and simulation.document.saves == 1  # Only the save before the app opened.
    status = gui_progress.read_json(simulation.sessions[-1] / "launcher_status.json")
    assert [entry["phase"] for entry in status["timeline"]] == ["prepare", "cancelled"]


def test_cancel_in_the_last_moment_before_export_does_not_export(simulation, monkeypatch):
    calls = exports(simulation, monkeypatch)
    original = simulation.gui.write_json

    def write_json(path, value):
        original(path, value)
        if isinstance(value, dict) and value.get("phase") == "export":
            (path.parent / "cancel_requested.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(simulation.gui, "write_json", write_json)
    assert "abgebrochen" in simulation.start()
    assert not calls


def test_gpu_is_checked_before_the_export(simulation, monkeypatch):
    calls = exports(simulation, monkeypatch)
    assert "ist fertig" in simulation.start("sam3")
    assert simulation.gpu_checks == ["sam3"] and calls == [1]

    def no_cuda(settings):
        raise RuntimeError("FlurSeg requires CUDA.")

    monkeypatch.setattr(backends.MODELS["flurseg"], "require_gpu", no_cuda)
    assert "requires CUDA" in simulation.start("flurseg")
    assert calls == [1]


def test_interrupted_script_does_not_leave_a_worker_on_the_gpu(simulation, monkeypatch):
    original = Process.work

    def work(process):
        if "stage2" in process.arguments:
            raise KeyboardInterrupt
        return original(process)

    monkeypatch.setattr(Process, "work", work)
    with pytest.raises(KeyboardInterrupt):
        simulation.gui.start()
    assert simulation.killed == ["msam3.gui_worker"]
    assert not (simulation.sessions[-1] / "worker.json").exists()
    assert gui_progress.read_run(simulation.sessions[-1]).state == "error"


def test_stop_process_really_ends_a_child(simulation):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        process = psutil.Process(child.pid)
        family = [process, *process.children(recursive=True)]
        assert simulation.gui.stop_process(child)
        psutil.wait_procs(family, timeout=10)
        assert not any(item.is_running() for item in family)
    finally:
        if child.poll() is None:
            child.kill()


@pytest.mark.parametrize("encoding", ["ascii", "cp932", "cp1251"])
def test_console_without_umlauts_never_aborts_a_run(simulation, monkeypatch, encoding):
    console = io.TextIOWrapper(io.BytesIO(), encoding=encoding)
    monkeypatch.setattr(sys, "stdout", console)
    simulation.document.chunks[0].point_cloud = None  # Makes the launcher print with umlauts.
    assert "ist fertig" in simulation.start()


EMBEDDED = [*sorted((ROOT / "metashape_scripts").glob("*.py")), ROOT / "start_in_metashape.py"]


@pytest.mark.parametrize("path", EMBEDDED, ids=lambda path: path.name)
def test_embedded_scripts_also_fit_python39_and_use_only_the_standard_library(path):
    # Metashape 2.3 embeds Python 3.12 (changelog); 2.2 is inferred to embed 3.9.
    tree = ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
    allowed = set(sys.stdlib_module_names) | {"Metashape", "metashape_scripts"}
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module]
        for name in names:
            assert name.split(".")[0] in allowed, name
        # These parse as 3.9 but fail on a 3.9 interpreter when evaluated.
        for attribute in ("annotation", "returns"):
            annotation = getattr(node, attribute, None)
            assert not isinstance(annotation, ast.BinOp), node.lineno
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "zip":
            assert "strict" not in [item.arg for item in node.keywords], node.lineno


def test_starter_reloads_every_embedded_module():
    starter = (ROOT / "start_in_metashape.py").read_text(encoding="utf-8")
    reloaded = set(re.findall(r"importlib\.reload\((\w+)\)", starter))
    modules = {path.stem for path in (ROOT / "metashape_scripts").glob("*.py")} - {"__init__"}
    assert reloaded == modules
