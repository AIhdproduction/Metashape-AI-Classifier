import asyncio
import http.client
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import pytest
from synthetic import write_flurseg_files

from msam3 import backends, cache, gui_options, gui_progress, install_check, user_classes
from msam3.app import browser, i18n, theme
from msam3.app.controller import ActionError, Controller
from msam3.app.lifecycle import Heartbeat, Lifecycle
from msam3.app.security import SessionGuard
from msam3.config import Classes, load_classes, load_settings
from msam3.gui_worker import settings_for_session
from msam3.segmentation.windows import window_count

ROOT = Path(__file__).parents[1]
SIZES = [[5280, 3956]] * 9


def make_session(root, name="session", **data):
    session = root / name
    session.mkdir(parents=True)
    values = {
        "project": "D:/Jobs/Site.psx",
        "image_count": len(SIZES),
        "image_sizes": SIZES,
        **data,
    }
    (session / "session.json").write_text(json.dumps(values), encoding="utf-8")
    return session


@pytest.fixture
def ready_settings(tmp_path, monkeypatch):
    settings = load_settings()
    settings.paths.flurseg_model = write_flurseg_files(tmp_path / "weights" / "flurseg")
    settings.paths.model = tmp_path / "weights" / "sam3.pt"
    settings.paths.model.write_bytes(b"sam")
    settings.paths.work_dir = tmp_path / "work"
    monkeypatch.setattr(backends.Model, "require_runtime", lambda self: None)
    return settings


# ---------------------------------------------------------------------------- options and jobs


def test_flurseg_start_writes_options_that_the_worker_accepts(tmp_path, ready_settings):
    session = make_session(tmp_path / "sessions")
    controller = Controller(session, settings=ready_settings)
    assert controller.options.backend == "flurseg" and controller.start_lock is None
    controller.select_profile("accurate")
    controller.set_option("flurseg_batch_size", 8.0)
    controller.start()

    payload = json.loads((session / "options.json").read_text(encoding="utf-8"))
    assert payload["model"]["title"] == "FlurSeg" and "classes" not in payload
    assert "sam" not in payload["settings_override"]
    settings = settings_for_session(session)
    assert settings.segmentation.backend == "flurseg"
    assert settings.flurseg.resolution == "accurate"
    assert settings.flurseg.window_batch_size == 8
    assert controller.run.state == "running"


def test_sam_start_snapshots_the_edited_classes(tmp_path, ready_settings):
    session = make_session(tmp_path / "sessions")
    controller = Controller(session, settings=ready_settings)
    controller.select_model("sam3")
    controller.select_profile("accurate")
    controller.save_class("Manhole", ["manhole cover"], "#112233")
    controller.set_class_enabled("water", False)
    controller.start()

    payload = json.loads((session / "options.json").read_text(encoding="utf-8"))
    names = [item.name for item in Classes.model_validate(payload["classes"]).classes]
    assert "manhole" in names and "water" not in names
    assert "flurseg" not in payload["settings_override"]
    settings = settings_for_session(session)
    assert settings.segmentation.backend == "sam3"
    assert (settings.sam.image_scale, settings.sam.tile_overlap) == (1.0, 0.3)
    assert settings.voting.min_views == 2
    assert settings.visibility.min_mask_confidence == 0.5


def test_profiles_and_custom_values():
    options = gui_options.RunOptions.from_settings(load_settings())
    assert options.profile == "balanced"
    options.backend = "sam3"
    assert options.profile == "balanced"
    options.sam_tile_overlap = 0.1
    assert options.profile == "custom"
    options.apply_profile("fast")
    assert (options.sam_image_scale, options.profile) == (0.5, "fast")


def test_invalid_value_is_refused_at_start(tmp_path, ready_settings):
    session = make_session(tmp_path / "sessions")
    controller = Controller(session, settings=ready_settings)
    controller.set_option("min_share", None)
    with pytest.raises(ActionError) as raised:
        controller.start()
    assert raised.value.key == "error.invalid_setting"
    assert not (session / "options.json").exists()


def test_start_is_locked_with_a_reason(tmp_path, ready_settings):
    standalone = Controller(settings=ready_settings, sessions_root=tmp_path / "none")
    assert standalone.start_lock == "lock.standalone"
    ready_settings.paths.flurseg_model = tmp_path / "missing"
    controller = Controller(make_session(tmp_path / "sessions"), settings=ready_settings)
    assert controller.start_lock == "lock.model_missing"
    with pytest.raises(ActionError):
        controller.start()
    controller.select_model("sam3")
    assert controller.start_lock is None  # Missing FlurSeg never blocks SAM 3.


def test_everything_is_locked_while_running(tmp_path, ready_settings):
    session = make_session(tmp_path / "sessions")
    controller = Controller(session, settings=ready_settings)
    controller.start()
    actions = (
        lambda: controller.select_model("sam3"),
        lambda: controller.select_profile("fast"),
        lambda: controller.set_option("road", False),
        lambda: controller.save_class("x", ["x"], "#112233"),
        controller.reset_classes,
        controller.start,
    )
    for action in actions:
        with pytest.raises(ActionError) as raised:
            action()
        assert raised.value.key == "lock.running"
    assert not controller.cancel_before_start()  # A started job is no longer a cancel.


def test_a_reopened_window_shows_the_runs_own_choice(tmp_path, ready_settings):
    session = make_session(tmp_path / "sessions")
    first = Controller(session, settings=ready_settings)
    first.select_model("sam3")
    first.select_profile("fast")
    first.start()
    reopened = Controller(session, settings=ready_settings)
    assert (reopened.options.backend, reopened.options.profile) == ("sam3", "fast")


def test_recent_projects_are_view_only(tmp_path, ready_settings):
    root = tmp_path / "sessions"
    for name, created in (("old", 100.0), ("new", 200.0)):
        session = make_session(root, name, created_at=created)
        (session / "options.json").write_text('{"model": {"title": "FlurSeg"}}', encoding="utf-8")
        (session / "launcher_status.json").write_text('{"phase": "complete"}', encoding="utf-8")
    make_session(root, "never-started")
    controller = Controller(settings=ready_settings, sessions_root=root)
    assert [entry["session"].name for entry in controller.recent] == ["new"]
    controller.open_recent(root / "new")
    assert controller.run.state == "done" and controller.options_lock == "lock.viewing"
    with pytest.raises(ActionError):
        controller.start()
    controller.close_recent()
    assert controller.session is None


def test_inactive_helpers_are_named(tmp_path, ready_settings):
    controller = Controller(make_session(tmp_path / "sessions"), settings=ready_settings)
    assert not controller.inactive_helpers("flurseg") and not controller.inactive_helpers("sam3")
    controller.set_class_enabled("road", False)
    controller.set_class_enabled("tree", False)
    inactive = controller.inactive_helpers("sam3")
    assert inactive["road"] == ["road"]
    assert inactive["vegetation_assist"] == ["tree"] and inactive["vegetation"] == ["tree"]


def test_download_command_hits_the_configured_path(tmp_path, ready_settings):
    default = Controller(settings=load_settings(), sessions_root=tmp_path / "none")
    command = default.download_command("flurseg")
    assert "5df53dbbf721778f1ecd4903369e9465926ae2fd" in command
    assert '--include "v1/*" --local-dir models/flurseg' in command
    ready_settings.paths.flurseg_model = tmp_path / "Modelle" / "v1"
    custom = Controller(settings=ready_settings, sessions_root=tmp_path / "none")
    assert str(tmp_path / "Modelle") in custom.download_command("flurseg")


# ---------------------------------------------------------------------------- progress, history


def log_lines(count=10, seconds=2, total=None, start=0):
    lines = []
    for index in range(1, count + 1):
        clock = start + index * seconds
        lines.append(
            f"2026-11-22 10:{clock // 60:02d}:{clock % 60:02d},123 INFO Chunk Nord (A): "
            f"camera {index}/{total or count} (DJI ({index})) complete; {index} images used"
        )
    return lines


def test_progress_parses_todays_and_older_lines(tmp_path):
    path = tmp_path / "stage2.log"
    old = "2026-10-09 08:01:00,000 INFO Chunk 1: image 3/9 (image-3) complete"
    path.write_text(old, encoding="utf-8")
    assert (gui_progress.image_progress(path).done, gui_progress.image_progress(path).total) == (
        3,
        9,
    )
    path.write_text("\n".join(log_lines(4, 5, total=20)), encoding="utf-8")
    progress = gui_progress.image_progress(path)
    assert (progress.done, progress.total, progress.name) == (4, 20, "DJI (4)")
    assert progress.seconds_per_image == 5 and progress.percent == 20
    assert progress.remaining_seconds == 80


def test_progress_counts_skips_and_several_chunks(tmp_path):
    lines = [
        "2026-11-22 10:00:00,000 INFO Segmentation plan: 2 chunks, 9 cameras",
        *log_lines(3, total=4),
        "2026-11-22 10:00:07,000 INFO Chunk Nord: camera 4/4 (D) skipped; no remaining cloud "
        "points in view",
        "2026-11-22 10:00:07,500 INFO Chunk Nord: all cameras processed; post-filter follows",
    ]
    path = tmp_path / "stage2.log"
    path.write_text("\n".join(lines), encoding="utf-8")
    progress = gui_progress.image_progress(path)
    assert (progress.done, progress.total, progress.finishing) == (4, 9, False)
    lines += log_lines(5, total=5, start=20)
    lines.append("2026-11-22 10:00:31,000 INFO Chunk Süd: all cameras processed; post-filter")
    path.write_text("\n".join(lines), encoding="utf-8")
    progress = gui_progress.image_progress(path)
    assert (progress.done, progress.total, progress.finishing) == (9, 9, True)
    assert progress.percent == 99


def history_run(root, name, backend, seconds, settings=None, cached=False, project=None):
    session = make_session(root, name, **({"project": project} if project else {}))
    values = dict(settings or {})
    if backend:
        values["segmentation"] = {"backend": backend}
    (session / "job.json").write_text(json.dumps({"settings": values}), encoding="utf-8")
    lines = log_lines(10, seconds)
    if cached:
        lines.append("2026-11-22 10:00:00,000 INFO Chunk A: camera 1/10 (D) reused from cache")
    (session / "stage2.log").write_text("\n".join(lines), encoding="utf-8")


def test_history_is_kept_per_model(tmp_path):
    history_run(tmp_path, "fast", "flurseg", 3, {"flurseg": {"resolution": "fast"}})
    history_run(tmp_path, "cached", "flurseg", 9, cached=True)
    history_run(tmp_path, "elsewhere", "flurseg", 11, project="D:/Other.psx")
    history_run(tmp_path, "old-sam", None, 47)  # Before the model choice: always SAM 3.
    rate = gui_options.historical_seconds_per_window(tmp_path, "D:/Jobs/Site.psx")
    assert rate == pytest.approx(3 / window_count(5280, 3956, "fast"))
    sam = gui_options.historical_seconds_per_image(tmp_path, SIZES, "D:/Jobs/Site.psx")
    assert sam == pytest.approx(47 * gui_options.batch_time_factor(8))


def test_estimate_counts_windows_and_never_divides_by_fifteen(tmp_path):
    options = gui_options.RunOptions.from_settings(load_settings())
    result = options.estimate(tmp_path, len(SIZES), SIZES)
    windows = len(SIZES) * window_count(5280, 3956, "balanced")
    assert (result.units, result.from_history) == (windows, False)
    expected = windows * 0.09 + gui_options.workflow_overhead(len(SIZES))
    assert result.seconds == pytest.approx(expected)
    options.apply_profile("accurate")
    assert options.estimate(tmp_path, len(SIZES), SIZES).seconds > result.seconds

    options.backend = "sam3"
    sam = options.estimate(tmp_path, len(SIZES), SIZES)
    tiles = gui_options.total_tile_count(SIZES, 0.75, 0.2)
    assert sam.units == tiles
    assert sam.seconds == pytest.approx(
        tiles * gui_options.SAM_SECONDS_PER_TILE + gui_options.workflow_overhead(len(SIZES))
    )


def test_run_states_and_times(tmp_path):
    session = make_session(tmp_path)
    assert gui_progress.read_run(session).state == "ready"
    (session / "options.json").write_text("{}", encoding="utf-8")
    assert gui_progress.read_run(session).phase == "prepare"
    names = ["prepare", "export", "classify", "import", "complete"]

    def status(count, **values):
        timeline = [{"phase": n, "at": 100 + 10 * i} for i, n in enumerate(names[:count])]
        data = {"phase": names[count - 1], "timeline": timeline, **values}
        (session / "launcher_status.json").write_text(json.dumps(data), encoding="utf-8")

    status(3)
    run = gui_progress.read_run(session, now=135)
    assert [step.state for step in run.steps] == ["done", "done", "active", "waiting"]
    assert [step.seconds for step in run.steps[:3]] == [10, 10, 15]
    assert run.total_seconds == 35 and run.can_cancel
    status(4)
    assert not gui_progress.read_run(session).can_cancel
    assert gui_progress.read_run(session, launcher_alive=False).interrupted
    status(5)
    run = gui_progress.read_run(session, launcher_alive=False)
    assert run.state == "done" and run.total_seconds == 40


def test_cancel_only_during_prepare_and_classification(tmp_path):
    session = make_session(tmp_path)
    status = session / "launcher_status.json"
    for phase in ("export", "import", "complete"):
        status.write_text(json.dumps({"phase": phase}), encoding="utf-8")
        assert not gui_progress.request_cancel(session)
    assert not (session / "cancel_requested.json").exists()
    status.write_text('{"phase": "classify"}', encoding="utf-8")
    assert gui_progress.request_cancel(session)
    assert (session / "cancel_requested.json").exists()


def test_cancel_never_kills_a_foreign_process(tmp_path, monkeypatch):
    session = make_session(tmp_path)
    (session / "launcher_status.json").write_text('{"phase": "classify"}', encoding="utf-8")
    (session / "worker.json").write_text(
        json.dumps({"pid": os.getpid(), "action": "stage2"}), encoding="utf-8"
    )

    def never(self):
        raise AssertionError("a process that is not the worker was terminated")

    monkeypatch.setattr(gui_progress.psutil.Process, "terminate", never)
    assert gui_progress.request_cancel(session)


def test_gpu_memory_is_not_queried_when_the_gpu_is_hidden(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(gui_options.subprocess, "run", lambda *a, **k: pytest.fail("queried"))
    assert gui_options.gpu_memory_gib() is None


def test_ui_modules_do_not_import_torch(tmp_path):
    program = (
        "import sys; from msam3 import gui_options; from msam3.app import controller, screen; "
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith("PYTEST")}
    environment["NICEGUI_STORAGE_PATH"] = str(tmp_path / "nicegui")
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=ROOT, env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------- lifecycle


class Clock:
    def __init__(self):
        self.now = 1000.3

    def __call__(self):
        return self.now


def lifecycle():
    clock = Clock()
    life = Lifecycle(clock=clock)

    def tick(state, windows, seconds=1, standalone=False):
        action = None
        for _ in range(seconds):
            clock.now += 1.01
            action = life.tick(state, windows, standalone) or action
        return action

    return life, clock, tick


def test_closing_before_start_cancels_after_the_reload_grace():
    _, _, tick = lifecycle()
    assert tick("ready", 1, 3) is None
    assert tick("ready", 0, 6) is None
    assert tick("ready", 1, 2) is None  # The reload came back in time.
    assert tick("ready", 0, 6) is None
    assert tick("ready", 0, 6) == "cancel"


def test_no_window_never_hangs_forever():
    _, _, tick = lifecycle()
    assert tick("ready", 0, 110) is None
    assert tick("ready", 0, 12) == "no_window"


def test_a_run_survives_reload_lost_socket_and_closed_window():
    _, _, tick = lifecycle()
    tick("ready", 1, 2)
    assert tick("running", 0, 4000) is None


def test_sleep_is_not_a_closed_window():
    life, clock, tick = lifecycle()
    tick("ready", 1, 2)
    clock.now += 3600
    assert life.tick("ready", 0) is None
    assert tick("ready", 0, 5) is None


def test_the_app_ends_after_the_final_state():
    _, _, tick = lifecycle()
    tick("running", 1, 2)
    assert tick("done", 1, 300) is None
    assert tick("done", 0, 11) == "exit"

    _, _, tick = lifecycle()
    tick("running", 1, 2)
    tick("running", 0, 500)
    assert tick("error", 0, 100) is None  # Not seen yet: wait longer, but not forever.
    assert tick("error", 0, 25) == "exit"


def test_standalone_ends_with_its_window():
    _, _, tick = lifecycle()
    tick("ready", 1, 2, standalone=True)
    assert tick("ready", 0, 12, standalone=True) == "exit"


def test_heartbeat_needs_an_event_loop_that_answers(tmp_path):
    path = tmp_path / "ui_alive"
    heartbeat = Heartbeat(path)
    scheduled = []
    heartbeat.tick(scheduled.append)
    heartbeat.tick(scheduled.append)
    assert not path.exists()  # The loop has not answered yet.
    scheduled.pop()()
    heartbeat.tick(scheduled.append)
    assert path.exists()
    stamp = path.stat().st_mtime_ns - 5_000_000_000
    os.utime(path, ns=(stamp, stamp))
    heartbeat.tick(scheduled.append)
    assert path.stat().st_mtime_ns == stamp  # A hung loop stops the heartbeat.


# ---------------------------------------------------------------------------- orphaned worker


@pytest.fixture
def orphan(tmp_path, ready_settings):
    """A worker that runs on although the launcher that waited for it is gone."""
    session = make_session(tmp_path / "sessions", "0f8fad5bd9cb469fa165", launcher_pid=2**22 + 9)
    (session / "options.json").write_text('{"model": {"backend": "flurseg"}}', encoding="utf-8")
    timeline = [{"phase": "prepare", "at": 10.0}, {"phase": "classify", "at": 20.0}]
    status = {"phase": "classify", "timeline": timeline}
    (session / "launcher_status.json").write_text(json.dumps(status), encoding="utf-8")
    started = []

    def start(*arguments, action="stage2"):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)", *map(str, arguments)]
        )
        started.append(child)
        (session / "worker.json").write_text(
            json.dumps({"pid": child.pid, "action": action}), encoding="utf-8"
        )
        return child

    yield session, ready_settings, start
    for child in started:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_stop_for_the_worker_of_a_dead_launcher(orphan):
    session, settings, start = orphan
    child = start("msam3.gui_worker", "stage2", "--session", session)
    controller = Controller(session, settings=settings)
    assert controller.run.interrupted and controller.run.worker_running
    assert not controller.run.can_cancel
    revision = controller.revision

    controller.stop_worker()

    child.wait(timeout=10)
    assert not controller.run.worker_running and controller.revision > revision
    with pytest.raises(ActionError) as raised:
        controller.stop_worker()
    assert raised.value.key == "error.worker_gone"


def test_stop_also_works_from_the_recent_projects(orphan):
    session, settings, start = orphan
    child = start("msam3.gui_worker", "stage2", "--session", session)
    controller = Controller(settings=settings, sessions_root=session.parent)
    controller.open_recent(session)
    assert controller.run.worker_running
    controller.stop_worker()
    child.wait(timeout=10)


@pytest.mark.parametrize("case", ["foreign", "other session", "similar name", "finalize"])
def test_stop_is_never_offered_for_another_process(orphan, case):
    session, settings, start = orphan
    arguments = {
        "foreign": ("some.other.tool", session),
        "other session": ("msam3.gui_worker", session.parent / "another-session"),
        "similar name": ("msam3.gui_worker_notes.txt", session),
        "finalize": ("msam3.gui_worker", session),
    }[case]
    child = start(*arguments, action="finalize" if case == "finalize" else "stage2")
    controller = Controller(session, settings=settings)
    assert controller.run.interrupted and not controller.run.worker_running
    with pytest.raises(ActionError):
        controller.stop_worker()
    assert child.poll() is None


def test_no_stop_while_the_launcher_still_waits(orphan):
    session, _, start = orphan
    start("msam3.gui_worker", "stage2", "--session", session)
    run = gui_progress.read_run(session, launcher_alive=True)
    assert run.state == "running" and run.can_cancel and not run.worker_running


# ---------------------------------------------------------------------------- token


def guarded_request(path="/", host="127.0.0.1:5123", query=b"", kind="http", **headers):
    reached, sent = [], []

    async def application(scope, receive, send):
        reached.append(scope["path"])

    async def send(message):
        sent.append(message)

    values = [(b"host", host.encode())]
    values += [(name.encode(), value.encode()) for name, value in headers.items()]
    scope = {"type": kind, "path": path, "query_string": query, "headers": values}
    asyncio.run(SessionGuard(application, "secret-token", 5123)(scope, None, send))
    return bool(reached), sent


def test_token_is_needed_for_pages_files_and_websocket():
    cookie = "msam3_5123=secret-token"
    assert guarded_request("/models", cookie=cookie)[0]
    assert guarded_request("/_nicegui_ws/socket.io/", kind="websocket", cookie=cookie)[0]
    assert guarded_request("/", cookie=f"broken[]=1; {cookie}")[0]

    for denied in (
        guarded_request("/models"),
        guarded_request("/models", cookie="msam3_5123=wrong"),
        guarded_request("/models", cookie="msam3_9999=secret-token"),
        guarded_request("/models", host="evil.example:5123", cookie=cookie),
        guarded_request("/models", cookie=cookie, origin="http://127.0.0.1:9999"),
    ):
        assert not denied[0] and denied[1][0]["status"] == 403

    reached, sent = guarded_request(
        "/_nicegui_ws/socket.io/", kind="websocket", cookie=cookie, origin="https://evil.example"
    )
    assert not reached and sent == [{"type": "websocket.close", "code": 1008}]


def test_start_link_becomes_a_strict_cookie():
    reached, sent = guarded_request(query=b"token=secret-token")
    headers = dict(sent[0]["headers"])
    assert not reached and sent[0]["status"] == 303 and headers[b"location"] == b"/"
    assert b"HttpOnly" in headers[b"set-cookie"] and b"SameSite=Strict" in headers[b"set-cookie"]
    assert guarded_request(query=b"token=wrong")[1][0]["status"] == 403


def test_server_is_loopback_only_protected_and_works_offline(tmp_path):
    session = make_session(tmp_path / "sessions")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("PYTEST")}
    environment["NICEGUI_STORAGE_PATH"] = str(tmp_path / "nicegui")
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    arguments = [sys.executable, "-m", "msam3.gui_options", "--session", str(session)]
    log_path = tmp_path / "ui.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [*arguments, "--no-browser"], cwd=ROOT, env=environment, stdout=log, stderr=log
        )
    try:
        deadline = time.monotonic() + 90
        while not (session / "ui.json").is_file():
            assert process.poll() is None, log_path.read_text(encoding="utf-8")
            assert time.monotonic() < deadline, log_path.read_text(encoding="utf-8")
            time.sleep(0.2)
        started = gui_progress.read_json(session / "ui.json")
        url = urllib.parse.urlsplit(started["url"])
        assert url.hostname == "127.0.0.1" and url.port not in (None, 8080)

        def get(path, cookie=None):
            connection = http.client.HTTPConnection(url.hostname, url.port, timeout=20)
            connection.request("GET", path, headers={"Cookie": cookie} if cookie else {})
            response = connection.getresponse()
            result = response.status, dict(response.getheaders()), response.read()
            connection.close()
            return result

        # The sign of life that the launcher watches, from the real monitor and event loop.
        alive = session / "ui_alive"
        while not alive.is_file():
            assert time.monotonic() < deadline, "no heartbeat"
            time.sleep(0.2)
        before = alive.stat().st_mtime_ns
        while alive.stat().st_mtime_ns == before:
            assert time.monotonic() < deadline, "the heartbeat stopped"
            time.sleep(0.2)
        assert get("/")[0] == 403
        code, headers, _ = get(f"/?{url.query}")
        assert code == 303
        cookie = {key.lower(): value for key, value in headers.items()}["set-cookie"].split(";")[0]
        code, headers, body = get("/", cookie)
        assert code == 200 and b"Metashape AI Classifier" in body
        policy = {key.lower(): value for key, value in headers.items()}["content-security-policy"]
        assert "https:" not in policy
        assert not re.search(rb"<(?:script|link)[^>]+(?:src|href)=\"https?://", body)
    finally:
        process.terminate()
        process.wait(timeout=20)


# ---------------------------------------------------------------------------- texts and design


def catalogues():
    return i18n.catalogue("de"), i18n.catalogue("en")


def test_languages_have_the_same_keys_and_placeholders():
    german, english = catalogues()
    assert set(german) == set(english)
    for key, text in german.items():
        assert text.strip() and english[key].strip()
        assert set(re.findall(r"{(\w+)}", text)) == set(re.findall(r"{(\w+)}", english[key])), key


def test_swiss_spelling_everywhere():
    sharp_s = "\u00df"
    assert not [key for key, text in catalogues()[0].items() if sharp_s in text]
    for folder in ("src", "metashape_scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            assert sharp_s not in path.read_text(encoding="utf-8"), path


def test_every_used_text_key_exists():
    used = set()
    for path in (ROOT / "src/msam3/app").rglob("*.py"):
        code = path.read_text(encoding="utf-8")
        used |= set(re.findall(r"\bt\(\s*\"([a-z0-9_]+\.[a-z0-9_.]+)\"", code))
        used |= set(re.findall(r"\"((?:lock|error|cancel)\.[a-z0-9_]+)\"", code))
        used -= {name for name in used if name.endswith("_")}  # Only the start of a key.
    used |= {f"step.{key}" for key in gui_progress.STEPS}
    used |= {f"running.{key}" for key in gui_progress.STEPS}
    used |= {f"recent.{key}" for key in (*gui_progress.STEPS, *gui_progress.FINAL, "open")}
    used |= {f"flow.{key}" for key in ("waiting", "active", "done", "error", "cancelled")}
    used |= {f"nav.{key}" for key in ("overview", "models", "classes", "settings", "log")}
    used |= {f"profile.{key}" for key in (*gui_options.PROFILE_NAMES, "custom")}
    used |= {f"class_error.{key}" for key in user_classes.UserClassError.MESSAGES}
    used |= {f"lock.model_{key}" for key in ("missing", "invalid", "runtime")}
    used |= {f"models.state.{key}" for key in ("ready", "missing", "invalid", "runtime")}
    for backend, model in backends.MODELS.items():
        used |= {f"profile.{backend}_{key}" for key in gui_options.PROFILE_NAMES}
        used |= {f"model.{backend}.{key}" for key in ("short", "choice", "target", "command")}
        classes = load_classes() if backend == "sam3" else model.classes()
        used |= {f"class.{item.name}" for item in classes.classes}
    assert sorted(used - set(catalogues()[0])) == []


def test_speed_wording_is_honest():
    german, english = catalogues()
    assert german["model.flurseg.short"] == "im Test ca. 6× schneller"
    assert english["model.flurseg.short"] == "approx. 6× faster in the test"
    assert german["model.speed_note"] == (
        "Im Testprojekt mit 291 Fotos: Klassierung und Voting 6,1× schneller, gesamter Ablauf "
        "5,1× schneller. Die früheren 15× bezogen sich nur auf die Bildsegmentierung in einem "
        "anderen Test."
    )


def test_language_is_saved():
    assert i18n.load_language() == "de"
    i18n.save_language("en")
    assert i18n.load_language() == "en"
    assert i18n.Translator("de").number(1234567) == "1\u2019234\u2019567"


def test_design_colours():
    expected = {
        "bg": "#09131d",
        "panel": "#0e1c28",
        "dropzone": "#0c1924",
        "canvas": "#050b12",
        "line": "#20394b",
        "line-soft": "rgba(113,154,181,.18)",
        "line-viewer": "#28475b",
        "line-field": "#456176",
        "line-strong": "#47718a",
        "text": "#eaf2f8",
        "text-2": "#c9d8e4",
        "muted": "#96aabe",
        "faint": "#7890a5",
        "icon": "#a8bfd1",
        "accent": "#26b7ef",
        "accent-light": "#8cdbfa",
        "on-accent": "#06202f",
        "success": "#4bc692",
        "error": "#ef6b73",
        "overlay": "rgba(9,19,29,.86)",
    }
    assert {key: theme.COLORS[key] for key in expected} == expected
    assert "<svg" in theme.LOGO_FILE.read_text(encoding="utf-8")


def test_app_window_prefers_edge_and_falls_back_to_the_default_browser(tmp_path, monkeypatch):
    edge = tmp_path / "Microsoft/Edge/Application/msedge.exe"
    edge.parent.mkdir(parents=True)
    edge.write_bytes(b"")
    found = browser.find_app_browser({"ProgramFiles": str(tmp_path)}, which=lambda name: None)
    assert Path(found) == edge

    started, opened = [], []
    monkeypatch.setattr(browser.subprocess, "Popen", lambda args, **kwargs: started.append(args))
    monkeypatch.setattr(browser.webbrowser, "open", lambda url: opened.append(url) or True)
    monkeypatch.setattr(browser, "find_app_browser", lambda: str(edge))
    assert browser.open_window("http://127.0.0.1:5123/")
    assert "--app=http://127.0.0.1:5123/" in started[0] and not opened
    monkeypatch.setattr(browser, "find_app_browser", lambda: None)
    assert browser.open_window("http://127.0.0.1:5123/")
    assert opened == ["http://127.0.0.1:5123/"]


def test_replace_waits_for_a_reader(tmp_path):
    attempts = []

    class Source:
        def replace(self, target):
            attempts.append(target)
            if len(attempts) < 3:
                raise PermissionError("being read")

    cache.replace_file(Source(), tmp_path / "status.json")
    assert len(attempts) == 3


def test_install_check_names_stale_packages(monkeypatch, capsys):
    found = install_check.problems({"no_such_module": ("no-such-distribution", None)})
    assert found == ["no-such-distribution fehlt"]
    assert "statt 0.0" in install_check.problems({"numpy": ("numpy", "0.0")})[0]
    monkeypatch.setattr(install_check, "problems", lambda: ["nicegui fehlt"])
    assert install_check.main() == 3
    assert "install.bat" in capsys.readouterr().out


def test_result_page_lists_the_checked_final_file_first():
    from msam3.app.pages.overview import delivered_files, size_text

    path = r"D:\Test Script\export\punktwolke_class.las"
    checked = {"path": path, "las_version": "1.4", "point_format": 7, "bytes": 3320154892}
    assert delivered_files({"files": [checked], "outputs": [path]}) == [checked]
    # Results written before the check existed still show their path.
    assert delivered_files({"outputs": [path]}) == [{"path": path}]
    assert delivered_files({}) == []
    assert size_text(checked["bytes"]) == "3.1 GB"
    german, english = catalogues()
    assert "{version}" in german["result.format"] and "31" in german["result.output_hint"]
    assert "{format}" in english["result.format"] and "31" in english["result.output_hint"]
