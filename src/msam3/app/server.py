"""Local server: loopback only, random port, session token, and it never outlives its purpose."""

import asyncio
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path

from msam3.app.lifecycle import Heartbeat, Lifecycle
from msam3.app.security import SessionGuard, new_token
from msam3.cache import atomic_json
from msam3.config import REPO_ROOT

LOG = logging.getLogger(__name__)
HOST = "127.0.0.1"
NO_BROWSER_VARIABLE = "MSAM3_NO_BROWSER"
EXIT_STARTED, EXIT_CANCELLED, EXIT_NO_WINDOW = 0, 1, 2


def free_port() -> int:
    while True:
        with socket.socket() as probe:
            probe.bind((HOST, 0))
            port = probe.getsockname()[1]
        if port != 8080:  # NiceGUI's default, often used by the user's other apps.
            return port


def session_environment() -> None:
    # pythonw has no console, and uvicorn fails on a missing stream.
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
    # NiceGUI would otherwise create .nicegui in the current folder.
    os.environ.setdefault("NICEGUI_STORAGE_PATH", str(REPO_ROOT / "work/.nicegui"))


def serve(session: Path | None = None, *, open_browser: bool = True) -> int:
    session_environment()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from nicegui import Client, app, ui

    from msam3.app import browser, theme
    from msam3.app.controller import Controller
    from msam3.app.screen import PAGES, Screen

    open_browser = open_browser and not os.environ.get(NO_BROWSER_VARIABLE)
    controller = Controller(session)
    port, token = free_port(), new_token()
    url = f"http://{HOST}:{port}/?token={token}"
    app.add_middleware(SessionGuard, token=token, port=port)

    def page_builder(key: str):
        # A closure, because NiceGUI turns a page function's arguments into URL parameters.
        def build():
            Screen(controller, key).build()

        return build

    for page in PAGES:
        ui.page(page.path)(page_builder(page.key))

    outcome = {"code": None}
    server = {"loop": None}
    heartbeat = Heartbeat(session / "ui_alive") if session else None
    stopped = threading.Event()

    def connected_windows() -> int:
        return sum(1 for client in list(Client.instances.values()) if client.has_socket_connection)

    def end(code: int) -> None:
        outcome["code"] = code
        stopped.set()
        app.shutdown()
        # A hanging socket must not leave an orphan process behind.
        watchdog = threading.Timer(15, os._exit, (code,))
        watchdog.daemon = True
        watchdog.start()

    def monitor() -> None:
        lifecycle = Lifecycle()
        cancelled_at = None
        hashing = None
        while not stopped.wait(1):
            try:
                controller.poll()
                if controller.weights_state == "checking" and not (hashing and hashing.is_alive()):
                    hashing = threading.Thread(target=controller.identify_weights, daemon=True)
                    hashing.start()
                if controller.cancelled_here:
                    # A moment for the window to show the state and close itself.
                    cancelled_at = cancelled_at or time.monotonic()
                    if time.monotonic() - cancelled_at >= 1.5:
                        end(EXIT_CANCELLED)
                    continue
                action = lifecycle.tick(
                    controller.run.state, connected_windows(), controller.standalone
                )
                if action == "cancel":
                    if controller.cancel_before_start():
                        LOG.info("Window closed before Start: cancelled")
                        end(EXIT_CANCELLED)
                elif action == "no_window":
                    LOG.error(
                        "Es hat sich kein Browserfenster verbunden. Bitte Microsoft Edge oder "
                        "Chrome prüfen und das Skript erneut starten."
                    )
                    end(EXIT_NO_WINDOW)
                elif action == "exit":
                    end(EXIT_STARTED)
                # Last, so that only a monitor that really works reports itself alive.
                if heartbeat and server["loop"] and not stopped.is_set():
                    heartbeat.tick(server["loop"].call_soon_threadsafe)
            except Exception:
                LOG.exception("App monitor failed; trying again")

    def announce() -> None:
        # No own deadline: ui.json tells the launcher that the app is up, and the launcher limits
        # the wait. Giving up earlier would get a slowly starting app ended.
        while not stopped.is_set():
            try:
                with socket.create_connection((HOST, port), timeout=1):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            return
        if session:
            # For reopening and tests. The link is not logged: the log is shown in the app.
            atomic_json(session / "ui.json", {"url": url, "port": port, "pid": os.getpid()})
        else:
            sys.stdout.write(f"Metashape AI Classifier: {url}\n")
            sys.stdout.flush()
        if open_browser and not browser.open_window(url):
            LOG.error("No browser window could be opened.")

    def on_startup() -> None:
        server["loop"] = asyncio.get_running_loop()
        threading.Thread(target=announce, daemon=True).start()

    app.on_startup(on_startup)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        ui.run(
            host=HOST,
            port=port,
            title="Metashape AI Classifier",
            favicon=theme.LOGO_FILE,
            dark=True,
            language="de" if controller.t.language == "de" else "en-US",
            reload=False,
            show=False,
            tailwind=False,
            show_welcome_message=False,
            reconnect_timeout=5.0,
        )
    finally:
        stopped.set()
    if outcome["code"] is not None:
        return outcome["code"]
    started = session is None or (session / "options.json").is_file()
    return EXIT_STARTED if started else EXIT_CANCELLED
