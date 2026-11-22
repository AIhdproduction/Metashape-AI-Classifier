"""Open the app as its own Edge/Chrome window, otherwise in the default browser."""

import logging
import os
import shutil
import subprocess
import webbrowser
from pathlib import Path

LOG = logging.getLogger(__name__)
CANDIDATES = (
    ("msedge", "Microsoft/Edge/Application/msedge.exe"),
    ("chrome", "Google/Chrome/Application/chrome.exe"),
)


def find_app_browser(environment=None, which=shutil.which) -> str | None:
    environment = os.environ if environment is None else environment
    roots = [environment.get(key) for key in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA")]
    for name, relative in CANDIDATES:
        for root in roots:
            if root and (Path(root) / relative).is_file():
                return str(Path(root) / relative)
        if found := which(name):
            return found
    return None


def open_window(url: str) -> bool:
    if executable := find_app_browser():
        try:
            # No inherited handles: the browser outlives the app and must not hold its log open.
            subprocess.Popen(
                [executable, f"--app={url}", "--window-size=1360,900", "--no-first-run"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            return True
        except OSError as error:
            LOG.warning("App window failed (%s); using the default browser", error)
    try:
        return webbrowser.open(url)
    except webbrowser.Error as error:
        LOG.warning("No browser could be opened: %s", error)
        return False
