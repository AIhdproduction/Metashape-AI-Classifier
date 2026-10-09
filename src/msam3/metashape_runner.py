"""Headless Metashape subprocess with logs and mandatory completion markers."""

import json
import logging
import os
import subprocess
from pathlib import Path

from msam3.config import REPO_ROOT, MetashapeConfig

LOG = logging.getLogger(__name__)


def command(config: MetashapeConfig, script: Path, job: Path) -> list[str]:
    exe = Path(config.exe)
    if not exe.is_file():
        raise FileNotFoundError(f"Metashape executable not found: {exe}. Set metashape.exe.")
    offscreen = config.offscreen == "always"
    if config.offscreen == "auto":
        offscreen = os.name != "nt" or (exe.parent / "plugins/platforms/qoffscreen.dll").is_file()
    if not offscreen and os.name == "nt":
        LOG.info("Using Windows headless -r mode without Qt's optional offscreen plugin")
    return [
        str(exe),
        *(["-platform", "offscreen"] if offscreen else []),
        "-r",
        str(script.resolve()),
        str(job.resolve()),
    ]


def run_metashape(stage: int, job_path: Path, work: Path, config: MetashapeConfig) -> dict:
    script_name = {1: "stage1_export.py", 3: "stage3_import.py"}[stage]
    marker = work / f"stage{stage}_done.json"
    marker.unlink(missing_ok=True)
    log_path = work / f"stage{stage}_metashape.log"
    args = command(config, REPO_ROOT / "metashape_scripts" / script_name, job_path)
    LOG.info("Starting Metashape stage %d; log: %s", stage, log_path)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            args,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    LOG.info("Metashape stage %d is still running; see %s", stage, log_path)
        except BaseException:
            process.terminate()
            process.wait(timeout=30)
            raise
    if code != 0 or not marker.is_file():
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-16:])
        raise RuntimeError(
            f"Metashape stage {stage} failed (exit {code}, "
            f"completion marker {'present' if marker.exists() else 'missing'}). "
            f"Check license, project lock, and {log_path}\n{tail}"
        )
    return json.loads(marker.read_text(encoding="utf-8"))
