import sys
from pathlib import Path

import pytest

from msam3.cli import main
from msam3.config import MetashapeConfig, Settings, load_classes
from msam3.driver import execute
from msam3.metashape_runner import run_metashape


def test_classes_cli(capsys):
    assert main(["classes"]) == 0
    output = capsys.readouterr().out
    assert "sidewalk" in output and "64" in output and "72" in output


def test_missing_checkpoint_before_metashape(tmp_path):
    settings = Settings()
    settings.paths.model = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="huggingface.co/facebook/sam3"):
        execute("run", tmp_path / "missing.psx", settings, load_classes())


def test_zero_exit_without_marker_is_failure(tmp_path, monkeypatch):
    # A subprocess that exits zero is insufficient evidence of a completed Metashape stage.
    monkeypatch.setattr(
        "msam3.metashape_runner.command", lambda *args: [sys.executable, "-c", "pass"]
    )
    with pytest.raises(RuntimeError, match="completion marker missing"):
        run_metashape(1, Path("job.json"), tmp_path, MetashapeConfig())
