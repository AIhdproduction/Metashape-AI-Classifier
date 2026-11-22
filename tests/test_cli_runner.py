import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

from msam3.cache import atomic_json, file_hash, json_hash
from msam3.cli import main
from msam3.config import MetashapeConfig, Settings, load_classes
from msam3.driver import execute, prepare_stage3_inputs
from msam3.metashape_runner import run_metashape


def test_classes_cli(capsys):
    assert main(["classes", "--model", "sam3"]) == 0
    output = capsys.readouterr().out
    assert "sidewalk" in output and "73" in output and "110" in output


def test_missing_checkpoint_before_metashape(tmp_path):
    settings = Settings()
    settings.segmentation.backend = "sam3"
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


def test_gui_stage3_contract_is_verified_before_live_import(tmp_path):
    """The direct-Metashape launcher uses the same LAS integrity contract as the CLI."""
    classes = load_classes()
    las = laspy.create(point_format=7, file_version="1.4")
    las.header.scales = np.full(3, 0.001)
    las.x, las.y, las.z = [1.0], [2.0], [3.0]
    path = tmp_path / "classified.las"
    las.write(path)
    atomic_json(tmp_path / "chunk.json", {"source_signature": "source"})
    source = {
        "directory": str(tmp_path),
        "classified_las": str(path),
        "classes_hash": json_hash(classes.model_dump()),
        "stage1_identity": "identity",
        "las_sha256": file_hash(path),
    }
    prepared = prepare_stage3_inputs([source], "identity", classes)
    assert prepared[0]["precision_m"] == 0.001
    assert prepared[0]["import_id"]
    assert "precision_m" not in source
