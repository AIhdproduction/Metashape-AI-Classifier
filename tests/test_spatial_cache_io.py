import csv
import importlib.util
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import laspy
import numpy as np
import pytest

from msam3.cache import LabelCache
from msam3.camera_model import Camera
from msam3.config import Classes
from msam3.pointcloud.las_io import merge_class_exports
from msam3.pointcloud.spatial import spatial_blocks, voxel_representatives


def test_embedded_legend_writes_color_column(tmp_path, monkeypatch):
    """Keep the stdlib-only Metashape helper compatible with palette-configured jobs."""
    monkeypatch.setitem(sys.modules, "Metashape", types.ModuleType("Metashape"))
    script = Path(__file__).parents[1] / "metashape_scripts" / "common.py"
    spec = importlib.util.spec_from_file_location("test_metashape_common", script)
    assert spec and spec.loader
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)

    common.write_legends(
        tmp_path,
        {
            "unknown_code": 1,
            "unknown_color": "#787878",
            "classes": [{"name": "ground", "code": 2, "color": "#663300", "prompts": ["ground"]}],
        },
    )

    with (tmp_path / "class_legend.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {"name": "Unclassified", "code": "1", "color": "#787878", "prompts": ""},
        {"name": "ground", "code": "2", "color": "#663300", "prompts": "ground"},
    ]


def test_metashape_launcher_reloads_embedded_modules():
    """A second Tools > Run Script must not retain a pre-palette common.py module."""
    root = Path(__file__).parents[1]
    program = textwrap.dedent(
        """
        import runpy
        import sys
        import types

        sys.modules["Metashape"] = types.ModuleType("Metashape")
        import metashape_scripts.common as common

        stale = object()
        common.write_legends = stale
        runpy.run_path("start_in_metashape.py", run_name="launcher_test")
        assert common.write_legends is not stale
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_gui_restores_photo_cloud_before_a_second_run(monkeypatch):
    monkeypatch.setitem(sys.modules, "Metashape", types.ModuleType("Metashape"))
    script = Path(__file__).parents[1] / "metashape_scripts" / "gui.py"
    spec = importlib.util.spec_from_file_location("test_metashape_gui", script)
    assert spec and spec.loader
    gui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gui)

    class Asset:
        def __init__(self, key, label, meta):
            self.key, self.label, self.meta = key, label, meta

    photo = Asset(7, "Original photo cloud", {})
    prior_result = Asset(
        8,
        "SAM3 classified",
        {"msam3.import_id": "prior", "msam3.source_point_cloud_key": "7"},
    )

    class Chunk:
        point_clouds = [photo, prior_result]
        point_cloud = prior_result

    class Document:
        chunks = [Chunk()]

    gui.restore_source_point_clouds(Document())
    assert Document.chunks[0].point_cloud is photo


def test_voxels_cross_batch_and_negative(tmp_path):
    points = np.array([[0, 0, 0], [0.01, 0, 0], [1, 0, 0], [-0.01, 0, 0], [1.01, 0, 0]])
    first = voxel_representatives(points, 0.03, tmp_path, batch_size=2, buckets=3)
    assert first.tolist() == [0, 2, 3]


def test_block_covering_entire_image_not_culled():
    camera = Camera(1, 20, 20, np.eye(4), {"f": 10})
    points = np.array([[-20, -20, 10], [20, 20, 10], [0, 0, 10]], dtype=float)
    blocks = spatial_blocks(points, 100)
    selected = np.concatenate(list(blocks.batches(camera, np.eye(4), 100)))
    assert 2 in selected


def test_cache_corruption_and_model_change(tmp_path):
    cache = LabelCache(tmp_path, "config", "model")
    identity = {"size": [2, 2]}
    labels = np.array([[0, 255], [1, 1]], np.uint8)
    cache.save("a", labels, np.ones((2, 2), np.float32), identity)
    assert cache.load("a", identity) is not None
    (tmp_path / "a.labels.png").write_bytes(b"broken")
    assert cache.load("a", identity) is None


def test_merge_metashape_class_partitions(tmp_path):
    classes = Classes.model_validate(
        {"classes": [{"name": "custom", "code": 64, "prompts": ["custom"], "color": "#123456"}]}
    )
    source = laspy.create(point_format=2, file_version="1.2")
    source.x = [1.0, 2.0]
    source.y, source.z = [0.0, 0.0], [0.0, 0.0]
    source.red, source.green, source.blue = [1234, 4321], [2345, 5432], [3456, 6543]
    source.classification = [0, 0]  # Native export wraps 64 -> 0.
    path = tmp_path / "class64.las"
    source.write(path)
    output = tmp_path / "final.las"
    counts = merge_class_exports(
        [{"path": str(path), "code": 64, "count": 2}],
        output,
        classes,
        'LOCAL_CS["Local",UNIT["metre",1]]',
        1,
    )
    result = laspy.read(output)
    assert list(result.classification) == [64, 64]
    assert list(result.red) == [1234, 4321]
    assert list(result.green) == [2345, 5432]
    assert list(result.blue) == [3456, 6543]
    assert counts[64] == 2
    with pytest.raises(ValueError, match="count"):
        merge_class_exports([{"path": str(path), "code": 64, "count": 3}], output, classes, "", 1)
