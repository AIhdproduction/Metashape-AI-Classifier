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

from msam3.cache import LabelCache, json_hash
from msam3.camera_model import Camera
from msam3.config import Classes, Settings
from msam3.driver import project_state
from msam3.pointcloud.las_io import describe_output, merge_class_exports
from msam3.pointcloud.spatial import (
    representative_class_mask,
    representative_classifications,
    spatial_blocks,
    voxel_representatives,
)


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
        rows = list(csv.DictReader(stream, delimiter=";"))
    assert rows == [
        {"name": "Unclassified", "code": "1", "color": "#787878", "prompts": ""},
        {"name": "ground", "code": "2", "color": "#663300", "prompts": "ground"},
    ]
    assert (tmp_path / "class_legend.csv").read_text(encoding="utf-8").splitlines()[
        0
    ] == "name;code;color;prompts"


def test_metashape_ground_assist_only_reclassifies_created_and_unclassified(tmp_path, monkeypatch):
    metashape = types.ModuleType("Metashape")
    metashape.PointClass = types.SimpleNamespace(Created=0, Unclassified=1)
    monkeypatch.setitem(sys.modules, "Metashape", metashape)
    script = Path(__file__).parents[1] / "metashape_scripts" / "stage1_export.py"
    spec = importlib.util.spec_from_file_location("test_stage1_ground_assist", script)
    assert spec and spec.loader
    stage1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage1)

    class Cloud:
        def __init__(self):
            self.calls = []
            self.point_count_by_class = {0: 5, 1: 7, 2: 3}

        def classifyGroundPoints(self, **kwargs):
            self.calls.append(kwargs)
            self.point_count_by_class[2] += self.point_count_by_class[kwargs["source_class"]]

    cloud = Cloud()
    result = stage1.classify_ground_support(
        types.SimpleNamespace(label="Chunk", point_cloud=cloud),
        {"settings": Settings().model_dump(mode="json")},
    )

    assert [call["source_class"] for call in cloud.calls] == [0, 1]
    assert all(call["keep_existing"] for call in cloud.calls)
    assert result["ground_before"] == 3
    assert result["ground_after"] == 15
    assert result["ground_added"] == 12

    class Asset:
        def __init__(self, label, count, meta=None):
            self.label, self.point_count, self.meta = label, count, meta or {}

    original = Asset("Original photo cloud", 100)
    old_result = Asset("SAM3 classified", 100, {"msam3.import_id": "prior"})

    class CloudSelection:
        point_cloud = None
        point_clouds = [old_result, original]

    assert stage1.source_point_cloud(CloudSelection()) is original

    project = tmp_path / "project.psx"
    project.write_bytes(b"project")
    settings = Settings()
    job = {
        "project": str(project),
        "settings": settings.model_dump(mode="json"),
        "stage1_identity": "stale",
    }
    expected = json_hash(
        {
            "schema": 1,
            "project": project_state(project),
            "chunks": settings.metashape.chunks,
            "depth_maps": settings.visibility.mode == "depth_maps",
            "ground_assist": settings.ground_assist.model_dump(),
        }
    )
    assert stage1.stage1_identity(job) == expected


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
    Document.chunks[0].point_cloud = None
    gui.restore_source_point_clouds(Document())
    assert Document.chunks[0].point_cloud is photo


def test_voxels_cross_batch_and_negative(tmp_path):
    points = np.array([[0, 0, 0], [0.01, 0, 0], [1, 0, 0], [-0.01, 0, 0], [1.01, 0, 0]])
    first = voxel_representatives(points, 0.03, tmp_path, batch_size=2, buckets=3)
    assert first.tolist() == [0, 2, 3]


def test_representative_class_mask_reads_only_requested_indices(tmp_path):
    cloud = laspy.create(point_format=7, file_version="1.4")
    cloud.x = np.arange(6)
    cloud.y = cloud.z = np.zeros(6)
    cloud.classification = [1, 2, 3, 2, 1, 2]
    path = tmp_path / "ground.las"
    cloud.write(path)

    mask = representative_class_mask(path, np.array([1, 2, 5]), 2, batch_size=2)

    assert mask.tolist() == [True, False, True]
    assert representative_classifications(path, np.array([1, 2, 5]), batch_size=2).tolist() == [
        2,
        3,
        2,
    ]
    cloud.red = [10, 20, 30, 40, 50, 60]
    cloud.green = [11, 21, 31, 41, 51, 61]
    cloud.blue = [12, 22, 32, 42, 52, 62]
    cloud.write(path)
    from msam3.pointcloud.spatial import representative_attributes

    labels, colors, normals = representative_attributes(
        path, np.array([1, 2, 5]), 2, include_rgb=True
    )
    assert labels.tolist() == [2, 3, 2]
    assert colors.tolist() == [[20, 21, 22], [30, 31, 32], [60, 61, 62]]
    assert normals is None


def test_block_covering_entire_image_not_culled():
    camera = Camera(1, 20, 20, np.eye(4), {"f": 10})
    points = np.array([[-20, -20, 10], [20, 20, 10], [0, 0, 10]], dtype=float)
    blocks = spatial_blocks(points, 100)
    selected = np.concatenate(list(blocks.batches(camera, np.eye(4), 100)))
    assert 2 in selected


def test_block_outside_camera_has_no_batches():
    camera = Camera(1, 20, 20, np.eye(4), {"f": 10})
    blocks = spatial_blocks(np.array([[100, 0, 10]], dtype=float), 1)

    assert list(blocks.batches(camera, np.eye(4), 100)) == []


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


def _metashape_part(path, code, count):
    """A class part as Metashape 2.3.1 writes it: LAS 1.2/PF2 with the full code in one byte."""
    part = laspy.create(point_format=2, file_version="1.2")
    part.x = np.arange(count, dtype=float)
    part.y, part.z = np.zeros(count), np.zeros(count)
    part.points.array["raw_classification"] = code
    part.write(path)
    return {"path": str(path), "code": code, "count": count}


def test_merge_keeps_codes_and_drops_flag_bits_of_truncated_codes(tmp_path):
    classes = Classes.model_validate(
        {
            "classes": [
                {"name": "ground", "code": 2, "prompts": ["g"], "color": "#111111"},
                {"name": "construction", "code": 67, "prompts": ["c"], "color": "#222222"},
                {"name": "pipe", "code": 110, "prompts": ["p"], "color": "#333333"},
            ]
        }
    )
    parts = [
        _metashape_part(tmp_path / "a.las", 2, 2),
        _metashape_part(tmp_path / "b.las", 67, 3),
        _metashape_part(tmp_path / "c.las", 110, 1),
    ]
    raw = laspy.read(parts[1]["path"])
    # The defect being repaired: 67 = 0b1000011 reads as class 3 with the key-point flag.
    assert list(raw.classification) == [3, 3, 3] and all(raw.key_point)
    output = tmp_path / "final.las"
    merge_class_exports(parts, output, classes, 'LOCAL_CS["Local",UNIT["metre",1]]', 2)
    result = laspy.read(output)
    assert str(result.header.version) == "1.4" and result.header.point_format.id == 7
    assert list(result.classification) == [2, 2, 67, 67, 67, 110]
    assert not any(result.key_point) and not any(result.synthetic) and not any(result.withheld)
    info = describe_output(output, parts)
    assert (info["las_version"], info["point_format"], info["point_count"]) == ("1.4", 7, 6)
    assert info["path"] == str(output.absolute()) and not info["compressed"]
    with pytest.raises(RuntimeError, match="class 64 was not written"):
        describe_output(output, [parts[0], {**parts[1], "code": 64}, parts[2]])


def test_final_output_refuses_las12_and_unknown_suffix(tmp_path):
    legacy = tmp_path / "legacy.las"
    _metashape_part(legacy, 67, 1)
    with pytest.raises(RuntimeError, match="LAS 1.2 point format 2"):
        describe_output(legacy)
    classes = Classes.model_validate(
        {"classes": [{"name": "c", "code": 67, "prompts": ["c"], "color": "#222222"}]}
    )
    part = _metashape_part(tmp_path / "p.las", 67, 1)
    with pytest.raises(ValueError, match=".las or .laz"):
        merge_class_exports([part], tmp_path / "final.txt", classes, "", 1)


def test_laz_output_is_really_compressed_or_refused(tmp_path):
    classes = Classes.model_validate(
        {"classes": [{"name": "c", "code": 67, "prompts": ["c"], "color": "#222222"}]}
    )
    part = _metashape_part(tmp_path / "p.las", 67, 2)
    output = tmp_path / "final.laz"
    if not laspy.LazBackend.detect_available():
        with pytest.raises(RuntimeError, match="lazrs"):
            merge_class_exports([part], output, classes, 'LOCAL_CS["L",UNIT["metre",1]]', 1)
        assert not output.exists()
        return
    merge_class_exports([part], output, classes, 'LOCAL_CS["L",UNIT["metre",1]]', 1)
    with laspy.open(output) as reader:
        assert reader.header.are_points_compressed
    assert list(laspy.read(output).classification) == [67, 67]
