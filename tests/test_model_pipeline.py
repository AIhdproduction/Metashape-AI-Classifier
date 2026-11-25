import json

import laspy
import numpy as np
import pytest
from synthetic import (
    SAM_NAMES,
    FakeFlurSeg,
    FakeSam,
    projections,
    site_cameras,
    site_points,
    write_flurseg_files,
    write_weights,
)

from msam3 import backends, user_classes
from msam3.cache import atomic_json
from msam3.config import load_classes, load_settings
from msam3.driver import _execute, export_identity
from msam3.pipeline import process_chunk
from msam3.segmentation.mask_filter import interior_labels
from msam3.segmentation.model_files import FLURSEG_LABELS


@pytest.fixture
def site(tmp_path):
    points, truth = site_points()
    chunk = tmp_path / "chunk-0"
    chunk.mkdir()
    las = laspy.create(point_format=7, file_version="1.4")
    las.header.offsets, las.header.scales = np.zeros(3), np.full(3, 0.0001)
    las.x, las.y, las.z = points.T
    las.red = np.arange(len(points)) % 60000
    las.classification = np.ones(len(points), np.uint8)
    las.write(chunk / "dense.las")
    cameras = site_cameras(chunk)
    atomic_json(chunk / "cameras.json", cameras)
    atomic_json(
        chunk / "chunk.json",
        {
            "key": 0,
            "label": "Süd",
            "point_count": len(points),
            "transform": np.eye(4).tolist(),
            "source_signature": "source",
        },
    )
    # The stored coordinates are quantised, so the samples come from the LAS itself.
    stored = laspy.read(chunk / "dense.las")
    samples = np.column_stack((stored.x, stored.y, stored.z))[::53]
    atomic_json(
        chunk / "validation_samples.json",
        {
            "samples": [{"internal": p.tolist(), "world": p.tolist()} for p in samples],
            "projections": projections(samples, cameras),
        },
    )
    settings = load_settings()
    settings.visibility.mode = "zbuffer"
    settings.visibility.zbuffer_scale = 1
    settings.visibility.splat_px = 0
    settings.gpu.device = "cpu"
    settings.voting.batch_points = 977
    settings.voting.voxel_buckets = 8
    settings.paths.flurseg_model = write_flurseg_files(tmp_path / "flurseg")
    settings.paths.model = tmp_path / "sam3.pt"
    settings.paths.model.write_bytes(b"sam weights")
    return tmp_path, chunk, settings, truth


def expected_codes(backend, truth, classes):
    code = {item.name: item.code for item in classes.classes}
    names = FLURSEG_LABELS
    if backend == "sam3":
        names = [SAM_NAMES[name] for name in names]
    return np.array([code[name] for name in names])[truth]


@pytest.mark.parametrize("backend", ["flurseg", "sam3"])
def test_end_to_end_with_both_class_sets(site, backend):
    _, chunk, settings, truth = site
    settings.segmentation.backend = backend
    # The synthetic site is one flat plane: every class lies on the terrain, so the terrain
    # model would rightly call its low vegetation ground. This test is about the class sets.
    settings.postprocess.dtm_ground.enabled = False
    model = backends.MODELS[backend]
    classes = model.classes()
    fake = FakeFlurSeg() if backend == "flurseg" else FakeSam()

    result = process_chunk(chunk, settings, classes, "hash", lambda: fake, device="cpu")

    written = np.asarray(laspy.read(result["classified_las"]).classification)
    expected = expected_codes(backend, truth, classes)
    assert np.mean(written == expected) > 0.8
    assert result["report"]["model"]["backend"] == backend
    legend = json.loads((chunk / "class_legend.json").read_text(encoding="utf-8"))
    assert {row["model"] for row in legend} == {model.title}
    header = (chunk / "class_legend.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "name;code;color;prompts"
    if backend == "flurseg":
        # Markings, kerbs and covers lie in the road and must survive the whole pipeline.
        for code in (82, 74, 75):
            assert np.mean(written[expected == code] == code) > 0.7, code
        assert {64, 2, 71, 3, 6, 91, 67}.issubset(set(written))


def test_without_the_erosion_exemption_a_thin_marking_gets_no_vote():
    labels = np.full((15, 15), 2, np.uint8)
    labels[:, 7:9] = 9
    confidence = np.random.default_rng(1).uniform(0.7, 0.99, labels.shape).astype(np.float32)
    assert not np.any(interior_labels(labels, confidence, 2, 0.4) == 9)
    kept = interior_labels(labels, confidence, 2, 0.4, no_erosion=[9])
    assert np.all(kept[:, 7:9] == 9)
    assert np.all(kept[2:-2, 6] == 255)  # The road still keeps away from the marking.
    confidence[:, 7] = 0.2
    assert np.all(interior_labels(labels, confidence, 2, 0.4, no_erosion=[9])[:, 7] == 255)


@pytest.mark.parametrize("first,second", [("flurseg", "sam3"), ("sam3", "flurseg")])
def test_switching_the_model_never_reuses_labels(site, first, second):
    _, chunk, settings, _ = site
    fakes = {"flurseg": FakeFlurSeg(), "sam3": FakeSam()}

    def run(backend, voting_only=False):
        settings.segmentation.backend = backend
        classes = backends.MODELS[backend].classes()
        # Even one equal weights hash must not let the models share their images.
        return process_chunk(
            chunk,
            settings,
            classes,
            "same-hash",
            lambda: fakes[backend],
            voting_only=voting_only,
            device="cpu",
        )

    run(first)
    with pytest.raises(RuntimeError, match="stale"):
        run(second, voting_only=True)
    run(second)
    assert fakes[second].calls > 0
    calls = fakes[first].calls
    assert run(first, voting_only=True)["report"]["model"]["backend"] == first
    assert fakes[first].calls == calls


def test_segmentation_cache_follows_weights_classes_and_options(site):
    _, chunk, settings, _ = site
    classes = backends.MODELS["flurseg"].classes()
    fake = FakeFlurSeg()
    process_chunk(chunk, settings, classes, "weights-a", lambda: fake, device="cpu")
    with pytest.raises(RuntimeError, match="stale"):
        process_chunk(chunk, settings, classes, "weights-b", voting_only=True, device="cpu")
    recoded = classes.model_copy(deep=True)
    recoded.classes[7].code = 68
    with pytest.raises(RuntimeError, match="stale"):
        process_chunk(chunk, settings, recoded, "weights-a", voting_only=True, device="cpu")
    changed = settings.model_copy(deep=True)
    changed.flurseg.resolution = "accurate"
    with pytest.raises(RuntimeError, match="stale"):
        process_chunk(chunk, changed, classes, "weights-a", voting_only=True, device="cpu")
    batch = settings.model_copy(deep=True)
    batch.flurseg.window_batch_size = 1
    process_chunk(chunk, batch, classes, "weights-a", voting_only=True, device="cpu")
    assert fake.calls == 3


@pytest.fixture
def driver(site, monkeypatch):
    root, chunk, settings, _ = site
    project = root / "site.psx"
    project.write_bytes(b"project")
    work = root / "work"
    work.mkdir()
    marker = {
        "schema": 1,
        "chunks": [{"key": 0, "label": "Süd", "directory": str(chunk)}],
        "stage1_identity": export_identity(project, settings),
    }
    atomic_json(work / "stage1_done.json", marker)
    fakes = {"flurseg": FakeFlurSeg(), "sam3": FakeSam()}
    for key, model in backends.MODELS.items():
        monkeypatch.setattr(model, "create", lambda settings, key=key: fakes[key])
    imported = []

    def stage_runner(stage, job_path, work, config):
        assert stage == 3, "the export is shared and must not run again"
        job = json.loads(job_path.read_text(encoding="utf-8"))
        imported.append(job)
        parts = []
        source = laspy.read(job["stage3_inputs"][0]["classified_las"])
        codes = np.asarray(source.classification)
        for code in np.unique(codes):
            path = work / f"part-{code}.las"
            las = laspy.LasData(source.header, source.points[codes == code].copy())
            las.write(path)
            parts.append(
                {"code": int(code), "count": int((codes == code).sum()), "path": str(path)}
            )
        output = {
            "chunk_key": 0,
            "asset_key": len(imported),
            "parts": parts,
            "crs_wkt": 'LOCAL_CS["Local",UNIT["metre",1]]',
            "directory": str(chunk),
            "output": str(root / "output" / "site.las"),
        }
        return {"chunks": [output]}

    def execute(action, backend, classes=None, change=None):
        run = settings.model_copy(deep=True)
        run.segmentation.backend = backend
        if change:
            change(run)
        classes = classes or backends.MODELS[backend].classes()
        output = root / "output"
        return _execute(
            action, project, work, output, run, classes, False, stage_runner=stage_runner
        )

    return execute, settings, imported


@pytest.mark.parametrize("first,second", [("flurseg", "sam3"), ("sam3", "flurseg")])
def test_stage3_never_imports_the_other_models_result(driver, first, second):
    execute, _, imported = driver
    execute("stage2", first)
    with pytest.raises(ValueError, match="changed since stage"):
        execute("stage3", second)
    assert not imported
    execute("stage3", first)
    job = imported[0]
    assert job["model"]["backend"] == first
    assert job["settings"]["metashape"]["imported_label"] == backends.MODELS[first].asset_label


@pytest.mark.parametrize("backend", ["flurseg", "sam3"])
def test_stage3_refuses_changed_weights(driver, backend):
    execute, settings, imported = driver
    execute("stage2", backend)
    if backend == "flurseg":
        write_weights(settings.paths.flurseg_model / "model.safetensors", "new")
    else:
        settings.paths.model.write_bytes(b"new sam weights")
    with pytest.raises(ValueError, match="weights or settings changed"):
        execute("stage3", backend)
    assert not imported


def test_stage3_refuses_changed_settings(driver):
    execute, _, imported = driver
    execute("stage2", "flurseg")
    with pytest.raises(ValueError, match="settings changed"):
        execute("stage3", "flurseg", change=lambda s: setattr(s.voting, "min_share", 0.6))
    assert not imported


def test_stage3_refuses_changed_classes(driver):
    execute, _, imported = driver
    execute("stage2", "sam3")
    standard = load_classes()
    edits = user_classes.save_class(
        standard, user_classes.UserClasses(), "manhole", ["manhole"], "#112233"
    )
    with pytest.raises(ValueError, match="Classes changed"):
        execute("stage3", "sam3", classes=user_classes.apply(standard, edits))
    assert not imported


def test_both_models_share_one_export(driver):
    execute, _, imported = driver
    execute("run", "flurseg")
    execute("run", "sam3")  # The fixture fails if stage 1 is started again.
    assert [job["model"]["backend"] for job in imported] == ["flurseg", "sam3"]


def test_identical_labels_of_other_weights_become_their_own_asset(driver):
    execute, settings, imported = driver
    execute("run", "flurseg")
    write_weights(settings.paths.flurseg_model / "model.safetensors", "new")
    execute("run", "flurseg")
    first, second = (job["stage3_inputs"][0]["import_id"] for job in imported)
    assert first != second
