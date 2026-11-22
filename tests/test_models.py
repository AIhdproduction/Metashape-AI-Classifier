import json
import types
from contextlib import nullcontext
from typing import get_args

import pytest
import torch
import yaml
from synthetic import write_flurseg_files, write_weights

from msam3 import backends, driver, gui_worker
from msam3.cli import main, parser
from msam3.config import REPO_ROOT, BackendName, Settings, load_classes, load_settings
from msam3.driver import execute, export_identity
from msam3.segmentation.model_files import FLURSEG_LABELS

FLURSEG, SAM3 = backends.MODELS["flurseg"], backends.MODELS["sam3"]


def test_flurseg_is_the_default_and_the_registry_matches_the_schema():
    assert Settings().segmentation.backend == "flurseg"
    settings = load_settings()
    assert backends.selected(settings) is FLURSEG
    assert tuple(backends.MODELS) == get_args(BackendName)
    assert settings.paths.flurseg_model.is_absolute()
    assert settings.paths.flurseg_model.parts[-3:] == ("models", "flurseg", "v1")


def test_config_from_before_the_model_choice_still_loads(tmp_path):
    old = yaml.safe_load(
        (REPO_ROOT / "config/default.yml").read_text(encoding="utf-8")
    )
    del old["segmentation"], old["flurseg"], old["paths"]["flurseg_model"]
    old["metashape"]["imported_label"] = "SAM3 classified"
    path = tmp_path / "old.yml"
    path.write_text(yaml.safe_dump(old), encoding="utf-8")

    settings = load_settings(path)

    assert settings.segmentation.backend == "flurseg"
    assert settings.paths.flurseg_model.is_absolute()
    # The old default label was never a choice, so it follows the model.
    assert FLURSEG.imported_label(settings) == "FlurSeg classified"
    assert SAM3.imported_label(settings) == "SAM3 classified"


def test_explicit_label_wins_for_both_models():
    settings = Settings()
    settings.metashape.imported_label = "Baustelle klassifiziert"
    for model in backends.MODELS.values():
        data = model.job_settings(settings)
        assert data["metashape"]["imported_label"] == "Baustelle klassifiziert"
    assert FLURSEG.job_settings(Settings())["metashape"]["imported_label"] == "FlurSeg classified"


def test_flurseg_classes_codes_colours_and_order():
    classes = FLURSEG.classes()
    assert classes.fixed and classes.unknown_code == 1
    assert [(item.name, item.code, item.color) for item in classes.classes] == [
        ("other", 64, "#9E9E9E"),
        ("ground", 2, "#A0522D"),
        ("paved_surface", 71, "#5C6BC0"),
        ("low_vegetation", 3, "#9CCC65"),
        ("tree", 5, "#1B5E20"),
        ("building", 6, "#E53935"),
        ("vehicle", 91, "#FFEB3B"),
        ("construction", 67, "#FF9800"),
        ("curb", 74, "#00BCD4"),
        ("road_marking", 82, "#FFFFFF"),
        ("utility_cover", 75, "#E040FB"),
    ]
    assert all(not item.prompts and item.code < 128 for item in classes.classes)
    with pytest.raises(ValueError, match="fixed classes"):
        FLURSEG.classes(REPO_ROOT / "config/classes.yml")


def test_flurseg_profile_covers_every_class_and_only_existing_ones():
    classes = FLURSEG.classes()
    settings = FLURSEG.apply_profile(load_settings(), classes)
    names = {item.name for item in classes.classes}
    post = settings.postprocess
    assert set(post.class_rules) == names
    used = {
        *settings.ground_assist.ground_like_classes,
        settings.ground_assist.fallback_class,
        settings.vegetation_assist.low_vegetation_class,
        settings.vegetation_assist.high_vegetation_class,
        post.road.road_class,
        *post.road.surface_replace_classes,
        post.building.building_class,
        post.building.ground_class,
        post.ground.ground_class,
        post.vegetation.low_vegetation_class,
        post.vegetation.high_vegetation_class,
        *post.planar_unknown_fill.class_names,
        *post.protected_classes,
        *post.cluster.min_same_overrides,
        *settings.smoothing.protected_classes,
        *settings.visibility.no_erosion_classes,
    }
    assert used <= names
    assert settings.ground_assist.ground_like_classes == ["ground", "paved_surface"]
    assert post.road.road_class == "paved_surface"
    assert post.vegetation.terrain_class_codes == [2, 71, 82, 75, 74]
    assert settings.visibility.min_mask_confidence == settings.flurseg.min_mask_confidence


def test_profile_keeps_the_users_switches_and_thresholds():
    settings = load_settings()
    settings.ground_assist.enabled = False
    settings.postprocess.road.enabled = False
    settings.postprocess.road.max_slope_degrees = 33
    settings.voting.min_share = 0.55
    result = FLURSEG.apply_profile(settings, FLURSEG.classes())
    assert not result.ground_assist.enabled and not result.postprocess.road.enabled
    assert result.postprocess.road.max_slope_degrees == 33
    assert result.voting.min_share == 0.55


def test_sam3_without_class_edits_keeps_todays_settings():
    settings = load_settings()
    settings.segmentation.backend = "sam3"
    classes = SAM3.classes()
    assert classes == load_classes()
    assert SAM3.apply_profile(settings, classes) == settings
    assert settings.visibility.min_mask_confidence == 0.5
    assert not settings.visibility.no_erosion_classes
    assert not settings.postprocess.protected_classes
    assert not settings.postprocess.cluster.min_same_overrides
    assert not settings.smoothing.protected_classes


def test_export_identity_is_shared_between_the_models(tmp_path):
    project = tmp_path / "site.psx"
    project.write_bytes(b"project")
    flurseg, sam = load_settings(), load_settings()
    sam.segmentation.backend = "sam3"
    assert export_identity(project, flurseg) == export_identity(project, sam)


def test_requirements_name_the_folder_and_the_command(tmp_path):
    settings = Settings()
    settings.paths.flurseg_model = tmp_path / "absent"
    with pytest.raises(FileNotFoundError, match="hf.exe download AIhdproduction/Flurseg"):
        FLURSEG.require_files(settings)
    assert FLURSEG.status(settings).state == "missing"
    with pytest.raises(FileNotFoundError, match="FlurSeg model files missing"):
        execute("run", tmp_path / "no.psx", settings, FLURSEG.classes())


def test_incomplete_or_foreign_flurseg_files_are_rejected(tmp_path):
    settings = Settings()
    directory = write_flurseg_files(tmp_path / "model")
    settings.paths.flurseg_model = directory
    FLURSEG.require_files(settings)
    weights = directory / "model.safetensors"
    weights.write_bytes(weights.read_bytes()[:-3])
    with pytest.raises(ValueError, match="incomplete"):
        FLURSEG.require_files(settings)
    weights.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    assert FLURSEG.status(settings).state == "invalid"
    swapped = list(FLURSEG_LABELS)
    swapped[8], swapped[9] = swapped[9], swapped[8]
    write_flurseg_files(directory, labels=swapped)
    with pytest.raises(ValueError, match="class list or order"):
        FLURSEG.require_files(settings)


def test_unknown_weights_are_allowed_and_named(tmp_path):
    settings = Settings()
    settings.paths.flurseg_model = write_flurseg_files(tmp_path / "model", "v2")
    weights = FLURSEG.fingerprint(settings)
    assert weights.info["weights"] == "nicht v1 / unbekannte Gewichte"
    write_weights(settings.paths.flurseg_model / "model.safetensors", "other")
    assert FLURSEG.fingerprint(settings).model_hash != weights.model_hash


@pytest.mark.parametrize("wanted", ["flurseg", "sam3"])
def test_prepare_checks_only_the_selected_model(tmp_path, monkeypatch, wanted):
    settings = load_settings()
    settings.paths.model = tmp_path / "none.pt"
    settings.paths.flurseg_model = tmp_path / "none"
    settings.paths.work_dir = tmp_path / "work"
    if wanted == "flurseg":
        settings.paths.flurseg_model = write_flurseg_files(tmp_path / "model")
    else:
        settings.paths.model = tmp_path / "sam3.pt"
        settings.paths.model.write_bytes(b"sam")
    monkeypatch.setattr(gui_worker, "load_settings", lambda: settings.model_copy(deep=True))
    monkeypatch.setattr(backends.Model, "require_runtime", lambda self: None)
    checked = []
    for key, model in backends.MODELS.items():
        monkeypatch.setattr(model, "require_gpu", lambda settings, key=key: checked.append(key))
    project = tmp_path / "site.psx"
    project.write_bytes(b"project")
    session = tmp_path / "session"
    session.mkdir()
    (session / "session.json").write_text(json.dumps({"project": str(project)}), encoding="utf-8")

    def choose(backend):
        options = {"settings_override": {"segmentation": {"backend": backend}}}
        (session / "options.json").write_text(json.dumps(options), encoding="utf-8")

    choose(wanted)
    gui_worker.prepare(session)
    job = json.loads((session / "job.json").read_text(encoding="utf-8"))
    assert job["model"]["backend"] == wanted
    assert checked == [wanted]
    assert len(job["classes"]["classes"]) == (11 if wanted == "flurseg" else 16)
    assert job["settings"]["metashape"]["imported_label"] == backends.MODELS[wanted].asset_label
    choose("sam3" if wanted == "flurseg" else "flurseg")
    with pytest.raises(FileNotFoundError):
        gui_worker.prepare(session)


def test_cli_model_choice(tmp_path, capsys):
    assert main(["classes"]) == 0
    output = capsys.readouterr().out
    assert "FlurSeg" in output and "paved_surface" in output and "sidewalk" not in output
    assert main(["classes", "--model", "sam3"]) == 0
    assert "SAM 3" in capsys.readouterr().out
    config = tmp_path / "sam.yml"
    config.write_text("segmentation: {backend: sam3}\n", encoding="utf-8")
    assert main(["classes", "--config", str(config)]) == 0
    assert "sidewalk" in capsys.readouterr().out
    for command in ("run", "stage1", "stage2", "stage3", "validate"):
        arguments = parser().parse_args([command, "--project", "x.psx", "--model", "sam3"])
        assert arguments.model == "sam3"


def test_cli_refuses_own_classes_for_flurseg(caplog):
    assert main(["classes", "--classes", str(REPO_ROOT / "config/classes.yml")]) == 1
    assert "eleven fixed classes" in caplog.text


@pytest.fixture
def gpu_job(tmp_path, monkeypatch):
    settings = load_settings()
    settings.paths.flurseg_model = write_flurseg_files(tmp_path / "model")
    settings.paths.model = tmp_path / "sam3.pt"
    settings.paths.model.write_bytes(b"sam")
    settings.paths.work_dir = tmp_path / "work"
    settings.paths.output_dir = tmp_path / "output"
    monkeypatch.setattr(backends.Model, "require_runtime", lambda self: None)

    def no_model(settings):
        pytest.fail("the GPU check loaded a model")

    for model in backends.MODELS.values():
        monkeypatch.setattr(model, "create", no_model)
    project = tmp_path / "site.psx"
    project.write_bytes(b"project")
    return settings, project


@pytest.mark.parametrize("backend", ["flurseg", "sam3"])
def test_prepare_fails_without_cuda(gpu_job, tmp_path, monkeypatch, backend):
    settings, project = gpu_job
    settings.segmentation.backend = backend
    monkeypatch.setattr(gui_worker, "load_settings", lambda: settings.model_copy(deep=True))
    session = tmp_path / "session"
    session.mkdir()
    (session / "session.json").write_text(json.dumps({"project": str(project)}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        gui_worker.prepare(session)
    assert not (session / "job.json").exists()


@pytest.mark.parametrize("backend", ["flurseg", "sam3"])
@pytest.mark.parametrize("action", ["run", "stage2"])
def test_cli_fails_without_cuda_before_metashape(gpu_job, monkeypatch, backend, action):
    settings, project = gpu_job
    settings.segmentation.backend = backend
    monkeypatch.setattr(driver, "_execute", lambda *a, **k: pytest.fail("the export could start"))
    with pytest.raises(RuntimeError, match="requires CUDA"):
        execute(action, project, settings, backends.MODELS[backend].classes())


@pytest.mark.parametrize(
    "action,voting_only",
    [("stage1", False), ("validate", False), ("stage3", False), ("stage2", True)],
)
def test_commands_without_inference_need_no_cuda(gpu_job, monkeypatch, action, voting_only):
    settings, project = gpu_job
    monkeypatch.setattr(driver, "_execute", lambda *a, **k: {"action": a[0]})
    classes = FLURSEG.classes()
    assert execute(action, project, settings, classes, voting_only) == {"action": action}


@pytest.mark.parametrize("backend", ["flurseg", "sam3"])
def test_gpu_check_passes_on_an_injected_gpu(gpu_job, monkeypatch, backend):
    settings, _ = gpu_job
    properties = types.SimpleNamespace(name="Fake GPU", total_memory=24 * 1024**3)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: properties)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    backends.MODELS[backend].require_gpu(settings)
