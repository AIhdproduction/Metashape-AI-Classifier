"""The selectable segmentation models: the only place that knows what differs between them.

The Metashape worker, the CLI driver, the pipeline and the UI ask this registry. It does not
import torch, so that the UI process stays off the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import yaml

from msam3 import user_classes
from msam3.cache import file_hash, json_hash
from msam3.config import (
    REPO_ROOT,
    Classes,
    ClassPostprocessRule,
    Settings,
    load_classes,
    merge_settings,
)
from msam3.segmentation import model_files

# Own SAM 3 classes get the cleanup of a small object unless default.yml has a rule for them.
CUSTOM_CLASS_RULE = ClassPostprocessRule(
    min_same_neighbors=2, fill_neighbors=5, fill_share=0.7, max_replaced_confidence=0.45
)


@dataclass(frozen=True)
class Weights:
    """``model_hash`` identifies everything on disk that influences the labels."""

    model_hash: str
    info: dict


@dataclass(frozen=True)
class ModelStatus:
    state: str  # ready | missing | invalid | runtime
    path: Path
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.state == "ready"


class Model:
    key: str
    title: str
    asset_label: str
    version: str
    package: str
    url: str
    editable_classes: bool

    def path(self, settings: Settings) -> Path:
        raise NotImplementedError

    def require_files(self, settings: Settings) -> None:
        raise NotImplementedError

    def require_runtime(self) -> None:
        if find_spec(self.package) is None:
            raise RuntimeError(
                f"The Python package '{self.package}' for {self.title} is missing. "
                "Run install.bat again."
            )

    def require_gpu(self, settings: Settings) -> None:
        """Fail before the long export. This loads no model; never call it in the UI process."""
        raise NotImplementedError

    def status(self, settings: Settings) -> ModelStatus:
        path = self.path(settings)
        try:
            self.require_files(settings)
        except FileNotFoundError as error:
            return ModelStatus("missing", path, str(error))
        except (OSError, ValueError) as error:
            return ModelStatus("invalid", path, str(error))
        try:
            self.require_runtime()
        except RuntimeError as error:
            return ModelStatus("runtime", path, str(error))
        return ModelStatus("ready", path)

    def fingerprint(self, settings: Settings) -> Weights:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"backend": self.key, "title": self.title, "version": self.version}

    def classes(self, path: Path | None = None, snapshot: dict | None = None) -> Classes:
        raise NotImplementedError

    def apply_profile(self, settings: Settings, classes: Classes) -> Settings:
        raise NotImplementedError

    def options(self, settings: Settings) -> dict:
        """Only the options that influence this model's per-image labels."""
        raise NotImplementedError

    def segmentation_identity(self, settings: Settings, classes: Classes) -> str:
        return json_hash(
            {
                "backend": self.key,
                "version": self.version,
                "options": self.options(settings),
                "cache": settings.cache.model_dump(),
                "classes": classes.model_dump(),
            }
        )

    def run_identity(self, settings: Settings, classes: Classes, model_hash: str) -> dict:
        """Stage 2 stores this; stage 3 refuses to import a result with another one."""
        effective = self.apply_profile(settings, classes)
        return {
            "backend": self.key,
            "model_hash": model_hash,
            "segmentation": self.segmentation_identity(settings, classes),
            "settings": json_hash(
                {
                    "visibility": effective.visibility.model_dump(),
                    "voting": effective.voting.model_dump(
                        exclude={"batch_points", "max_ram_gb", "voxel_buckets"}
                    ),
                    "ground_assist": effective.ground_assist.model_dump(),
                    "vegetation_assist": effective.vegetation_assist.model_dump(),
                    "smoothing": effective.smoothing.model_dump(),
                    "postprocess": effective.postprocess.model_dump(),
                }
            ),
        }

    def imported_label(self, settings: Settings) -> str:
        label = settings.metashape.imported_label
        # The two model labels were never a choice: older default.yml copies contain one of them.
        if not label or label in {model.asset_label for model in MODELS.values()}:
            return self.asset_label
        return label

    def job_settings(self, settings: Settings) -> dict:
        """Settings for job.json. The embedded scripts cannot resolve the label themselves."""
        data = settings.model_dump(mode="json")
        data["metashape"]["imported_label"] = self.imported_label(settings)
        return data

    def create(self, settings: Settings):
        raise NotImplementedError

    def segment(self, backend, image, classes: Classes, settings: Settings, scratch: Path):
        raise NotImplementedError


class FlurSeg(Model):
    key = "flurseg"
    title = "FlurSeg"
    asset_label = "FlurSeg classified"
    version = model_files.FLURSEG_VERSION
    package = "transformers"
    url = model_files.FLURSEG_URL
    editable_classes = False

    def path(self, settings):
        return settings.paths.flurseg_model

    def require_files(self, settings):
        model_files.require_flurseg(self.path(settings))

    def require_gpu(self, settings):
        from msam3.segmentation.flurseg_backend import check_cuda

        check_cuda(settings.gpu)

    def fingerprint(self, settings):
        hashes = {name: file_hash(self.path(settings) / name) for name in model_files.FLURSEG_FILES}
        sha256 = hashes["model.safetensors"]
        known = sha256 == model_files.FLURSEG_V1_SHA256
        info = {
            **self.describe(),
            "weights_sha256": sha256,
            "weights": "v1" if known else "nicht v1 / unbekannte Gewichte",
        }
        return Weights(json_hash(hashes), info)

    def classes(self, path=None, snapshot=None):
        classes = load_classes(path or REPO_ROOT / "config/classes.flurseg.yml")
        if [item.name for item in classes.classes] != list(model_files.FLURSEG_LABELS):
            raise ValueError(
                "FlurSeg has eleven fixed classes in a fixed order. "
                "Own classes need SAM 3 (--model sam3)."
            )
        return classes

    def apply_profile(self, settings, classes):
        path = REPO_ROOT / "config/profile.flurseg.yml"
        data = settings.model_dump()
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
        # Rules describe a whole class set: never mix them with the SAM 3 ones.
        merge_settings(data, profile, replace=("class_rules",))
        data["visibility"]["min_mask_confidence"] = settings.flurseg.min_mask_confidence
        return Settings.model_validate(data)

    def options(self, settings):
        options = settings.flurseg
        # The window batch is left out: it changes speed and memory, not the labels.
        return {
            "resolution": options.resolution,
            "precision": options.precision,
            "background_threshold": options.background_threshold,
        }

    def create(self, settings):
        from msam3.segmentation.flurseg_backend import FlurSegBackend

        return FlurSegBackend.load(self.path(settings), settings.gpu, settings.flurseg)

    def segment(self, backend, image, classes, settings, scratch):
        return backend.segment(image)


class Sam3(Model):
    key = "sam3"
    title = "SAM 3"
    asset_label = "SAM3 classified"
    version = model_files.SAM_VERSION
    package = "sam3"
    url = model_files.SAM_URL
    editable_classes = True

    def path(self, settings):
        return settings.paths.model

    def require_files(self, settings):
        model_files.require_checkpoint(self.path(settings))

    def require_gpu(self, settings):
        from msam3.segmentation.sam3_backend import check_gpu

        check_gpu(settings.gpu)

    def fingerprint(self, settings):
        sha256 = file_hash(self.path(settings))
        return Weights(sha256, {**self.describe(), "weights_sha256": sha256})

    def classes(self, path=None, snapshot=None):
        if snapshot is not None:
            classes = Classes.model_validate(snapshot)
        elif path is not None:
            classes = load_classes(path)
        else:
            classes = user_classes.apply(load_classes(), user_classes.read())
        if classes.fixed:
            raise ValueError("SAM 3 needs classes with text prompts, not a fixed class list")
        return classes

    def apply_profile(self, settings, classes):
        rules = settings.postprocess.class_rules
        if not rules:
            # No rules at all means that the neighbourhood filter is not configured.
            return settings
        standard = {item.name for item in load_classes().classes}
        own = [c.name for c in classes.classes if c.name not in standard and c.name not in rules]
        if not own:
            return settings
        result = settings.model_copy(deep=True)
        for name in own:
            result.postprocess.class_rules[name] = CUSTOM_CLASS_RULE.model_copy()
        return result

    def options(self, settings):
        return {
            "sam": settings.sam.model_dump(),
            "dtype": settings.gpu.dtype,
            "revision": model_files.SAM_REVISION,
        }

    def create(self, settings):
        from msam3.segmentation.sam3_backend import SAM3Backend

        return SAM3Backend(self.path(settings), settings.gpu, settings.sam)

    def segment(self, backend, image, classes, settings, scratch):
        from msam3.segmentation.tiling import segment_image

        return segment_image(image, backend, classes, settings.sam, scratch)


# The order is the order in the UI; the first one is the default.
MODELS: dict[str, Model] = {model.key: model for model in (FlurSeg(), Sam3())}


def selected(settings: Settings) -> Model:
    return MODELS[settings.segmentation.backend]
