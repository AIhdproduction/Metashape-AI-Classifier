"""Validated settings and deterministic LAS class allocation."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Code = Annotated[int, Field(ge=0, le=255, strict=True)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class SemanticClass(StrictModel):
    name: Annotated[str, Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")]
    code: Code | None = None
    prompts: Annotated[list[Annotated[str, Field(min_length=1)]], Field(min_length=1)]

    @model_validator(mode="after")
    def check_prompts(self):
        if any(not p.strip() for p in self.prompts):
            raise ValueError(f"Empty prompt in class {self.name}")
        return self


class Classes(StrictModel):
    unknown_code: Code = 1
    custom_code_start: Code = 64
    classes: Annotated[list[SemanticClass], Field(min_length=1, max_length=255)]

    @model_validator(mode="after")
    def resolve(self):
        names = [c.name for c in self.classes]
        explicit = [c.code for c in self.classes if c.code is not None]
        if len(names) != len(set(names)):
            raise ValueError("Class names must be unique")
        if len(explicit) != len(set(explicit)):
            raise ValueError("Class codes must be unique")
        if self.unknown_code in explicit:
            raise ValueError("A semantic class cannot use unknown_code")
        used = set(explicit) | {self.unknown_code}
        candidate = self.custom_code_start
        for cls in self.classes:
            if cls.code is None:
                while candidate in used:
                    candidate += 1
                if candidate > 255:
                    raise ValueError("No free LAS class code remains in 0..255")
                cls.code = candidate
                used.add(candidate)
                candidate += 1
        return self


class MetashapeConfig(StrictModel):
    exe: str = "C:/Program Files/Agisoft/Metashape Pro/metashape.exe"
    chunks: list[str] = Field(default_factory=list)
    imported_label: str = "SAM3 classified"
    make_default: bool = False
    offscreen: Literal["auto", "always", "never"] = "auto"


class PathsConfig(StrictModel):
    model: Path = Path("models/sam3.pt")
    work_dir: Path = Path("work")
    output_dir: Path = Path("output")
    image_path_remap: dict[str, str] = Field(default_factory=dict)


class GPUConfig(StrictModel):
    device: str = "cuda:0"
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    min_vram_gb: Positive = 16


class SAMConfig(StrictModel):
    input_size: Literal[1008] = 1008
    prompt_batch_size: Annotated[int, Field(ge=1, le=16)] = 4
    image_scale: Annotated[float, Field(gt=0, le=1)] = 1.0
    tile_overlap: Annotated[float, Field(ge=0, lt=1)] = 0.2
    instance_conf_threshold: Probability = 0.3
    background_threshold: Probability = 0.3


class CacheConfig(StrictModel):
    scale: Annotated[float, Field(gt=0, le=1)] = 0.5


class ValidationConfig(StrictModel):
    max_reprojection_px: Positive = 0.5
    max_tie_to_dense_m: Positive = 0.10


class VisibilityConfig(StrictModel):
    mode: Literal["zbuffer", "depth_maps"] = "zbuffer"
    zbuffer_scale: Annotated[float, Field(gt=0, le=1)] = 0.25
    splat_px: Annotated[int, Field(ge=0, le=16)] = 2
    rel_tolerance: Probability = 0.02
    abs_tolerance_m: Annotated[float, Field(ge=0)] = 0.05


class VotingConfig(StrictModel):
    voxel_size_m: Positive = 0.03
    block_size_m: Positive = 10.0
    min_views: Annotated[int, Field(ge=1)] = 3
    min_share: Probability = 0.4
    use_incidence_weight: bool = False
    max_ram_gb: Positive = 16
    batch_points: Annotated[int, Field(ge=1)] = 500_000
    voxel_buckets: Annotated[int, Field(ge=1, le=4096)] = 256


class SmoothingConfig(StrictModel):
    enabled: bool = False
    k: Annotated[int, Field(ge=1)] = 8
    max_share_to_change: Probability = 0.6


class ClassPostprocessRule(StrictModel):
    """Neighbourhood requirements for one semantic class after image voting."""

    min_same_neighbors: Annotated[int, Field(ge=0)] = 3
    fill_neighbors: Annotated[int, Field(ge=1)] = 8
    fill_share: Probability = 0.75
    max_replaced_confidence: Probability = 0.55


class BuildingPostprocessConfig(StrictModel):
    enabled: bool = True
    ground_class: str = "ground"
    building_class: str = "building"
    horizontal_radius_m: Positive = 0.75
    min_vertical_drop_m: Annotated[float, Field(ge=0)] = 1.0
    min_building_neighbors: Annotated[int, Field(ge=1)] = 4


class RoadPostprocessConfig(StrictModel):
    enabled: bool = True
    road_class: str = "road"
    max_slope_degrees: Annotated[float, Field(gt=0, le=90)] = 40.0
    plane_neighbors: Annotated[int, Field(ge=3)] = 12


class PostprocessConfig(StrictModel):
    """Conservative geometry cleanup applied after SAM image voting."""

    enabled: bool = True
    output_subdir: str = "post_sam_filter"
    neighborhood_radius_m: Positive = 0.35
    neighbors: Annotated[int, Field(ge=2, le=64)] = 16
    npi_preserve_confidence: Probability = 0.9
    class_rules: dict[str, ClassPostprocessRule] = Field(default_factory=dict)
    building: BuildingPostprocessConfig = Field(default_factory=BuildingPostprocessConfig)
    road: RoadPostprocessConfig = Field(default_factory=RoadPostprocessConfig)


class DebugConfig(StrictModel):
    save_overlays: bool = False
    overlay_every_n: Annotated[int, Field(ge=1)] = 25


class Settings(StrictModel):
    metashape: MetashapeConfig = Field(default_factory=MetashapeConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    sam: SAMConfig = Field(default_factory=SAMConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    visibility: VisibilityConfig = Field(default_factory=VisibilityConfig)
    voting: VotingConfig = Field(default_factory=VotingConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)


def load_settings(path: Path | None = None) -> Settings:
    path = path or REPO_ROOT / "config/default.yml"
    with path.open(encoding="utf-8") as stream:
        settings = Settings.model_validate(yaml.safe_load(stream) or {})
    # Paths are relative to the repository, independent of the caller's working directory.
    for name in ("model", "work_dir", "output_dir"):
        value = getattr(settings.paths, name)
        setattr(settings.paths, name, (REPO_ROOT / value).resolve())
    return settings


def load_classes(path: Path | None = None) -> Classes:
    with (path or REPO_ROOT / "config/classes.yml").open(encoding="utf-8") as stream:
        return Classes.model_validate(yaml.safe_load(stream))
