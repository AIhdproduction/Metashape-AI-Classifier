"""Validated settings and deterministic LAS class allocation."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Code = Annotated[int, Field(ge=0, le=255, strict=True)]
HexColor = Annotated[str, Field(pattern=r"^#[0-9A-Fa-f]{6}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class SemanticClass(StrictModel):
    name: Annotated[str, Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")]
    code: Code | None = None
    prompts: Annotated[list[Annotated[str, Field(min_length=1)]], Field(min_length=1)]
    color: HexColor = "#808080"

    @model_validator(mode="after")
    def check_prompts(self):
        if any(not p.strip() for p in self.prompts):
            raise ValueError(f"Empty prompt in class {self.name}")
        return self


class Classes(StrictModel):
    unknown_code: Code = 1
    unknown_color: HexColor = "#808080"
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
    display_mode: Literal["classification", "class_palette", "image_colors"] = "image_colors"
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
    mask_interior_px: Annotated[int, Field(ge=0, le=8)] = 2
    min_mask_confidence: Probability = 0.5


class VotingConfig(StrictModel):
    voxel_size_m: Positive = 0.03
    block_size_m: Positive = 10.0
    min_views: Annotated[int, Field(ge=1)] = 3
    min_share: Probability = 0.4
    use_incidence_weight: bool = False
    max_ram_gb: Positive = 16
    batch_points: Annotated[int, Field(ge=1)] = 500_000
    voxel_buckets: Annotated[int, Field(ge=1, le=4096)] = 256


class GroundAssistConfig(StrictModel):
    """Optional Metashape terrain evidence that may resolve otherwise unknown SAM votes."""

    enabled: bool = True
    classify_before_export: bool = True
    source_classes: list[Literal["created", "unclassified"]] = ["created", "unclassified"]
    max_angle_degrees: Annotated[float, Field(gt=0, le=90)] = 15.0
    max_distance_m: Positive = 0.3
    max_terrain_slope_degrees: Annotated[float, Field(gt=0, le=90)] = 35.0
    cell_size_m: Positive = 10.0
    erosion_radius_m: Annotated[float, Field(ge=0)] = 0.5
    ground_like_classes: list[str] = ["ground", "road", "sidewalk"]
    min_sam_ground_share: Probability = 0.15
    min_sam_ground_views: Annotated[int, Field(ge=1)] = 1
    fallback_class: str = "ground"
    fallback_confidence: Probability = 0.4


class VegetationAssistConfig(StrictModel):
    """Use existing Metashape vegetation classes only where SAM has no decision."""

    enabled: bool = True
    source_low_codes: list[Code] = [3]
    source_high_codes: list[Code] = [5]
    low_vegetation_class: str = "low_vegetation"
    high_vegetation_class: str = "tree"
    fallback_confidence: Probability = 0.35

    @model_validator(mode="after")
    def check_source_codes(self):
        if set(self.source_low_codes) & set(self.source_high_codes):
            raise ValueError("Metashape low and high vegetation source codes must not overlap")
        return self


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


class GroundPostprocessConfig(StrictModel):
    """Reject ground-labelled points that are detached from the local ground plane."""

    enabled: bool = True
    ground_class: str = "ground"
    horizontal_radius_m: Positive = 0.75
    neighbors: Annotated[int, Field(ge=3, le=64)] = 16
    min_ground_neighbors: Annotated[int, Field(ge=3, le=64)] = 8
    max_plane_slope_degrees: Annotated[float, Field(gt=0, le=90)] = 55.0
    max_offset_m: Annotated[float, Field(gt=0)] = 0.15


class VegetationPostprocessConfig(StrictModel):
    """Split mixed vegetation clusters by height above an in-memory terrain DEM."""

    enabled: bool = True
    low_vegetation_class: str = "low_vegetation"
    high_vegetation_class: str = "tree"
    terrain_class_codes: list[Code] = [2, 9, 11, 73, 74]
    fallback_terrain_class_codes: list[Code] = [14, 77]
    dem_grid_size_m: Positive = 0.25
    max_dem_nodes: Annotated[int, Field(ge=1000)] = 500_000
    interpolation_radius_m: Positive = 2.0
    interpolation_neighbors: Annotated[int, Field(ge=3, le=64)] = 8
    min_dem_neighbors: Annotated[int, Field(ge=3, le=64)] = 6
    max_dem_slope_degrees: Annotated[float, Field(gt=0, le=90)] = 55.0
    max_dem_residual_m: Positive = 0.15
    high_vegetation_min_height_m: Annotated[float, Field(ge=0)] = 1.2
    threshold_tolerance: Probability = 0.1
    cluster_cell_size_m: Positive = 0.5
    tree_cluster_min_height_m: Positive = 1.5
    tree_cluster_radius_m: Positive = 2.5
    tree_seed_min_sam_share: Probability = 0.65


class UnknownFillConfig(StrictModel):
    """Conservatively extend a class into immediately adjacent unclassified points."""

    enabled: bool = True
    min_same_neighbors: Annotated[int, Field(ge=1, le=64)] = 5
    min_share: Probability = 0.75


class RoadPostprocessConfig(StrictModel):
    enabled: bool = True
    road_class: str = "road"
    max_slope_degrees: Annotated[float, Field(gt=0, le=90)] = 40.0
    plane_neighbors: Annotated[int, Field(ge=3)] = 12
    surface_radius_m: Positive = 1.5
    surface_neighbors: Annotated[int, Field(ge=3, le=64)] = 24
    min_surface_road_neighbors: Annotated[int, Field(ge=3, le=64)] = 12
    min_surface_coverage_sectors: Annotated[int, Field(ge=1, le=8)] = 6
    max_surface_offset_m: Annotated[float, Field(ge=0)] = 0.08
    surface_replace_classes: list[str] = Field(default_factory=list)
    fill_unknown_surface: bool = True


class PostprocessConfig(StrictModel):
    """Conservative geometry cleanup applied after SAM image voting."""

    enabled: bool = True
    output_subdir: str = "post_sam_filter"
    neighborhood_radius_m: Positive = 0.35
    neighbors: Annotated[int, Field(ge=2, le=64)] = 16
    npi_preserve_confidence: Probability = 0.9
    class_rules: dict[str, ClassPostprocessRule] = Field(default_factory=dict)
    building: BuildingPostprocessConfig = Field(default_factory=BuildingPostprocessConfig)
    ground: GroundPostprocessConfig = Field(default_factory=GroundPostprocessConfig)
    vegetation: VegetationPostprocessConfig = Field(default_factory=VegetationPostprocessConfig)
    unknown_fill: UnknownFillConfig = Field(default_factory=UnknownFillConfig)
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
    ground_assist: GroundAssistConfig = Field(default_factory=GroundAssistConfig)
    vegetation_assist: VegetationAssistConfig = Field(default_factory=VegetationAssistConfig)
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
