"""Validated settings and deterministic LAS class allocation."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
BackendName = Literal["flurseg", "sam3"]
ResolutionName = Literal["fast", "balanced", "accurate"]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Code = Annotated[int, Field(ge=0, le=255, strict=True)]
HexColor = Annotated[str, Field(pattern=r"^#[0-9A-Fa-f]{6}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class SemanticClass(StrictModel):
    name: Annotated[str, Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")]
    code: Code | None = None
    prompts: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)
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
    # A fixed class list comes with the model weights and has no text prompts.
    fixed: bool = False
    classes: Annotated[list[SemanticClass], Field(min_length=1, max_length=255)]

    @model_validator(mode="after")
    def resolve(self):
        names = [c.name for c in self.classes]
        explicit = [c.code for c in self.classes if c.code is not None]
        if not self.fixed:
            for cls in self.classes:
                if not cls.prompts:
                    raise ValueError(f"Class {cls.name} needs at least one prompt")
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
    # None follows the selected model ("FlurSeg classified" / "SAM3 classified").
    imported_label: str | None = None
    make_default: bool = False
    display_mode: Literal["classification", "class_palette", "image_colors"] = "image_colors"
    offscreen: Literal["auto", "always", "never"] = "auto"


class SegmentationConfig(StrictModel):
    backend: BackendName = "flurseg"


class FlurSegConfig(StrictModel):
    resolution: ResolutionName = "balanced"
    window_batch_size: Annotated[int, Field(ge=1, le=32)] = 4
    precision: Literal["fp16", "fp32"] = "fp16"
    # Scores are normalised over the eleven classes, so both thresholds are class shares.
    background_threshold: Probability = 0.3
    min_mask_confidence: Probability = 0.4


class PathsConfig(StrictModel):
    model: Path = Path("models/sam3.pt")
    flurseg_model: Path = Path("models/flurseg/v1")
    work_dir: Path = Path("work")
    output_dir: Path = Path("output")
    image_path_remap: dict[str, str] = Field(default_factory=dict)
    # False deletes the project's work folder after a verified final LAS. True keeps the
    # export, depth maps and image caches, so a rerun skips the export and the GPU.
    keep_work_data: bool = False


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
    min_tie_points_inside_dense_share: Probability = 0.8


class VisibilityConfig(StrictModel):
    mode: Literal["zbuffer", "depth_maps"] = "zbuffer"
    zbuffer_scale: Annotated[float, Field(gt=0, le=1)] = 0.25
    splat_px: Annotated[int, Field(ge=0, le=16)] = 2
    rel_tolerance: Probability = 0.02
    abs_tolerance_m: Annotated[float, Field(ge=0)] = 0.05
    mask_interior_px: Annotated[int, Field(ge=0, le=8)] = 2
    # Classes in their neighbour's plane that would vanish entirely by mask erosion.
    no_erosion_classes: list[str] = Field(default_factory=list)
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
    """Optional Metashape terrain evidence that may resolve otherwise unknown votes."""

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
    """Use existing Metashape vegetation classes only where the model has no decision."""

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
    protected_classes: list[str] = Field(default_factory=list)


class ClassPostprocessRule(StrictModel):
    """Neighbourhood requirements for one semantic class after image voting."""

    min_same_neighbors: Annotated[int, Field(ge=0)] = 3
    fill_neighbors: Annotated[int, Field(ge=1)] = 8
    fill_share: Probability = 0.75
    max_replaced_confidence: Probability = 0.55


class ClusterPostprocessConfig(StrictModel):
    """Reject isolated semantic fragments for every configured output class."""

    enabled: bool = True
    radius_m: Positive = 0.45
    neighbors: Annotated[int, Field(ge=1, le=64)] = 12
    min_same_neighbors: Annotated[int, Field(ge=1, le=64)] = 4
    min_same_overrides: dict[str, Annotated[int, Field(ge=1, le=64)]] = Field(default_factory=dict)


class BuildingPostprocessConfig(StrictModel):
    enabled: bool = True
    ground_class: str = "ground"
    building_class: str = "building"
    horizontal_radius_m: Positive = 3.0
    min_vertical_drop_m: Positive = 1.0
    min_building_neighbors: Annotated[int, Field(ge=4, le=48)] = 8
    max_roof_slope_degrees: Annotated[float, Field(gt=0, lt=90)] = 65.0
    terrain_radius_m: Positive = 12.0


class GroundPostprocessConfig(StrictModel):
    """Reject ground-labelled points that are detached from the local ground plane."""

    enabled: bool = True
    ground_class: str = "ground"
    horizontal_radius_m: Positive = 0.75
    neighbors: Annotated[int, Field(ge=3, le=64)] = 16
    min_ground_neighbors: Annotated[int, Field(ge=3, le=64)] = 8
    max_plane_slope_degrees: Annotated[float, Field(gt=0, le=90)] = 55.0
    max_offset_m: Annotated[float, Field(gt=0)] = 0.15


class ColorGroundPostprocessConfig(StrictModel):
    """Restore bare terrain that the image model called low vegetation."""

    enabled: bool = False
    vegetation_class: str = "low_vegetation"
    ground_class: str = "ground"
    max_excess_green: Annotated[float, Field(ge=-1, le=1)] = 0.08
    min_brightness: Probability = 0.05


class OtherBuildingPostprocessConfig(StrictModel):
    """Extend trusted building points into nearby 'other' roof and wall points."""

    enabled: bool = False
    other_class: str = "other"
    building_class: str = "building"
    radius_m: Positive = 0.75
    # Kerbs, markings and covers that the model called 'other' often lie beside protected
    # points; candidates this close to a protected class keep their label.
    protected_clearance_m: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.25


SurfaceKind = Literal["horizontal", "vertical", "any"]
Sectors = Annotated[int, Field(ge=0, le=8)]


class GeometryReviewRule(StrictModel):
    """Move a point to the target class only where the target's local surface carries it.

    The candidate must lie on a plane fitted to nearby target points, its own Metashape normal
    must agree with that plane, and the plane must surround it. Unclassified points face the
    stricter ``unknown_*`` limits: they are filled inside a surface, never grown past its edge.
    """

    source_classes: list[str] = Field(default_factory=list)
    include_unknown: bool = False
    # Roofs and facades are separate planes; fitting them together blurs both.
    surfaces: list[SurfaceKind] = Field(default_factory=lambda: ["any"], min_length=1)
    # Optional: a source counts only where its own normal faces this way, e.g. {other: [vertical]}.
    source_surfaces: dict[str, list[SurfaceKind]] = Field(default_factory=dict)
    normal_vertical_boundary: Probability = 0.55  # |normal z| from here on is horizontal.
    support_voxel_m: Positive = 0.12
    horizontal_radius_m: Positive = 4.0
    vertical_radius_m: Positive = 2.0
    neighbors: Annotated[int, Field(ge=3, le=64)] = 16
    min_support_neighbors: Annotated[int, Field(ge=3, le=64)] = 8
    min_planar_spread_m: Positive = 0.08
    max_plane_offset_m: Positive = 0.12
    min_normal_alignment: Probability = 0.8
    min_coverage_sectors: Sectors = 0
    unknown_radius_m: Positive = 0.2
    unknown_max_plane_offset_m: Positive = 0.08
    unknown_min_normal_alignment: Probability = 0.85
    unknown_min_coverage_sectors: Sectors = 7
    # Candidates this close to a protected thin class (kerb, marking, cover) stay unchanged.
    protected_clearance_m: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.25

    @model_validator(mode="after")
    def check_support_count(self):
        if self.min_support_neighbors > self.neighbors:
            raise ValueError("geometry min_support_neighbors exceeds neighbors")
        if len(set(self.surfaces)) != len(self.surfaces):
            raise ValueError("geometry surfaces are listed twice")
        return self


class CompactClassRule(StrictModel):
    """A class of small, flat objects: groups outside these limits take their surroundings."""

    min_length_m: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.2
    max_length_m: Positive = 1.5
    max_height_span_m: Positive = 0.15
    link_m: Positive = 0.1  # Points closer than this belong to one object.

    @model_validator(mode="after")
    def check_lengths(self):
        if self.min_length_m >= self.max_length_m:
            raise ValueError("compact class min_length_m must be below max_length_m")
        return self


class DtmGroundPostprocessConfig(StrictModel):
    """Low vegetation lying on a terrain model of the lowest points becomes ground.

    Photogrammetry sees the top of a meadow, not the soil. Per grid cell, a low percentile of
    the ground and low-vegetation heights (along the true vertical) forms a terrain node; nodes
    far below or above their neighbours are dropped, and each node gets a local plane. Low
    vegetation at most ``max_height_m`` above that plane is ground. Runs after the neighbourhood
    filters, which would otherwise dissolve the low vegetation left in between.
    """

    enabled: bool = True
    ground_class: str = "ground"
    vegetation_class: str = "low_vegetation"
    support_classes: list[str] = Field(default_factory=lambda: ["ground", "low_vegetation"])
    cell_size_m: Positive = 1.0
    percentile: Probability = 0.05
    min_cell_points: Annotated[int, Field(ge=1)] = 5
    max_height_m: Positive = 0.05
    max_depth_m: Positive = 0.3  # Further below the terrain is noise and keeps its label.
    max_node_drop_m: Positive = 0.25
    max_node_rise_m: Positive = 1.0
    max_slope_degrees: Annotated[float, Field(gt=0, le=90)] = 60


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
    sam_high_min_height_m: Positive = 0.4
    sam_high_min_share: Probability = 0.65


class UnknownFillConfig(StrictModel):
    """Conservatively extend a class into immediately adjacent unclassified points."""

    enabled: bool = True
    min_same_neighbors: Annotated[int, Field(ge=1, le=64)] = 5
    min_share: Probability = 0.75


class PlanarUnknownFillConfig(StrictModel):
    """Restore unknown edge points only where one labelled surface supplies a local plane."""

    enabled: bool = True
    class_names: list[str] = ["ground", "road", "sidewalk", "building"]
    radius_m: Positive = 0.5
    neighbors: Annotated[int, Field(ge=3, le=64)] = 16
    min_same_neighbors: Annotated[int, Field(ge=3, le=64)] = 8
    min_planar_spread_m: Positive = 0.05
    max_plane_offset_m: Annotated[float, Field(ge=0)] = 0.08


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
    """Conservative geometry cleanup applied after image voting."""

    enabled: bool = True
    output_subdir: str = "post_sam_filter"
    # Metashape's internal frame is not levelled (31.7 degrees in the 291-photo test project).
    # True rotates all post-filters to the vertical estimated from the point cloud; their
    # thresholds were calibrated in the internal frame, so this is off until recalibrated.
    # Roof/facade checks always use that vertical.
    level_to_vertical: bool = False
    neighborhood_radius_m: Positive = 0.35
    neighbors: Annotated[int, Field(ge=2, le=64)] = 16
    class_rules: dict[str, ClassPostprocessRule] = Field(default_factory=dict)
    # Never overwritten by a neighbouring class (thin markings, covers, kerbs).
    protected_classes: list[str] = Field(default_factory=list)
    cluster: ClusterPostprocessConfig = Field(default_factory=ClusterPostprocessConfig)
    building: BuildingPostprocessConfig = Field(default_factory=BuildingPostprocessConfig)
    ground: GroundPostprocessConfig = Field(default_factory=GroundPostprocessConfig)
    color_ground: ColorGroundPostprocessConfig = Field(default_factory=ColorGroundPostprocessConfig)
    other_building: OtherBuildingPostprocessConfig = Field(
        default_factory=OtherBuildingPostprocessConfig
    )
    geometry_rules: dict[str, GeometryReviewRule] = Field(default_factory=dict)
    vegetation: VegetationPostprocessConfig = Field(default_factory=VegetationPostprocessConfig)
    dtm_ground: DtmGroundPostprocessConfig = Field(default_factory=DtmGroundPostprocessConfig)
    compact_classes: dict[str, CompactClassRule] = Field(default_factory=dict)
    unknown_fill: UnknownFillConfig = Field(default_factory=UnknownFillConfig)
    planar_unknown_fill: PlanarUnknownFillConfig = Field(default_factory=PlanarUnknownFillConfig)
    road: RoadPostprocessConfig = Field(default_factory=RoadPostprocessConfig)


class DebugConfig(StrictModel):
    save_overlays: bool = False
    overlay_every_n: Annotated[int, Field(ge=1)] = 25


class Settings(StrictModel):
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    metashape: MetashapeConfig = Field(default_factory=MetashapeConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    sam: SAMConfig = Field(default_factory=SAMConfig)
    flurseg: FlurSegConfig = Field(default_factory=FlurSegConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    visibility: VisibilityConfig = Field(default_factory=VisibilityConfig)
    voting: VotingConfig = Field(default_factory=VotingConfig)
    ground_assist: GroundAssistConfig = Field(default_factory=GroundAssistConfig)
    vegetation_assist: VegetationAssistConfig = Field(default_factory=VegetationAssistConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)


def merge_settings(base: dict, override: dict, replace: tuple[str, ...] = ()) -> dict:
    """Merge nested settings in place. Keys in ``replace`` are never mixed with their old value."""
    for key, value in override.items():
        if key not in replace and isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_settings(base[key], value, replace)
        else:
            base[key] = value
    return base


def load_settings(path: Path | None = None) -> Settings:
    path = path or REPO_ROOT / "config/default.yml"
    with path.open(encoding="utf-8") as stream:
        settings = Settings.model_validate(yaml.safe_load(stream) or {})
    # Paths are relative to the repository, independent of the caller's working directory.
    for name in ("model", "flurseg_model", "work_dir", "output_dir"):
        value = getattr(settings.paths, name)
        setattr(settings.paths, name, (REPO_ROOT / value).resolve())
    return settings


def load_classes(path: Path | None = None) -> Classes:
    with (path or REPO_ROOT / "config/classes.yml").open(encoding="utf-8") as stream:
        return Classes.model_validate(yaml.safe_load(stream))
