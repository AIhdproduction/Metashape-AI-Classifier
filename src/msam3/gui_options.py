"""Run options and estimates for the app, and the entry point that the Metashape launcher starts.

Everything here is free of NiceGUI and torch, so it can be tested and never allocates VRAM.
"""

import argparse
import logging
import os
import statistics
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from msam3 import backends
from msam3.config import Classes, Settings, merge_settings
from msam3.gui_progress import CACHED_LINE, IMAGE_LINE, read_json
from msam3.segmentation.windows import window_count

LOG = logging.getLogger(__name__)
PROFILE_NAMES = ("fast", "balanced", "accurate")
SAM_PROFILES = {
    "fast": {"image_scale": 0.50, "tile_overlap": 0.10, "min_views": 1},
    "balanced": {"image_scale": 0.75, "tile_overlap": 0.20, "min_views": 1},
    "accurate": {"image_scale": 1.00, "tile_overlap": 0.30, "min_views": 2},
}
# Planning values while this project has no own history (both measured on an RTX 4090):
# FlurSeg in fp32, SAM 3 per 1008 px tile at prompt batch 8.
FLURSEG_SECONDS_PER_WINDOW = 0.09
SAM_SECONDS_PER_TILE = 1.4
SAM_SECONDS_PER_IMAGE = 45.0


def tile_count(width: int, height: int, image_scale: float, tile_overlap: float) -> int:
    """Mirror the production sliding-window count without loading an image."""
    scaled_width = max(1, round(width * image_scale))
    scaled_height = max(1, round(height * image_scale))
    stride = max(1, round(1008 * (1 - tile_overlap)))

    def starts(length: int) -> int:
        last = max(0, length - 1008)
        return len(set([*range(0, last + 1, stride), last]))

    return starts(scaled_width) * starts(scaled_height)


def total_tile_count(image_sizes, image_scale: float, tile_overlap: float) -> int:
    return sum(
        count * tile_count(int(width), int(height), image_scale, tile_overlap)
        for (width, height), count in Counter(map(tuple, image_sizes)).items()
    )


def total_window_count(image_sizes, resolution: str) -> int:
    return sum(
        count * window_count(int(width), int(height), resolution)
        for (width, height), count in Counter(map(tuple, image_sizes)).items()
    )


def batch_time_factor(prompt_batch_size: int) -> float:
    """The image encoder dominates; more text prompts still shorten each tile measurably."""
    return (8 / prompt_batch_size) ** 0.20


def _log_seconds_per_image(log_path: Path, shortest: float = 1) -> float | None:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    if any(CACHED_LINE.search(line) for line in lines):
        return None  # A resumed run says nothing about the model's speed.
    chunks = [[]]
    for line in lines:
        if match := IMAGE_LINE.match(line):
            if chunks[-1] and int(match["done"]) <= int(chunks[-1][-1]["done"]):
                chunks.append([])
            chunks[-1].append(match)
    records = max(chunks, key=len)
    if len(records) < 8:
        return None
    first = datetime.strptime(records[0]["time"], "%Y-%m-%d %H:%M:%S")
    last = datetime.strptime(records[-1]["time"], "%Y-%m-%d %H:%M:%S")
    # Skipped cameras do not count: they take no time.
    seconds = (last - first).total_seconds() / (len(records) - 1)
    return seconds if shortest <= seconds <= 900 else None


def _previous_runs(sessions_root: Path, project: str, backend: str):
    """Yield (log, session, settings) of this project's runs with the same model."""
    for log_path in sessions_root.glob("*/stage2.log"):
        session = read_json(log_path.parent / "session.json")
        if not session or (project and session.get("project") != project):
            continue
        job = read_json(log_path.parent / "job.json")
        if not job:
            continue
        settings = job.get("settings", {})
        # Jobs from before the model choice were always SAM 3.
        if settings.get("segmentation", {}).get("backend", "sam3") == backend:
            yield log_path, session, settings


def historical_seconds_per_image(
    sessions_root: Path,
    image_sizes=(),
    project: str = "",
    default: float | None = SAM_SECONDS_PER_IMAGE,
) -> float | None:
    """Normalize matching SAM 3 logs to the standard 75% / 20% / batch-8 profile."""
    samples = []
    standard_tiles = total_tile_count(image_sizes, 0.75, 0.20) if image_sizes else 0
    for log_path, _, settings in _previous_runs(sessions_root, project, "sam3"):
        seconds = _log_seconds_per_image(log_path)
        if seconds is None:
            continue
        sam = settings.get("sam", {})
        scale = float(sam.get("image_scale", 0.75))
        overlap = float(sam.get("tile_overlap", 0.20))
        batch = int(sam.get("prompt_batch_size", 8))
        if standard_tiles:
            source_tiles = total_tile_count(image_sizes, scale, overlap)
            if source_tiles:
                seconds *= standard_tiles / source_tiles
        else:
            seconds /= (scale / 0.75) ** 2 * ((1 - 0.20) / (1 - overlap)) ** 2
        samples.append(seconds / batch_time_factor(batch))
    return statistics.median(samples) if samples else default


def historical_seconds_per_window(sessions_root: Path, project: str = "") -> float | None:
    samples = []
    for log_path, session, settings in _previous_runs(sessions_root, project, "flurseg"):
        sizes = session.get("image_sizes") or []
        seconds = _log_seconds_per_image(log_path, shortest=0.05)
        if seconds is None or not sizes:
            continue
        resolution = settings.get("flurseg", {}).get("resolution", "balanced")
        samples.append(seconds * len(sizes) / total_window_count(sizes, resolution))
    return statistics.median(samples) if samples else None


def workflow_overhead(image_count: int) -> float:
    """Export, import and LAS assembly do not depend on the model."""
    return max(180, min(1200, image_count * 2.0))


def estimate_seconds(
    image_count: int,
    image_scale: float,
    tile_overlap: float,
    min_views: int,
    baseline: float,
    prompt_batch_size: int = 8,
    image_sizes=(),
) -> float:
    """SAM 3: estimate from comparable logs and the exact tile count for the selected settings."""
    del min_views  # All enabled photographs are processed regardless of the vote threshold.
    if image_sizes:
        standard_tiles = total_tile_count(image_sizes, 0.75, 0.20)
        selected_tiles = total_tile_count(image_sizes, image_scale, tile_overlap)
        tile_factor = selected_tiles / max(1, standard_tiles)
    else:
        tile_factor = (image_scale / 0.75) ** 2 * ((1 - 0.20) / (1 - tile_overlap)) ** 2
    classification = image_count * baseline * tile_factor * batch_time_factor(prompt_batch_size)
    return classification + workflow_overhead(image_count)


def format_duration(seconds: float, about: str = "ca.") -> str:
    minutes = max(1, round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{about} {hours} h {minutes:02d} min" if hours else f"{about} {minutes} min"


def gpu_memory_gib(device: str = "cuda:0") -> tuple[float, float] | None:
    """Free and total GPU memory from the driver; unlike a CUDA query this allocates nothing."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip() in ("", "-1"):
        return None
    index = device.partition(":")[2] or "0"
    command = [
        "nvidia-smi",
        f"--id={index}",
        "--query-gpu=memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        output = subprocess.run(
            command, capture_output=True, text=True, timeout=8, check=True, creationflags=flags
        ).stdout
        free, total = (float(value) / 1024 for value in output.splitlines()[0].split(","))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
        LOG.info("GPU memory cannot be read: %s", error)
        return None
    return free, total


def maximum_safe_batch(
    free_gib: float, total_gib: float, image_scale: float, tile_overlap: float
) -> int | None:
    """Select the largest batch up to 16 while reserving room for SAM and Windows."""
    workload = (image_scale / 0.75) ** 2 * (1 + 1.5 * abs(tile_overlap - 0.20))
    windows_reserve = max(2.0, total_gib * 0.12)
    sam_model_reserve = 10.5 + 1.2 * workload
    prompt_reserve = 0.55 * workload
    available = free_gib - windows_reserve
    for batch in (16, 12, 8, 4):
        if sam_model_reserve + batch * prompt_reserve <= available:
            return batch
    return None


def settings_override(
    image_scale: float,
    tile_overlap: float,
    prompt_batch_size: int,
    min_views: int,
    ground_assist: bool,
    postprocess: bool,
    vegetation: bool,
    road: bool,
    smoothing: bool,
    vegetation_assist: bool = True,
) -> dict:
    """Translate the visible SAM 3 controls to the validated job settings schema."""
    return {
        "sam": {
            "image_scale": image_scale,
            "tile_overlap": tile_overlap,
            "prompt_batch_size": prompt_batch_size,
        },
        "voting": {"min_views": min_views},
        "ground_assist": {"enabled": ground_assist},
        "vegetation_assist": {"enabled": vegetation_assist},
        "postprocess": {
            "enabled": postprocess,
            "vegetation": {"enabled": vegetation},
            "road": {"enabled": road},
        },
        "smoothing": {"enabled": smoothing},
    }


@dataclass(frozen=True)
class Estimate:
    seconds: float
    units: int  # FlurSeg windows or SAM 3 tiles; 0 when the image sizes are unknown
    from_history: bool


@dataclass
class RunOptions:
    """What the app lets the user choose. It applies to one run only."""

    backend: str
    sam_image_scale: float
    sam_tile_overlap: float
    sam_prompt_batch_size: int
    sam_min_views: int
    sam_instance_threshold: float
    sam_background_threshold: float
    sam_min_mask_confidence: float
    flurseg_resolution: str
    flurseg_batch_size: int
    flurseg_precision: str
    flurseg_min_views: int
    flurseg_background_threshold: float
    flurseg_min_mask_confidence: float
    min_share: float
    voxel_size_m: float
    mask_interior_px: int
    ground_assist: bool
    vegetation_assist: bool
    postprocess: bool
    vegetation: bool
    road: bool
    smoothing: bool

    @classmethod
    def from_settings(cls, settings: Settings) -> "RunOptions":
        return cls(
            backend=settings.segmentation.backend,
            sam_image_scale=settings.sam.image_scale,
            sam_tile_overlap=settings.sam.tile_overlap,
            sam_prompt_batch_size=settings.sam.prompt_batch_size,
            sam_min_views=settings.voting.min_views,
            sam_instance_threshold=settings.sam.instance_conf_threshold,
            sam_background_threshold=settings.sam.background_threshold,
            sam_min_mask_confidence=settings.visibility.min_mask_confidence,
            flurseg_resolution=settings.flurseg.resolution,
            flurseg_batch_size=settings.flurseg.window_batch_size,
            flurseg_precision=settings.flurseg.precision,
            flurseg_min_views=settings.voting.min_views,
            flurseg_background_threshold=settings.flurseg.background_threshold,
            flurseg_min_mask_confidence=settings.flurseg.min_mask_confidence,
            min_share=settings.voting.min_share,
            voxel_size_m=settings.voting.voxel_size_m,
            mask_interior_px=settings.visibility.mask_interior_px,
            ground_assist=settings.ground_assist.enabled,
            vegetation_assist=settings.vegetation_assist.enabled,
            postprocess=settings.postprocess.enabled,
            vegetation=settings.postprocess.vegetation.enabled,
            road=settings.postprocess.road.enabled,
            smoothing=settings.smoothing.enabled,
        )

    @property
    def profile(self) -> str:
        if self.backend == "flurseg":
            return self.flurseg_resolution
        current = {
            "image_scale": self.sam_image_scale,
            "tile_overlap": self.sam_tile_overlap,
            "min_views": self.sam_min_views,
        }
        return next((name for name, v in SAM_PROFILES.items() if v == current), "custom")

    def apply_profile(self, name: str) -> None:
        if self.backend == "flurseg":
            self.flurseg_resolution = name
            return
        values = SAM_PROFILES[name]
        self.sam_image_scale = values["image_scale"]
        self.sam_tile_overlap = values["tile_overlap"]
        self.sam_min_views = values["min_views"]

    def override(self) -> dict:
        is_flurseg = self.backend == "flurseg"
        result = settings_override(
            self.sam_image_scale,
            self.sam_tile_overlap,
            self.sam_prompt_batch_size,
            self.flurseg_min_views if is_flurseg else self.sam_min_views,
            self.ground_assist,
            self.postprocess,
            self.vegetation,
            self.road,
            self.smoothing,
            self.vegetation_assist,
        )
        result["segmentation"] = {"backend": self.backend}
        result["voting"].update(min_share=self.min_share, voxel_size_m=self.voxel_size_m)
        result["visibility"] = {"mask_interior_px": self.mask_interior_px}
        if is_flurseg:
            del result["sam"]
            result["flurseg"] = {
                "resolution": self.flurseg_resolution,
                "window_batch_size": self.flurseg_batch_size,
                "precision": self.flurseg_precision,
                "background_threshold": self.flurseg_background_threshold,
                "min_mask_confidence": self.flurseg_min_mask_confidence,
            }
        else:
            result["sam"].update(
                instance_conf_threshold=self.sam_instance_threshold,
                background_threshold=self.sam_background_threshold,
            )
            result["visibility"]["min_mask_confidence"] = self.sam_min_mask_confidence
        return result

    def settings(self, base: Settings) -> Settings:
        """Validated settings of this run; raises on an invalid value."""
        return Settings.model_validate(
            merge_settings(base.model_dump(mode="json"), self.override())
        )

    def estimate(
        self, sessions_root: Path, image_count: int, image_sizes=(), project: str = ""
    ) -> Estimate:
        if self.backend == "flurseg":
            rate = historical_seconds_per_window(sessions_root, project)
        else:
            rate = historical_seconds_per_image(sessions_root, image_sizes, project, None)
        return self.estimate_from(rate, image_count, image_sizes)

    def estimate_from(self, rate: float | None, image_count: int, image_sizes=()) -> Estimate:
        """``rate`` is the history: seconds per window (FlurSeg) or per standard image (SAM 3)."""
        measured = rate is not None
        if self.backend == "flurseg":
            rate = rate if measured else FLURSEG_SECONDS_PER_WINDOW
            if image_sizes:
                units = total_window_count(image_sizes, self.flurseg_resolution)
                windows = units * image_count / len(image_sizes)
            else:
                # Without sizes assume a common 20 MP drone photo.
                units = 0
                windows = image_count * window_count(5280, 3956, self.flurseg_resolution)
            seconds = windows * rate + workflow_overhead(image_count)
            return Estimate(seconds, units, measured)
        units = 0
        if image_sizes:
            units = total_tile_count(image_sizes, self.sam_image_scale, self.sam_tile_overlap)
            if not measured:
                standard = total_tile_count(image_sizes, 0.75, 0.20)
                rate = SAM_SECONDS_PER_TILE * standard / len(image_sizes)
        seconds = estimate_seconds(
            image_count,
            self.sam_image_scale,
            self.sam_tile_overlap,
            self.sam_min_views,
            rate if rate is not None else SAM_SECONDS_PER_IMAGE,
            self.sam_prompt_batch_size,
            image_sizes,
        )
        return Estimate(seconds, units, measured)


def options_payload(
    options: RunOptions, base: Settings, classes: Classes, estimated_seconds: float
) -> dict:
    """The content of options.json, validated before Metashape continues with it."""
    options.settings(base)
    model = backends.MODELS[options.backend]
    payload = {
        "model": model.describe(),
        "profile": options.profile,
        "estimated_seconds": round(estimated_seconds),
        "settings_override": options.override(),
    }
    if model.editable_classes:
        payload["classes"] = classes.model_dump()
    return payload


def inactive_helpers(settings: Settings, classes: Classes) -> dict[str, list[str]]:
    """Switched-on assists and filters that cannot work, with the classes they miss."""
    settings = backends.selected(settings).apply_profile(settings, classes)
    available = {item.name for item in classes.classes}
    ground, vegetation = settings.ground_assist, settings.vegetation_assist
    post = settings.postprocess
    split, road, roof = post.vegetation, post.road, post.building
    needs = {
        "ground_assist": (ground.enabled, [ground.fallback_class]),
        "vegetation_assist": (
            vegetation.enabled,
            [vegetation.low_vegetation_class, vegetation.high_vegetation_class],
        ),
        "vegetation": (
            post.enabled and split.enabled,
            [split.low_vegetation_class, split.high_vegetation_class],
        ),
        "road": (post.enabled and road.enabled, [road.road_class]),
        "roof": (post.enabled and roof.enabled, [roof.ground_class, roof.building_class]),
        "ground_height": (post.enabled and post.ground.enabled, [post.ground.ground_class]),
    }
    inactive = {}
    for key, (enabled, names) in needs.items():
        missing = [name for name in names if name not in available]
        if enabled and missing:
            inactive[key] = missing
    return inactive


def main() -> int:
    """Entry point of the launcher: the app writes options.json, or options_cancelled.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    from msam3.app.server import serve

    return serve(args.session.resolve(), open_browser=not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
