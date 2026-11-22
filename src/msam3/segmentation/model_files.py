"""Checks of the local model files. No torch and no network: the UI runs these as well."""

import json
from pathlib import Path

from msam3.segmentation.windows import IMAGE_SIZE

SAM_REVISION = "0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc"
SAM_VERSION = "raw-grounding-v2-batched"
SAM_URL = "https://huggingface.co/facebook/sam3"

FLURSEG_VERSION = "eomt-dinov3-hann-band-v1"
FLURSEG_URL = "https://huggingface.co/AIhdproduction/Flurseg"
FLURSEG_REVISION = "5df53dbbf721778f1ecd4903369e9465926ae2fd"
FLURSEG_V1_BYTES = 1_259_264_920
FLURSEG_V1_SHA256 = "83ad8990953cadbdbc4ae831eb41eeddf100312709cee8042ff412f211a789a9"
FLURSEG_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")
FLURSEG_LABELS = (
    "other",
    "ground",
    "paved_surface",
    "low_vegetation",
    "tree",
    "building",
    "vehicle",
    "construction",
    "curb",
    "road_marking",
    "utility_cover",
)
FLURSEG_DOWNLOAD = (
    r".venv\Scripts\hf.exe download AIhdproduction/Flurseg "
    f'--revision {FLURSEG_REVISION} --include "v1/*" --local-dir models/flurseg'
)


def require_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"SAM 3 checkpoint missing: {path}. Request access at "
            f"{SAM_URL} (access request required), accept Meta's "
            "SAM License, download the checkpoint and place it at models/sam3.pt "
            "(or set paths.model). Weights are not included in this repository."
        )


def check_safetensors(path: Path) -> None:
    """Catch pointer files and interrupted downloads before the long Metashape export."""
    size = path.stat().st_size
    try:
        with path.open("rb") as stream:
            length = int.from_bytes(stream.read(8), "little")
            if not 2 <= length <= size - 8:
                raise ValueError("header length")
            header = json.loads(stream.read(length))
        ends = [value["data_offsets"][1] for key, value in header.items() if key != "__metadata__"]
        if 8 + length + max(ends, default=0) != size:
            raise ValueError("file length")
    except (ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(
            f"{path} is incomplete or not a safetensors file. Download FlurSeg again: "
            f"{FLURSEG_DOWNLOAD}"
        ) from error


def read_flurseg_metadata(directory: Path) -> tuple[list[float], list[float]]:
    """Validate architecture and label order; return the normalisation mean and std."""
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        processor = json.loads((directory / "preprocessor_config.json").read_text(encoding="utf-8"))
        id2label = {int(key): str(value) for key, value in config["id2label"].items()}
        mean = [float(value) for value in processor["image_mean"]]
        std = [float(value) for value in processor["image_std"]]
        model_type, image_size = config["model_type"], config["image_size"]
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Invalid FlurSeg configuration in {directory}: {error!r}") from error
    if model_type != "eomt_dinov3" or image_size != IMAGE_SIZE:
        raise ValueError(
            f"{directory} is not a FlurSeg model: expected eomt_dinov3 with {IMAGE_SIZE} pixels, "
            f"found {model_type} with {image_size}"
        )
    check_labels(id2label)
    if len(mean) != 3 or len(std) != 3 or min(std) <= 0:
        raise ValueError(f"Invalid image_mean/image_std in {directory}")
    return mean, std


def check_labels(id2label: dict) -> None:
    found = [id2label[index] for index in sorted(id2label)]
    if sorted(id2label) != list(range(len(FLURSEG_LABELS))) or found != list(FLURSEG_LABELS):
        raise ValueError(
            f"Unexpected FlurSeg class list or order {found}; expected {list(FLURSEG_LABELS)}"
        )


def require_flurseg(directory: Path) -> None:
    missing = [name for name in FLURSEG_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"FlurSeg model files missing in {directory}: {', '.join(missing)}. "
            f"Download them from {FLURSEG_URL}/tree/main/v1 or run in the repository: "
            f"{FLURSEG_DOWNLOAD}"
        )
    read_flurseg_metadata(directory)
    check_safetensors(directory / "model.safetensors")
