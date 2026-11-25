"""Atomic per-image caches whose identity excludes voting-only settings."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


def json_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False, default=str), encoding="utf-8")
    replace_file(temp, path)


def replace_file(source: Path, target: Path, attempts: int = 60) -> None:
    """Windows refuses a replace for the moment in which the UI reads the same file."""
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05)


class LabelCache:
    def __init__(self, root: Path, config_hash: str, model_hash: str):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.config_hash = config_hash
        self.model_hash = model_hash

    def identity(self, camera, image: Path, size: tuple[int, int]) -> dict:
        stat = image.stat()
        return {
            "schema": 1,
            "config_hash": self.config_hash,
            "model_hash": self.model_hash,
            "image": str(image.resolve()),
            "image_bytes": stat.st_size,
            "image_mtime_ns": stat.st_mtime_ns,
            "camera_key": camera.key,
            "raw_size": [camera.width, camera.height],
            "size": list(size),
            "orientation": camera.orientation,
        }

    def load(self, key: str, identity: dict):
        sidecar = self.root / f"{key}.json"
        try:
            stored = json.loads(sidecar.read_text(encoding="utf-8"))
            if stored.get("identity") != identity:
                return None
            paths = [self.root / f"{key}.{kind}.png" for kind in ("labels", "confidence")]
            if [file_hash(p) for p in paths] != stored["png_hashes"]:
                return None
            arrays = []
            for path in paths:
                with Image.open(path) as image:
                    arrays.append(np.array(image, dtype=np.uint8))
            expected = tuple(reversed(identity["size"]))
            if any(a.shape != expected for a in arrays):
                return None
            return arrays[0], arrays[1].astype(np.float32) / 255
        except (OSError, ValueError, KeyError):
            return None

    def save(self, key: str, labels: np.ndarray, confidence: np.ndarray, identity: dict):
        size = tuple(identity["size"])
        # Sample both maps identically. Interpolating confidence across labels invents evidence.
        label_image = Image.fromarray(labels).resize(size, Image.Resampling.NEAREST)
        conf_image = Image.fromarray(np.rint(confidence * 255).clip(0, 255).astype(np.uint8))
        conf_image = conf_image.resize(size, Image.Resampling.NEAREST)
        hashes = []
        for kind, image in (("labels", label_image), ("confidence", conf_image)):
            path = self.root / f"{key}.{kind}.png"
            temp = path.with_suffix(".tmp")
            # Confidence maps compress poorly at level 6. Level 1 saved about
            # 0.1 s for one representative test image while keeping lossless PNGs.
            image.save(temp, format="PNG", compress_level=1)
            temp.replace(path)
            hashes.append(file_hash(path))
        atomic_json(self.root / f"{key}.json", {"identity": identity, "png_hashes": hashes})
        return np.array(label_image), np.array(conf_image, dtype=np.float32) / 255
