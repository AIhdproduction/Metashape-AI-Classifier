"""Sliding windows and maximum merge before the single final argmax."""

from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from msam3.config import Classes, SAMConfig


class TileBackend(Protocol):
    def predict_tile(self, image: Image.Image, classes: Classes) -> np.ndarray:
        """Return C x H x W fused float probabilities."""
        ...


def tile_boxes(width: int, height: int, size: int, overlap: float):
    stride = max(1, round(size * (1 - overlap)))

    def starts(length):
        last = max(0, length - size)
        return sorted(set([*range(0, last + 1, stride), last]))

    for y in starts(height):
        for x in starts(width):
            yield x, y, min(width, x + size), min(height, y + size)


def segment_image(
    image: Image.Image, backend: TileBackend, classes: Classes, settings: SAMConfig, scratch: Path
):
    width = max(1, round(image.width * settings.image_scale))
    height = max(1, round(image.height * settings.image_scale))
    scaled = image.resize((width, height), Image.Resampling.LANCZOS)
    # Full per-class image scores live on disk, not in GPU memory.
    merged = np.memmap(
        scratch, mode="w+", dtype="float32", shape=(len(classes.classes), height, width)
    )
    merged[:] = 0
    try:
        for x0, y0, x1, y1 in tile_boxes(width, height, settings.input_size, settings.tile_overlap):
            tile = scaled.crop((x0, y0, x1, y1))
            probabilities = backend.predict_tile(tile, classes)
            expected = (len(classes.classes), y1 - y0, x1 - x0)
            if probabilities.shape != expected or not np.isfinite(probabilities).all():
                raise ValueError(
                    f"Backend returned invalid tile probabilities; expected {expected}"
                )
            if np.any((probabilities < 0) | (probabilities > 1)):
                raise ValueError("Backend probabilities must be in [0, 1]")
            region = merged[:, y0:y1, x0:x1]
            np.maximum(region, probabilities, out=region)
        labels = np.empty((height, width), dtype=np.uint8)
        confidence = np.empty((height, width), dtype=np.float32)
        for y in range(0, height, 256):
            scores = merged[:, y : y + 256]
            labels[y : y + 256] = scores.argmax(axis=0)
            confidence[y : y + 256] = scores.max(axis=0)
        labels[confidence < settings.background_threshold] = 255
        return labels, confidence
    finally:
        merged.flush()
        # NumPy slice views can keep the mapping alive after del merged on Windows.
        merged._mmap.close()
        del merged
        scratch.unlink(missing_ok=True)
        scaled.close()
