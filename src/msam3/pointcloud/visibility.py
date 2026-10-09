"""GPU z-buffer built across all spatial batches before any visibility queries."""

import math

import torch

from msam3.config import VisibilityConfig


class ZBuffer:
    def __init__(self, width: int, height: int, config: VisibilityConfig, device: str):
        self.width = max(1, math.ceil(width * config.zbuffer_scale))
        self.height = max(1, math.ceil(height * config.zbuffer_scale))
        self.sx, self.sy = self.width / width, self.height / height
        self.config = config
        self.depth = torch.full((self.width * self.height,), float("inf"), device=device)

    def pixels(self, uv):
        return (uv[:, 0] * self.sx).long(), (uv[:, 1] * self.sy).long()

    def add(self, uv, depth):
        x, y = self.pixels(uv)
        radius = self.config.splat_px
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                xx, yy = x + dx, y + dy
                valid = (xx >= 0) & (xx < self.width) & (yy >= 0) & (yy < self.height)
                indices = yy[valid] * self.width + xx[valid]
                self.depth.scatter_reduce_(
                    0, indices, depth[valid], reduce="amin", include_self=True
                )

    def visible(self, uv, depth):
        x, y = self.pixels(uv)
        reference = self.depth[y * self.width + x]
        return torch.isfinite(reference) & (
            depth <= reference * (1 + self.config.rel_tolerance) + self.config.abs_tolerance_m
        )
