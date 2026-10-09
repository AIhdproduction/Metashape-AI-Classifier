"""Weighted observations and bounded-memory final decisions."""

from pathlib import Path

import numpy as np

from msam3.config import Classes, VotingConfig


class Votes:
    def __init__(self, count: int, classes: Classes, config: VotingConfig, directory: Path):
        self.classes, self.config = classes, config
        shape = (count, len(classes.classes))
        self.memmap_path = None
        if count * shape[1] * 4 > config.max_ram_gb * 1024**3:
            self.memmap_path = directory / "votes.dat"
            self.weights = np.memmap(self.memmap_path, mode="w+", dtype="float32", shape=shape)
            self.weights[:] = 0
        else:
            self.weights = np.zeros(shape, np.float32)
        self.views = np.zeros(count, np.uint32)

    def add(self, indices, labels, confidence, view_weight=1.0):
        indices, labels = np.asarray(indices), np.asarray(labels)
        weights = np.asarray(confidence) * view_weight
        valid = (labels != 255) & (labels < self.weights.shape[1])
        valid &= np.isfinite(weights) & (weights > 0)
        ids = indices[valid]
        np.add.at(self.weights, (ids, labels[valid]), weights[valid])
        np.add.at(self.views, ids, 1)

    def decide(self):
        count = len(self.views)
        labels = np.full(count, self.classes.unknown_code, np.uint8)
        shares = np.zeros(count, np.float32)
        code_lookup = np.array([c.code for c in self.classes.classes], dtype=np.uint8)
        for start in range(0, count, self.config.batch_points):
            weights = self.weights[start : start + self.config.batch_points]
            best = weights.argmax(axis=1)
            total = weights.sum(axis=1)
            maximum = weights[np.arange(len(best)), best]
            share = np.divide(maximum, total, out=np.zeros_like(total), where=total > 0)
            valid = self.views[start : start + len(best)] >= self.config.min_views
            valid &= (share >= self.config.min_share) & (total > 0)
            labels[start : start + len(best)][valid] = code_lookup[best[valid]]
            shares[start : start + len(best)] = share
        return labels, shares
