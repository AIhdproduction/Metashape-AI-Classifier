"""Weighted observations and bounded-memory final decisions."""

from pathlib import Path

import numpy as np

from msam3.config import Classes, GroundAssistConfig, VotingConfig


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


def apply_ground_assist(labels, shares, votes: Votes, ground_hint, config: GroundAssistConfig):
    """Use Metashape terrain only for points that SAM left undecided."""
    if not config.enabled:
        return labels, shares, {"metashape_ground_points": 0, "resolved": {}}
    names = {item.name: (index, item.code) for index, item in enumerate(votes.classes.classes)}
    candidates = [names[name] for name in config.ground_like_classes if name in names]
    if not candidates or config.fallback_class not in names:
        return labels, shares, {"metashape_ground_points": 0, "resolved": {}}

    ground_hint = np.asarray(ground_hint, dtype=bool)
    if ground_hint.shape != labels.shape:
        raise ValueError("Metashape ground hint does not match voting points")
    targets = ground_hint & (labels == votes.classes.unknown_code)
    if not np.any(targets):
        return labels, shares, {"metashape_ground_points": int(ground_hint.sum()), "resolved": {}}

    candidate_indices = np.array([index for index, _ in candidates], dtype=np.intp)
    candidate_codes = np.array([code for _, code in candidates], dtype=np.uint8)
    target_ids = np.flatnonzero(targets)
    support = votes.weights[target_ids][:, candidate_indices]
    best = support.argmax(axis=1)
    strongest = support[np.arange(len(target_ids)), best]
    total = votes.weights[target_ids].sum(axis=1)
    support_share = np.divide(strongest, total, out=np.zeros_like(strongest), where=total > 0)
    sam_supported = votes.views[target_ids] >= config.min_sam_ground_views
    sam_supported &= strongest > 0
    sam_supported &= support_share >= config.min_sam_ground_share

    resolved = {}
    if np.any(sam_supported):
        ids = target_ids[sam_supported]
        codes = candidate_codes[best[sam_supported]]
        labels[ids] = codes
        shares[ids] = support_share[sam_supported]
        resolved.update({str(code): int((codes == code).sum()) for code in np.unique(codes)})
    fallback_ids = target_ids[~sam_supported]
    if len(fallback_ids):
        fallback_code = names[config.fallback_class][1]
        labels[fallback_ids] = fallback_code
        shares[fallback_ids] = np.maximum(shares[fallback_ids], config.fallback_confidence)
        resolved[str(fallback_code)] = resolved.get(str(fallback_code), 0) + len(fallback_ids)
    return labels, shares, {"metashape_ground_points": int(ground_hint.sum()), "resolved": resolved}
