"""Conservative, configurable geometry cleanup after SAM image voting."""

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from msam3.cache import atomic_json
from msam3.config import Classes, PostprocessConfig


def _class_codes(classes: Classes, names) -> dict[str, int]:
    available = {item.name: item.code for item in classes.classes}
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError("postprocess refers to unknown classes: " + ", ".join(unknown))
    return {name: available[name] for name in names}


def _neighbours(tree, points, ids, count, radius, batch_size):
    """Yield metric neighbour indices without materialising a full point-cloud matrix."""
    for start in range(0, len(ids), batch_size):
        selected = ids[start : start + batch_size]
        distances, indices = tree.query(
            points[selected], k=count + 1, distance_upper_bound=radius, workers=-1
        )
        yield selected, distances[:, 1:], indices[:, 1:]


def _support(labels, indices, distances, code):
    valid = (indices < len(labels)) & np.isfinite(distances)
    safe = np.minimum(indices, len(labels) - 1)
    return np.sum(valid & (labels[safe] == code), axis=1), valid


def _slopes(tree, points, ids, config: PostprocessConfig, batch_size):
    """Estimate local plane slope from metric k-nearest neighbours in degrees."""
    # A sparse neighbourhood has no trustworthy plane. It must not be treated as vertical.
    slopes = np.full(len(ids), np.nan, dtype=np.float32)
    cursor = 0
    for selected, distances, indices in _neighbours(
        tree,
        points,
        ids,
        config.road.plane_neighbors,
        config.neighborhood_radius_m,
        batch_size,
    ):
        valid = (indices < len(points)) & np.isfinite(distances)
        safe = np.minimum(indices, len(points) - 1)
        neighbours = points[safe].astype(np.float64, copy=False)
        center = points[selected].astype(np.float64, copy=False)
        offsets = neighbours - center[:, None, :]
        valid_count = valid.sum(axis=1)
        usable = valid_count >= 3
        if np.any(usable):
            masked = offsets * valid[:, :, None]
            covariance = np.einsum("bki,bkj->bij", masked, masked)
            covariance /= np.maximum(valid_count, 1)[:, None, None]
            _, vectors = np.linalg.eigh(covariance[usable])
            normal = vectors[:, :, 0]
            vertical = np.clip(np.abs(normal[:, 2]), 0, 1)
            slopes[cursor : cursor + len(selected)][usable] = np.degrees(np.arccos(vertical))
        cursor += len(selected)
    return slopes


def _remove_isolated(tree, points, labels, shares, unknown, rules, config, batch_size):
    removed = Counter()
    snapshot = labels.copy()
    for code, rule in rules.items():
        if not rule.min_same_neighbors:
            continue
        ids = np.flatnonzero(snapshot == code)
        for selected, distances, indices in _neighbours(
            tree,
            points,
            ids,
            config.neighbors,
            config.neighborhood_radius_m,
            batch_size,
        ):
            support, _ = _support(snapshot, indices, distances, code)
            reject = (support < rule.min_same_neighbors) & (
                shares[selected] < config.npi_preserve_confidence
            )
            if np.any(reject):
                labels[selected[reject]] = unknown
                removed[int(code)] += int(reject.sum())
    return removed


def _fill_small_holes(tree, points, labels, shares, rules, config, batch_size, road_code):
    filled = Counter()
    if not rules:
        return filled
    snapshot = labels.copy()
    maximum = max(rule.max_replaced_confidence for rule in rules.values())
    candidates = np.flatnonzero(shares <= maximum)
    for selected, distances, indices in _neighbours(
        tree,
        points,
        candidates,
        config.neighbors,
        config.neighborhood_radius_m,
        batch_size,
    ):
        valid = (indices < len(labels)) & np.isfinite(distances)
        safe = np.minimum(indices, len(labels) - 1)
        neighbour_labels = snapshot[safe]
        valid_count = valid.sum(axis=1)
        proposal = labels[selected].copy()
        for code, rule in rules.items():
            support = np.sum(valid & (neighbour_labels == code), axis=1)
            accepted = support >= rule.fill_neighbors
            accepted &= support >= np.ceil(valid_count * rule.fill_share)
            accepted &= shares[selected] <= rule.max_replaced_confidence
            if code == road_code and np.any(accepted):
                slopes = _slopes(tree, points, selected[accepted], config, batch_size)
                accepted_ids = np.flatnonzero(accepted)
                accepted[accepted_ids] &= slopes <= config.road.max_slope_degrees
            replace = accepted & (proposal != code)
            proposal[replace] = code
            filled[int(code)] += int(replace.sum())
        labels[selected] = proposal
    return filled


def _repair_flat_roofs(tree, points, labels, classes, config, batch_size):
    if not config.building.enabled:
        return 0
    names = [config.building.ground_class, config.building.building_class]
    available = {item.name for item in classes.classes}
    if not set(names).issubset(available):
        return 0
    ground, building = _class_codes(classes, names).values()
    snapshot = labels.copy()
    candidates = np.flatnonzero(snapshot == ground)
    repaired = 0
    # A roof is intentionally above the facade. Use XY distance here, rather than a 3D radius.
    horizontal_tree = cKDTree(points[:, :2])
    for start in range(0, len(candidates), batch_size):
        selected = candidates[start : start + batch_size]
        distances, indices = horizontal_tree.query(
            points[selected, :2],
            k=config.neighbors + 1,
            distance_upper_bound=config.building.horizontal_radius_m,
            workers=-1,
        )
        distances, indices = distances[:, 1:], indices[:, 1:]
        valid = (indices < len(labels)) & np.isfinite(distances)
        safe = np.minimum(indices, len(labels) - 1)
        building_neighbours = valid & (snapshot[safe] == building)
        vertical_drop = points[selected, 2, None] - points[safe, 2]
        support = np.sum(
            building_neighbours & (vertical_drop >= config.building.min_vertical_drop_m), axis=1
        )
        replace = support >= config.building.min_building_neighbors
        labels[selected[replace]] = building
        repaired += int(replace.sum())
    return repaired


def _fit_height_planes(neighbour_xy, neighbour_z, valid):
    """Return z=ax+by+c fits for one batch of valid local terrain neighbours."""
    design = np.concatenate((neighbour_xy, np.ones((*neighbour_xy.shape[:2], 1))), axis=2)
    weighted = design * valid[:, :, None]
    matrix = np.einsum("bki,bkj->bij", weighted, weighted)
    rhs = np.einsum("bki,bk->bi", weighted, neighbour_z * valid)
    coefficients = np.einsum("bij,bj->bi", np.linalg.pinv(matrix), rhs)
    return design, coefficients


def _reject_ground_height_outliers(points, labels, classes, config, batch_size):
    """Remove ground labels that are vertically detached from a supported local plane."""
    ground_config = config.ground
    if not ground_config.enabled or ground_config.ground_class not in {
        item.name for item in classes.classes
    }:
        return 0
    ground_code = _class_codes(classes, [ground_config.ground_class])[ground_config.ground_class]
    snapshot = labels.copy()
    ground_ids = np.flatnonzero(snapshot == ground_code)
    if len(ground_ids) <= ground_config.min_ground_neighbors:
        return 0

    horizontal_tree = cKDTree(points[ground_ids, :2])
    rejected = 0
    neighbour_count = min(ground_config.neighbors + 1, len(ground_ids))
    for start in range(0, len(ground_ids), batch_size):
        selected = ground_ids[start : start + batch_size]
        distances, indices = horizontal_tree.query(
            points[selected, :2],
            k=neighbour_count,
            distance_upper_bound=ground_config.horizontal_radius_m,
            workers=-1,
        )
        distances = np.asarray(distances).reshape(len(selected), -1)
        indices = np.asarray(indices).reshape(len(selected), -1)
        valid = (indices < len(ground_ids)) & np.isfinite(distances)
        safe = np.minimum(indices, len(ground_ids) - 1)
        neighbour_ids = ground_ids[safe]
        # Do not let a candidate itself define the plane it is tested against.
        valid &= neighbour_ids != selected[:, None]
        neighbours = points[neighbour_ids].astype(np.float64, copy=False)
        valid_count = valid.sum(axis=1)
        usable = valid_count >= ground_config.min_ground_neighbors
        if not np.any(usable):
            continue
        selected_usable = selected[usable]
        neighbour_xy = neighbours[usable, :, :2]
        neighbour_z = neighbours[usable, :, 2]
        usable_valid = valid[usable]

        # z=ax+by+c deliberately refuses vertical planes: a vertically stacked artefact
        # must not be able to validate itself as ground.
        design, coefficients = _fit_height_planes(neighbour_xy, neighbour_z, usable_valid)
        neighbour_residual = np.abs(neighbour_z - np.einsum("bki,bi->bk", design, coefficients))
        inliers = usable_valid & (neighbour_residual <= ground_config.max_offset_m)
        inlier_count = inliers.sum(axis=1)
        can_refit = inlier_count >= ground_config.min_ground_neighbors
        if np.any(can_refit):
            coefficients[can_refit] = _fit_height_planes(neighbour_xy, neighbour_z, inliers)[1][
                can_refit
            ]
        candidate_design = np.column_stack(
            (points[selected_usable, :2], np.ones(len(selected_usable)))
        )
        residual = np.abs(
            points[selected_usable, 2] - np.einsum("bi,bi->b", candidate_design, coefficients)
        )
        slopes = np.degrees(np.arctan(np.linalg.norm(coefficients[:, :2], axis=1)))
        reject = (
            (slopes <= ground_config.max_plane_slope_degrees)
            & (residual > ground_config.max_offset_m)
            & can_refit
        )
        labels[selected_usable[reject]] = classes.unknown_code
        rejected += int(reject.sum())
    return rejected


def _split_vegetation_height(points, labels, classes, config, batch_size):
    """Classify low/high vegetation from terrain-relative height without changing other classes."""
    vegetation = config.vegetation
    available = {item.name for item in classes.classes}
    required = {vegetation.low_vegetation_class, vegetation.high_vegetation_class}
    if not vegetation.enabled or not required.issubset(available):
        return Counter()
    terrain_names = [name for name in vegetation.terrain_classes if name in available]
    if not terrain_names:
        return Counter()
    codes = _class_codes(
        classes, [vegetation.low_vegetation_class, vegetation.high_vegetation_class, *terrain_names]
    )
    low_code = codes[vegetation.low_vegetation_class]
    high_code = codes[vegetation.high_vegetation_class]
    snapshot = labels.copy()
    terrain_codes = np.array([codes[name] for name in terrain_names], dtype=np.uint8)
    terrain_ids = np.flatnonzero(np.isin(snapshot, terrain_codes))
    candidates = np.flatnonzero(np.isin(snapshot, [low_code, high_code]))
    if len(terrain_ids) < vegetation.min_terrain_neighbors or not len(candidates):
        return Counter()

    terrain_tree = cKDTree(points[terrain_ids, :2])
    neighbour_count = min(vegetation.neighbors, len(terrain_ids))
    changed = Counter()
    for start in range(0, len(candidates), batch_size):
        selected = candidates[start : start + batch_size]
        distances, indices = terrain_tree.query(
            points[selected, :2],
            k=neighbour_count,
            distance_upper_bound=vegetation.horizontal_radius_m,
            workers=-1,
        )
        distances = np.asarray(distances).reshape(len(selected), -1)
        indices = np.asarray(indices).reshape(len(selected), -1)
        valid = (indices < len(terrain_ids)) & np.isfinite(distances)
        valid_count = valid.sum(axis=1)
        usable = valid_count >= vegetation.min_terrain_neighbors
        if not np.any(usable):
            continue
        safe = np.minimum(indices, len(terrain_ids) - 1)
        neighbours = points[terrain_ids[safe]].astype(np.float64, copy=False)
        selected_usable = selected[usable]
        neighbour_xy = neighbours[usable, :, :2]
        neighbour_z = neighbours[usable, :, 2]
        usable_valid = valid[usable]
        design, coefficients = _fit_height_planes(neighbour_xy, neighbour_z, usable_valid)
        neighbour_residual = np.abs(neighbour_z - np.einsum("bki,bi->bk", design, coefficients))
        inliers = usable_valid & (neighbour_residual <= vegetation.max_terrain_residual_m)
        inlier_count = inliers.sum(axis=1)
        supported = inlier_count >= vegetation.min_terrain_neighbors
        if np.any(supported):
            coefficients[supported] = _fit_height_planes(neighbour_xy, neighbour_z, inliers)[1][
                supported
            ]
        candidate_design = np.column_stack(
            (points[selected_usable, :2], np.ones(len(selected_usable)))
        )
        height = points[selected_usable, 2] - np.einsum("bi,bi->b", candidate_design, coefficients)
        slopes = np.degrees(np.arctan(np.linalg.norm(coefficients[:, :2], axis=1)))
        valid_surface = supported & (slopes <= vegetation.max_terrain_slope_degrees)
        proposed = np.where(
            height >= vegetation.high_vegetation_min_height_m, high_code, low_code
        ).astype(np.uint8)
        replace = valid_surface & (labels[selected_usable] != proposed)
        for code in np.unique(proposed[replace]):
            changed[int(code)] += int((proposed[replace] == code).sum())
        labels[selected_usable[replace]] = proposed[replace]
    return changed


def _restrict_road_slope(tree, points, labels, classes, config, batch_size):
    if not config.road.enabled:
        return 0, None
    if config.road.road_class not in {item.name for item in classes.classes}:
        return 0, None
    road = _class_codes(classes, [config.road.road_class])[config.road.road_class]
    ids = np.flatnonzero(labels == road)
    if not len(ids):
        return 0, road
    slopes = _slopes(tree, points, ids, config, batch_size)
    rejected = np.isfinite(slopes) & (slopes > config.road.max_slope_degrees)
    labels[ids[rejected]] = classes.unknown_code
    return int(rejected.sum()), road


def _repair_road_surface(points, labels, classes, config, batch_size, road_code):
    """Replace configured false classes only inside a locally enclosing road plane."""
    road_config = config.road
    if not road_config.enabled or road_code is None:
        return 0
    names = [road_config.road_class, *road_config.surface_replace_classes]
    codes = _class_codes(classes, names)
    target_codes = [
        codes[name] for name in road_config.surface_replace_classes if codes[name] != road_code
    ]
    if road_config.fill_unknown_surface:
        target_codes.append(classes.unknown_code)
    targets = np.array(target_codes, dtype=np.uint8)
    road_ids = np.flatnonzero(labels == road_code)
    candidates = np.flatnonzero(np.isin(labels, targets))
    if not len(road_ids) or not len(candidates):
        return 0
    road_tree = cKDTree(points[road_ids, :2])
    repaired = 0
    neighbour_count = min(road_config.surface_neighbors, len(road_ids))
    for start in range(0, len(candidates), batch_size):
        candidate_ids = candidates[start : start + batch_size]
        candidate_xy = points[candidate_ids, :2]
        nearest_distance, _ = road_tree.query(
            candidate_xy,
            k=1,
            distance_upper_bound=road_config.surface_radius_m,
            workers=-1,
        )
        nearby = np.flatnonzero(np.isfinite(nearest_distance))
        if not len(nearby):
            continue
        selected = candidate_ids[nearby]
        distances, nearest = road_tree.query(
            points[selected, :2],
            k=neighbour_count,
            distance_upper_bound=road_config.surface_radius_m,
            workers=-1,
        )
        distances = np.asarray(distances).reshape(len(selected), -1)
        nearest = np.asarray(nearest).reshape(len(selected), -1)
        valid = (nearest < len(road_ids)) & np.isfinite(distances)
        safe = np.minimum(nearest, len(road_ids) - 1)
        neighbours = points[road_ids[safe]].astype(np.float64, copy=False)
        valid_count = valid.sum(axis=1)
        mean = (neighbours * valid[:, :, None]).sum(axis=1) / np.maximum(valid_count, 1)[:, None]
        centered = (neighbours - mean[:, None, :]) * valid[:, :, None]
        covariance = np.einsum("bki,bkj->bij", centered, centered)
        covariance /= np.maximum(valid_count, 1)[:, None, None]
        _, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, :, 0]
        slopes = np.degrees(np.arccos(np.clip(np.abs(normal[:, 2]), 0, 1)))
        offsets = neighbours[:, :, :2] - points[selected, None, :2]
        angles = np.floor((np.arctan2(offsets[:, :, 1], offsets[:, :, 0]) + np.pi) * 4 / np.pi)
        sectors = np.mod(angles.astype(np.int8), 8)
        coverage = np.zeros(len(selected), dtype=np.int8)
        for sector in range(8):
            coverage += np.any(valid & (sectors == sector), axis=1)
        residual = np.abs(np.sum((points[selected] - mean) * normal, axis=1))
        accepted = valid_count >= road_config.min_surface_road_neighbors
        accepted &= coverage >= road_config.min_surface_coverage_sectors
        accepted &= slopes <= road_config.max_slope_degrees
        accepted &= residual <= road_config.max_surface_offset_m
        labels[selected[accepted]] = road_code
        repaired += int(accepted.sum())
    return repaired


def apply_postprocess(points, labels, shares, classes, config, directory, batch_size):
    """Apply NPI outlier removal plus ground, vegetation, roof, and road geometry rules."""
    labels = np.asarray(labels, dtype=np.uint8).copy()
    if not config.enabled or not len(labels):
        return labels
    codes = _class_codes(classes, config.class_rules)
    rules = {codes[name]: rule for name, rule in config.class_rules.items()}
    tree = cKDTree(points)
    before = Counter(int(code) for code in labels)
    roofs = _repair_flat_roofs(tree, points, labels, classes, config, batch_size)
    ground_height_rejected = _reject_ground_height_outliers(
        points, labels, classes, config, batch_size
    )
    vegetation_height = _split_vegetation_height(points, labels, classes, config, batch_size)
    removed = _remove_isolated(
        tree, points, labels, shares, classes.unknown_code, rules, config, batch_size
    )
    road_rejected, road_code = _restrict_road_slope(
        tree, points, labels, classes, config, batch_size
    )
    road_surface = _repair_road_surface(points, labels, classes, config, batch_size, road_code)
    filled = _fill_small_holes(tree, points, labels, shares, rules, config, batch_size, road_code)
    output = Path(directory) / config.output_subdir
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "postprocess_report.json",
        {
            "before_counts": dict(sorted(before.items())),
            "after_counts": dict(sorted(Counter(int(code) for code in labels).items())),
            "npi_removed": dict(sorted(removed.items())),
            "road_slope_removed": road_rejected,
            "road_surface_to_road": road_surface,
            "holes_filled": dict(sorted(filled.items())),
            "flat_roof_ground_to_building": roofs,
            "ground_height_to_unknown": ground_height_rejected,
            "vegetation_height_reclassified": dict(sorted(vegetation_height.items())),
        },
    )
    return labels
