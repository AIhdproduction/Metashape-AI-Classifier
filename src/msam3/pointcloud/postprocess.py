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


def _fill_unknown_edges(tree, points, labels, classes, rules, config, batch_size, road_code):
    """Extend a well-supported local class into nearby unclassified edge points only."""
    unknown_config = config.unknown_fill
    if not unknown_config.enabled or not rules:
        return Counter()
    snapshot = labels.copy()
    candidates = np.flatnonzero(snapshot == classes.unknown_code)
    filled = Counter()
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
        proposal = np.full(len(selected), classes.unknown_code, dtype=np.uint8)
        best_support = np.zeros(len(selected), dtype=np.intp)
        for code in rules:
            support = np.sum(valid & (neighbour_labels == code), axis=1)
            better = support > best_support
            proposal[better] = code
            best_support[better] = support[better]
        accepted = best_support >= unknown_config.min_same_neighbors
        accepted &= best_support >= np.ceil(valid_count * unknown_config.min_share)
        if road_code is not None and np.any(accepted & (proposal == road_code)):
            road_positions = np.flatnonzero(accepted & (proposal == road_code))
            slopes = _slopes(tree, points, selected[road_positions], config, batch_size)
            accepted[road_positions] &= slopes <= config.road.max_slope_degrees
        for code in np.unique(proposal[accepted]):
            filled[int(code)] += int((proposal[accepted] == code).sum())
        labels[selected[accepted]] = proposal[accepted]
    return filled


def _fill_unknown_planar_edges(points, labels, classes, config, batch_size):
    """Assign an unknown point only when it lies on one nearby labelled surface plane."""
    plane_config = config.planar_unknown_fill
    if not plane_config.enabled:
        return Counter()
    available = {item.name: item.code for item in classes.classes}
    names = [name for name in plane_config.class_names if name in available]
    snapshot = labels.copy()
    candidates = np.flatnonzero(snapshot == classes.unknown_code)
    if not names or not len(candidates):
        return Counter()

    proposal = np.full(len(candidates), classes.unknown_code, dtype=np.uint8)
    best_support = np.zeros(len(candidates), dtype=np.intp)
    best_residual = np.full(len(candidates), np.inf, dtype=np.float64)
    for name in names:
        code = available[name]
        support_ids = np.flatnonzero(snapshot == code)
        if len(support_ids) < plane_config.min_same_neighbors:
            continue
        support_tree = cKDTree(points[support_ids])
        neighbour_count = min(plane_config.neighbors, len(support_ids))
        for start in range(0, len(candidates), batch_size):
            selected = candidates[start : start + batch_size]
            distances, nearest = support_tree.query(
                points[selected],
                k=neighbour_count,
                distance_upper_bound=plane_config.radius_m,
                workers=-1,
            )
            distances = np.asarray(distances).reshape(len(selected), -1)
            nearest = np.asarray(nearest).reshape(len(selected), -1)
            valid = (nearest < len(support_ids)) & np.isfinite(distances)
            valid_count = valid.sum(axis=1)
            usable = valid_count >= plane_config.min_same_neighbors
            if not np.any(usable):
                continue
            safe = np.minimum(nearest, len(support_ids) - 1)
            neighbours = points[support_ids[safe]].astype(np.float64, copy=False)
            mean = (neighbours * valid[:, :, None]).sum(axis=1) / np.maximum(valid_count, 1)[
                :, None
            ]
            centered = (neighbours - mean[:, None, :]) * valid[:, :, None]
            covariance = np.einsum("bki,bkj->bij", centered, centered)
            covariance /= np.maximum(valid_count, 1)[:, None, None]
            eigenvalues, vectors = np.linalg.eigh(covariance)
            normal = vectors[:, :, 0]
            residual = np.abs(np.sum((points[selected] - mean) * normal, axis=1))
            planar = eigenvalues[:, 1] >= plane_config.min_planar_spread_m**2
            accepted = usable & planar & (residual <= plane_config.max_plane_offset_m)
            global_positions = start + np.flatnonzero(accepted)
            stronger = valid_count[accepted] > best_support[global_positions]
            equal_support = valid_count[accepted] == best_support[global_positions]
            closer = residual[accepted] < best_residual[global_positions]
            replace = stronger | (equal_support & closer)
            positions = global_positions[replace]
            proposal[positions] = code
            best_support[positions] = valid_count[accepted][replace]
            best_residual[positions] = residual[accepted][replace]
    accepted = proposal != classes.unknown_code
    labels[candidates[accepted]] = proposal[accepted]
    return Counter(int(code) for code in proposal[accepted])


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


def _terrain_dem_nodes(points, terrain_ids, config):
    """Build a bounded in-memory 2.5D terrain mesh from the requested class codes."""
    grid_size = config.dem_grid_size_m
    while True:
        cells = np.floor(points[terrain_ids, :2] / grid_size).astype(np.int64)
        unique, inverse = np.unique(cells, axis=0, return_inverse=True)
        if len(unique) <= config.max_dem_nodes:
            break
        grid_size *= np.sqrt(len(unique) / config.max_dem_nodes) * 1.01
    heights = np.full(len(unique), np.inf, dtype=np.float64)
    # The lowest support in each cell prevents curbs, fences, and wires from lifting the DTM.
    np.minimum.at(heights, inverse, points[terrain_ids, 2])
    nodes = np.column_stack(((unique.astype(np.float64) + 0.5) * grid_size, heights))
    return nodes


def _sample_dem_heights(points, candidate_ids, nodes, config, batch_size):
    """Sample local terrain planes from an in-memory DTM without writing temporary files."""
    heights = np.full(len(candidate_ids), np.nan, dtype=np.float64)
    supported_surface = np.zeros(len(candidate_ids), dtype=bool)
    if len(nodes) < config.min_dem_neighbors or not len(candidate_ids):
        return heights, supported_surface

    dem_tree = cKDTree(nodes[:, :2])
    neighbour_count = min(config.interpolation_neighbors, len(nodes))
    for start in range(0, len(candidate_ids), batch_size):
        selected = candidate_ids[start : start + batch_size]
        distances, indices = dem_tree.query(
            points[selected, :2],
            k=neighbour_count,
            distance_upper_bound=config.interpolation_radius_m,
            workers=-1,
        )
        distances = np.asarray(distances).reshape(len(selected), -1)
        indices = np.asarray(indices).reshape(len(selected), -1)
        valid = (indices < len(nodes)) & np.isfinite(distances)
        usable = valid.sum(axis=1) >= config.min_dem_neighbors
        if not np.any(usable):
            continue
        safe = np.minimum(indices, len(nodes) - 1)
        selected_usable = selected[usable]
        neighbours = nodes[safe][usable]
        neighbour_xy = neighbours[:, :, :2]
        neighbour_z = neighbours[:, :, 2]
        usable_valid = valid[usable]
        design, coefficients = _fit_height_planes(neighbour_xy, neighbour_z, usable_valid)
        residual = np.abs(neighbour_z - np.einsum("bki,bi->bk", design, coefficients))
        inliers = usable_valid & (residual <= config.max_dem_residual_m)
        supported = inliers.sum(axis=1) >= config.min_dem_neighbors
        if np.any(supported):
            coefficients[supported] = _fit_height_planes(neighbour_xy, neighbour_z, inliers)[1][
                supported
            ]
        candidate_design = np.column_stack(
            (points[selected_usable, :2], np.ones(len(selected_usable)))
        )
        sampled_height = points[selected_usable, 2] - np.einsum(
            "bi,bi->b", candidate_design, coefficients
        )
        slopes = np.degrees(np.arctan(np.linalg.norm(coefficients[:, :2], axis=1)))
        accepted = supported & (slopes <= config.max_dem_slope_degrees)
        accepted_positions = start + np.flatnonzero(usable)[accepted]
        heights[accepted_positions] = sampled_height[accepted]
        supported_surface[accepted_positions] = True
    return heights, supported_surface


def _split_vegetation_height(points, labels, shares, classes, config, batch_size):
    """Split vegetation by height above local terrain without radial tree expansion."""
    vegetation = config.vegetation
    available = {item.name for item in classes.classes}
    required = {vegetation.low_vegetation_class, vegetation.high_vegetation_class}
    empty = {
        "dem_nodes": 0,
        "trusted_dem_nodes": 0,
        "fallback_dem_nodes": 0,
        "fallback_points_used": 0,
        "strong_sam_high_points": 0,
        "reclassified": {},
    }
    if not vegetation.enabled or not required.issubset(available):
        return empty
    codes = _class_codes(
        classes, [vegetation.low_vegetation_class, vegetation.high_vegetation_class]
    )
    low_code = codes[vegetation.low_vegetation_class]
    high_code = codes[vegetation.high_vegetation_class]
    snapshot = labels.copy()
    trusted_codes = np.asarray(vegetation.terrain_class_codes, dtype=np.uint8)
    fallback_codes = np.asarray(vegetation.fallback_terrain_class_codes, dtype=np.uint8)
    trusted_ids = np.flatnonzero(np.isin(snapshot, trusted_codes))
    fallback_ids = np.flatnonzero(np.isin(snapshot, fallback_codes))
    candidates = np.flatnonzero(np.isin(snapshot, [low_code, high_code]))
    if not len(candidates):
        return empty

    trusted_nodes = (
        _terrain_dem_nodes(points, trusted_ids, vegetation)
        if len(trusted_ids) >= vegetation.min_dem_neighbors
        else np.empty((0, 3), dtype=np.float64)
    )
    # Wires and guard rails are only a reserve. They are consulted after trusted terrain
    # (ground, water, road, sidewalk, curb) could not support a local DTM sample.
    all_terrain_ids = np.flatnonzero(
        np.isin(snapshot, np.concatenate((trusted_codes, fallback_codes)))
    )
    fallback_nodes = (
        _terrain_dem_nodes(points, all_terrain_ids, vegetation)
        if len(fallback_ids) and len(all_terrain_ids) >= vegetation.min_dem_neighbors
        else np.empty((0, 3), dtype=np.float64)
    )
    if (
        len(trusted_nodes) < vegetation.min_dem_neighbors
        and len(fallback_nodes) < vegetation.min_dem_neighbors
    ):
        return empty

    heights, valid_surface = _sample_dem_heights(
        points, candidates, trusted_nodes, vegetation, batch_size
    )
    fallback_points_used = 0
    missing = np.flatnonzero(~valid_surface)
    if len(missing) and len(fallback_nodes) >= vegetation.min_dem_neighbors:
        fallback_heights, fallback_valid = _sample_dem_heights(
            points, candidates[missing], fallback_nodes, vegetation, batch_size
        )
        accepted_missing = missing[fallback_valid]
        heights[accepted_missing] = fallback_heights[fallback_valid]
        valid_surface[accepted_missing] = True
        fallback_points_used = int(len(accepted_missing))

    valid_heights = valid_surface & np.isfinite(heights)
    strong_sam_high = (
        (snapshot[candidates] == high_code)
        & valid_heights
        & (shares[candidates] >= vegetation.sam_high_min_share)
        & (heights >= vegetation.sam_high_min_height_m)
    )
    tolerance = vegetation.high_vegetation_min_height_m * vegetation.threshold_tolerance
    proposed = snapshot[candidates].copy()
    # Geometry determines the general split. This leaves no fixed-radius halo around a tree.
    proposed[valid_surface & (heights < vegetation.high_vegetation_min_height_m - tolerance)] = (
        low_code
    )
    proposed[valid_surface & (heights >= vegetation.high_vegetation_min_height_m + tolerance)] = (
        high_code
    )
    # Preserve the actual SAM mask of a reliable hedge/tree near the threshold. Unlike the old
    # radius rule, this never expands into a circular area beyond what SAM identified.
    proposed[strong_sam_high] = high_code
    replace = labels[candidates] != proposed
    changed = Counter(int(code) for code in proposed[replace])
    labels[candidates[replace]] = proposed[replace]
    return {
        "dem_nodes": int(len(trusted_nodes) + len(fallback_nodes)),
        "trusted_dem_nodes": int(len(trusted_nodes)),
        "fallback_dem_nodes": int(len(fallback_nodes)),
        "fallback_points_used": fallback_points_used,
        "strong_sam_high_points": int(strong_sam_high.sum()),
        "reclassified": dict(sorted(changed.items())),
    }


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
    vegetation_height = _split_vegetation_height(
        points, labels, shares, classes, config, batch_size
    )
    removed = _remove_isolated(
        tree, points, labels, shares, classes.unknown_code, rules, config, batch_size
    )
    road_rejected, road_code = _restrict_road_slope(
        tree, points, labels, classes, config, batch_size
    )
    road_surface = _repair_road_surface(points, labels, classes, config, batch_size, road_code)
    filled = _fill_small_holes(tree, points, labels, shares, rules, config, batch_size, road_code)
    unknown_edges = _fill_unknown_edges(
        tree, points, labels, classes, rules, config, batch_size, road_code
    )
    planar_edges = _fill_unknown_planar_edges(points, labels, classes, config, batch_size)
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
            "unknown_edges_filled": dict(sorted(unknown_edges.items())),
            "planar_unknown_edges_filled": dict(sorted(planar_edges.items())),
            "flat_roof_ground_to_building": roofs,
            "ground_height_to_unknown": ground_height_rejected,
            "vegetation_dem": vegetation_height,
        },
    )
    return labels
