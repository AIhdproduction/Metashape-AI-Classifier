"""Conservative, configurable geometry cleanup after image voting."""

import logging
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from msam3.cache import atomic_json
from msam3.config import Classes, PostprocessConfig

LOG = logging.getLogger(__name__)


def _class_codes(classes: Classes, names) -> dict[str, int]:
    available = {item.name: item.code for item in classes.classes}
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError("postprocess refers to unknown classes: " + ", ".join(unknown))
    return {name: available[name] for name in names}


def _available_codes(classes: Classes, names) -> dict[str, int]:
    """Codes of the named classes that exist: a disabled class only switches off its own step."""
    available = {item.name: item.code for item in classes.classes}
    return {name: available[name] for name in names if name in available}


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


def _remove_unclustered(tree, points, labels, classes, config, batch_size):
    """Remove every labelled point that lacks a local cluster of its own class."""
    cluster = config.cluster
    if not cluster.enabled:
        return Counter()
    removed = Counter()
    snapshot = labels.copy()
    for item in classes.classes:
        code = item.code
        required = cluster.min_same_overrides.get(item.name, cluster.min_same_neighbors)
        ids = np.flatnonzero(snapshot == code)
        if not len(ids):
            continue
        for selected, distances, indices in _neighbours(
            tree,
            points,
            ids,
            cluster.neighbors,
            cluster.radius_m,
            batch_size,
        ):
            support, _ = _support(snapshot, indices, distances, code)
            reject = support < required
            if np.any(reject):
                labels[selected[reject]] = classes.unknown_code
                removed[int(code)] += int(reject.sum())
    return removed


def _remove_isolated(tree, points, labels, unknown, rules, config, batch_size):
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
            # A confident single point is still a reconstruction artefact. Confidence
            # describes the model's image vote, not whether a 3D point belongs to an object.
            reject = support < rule.min_same_neighbors
            if np.any(reject):
                labels[selected[reject]] = unknown
                removed[int(code)] += int(reject.sum())
    return removed


def _fill_small_holes(
    tree, points, labels, shares, rules, config, batch_size, road_code, protected=()
):
    filled = Counter()
    if not rules:
        return filled
    snapshot = labels.copy()
    maximum = max(rule.max_replaced_confidence for rule in rules.values())
    replaceable = shares <= maximum
    if protected:
        replaceable &= ~np.isin(snapshot, list(protected))
    candidates = np.flatnonzero(replaceable)
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


def _repair_flat_roofs(points, labels, classes, config, batch_size):
    if not config.building.enabled:
        return 0
    names = [config.building.ground_class, config.building.building_class]
    available = {item.name for item in classes.classes}
    if not set(names).issubset(available):
        return 0
    ground, building = _class_codes(classes, names).values()
    snapshot = labels.copy()
    candidates = np.flatnonzero(snapshot == ground)
    building_ids = np.flatnonzero(snapshot == building)
    if not len(candidates) or not len(building_ids):
        return 0
    repaired = 0
    # Query only known building points. Querying the whole cloud would let nearby roof
    # points consume the neighbour slots and hide the building facade below a wide roof.
    horizontal_tree = cKDTree(points[building_ids, :2])
    neighbour_count = min(config.neighbors, len(building_ids))
    for start in range(0, len(candidates), batch_size):
        selected = candidates[start : start + batch_size]
        distances, indices = horizontal_tree.query(
            points[selected, :2],
            k=neighbour_count,
            distance_upper_bound=config.building.horizontal_radius_m,
            workers=-1,
        )
        distances = np.asarray(distances).reshape(len(selected), -1)
        indices = np.asarray(indices).reshape(len(selected), -1)
        valid = (indices < len(building_ids)) & np.isfinite(distances)
        safe = np.minimum(indices, len(building_ids) - 1)
        neighbours = points[building_ids[safe]]
        vertical_drop = points[selected, 2, None] - neighbours[:, :, 2]
        support = np.sum(valid & (vertical_drop >= config.building.min_vertical_drop_m), axis=1)
        replace = support >= config.building.min_building_neighbors
        labels[selected[replace]] = building
        repaired += int(replace.sum())
    return repaired


def _restore_bare_ground(labels, classes, config, source_ground_hint, colors, batch_size):
    """Correct non-green low vegetation on Metashape terrain candidates."""
    rule = config.color_ground
    codes = _available_codes(classes, [rule.vegetation_class, rule.ground_class])
    if not rule.enabled or len(codes) != 2 or source_ground_hint is None or colors is None:
        return 0
    candidates = np.flatnonzero(
        (labels == codes[rule.vegetation_class]) & np.asarray(source_ground_hint, dtype=bool)
    )
    corrected = 0
    for start in range(0, len(candidates), batch_size):
        selected = candidates[start : start + batch_size]
        rgb = np.asarray(colors[selected], dtype=np.float32)
        brightness = rgb.sum(axis=1)
        excess_green = (2 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]) / np.maximum(
            brightness, rule.min_brightness * 65535
        )
        restore = (brightness >= rule.min_brightness * 65535) & (
            excess_green < rule.max_excess_green
        )
        labels[selected[restore]] = codes[rule.ground_class]
        corrected += int(restore.sum())
    return corrected


def _outside_protected(points, labels, classes, config, candidates, rule):
    """Drop candidates within ``rule.protected_clearance_m`` of a protected thin class."""
    protected = list(_available_codes(classes, config.protected_classes).values())
    if not protected or rule.protected_clearance_m <= 0 or not len(candidates):
        return candidates
    protected_ids = np.flatnonzero(np.isin(labels, protected))
    if not len(protected_ids):
        return candidates
    distance, _ = cKDTree(points[protected_ids]).query(
        points[candidates], k=1, distance_upper_bound=rule.protected_clearance_m, workers=-1
    )
    return candidates[~np.isfinite(distance)]


def _restore_other_buildings(points, labels, classes, config, batch_size):
    """Repair nearby roof and facade points predicted as the generic 'other' class."""
    rule = config.other_building
    codes = _available_codes(classes, [rule.other_class, rule.building_class])
    if not rule.enabled or len(codes) != 2:
        return 0
    candidates = np.flatnonzero(labels == codes[rule.other_class])
    building_ids = np.flatnonzero(labels == codes[rule.building_class])
    if not len(candidates) or not len(building_ids):
        return 0
    candidates = _outside_protected(points, labels, classes, config, candidates, rule)
    tree = cKDTree(points[building_ids])
    corrected = 0
    for start in range(0, len(candidates), batch_size):
        selected = candidates[start : start + batch_size]
        distance, _ = tree.query(
            points[selected], k=1, distance_upper_bound=rule.radius_m, workers=-1
        )
        restore = np.isfinite(distance)
        labels[selected[restore]] = codes[rule.building_class]
        corrected += int(restore.sum())
    return corrected


_POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.int8)


def _thin_support(points, ids, voxel):
    """Keep one target point per voxel so dense patches cannot crowd out a plane's extent."""
    if not len(ids):
        return ids
    cells = np.floor(points[ids] / voxel).astype(np.int64)
    _, first = np.unique(cells, axis=0, return_index=True)
    return ids[np.sort(first)]


def _surface_support(points, normals, support, rule, up):
    """Split target points into the requested surface kinds with their search radius."""
    horizontal = np.abs(normals[support] @ up) >= rule.normal_vertical_boundary
    choices = {
        "horizontal": (horizontal, rule.horizontal_radius_m),
        "vertical": (~horizontal, rule.vertical_radius_m),
        "any": (np.ones(len(support), dtype=bool), rule.horizontal_radius_m),
    }
    for kind in rule.surfaces:
        mask, radius = choices[kind]
        ids = _thin_support(points, support[mask], rule.support_voxel_m)
        if len(ids) >= rule.min_support_neighbors:
            yield kind, ids, radius


def _plane_review(points, normals, candidates, strict, support_ids, radius, rule, batch_size):
    """Return (accepted, residual) for candidates tested against one support surface."""
    accepted = np.zeros(len(candidates), dtype=bool)
    residuals = np.full(len(candidates), np.inf, dtype=np.float32)
    tree = cKDTree(points[support_ids])
    neighbour_count = min(rule.neighbors, len(support_ids))
    # The neighbour tensor is batch x k x 3 float64; 50,000 rows keep it near 20 MB.
    geometry_batch = max(1, min(batch_size, 50_000))
    for start in range(0, len(candidates), geometry_batch):
        stop = min(start + geometry_batch, len(candidates))
        selected = candidates[start:stop]
        distances, indices = tree.query(
            points[selected], k=neighbour_count, distance_upper_bound=radius, workers=-1
        )
        distances = np.asarray(distances).reshape(len(selected), -1)
        indices = np.asarray(indices).reshape(len(selected), -1)
        valid = np.isfinite(distances)
        count = valid.sum(axis=1)
        is_strict = strict[start:stop]
        usable = count >= rule.min_support_neighbors
        # Unclassified points must touch the surface; others may sit inside a sparse patch.
        usable &= ~is_strict | (distances[:, 0] <= rule.unknown_radius_m)
        positions = np.flatnonzero(usable)
        if not len(positions):
            continue
        selected = selected[positions]
        valid = valid[positions]
        count = count[positions]
        is_strict = is_strict[positions]
        safe = np.minimum(indices[positions], len(support_ids) - 1)
        neighbours = points[support_ids[safe]].astype(np.float64)
        weights = valid[:, :, None]
        mean = (neighbours * weights).sum(axis=1) / count[:, None]
        centered = (neighbours - mean[:, None, :]) * weights
        covariance = np.einsum("bki,bkj->bij", centered, centered) / count[:, None, None]
        eigenvalues, vectors = np.linalg.eigh(covariance)
        plane_normal = vectors[:, :, 0]
        candidate = points[selected].astype(np.float64)
        residual = np.abs(np.einsum("bi,bi->b", candidate - mean, plane_normal))
        alignment = np.abs(np.einsum("bi,bi->b", normals[selected], plane_normal))
        # Directions of the support around the candidate, measured inside the fitted plane.
        offsets = neighbours - candidate[:, None, :]
        along = np.einsum("bki,bi->bk", offsets, vectors[:, :, 2])
        across = np.einsum("bki,bi->bk", offsets, vectors[:, :, 1])
        sectors = np.floor((np.arctan2(across, along) + np.pi) * 4 / np.pi).astype(np.int64) % 8
        occupied = np.bitwise_or.reduce(np.where(valid, 1 << sectors, 0), axis=1)
        coverage = _POPCOUNT[occupied]
        good = eigenvalues[:, 1] >= rule.min_planar_spread_m**2
        good &= np.where(
            is_strict,
            (residual <= rule.unknown_max_plane_offset_m)
            & (alignment >= rule.unknown_min_normal_alignment)
            & (coverage >= rule.unknown_min_coverage_sectors),
            (residual <= rule.max_plane_offset_m)
            & (alignment >= rule.min_normal_alignment)
            & (coverage >= rule.min_coverage_sectors),
        )
        local = start + positions
        accepted[local] = good
        residuals[local[good]] = residual[good]
    return accepted, residuals


def _review_geometry_by_class(points, labels, normals, classes, config, batch_size, up=None):
    """Reassign allowed labels only where a target class supplies the same local surface.

    Every rule reads the labels from before this step, so the order of the rules does not
    matter. A point that two targets accept goes to the one whose plane fits it more closely.
    """
    if not config.geometry_rules:
        return {}
    if normals is None:
        LOG.warning("Per-class geometry review skipped: source LAS has no normals")
        return {}
    # Roofs and facades are told apart by the true vertical, not by the frame's z axis.
    up = np.array([0, 0, 1] if up is None else up, dtype=np.float32)
    names = {item.name: item.code for item in classes.classes}
    snapshot = labels.copy()
    protected = set(_available_codes(classes, config.protected_classes).values())
    best_target = np.zeros(len(labels), dtype=np.uint8)
    best_residual = np.full(len(labels), np.inf, dtype=np.float32)
    for target_name, rule in config.geometry_rules.items():
        if target_name not in names:
            continue
        target_code = names[target_name]
        if target_code in protected:
            LOG.warning("Geometry rule for protected class %s ignored", target_name)
            continue
        source_codes = [
            names[name]
            for name in rule.source_classes
            if name in names and names[name] != target_code and names[name] not in protected
        ]
        if rule.include_unknown:
            source_codes.append(classes.unknown_code)
        if not source_codes:
            continue
        candidates = np.flatnonzero(np.isin(snapshot, source_codes))
        support = np.flatnonzero(snapshot == target_code)
        if len(support) < rule.min_support_neighbors or not len(candidates):
            continue
        # A point without a usable Metashape normal cannot confirm any plane.
        candidates = candidates[np.any(normals[candidates] != 0, axis=1)]
        candidates = _outside_protected(points, snapshot, classes, config, candidates, rule)
        if len(candidates) and rule.source_surfaces:
            # Some sources count only where the point itself faces the right way, e.g. 'other'
            # on facades: flat 'other' beside a building is mostly not building.
            horizontal = np.abs(normals[candidates] @ up) >= rule.normal_vertical_boundary
            keep = np.ones(len(candidates), dtype=bool)
            for name, kinds in rule.source_surfaces.items():
                if name not in names:
                    continue
                facing = np.zeros(len(candidates), dtype=bool)
                if "any" in kinds:
                    facing[:] = True
                if "horizontal" in kinds:
                    facing |= horizontal
                if "vertical" in kinds:
                    facing |= ~horizontal
                keep &= (snapshot[candidates] != names[name]) | facing
            candidates = candidates[keep]
        if not len(candidates):
            continue
        strict = snapshot[candidates] == classes.unknown_code
        accepted = np.zeros(len(candidates), dtype=bool)
        residual = np.full(len(candidates), np.inf, dtype=np.float32)
        for _, support_ids, radius in _surface_support(points, normals, support, rule, up):
            # A candidate accepted on one surface is not tested again on the next one.
            pending = np.flatnonzero(~accepted)
            if not len(pending):
                break
            ok, fit = _plane_review(
                points,
                normals,
                candidates[pending],
                strict[pending],
                support_ids,
                radius,
                rule,
                batch_size,
            )
            accepted[pending[ok]] = True
            residual[pending[ok]] = fit[ok]
        chosen = candidates[accepted]
        closer = residual[accepted] < best_residual[chosen]
        best_target[chosen[closer]] = target_code
        best_residual[chosen[closer]] = residual[accepted][closer]
    changes = {}
    # A finite residual marks an accepted point; code 0 is a valid target as well.
    moved = np.flatnonzero(np.isfinite(best_residual))
    code_names = {code: name for name, code in names.items()}
    for target_code in np.unique(best_target[moved]):
        ids = moved[best_target[moved] == target_code]
        source_counts = Counter(int(code) for code in snapshot[ids])
        changes[code_names[int(target_code)]] = dict(sorted(source_counts.items()))
        labels[ids] = target_code
    return changes


def _fit_height_planes(neighbour_xy, neighbour_z, valid):
    """Return z=ax+by+c least-squares fits for one batch of valid local terrain neighbours.

    Centred on the neighbours' mean, the normal equations split into the mean height and a
    2x2 system for the slope, solved in closed form. This replaces a batched pseudo-inverse
    (SVD), which took about half of the vegetation filter's time on 22 million points.

    Neighbours on one line (or on one spot) determine no plane. Those rare rows still use the
    pseudo-inverse, so their results stay exactly as before. Note: in absolute coordinates it
    returns a tilt across the line that the data do not determine.
    """
    design = np.concatenate((neighbour_xy, np.ones((*neighbour_xy.shape[:2], 1))), axis=2)
    weights = valid.astype(np.float64)
    count = np.maximum(weights.sum(axis=1), 1)
    centre = np.einsum("bk,bki->bi", weights, neighbour_xy) / count[:, None]
    mean_z = np.einsum("bk,bk->b", weights, neighbour_z) / count
    dx = (neighbour_xy[:, :, 0] - centre[:, None, 0]) * weights
    dy = (neighbour_xy[:, :, 1] - centre[:, None, 1]) * weights
    dz = (neighbour_z - mean_z[:, None]) * weights
    sxx, syy, sxy = (dx * dx).sum(axis=1), (dy * dy).sum(axis=1), (dx * dy).sum(axis=1)
    sxz, syz = (dx * dz).sum(axis=1), (dy * dz).sum(axis=1)
    determinant = sxx * syy - sxy * sxy
    scale = (sxx + syy) ** 2
    regular = determinant > 1e-12 * np.maximum(scale, np.finfo(np.float64).tiny)
    safe = np.where(regular, determinant, 1)
    slope_x = (syy * sxz - sxy * syz) / safe
    slope_y = (sxx * syz - sxy * sxz) / safe
    intercept = mean_z - slope_x * centre[:, 0] - slope_y * centre[:, 1]
    coefficients = np.column_stack((slope_x, slope_y, intercept))
    degenerate = np.flatnonzero(~regular)
    if len(degenerate):
        weighted = design[degenerate] * weights[degenerate, :, None]
        matrix = np.einsum("bki,bkj->bij", weighted, weighted)
        rhs = np.einsum("bki,bk->bi", weighted, neighbour_z[degenerate] * weights[degenerate])
        coefficients[degenerate] = np.einsum("bij,bj->bi", np.linalg.pinv(matrix), rhs)
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
    # Preserve the actual mask of a reliable hedge/tree near the threshold. Unlike the old
    # radius rule, this never expands into a circular area beyond what the model identified.
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
    codes = _available_codes(classes, names)
    target_codes = [
        codes[name]
        for name in road_config.surface_replace_classes
        if name in codes and codes[name] != road_code and name not in config.protected_classes
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


def _cell_percentile_nodes(level, cell, percentile, min_points):
    """One terrain node per grid cell: the point at a low height percentile of that cell."""
    cells = np.floor(level[:, :2] / cell).astype(np.int64)
    cells -= cells.min(axis=0)
    key = cells[:, 0] * (int(cells[:, 1].max()) + 1) + cells[:, 1]
    del cells
    order = np.lexsort((level[:, 2], key))
    sorted_key = key[order]
    del key
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_key)) + 1]
    counts = np.diff(np.r_[starts, len(sorted_key)])
    chosen = order[starts + np.floor(percentile * (counts - 1)).astype(np.int64)]
    return level[chosen[counts >= min_points]].astype(np.float64)


def _node_planes(nodes, neighbours, radius, exclude_self):
    tree = cKDTree(nodes[:, :2])
    distances, indices = tree.query(
        nodes[:, :2], k=min(neighbours, len(nodes)), distance_upper_bound=radius, workers=-1
    )
    distances = np.asarray(distances).reshape(len(nodes), -1)
    indices = np.asarray(indices).reshape(len(nodes), -1)
    valid = np.isfinite(distances)
    if exclude_self:
        valid[:, 0] = False
    near = nodes[np.minimum(indices, len(nodes) - 1)]
    _, planes = _fit_height_planes(near[:, :, :2], near[:, :, 2], valid)
    planes[valid.sum(axis=1) < 4] = np.nan
    return tree, planes


def _restore_dtm_ground(points, labels, classes, config, batch_size, up=None):
    """Turn low vegetation lying on the terrain of the lowest points into ground."""
    rule = config.dtm_ground
    codes = _available_codes(classes, [rule.ground_class, rule.vegetation_class])
    support_codes = list(_available_codes(classes, rule.support_classes).values())
    empty = {"nodes": 0, "low_vegetation_to_ground": 0}
    if not rule.enabled or len(codes) != 2 or not support_codes:
        return empty
    support = np.flatnonzero(np.isin(labels, support_codes))
    if len(support) < rule.min_cell_points * 4:
        return empty
    # Heights along the true vertical: in a tilted frame "lowest" would mean the wrong thing.
    rotation = level_rotation([0, 0, 1] if up is None else up).T.astype(np.float32)
    level = np.asarray(points[support], dtype=np.float32) @ rotation
    nodes = _cell_percentile_nodes(level, rule.cell_size_m, rule.percentile, rule.min_cell_points)
    if len(nodes) < 5:
        return empty
    # A node far below its neighbours is noise under the ground; far above, a roof or a car.
    _, planes = _node_planes(nodes, 10, 3 * rule.cell_size_m, exclude_self=True)
    offset = nodes[:, 2] - (planes[:, 0] * nodes[:, 0] + planes[:, 1] * nodes[:, 1] + planes[:, 2])
    keep = ~(offset < -rule.max_node_drop_m) & ~(offset > rule.max_node_rise_m)
    nodes = nodes[keep]
    if len(nodes) < 5:
        return empty
    tree, planes = _node_planes(nodes, 9, 2.5 * rule.cell_size_m, exclude_self=False)
    slope = np.degrees(np.arctan(np.hypot(planes[:, 0], planes[:, 1])))
    planes[~(slope <= rule.max_slope_degrees)] = np.nan
    candidates = np.flatnonzero(labels[support] == codes[rule.vegetation_class])
    changed = 0
    for start in range(0, len(candidates), batch_size):
        local = candidates[start : start + batch_size]
        xyz = level[local].astype(np.float64)
        _, nearest = tree.query(xyz[:, :2], k=1, distance_upper_bound=1.5 * rule.cell_size_m)
        found = nearest < len(nodes)
        plane = planes[np.minimum(nearest, len(nodes) - 1)]
        height = xyz[:, 2] - (plane[:, 0] * xyz[:, 0] + plane[:, 1] * xyz[:, 1] + plane[:, 2])
        ground = found & (height <= rule.max_height_m) & (height >= -rule.max_depth_m)
        labels[support[local[ground]]] = codes[rule.ground_class]
        changed += int(ground.sum())
    return {"nodes": int(len(nodes)), "low_vegetation_to_ground": changed}


def _object_groups(level, link):
    """Connected groups of points closer than ``link``; returns (count, group of each point)."""
    pairs = cKDTree(level).query_pairs(link, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
        shape=(len(level), len(level)),
    )
    return connected_components(graph, directed=False)


def _check_compact_classes(tree, points, labels, classes, config, up=None):
    """Small flat classes (manhole covers): groups too long, too short or not flat move away.

    Each group is measured along the true vertical: length along its main horizontal axis
    (2nd to 98th percentile) and height span (5th to 95th percentile). A rejected group takes
    the most common other class within 0.5 m, or becomes unclassified without neighbours.
    """
    names = {item.name: item.code for item in classes.classes}
    rotation = level_rotation([0, 0, 1] if up is None else up).T.astype(np.float32)
    report = {}
    for name, rule in config.compact_classes.items():
        if name not in names:
            continue
        code = names[name]
        ids = np.flatnonzero(labels == code)
        if not len(ids):
            continue
        level = np.asarray(points[ids], dtype=np.float32) @ rotation
        count, group = _object_groups(level, rule.link_m)
        order = np.argsort(group, kind="stable")
        bounds = np.r_[0, np.flatnonzero(np.diff(group[order])) + 1, len(order)]
        rejected, reasons = [], Counter()
        for left, right in zip(bounds[:-1], bounds[1:], strict=True):
            members = order[left:right]
            part = level[members].astype(np.float64)
            flat = part[:, :2] - part[:, :2].mean(axis=0)
            if len(members) >= 3:
                axis = np.linalg.eigh(flat.T @ flat)[1][:, 1]
                along = flat @ axis
                length = np.percentile(along, 98) - np.percentile(along, 2)
            else:
                length = 0.0
            span = np.percentile(part[:, 2], 95) - np.percentile(part[:, 2], 5)
            reason = (
                "too_small"
                if length < rule.min_length_m
                else "too_long"
                if length > rule.max_length_m
                else "not_flat"
                if span > rule.max_height_span_m
                else None
            )
            if reason:
                rejected.append(members)
                reasons[reason] += 1
        vertical = rotation[:, 2]
        moved = Counter()
        for members in rejected:
            selected = ids[members]
            _, near = tree.query(points[selected], k=32, distance_upper_bound=0.5, workers=-1)
            valid = near < len(labels)
            safe = np.minimum(near, len(labels) - 1)
            # Only neighbours at the same height: a drain beside a facade lies on the ground.
            rise = (points[safe] - points[selected][:, None, :]) @ vertical
            around = labels[safe][valid & (np.abs(rise) <= 0.2)]
            around = around[(around != code) & (around != classes.unknown_code)]
            target = int(np.bincount(around).argmax()) if len(around) else int(classes.unknown_code)
            labels[selected] = target
            moved[target] += len(selected)
        report[name] = {
            "groups": int(count),
            "rejected_groups": dict(sorted(reasons.items())),
            "moved_points": dict(sorted(moved.items())),
        }
    return report


def level_rotation(up) -> np.ndarray:
    """Rotation that turns the unit vector ``up`` into +z (Rodrigues' formula)."""
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    axis = np.cross(up, [0.0, 0.0, 1.0])
    cosine = float(up[2])
    if cosine <= -1 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    skew = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float64
    )
    return np.eye(3) + skew + skew @ skew / (1 + cosine)


def apply_postprocess(
    points,
    labels,
    shares,
    classes,
    config,
    directory,
    batch_size,
    *,
    source_ground_hint=None,
    colors=None,
    normals=None,
    up=None,
    up_info=None,
):
    """Apply geometry, cluster, and surface cleanup after image voting.

    ``up`` is the true vertical in the frame of ``points``. Without it, +z is taken as up.
    """
    labels = np.asarray(labels, dtype=np.uint8).copy()
    if not config.enabled or not len(labels):
        return labels
    if up is not None and config.level_to_vertical:
        # A temporary levelled copy: heights, slopes and DEMs then refer to the true vertical.
        rotation = level_rotation(up).T.astype(np.float32)
        points = np.asarray(points, dtype=np.float32) @ rotation
        normals = None if normals is None else np.asarray(normals, dtype=np.float32) @ rotation
        up = None
    codes = _available_codes(classes, config.class_rules)
    skipped = sorted(set(config.class_rules) - set(codes))
    if skipped:
        LOG.info("Post-filter rules skipped, classes not in this run: %s", ", ".join(skipped))
    rules = {codes[name]: rule for name, rule in config.class_rules.items() if name in codes}
    protected = set(_available_codes(classes, config.protected_classes).values())
    tree = cKDTree(points)
    before = Counter(int(code) for code in labels)
    bare_ground = _restore_bare_ground(
        labels, classes, config, source_ground_hint, colors, batch_size
    )
    roofs = _repair_flat_roofs(points, labels, classes, config, batch_size)
    other_buildings = _restore_other_buildings(points, labels, classes, config, batch_size)
    ground_height_rejected = _reject_ground_height_outliers(
        points, labels, classes, config, batch_size
    )
    vegetation_height = _split_vegetation_height(
        points, labels, shares, classes, config, batch_size
    )
    cluster_removed = _remove_unclustered(tree, points, labels, classes, config, batch_size)
    removed = _remove_isolated(
        tree, points, labels, classes.unknown_code, rules, config, batch_size
    )
    road_rejected, road_code = _restrict_road_slope(
        tree, points, labels, classes, config, batch_size
    )
    road_surface = _repair_road_surface(points, labels, classes, config, batch_size, road_code)
    filled = _fill_small_holes(
        tree, points, labels, shares, rules, config, batch_size, road_code, protected
    )
    unknown_edges = _fill_unknown_edges(
        tree, points, labels, classes, rules, config, batch_size, road_code
    )
    planar_edges = _fill_unknown_planar_edges(points, labels, classes, config, batch_size)
    compact = _check_compact_classes(tree, points, labels, classes, config, up)
    geometry_changes = _review_geometry_by_class(
        points, labels, normals, classes, config, batch_size, up
    )
    # Last: the neighbourhood filters would dissolve the low vegetation left between ground.
    dtm_ground = _restore_dtm_ground(points, labels, classes, config, batch_size, up)
    output = Path(directory) / config.output_subdir
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "postprocess_report.json",
        {
            "before_counts": dict(sorted(before.items())),
            "after_counts": dict(sorted(Counter(int(code) for code in labels).items())),
            "npi_removed": dict(sorted(removed.items())),
            "cluster_to_unknown": dict(sorted(cluster_removed.items())),
            "road_slope_removed": road_rejected,
            "road_surface_to_road": road_surface,
            "holes_filled": dict(sorted(filled.items())),
            "unknown_edges_filled": dict(sorted(unknown_edges.items())),
            "planar_unknown_edges_filled": dict(sorted(planar_edges.items())),
            "flat_roof_ground_to_building": roofs,
            "color_vegetation_to_ground": bare_ground,
            "other_to_building": other_buildings,
            "geometry_reclassified": geometry_changes,
            "vertical": up_info,
            "dtm_ground": dtm_ground,
            "compact_classes": compact,
            "ground_height_to_unknown": ground_height_rejected,
            "vegetation_dem": vegetation_height,
        },
    )
    return labels
