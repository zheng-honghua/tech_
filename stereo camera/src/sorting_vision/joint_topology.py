"""Partial metric evidence graph, never a completed or watertight mesh.

Only measured RGB-D points can locate an edge. Unsupported side rays stay
unresolved; neither a box nor the classifier can manufacture missing depth.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .rgbd import fit_plane_ransac


GRAPH_FEATURE_NAMES = (
    "graph_nodes", "graph_edges", "graph_faces", "graph_shared_edges",
    "graph_side_supported_edges", "graph_unresolved_ratio",
    "graph_junction_ratio", "graph_cycle_rank", "graph_plane_coverage",
    "graph_edge_residual", "graph_quality", "graph_components",
)


def _fit_edge(points: np.ndarray, minimum_span: float) -> tuple[np.ndarray, float] | None:
    if len(points) < 8:
        return None
    center = np.median(points, axis=0)
    _, singular, axes = np.linalg.svd(points - center, full_matrices=False)
    if singular[0] <= 1e-8 or singular[1] / singular[0] > .12:
        return None
    along = (points - center) @ axes[0]
    residual = np.linalg.norm(points - center - along[:, None] * axes[0], axis=1)
    low, high = np.percentile(along, [5, 95])
    if high - low < minimum_span or float(np.percentile(residual, 90)) > 2.0:
        return None
    return np.stack((center + low * axes[0], center + high * axes[0])), float(np.median(residual))


def build_joint_topology(points_primary_mm: np.ndarray, side_segments_px: np.ndarray, calibration: Any) -> dict[str, Any]:
    cloud = np.asarray(points_primary_mm, np.float64).reshape(-1, 3)
    cloud = cloud[np.all(np.isfinite(cloud), axis=1) & (cloud[:, 2] > 0)]
    segments = np.asarray(side_segments_px, np.float64).reshape(-1, 4)
    if len(cloud) < 40:
        raise ValueError("joint topology needs forty finite positive-depth points")
    # Bounded work and deterministic sampling, independent of class labels.
    cloud = cloud[::max(1, int(np.ceil(len(cloud) / 1600)))]
    extent = max(float(np.linalg.norm(np.ptp(cloud, axis=0))), 1.0)
    minimum_span = max(4.0, .08 * extent)
    remaining = cloud.copy()
    faces: list[dict[str, Any]] = []
    planes = []
    for index in range(6):
        if len(remaining) < max(30, .08 * len(cloud)):
            break
        plane = fit_plane_ransac(remaining, threshold_mm=1.0, iterations=45, seed=71 + index)
        selected = np.abs(plane.signed_distance(remaining)) < 1.0
        if selected.sum() < max(30, .08 * len(cloud)):
            break
        # Do not split one measured plane into multiple graph faces.
        if any(abs(float(plane.normal @ old.normal)) > .995 and
               abs(float(old.signed_distance(np.median(remaining[selected], axis=0)))) < 2.0 for old in planes):
            break
        faces.append({"face_id": len(faces), "normal": plane.normal.tolist(),
                      "offset_mm": float(plane.offset), "rmse_mm": float(plane.rmse_mm),
                      "support_points": int(selected.sum()), "source": "PRIMARY_DEPTH"})
        planes.append(plane)
        remaining = remaining[~selected]

    candidates: list[dict[str, Any]] = []
    for first, plane in enumerate(planes):
        for second in range(first + 1, len(planes)):
            other = planes[second]
            if abs(float(plane.normal @ other.normal)) > .97:
                continue
            near = (np.abs(plane.signed_distance(cloud)) < 1.5) & (np.abs(other.signed_distance(cloud)) < 1.5)
            fitted = _fit_edge(cloud[near], minimum_span)
            if fitted is not None:
                endpoints, residual = fitted
                candidates.append({"endpoints": endpoints, "residual_mm": residual,
                                   "sources": ["TOP_PLANE_INTERSECTION"], "face_ids": [first, second],
                                   "side_segment_ids": [], "support_points": int(near.sum())})

    projected = calibration.project_primary_points(cloud)
    unresolved = []
    for index, segment in enumerate(segments):
        direction = segment[2:] - segment[:2]
        length2 = float(direction @ direction)
        if not np.all(np.isfinite(segment)) or length2 < 64:
            unresolved.append({"segment_id": index, "reason": "SHORT_OR_INVALID"})
            continue
        delta = projected - segment[:2]
        along = delta @ direction / length2
        distance = np.linalg.norm(delta - along[:, None] * direction, axis=1)
        selected = np.all(np.isfinite(projected), axis=1) & (distance <= 2.0) & (along >= 0) & (along <= 1)
        fitted = _fit_edge(cloud[selected], minimum_span)
        if fitted is None or not np.any(selected) or np.ptp(along[selected]) < .40:
            unresolved.append({"segment_id": index, "reason": "NO_RELIABLE_METRIC_LINE_SUPPORT"})
            continue
        endpoints, residual = fitted
        candidates.append({"endpoints": endpoints, "residual_mm": residual,
                           "sources": ["SIDE_DEPTH_SUPPORTED"], "face_ids": [],
                           "side_segment_ids": [index], "support_points": int(selected.sum())})

    # Match overlapping collinear metric estimates, not all nearby endpoints.
    edges: list[dict[str, Any]] = []
    for candidate in candidates:
        endpoints = candidate["endpoints"]
        direction = endpoints[1] - endpoints[0]
        direction /= np.linalg.norm(direction)
        matched = None
        for old in edges:
            old_points = old["endpoints"]
            old_direction = old_points[1] - old_points[0]
            old_length = float(np.linalg.norm(old_direction))
            old_direction /= old_length
            delta = endpoints - old_points[0]
            along = delta @ old_direction
            residual = np.linalg.norm(delta - along[:, None] * old_direction, axis=1)
            overlap = min(float(along.max()), old_length) - max(float(along.min()), 0.0)
            if abs(float(direction @ old_direction)) > .985 and residual.max() <= 2.0 and overlap >= .4 * min(old_length, float(np.linalg.norm(endpoints[1] - endpoints[0]))):
                matched = old
                break
        if matched is None:
            edges.append(candidate)
        else:
            for name in ("sources", "face_ids", "side_segment_ids"):
                matched[name] = sorted(set(matched[name] + candidate[name]))
            matched["support_points"] = max(matched["support_points"], candidate["support_points"])

    nodes: list[np.ndarray] = []
    degree: list[int] = []
    graph_edges = []
    for edge in edges:
        ids = []
        for endpoint in edge["endpoints"]:
            distances = np.asarray([np.linalg.norm(endpoint - node) for node in nodes])
            if len(distances) and distances.min() <= 2.0:
                node_id = int(distances.argmin())
            else:
                node_id = len(nodes)
                nodes.append(endpoint.copy())
                degree.append(0)
            ids.append(node_id)
        if ids[0] == ids[1]:
            continue
        degree[ids[0]] += 1
        degree[ids[1]] += 1
        graph_edges.append({**{k: v for k, v in edge.items() if k != "endpoints"},
                            "node_ids": ids, "endpoints_primary_mm": edge["endpoints"].tolist()})
    parents = list(range(len(nodes)))
    def root(index: int) -> int:
        while parents[index] != index:
            index = parents[index]
        return index
    for edge in graph_edges:
        first, second = edge["node_ids"]
        parents[root(first)] = root(second)
    components = len({root(index) for index in range(len(nodes))})
    coverage = sum(face["support_points"] for face in faces) / len(cloud)
    unresolved_ratio = len(unresolved) / max(len(segments), 1)
    median_residual = float(np.median([edge["residual_mm"] for edge in graph_edges])) if graph_edges else 2.0
    quality = float(np.clip(coverage * (1 - median_residual / 2) * min(len(graph_edges) / 4, 1), 0, 1))
    metrics = [len(nodes) / 24, len(graph_edges) / 24, len(faces) / 6,
               sum(len(edge["sources"]) > 1 for edge in graph_edges) / max(len(graph_edges), 1),
               sum("SIDE_DEPTH_SUPPORTED" in edge["sources"] for edge in graph_edges) / max(len(graph_edges), 1),
               unresolved_ratio, sum(value >= 3 for value in degree) / max(len(nodes), 1),
               max(0, len(graph_edges) - len(nodes) + components) / 12, coverage,
               median_residual / 2, quality, components / max(len(nodes), 1)]
    edges_side_px = []
    for edge in graph_edges:
        pixels = calibration.project_primary_points(np.asarray(edge["endpoints_primary_mm"]))
        if np.all(np.isfinite(pixels)):
            edges_side_px.append(pixels.reshape(4).tolist())
    return {"version": 1, "coordinate_frame": "PRIMARY_CAMERA_MM", "complete_mesh": False,
            "edges_side_px": edges_side_px,
            "nodes": [{"node_id": index, "position_mm": node.tolist(), "degree": degree[index]} for index, node in enumerate(nodes)],
            "edges": graph_edges, "faces": faces, "unresolved_side_segments": unresolved,
            "features": dict(zip(GRAPH_FEATURE_NAMES, metrics)),
            "limitations": ["PARTIAL_VISIBLE_SURFACES", "NO_SIDE_ONLY_METRIC_DEPTH", "NO_GRASP_GEOMETRY_UPGRADE"]}
