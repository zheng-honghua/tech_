from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from .geometry_edges import (
    EdgeLine,
    _angle_difference,
    _junctions,
    _line_from_points,
    extract_edge_topology,
)


STRUCTURAL_VECTOR_LENGTH = 56


@dataclass(frozen=True)
class StructuralLine:
    x1: float
    y1: float
    x2: float
    y2: float
    source: str
    confidence: float
    support: float

    def points(self) -> np.ndarray:
        return np.asarray(((self.x1, self.y1), (self.x2, self.y2)), np.float32)


@dataclass(frozen=True)
class StructuralVertex:
    x: float
    y: float
    kind: str
    degree: int
    confidence: float


@dataclass(frozen=True)
class StructuralContour:
    clean_mask: np.ndarray
    raw_edge_map: np.ndarray
    clean_line_map: np.ndarray
    vertex_heatmap: np.ndarray
    outer_polygon: np.ndarray
    outer_lines: tuple[StructuralLine, ...]
    internal_lines: tuple[StructuralLine, ...]
    vertices: tuple[StructuralVertex, ...]
    outer_iou: float
    contour_fit_error_px: float
    rejected_candidate_ratio: float
    dangling_endpoint_count: int
    quality: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "outer_polygon": self.outer_polygon.astype(float).tolist(),
            "outer_lines": [asdict(item) for item in self.outer_lines],
            "internal_lines": [asdict(item) for item in self.internal_lines],
            "vertices": [asdict(item) for item in self.vertices],
            "outer_iou": self.outer_iou,
            "contour_fit_error_px": self.contour_fit_error_px,
            "rejected_candidate_ratio": self.rejected_candidate_ratio,
            "dangling_endpoint_count": self.dangling_endpoint_count,
            "quality": self.quality,
            "reason": self.reason,
        }


def _largest_clean_mask(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        raise ValueError("structural contour requires a non-empty object mask")
    selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    output = (labels == selected).astype(np.uint8) * 255
    contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(output)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return filled


def _polygon_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    output = np.zeros(shape, np.uint8)
    cv2.fillPoly(output, [np.rint(polygon).astype(np.int32)], 255)
    return output


def _fit_outer_polygon(mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    area = max(float(cv2.contourArea(contour)), 1.0)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    source = hull if area / hull_area >= 0.86 else contour
    perimeter = max(float(cv2.arcLength(source, True)), 1.0)
    best: tuple[float, np.ndarray, float] | None = None
    for epsilon_ratio in np.linspace(0.006, 0.035, 18):
        candidate = cv2.approxPolyDP(source, epsilon_ratio * perimeter, True).reshape(-1, 2)
        if not 3 <= len(candidate) <= 12:
            continue
        candidate_mask = _polygon_mask(mask.shape, candidate)
        intersection = cv2.countNonZero(cv2.bitwise_and(mask, candidate_mask))
        union = cv2.countNonZero(cv2.bitwise_or(mask, candidate_mask))
        iou = intersection / max(union, 1)
        # A regular solid silhouette should not need dozens of tiny segments.
        score = iou - 0.004 * max(len(candidate) - 6, 0)
        if best is None or score > best[0]:
            best = (score, candidate.astype(np.float32), iou)
    if best is None:
        raise ValueError("unable to fit a stable outer polygon")
    polygon = _prune_short_polygon_edges(best[1], mask)
    polygon_mask = _polygon_mask(mask.shape, polygon)
    intersection = cv2.countNonZero(cv2.bitwise_and(mask, polygon_mask))
    union = cv2.countNonZero(cv2.bitwise_or(mask, polygon_mask))
    final_iou = intersection / max(union, 1)
    distances = [
        abs(cv2.pointPolygonTest(
            polygon.reshape(-1, 1, 2),
            (float(point[0, 0]), float(point[0, 1])),
            True,
        ))
        for point in contour[:: max(1, len(contour) // 256)]
    ]
    return polygon, float(final_iou), float(np.mean(distances))


def _prune_short_polygon_edges(polygon: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = polygon.astype(np.float32).copy()
    while len(output) > 3:
        lengths = np.linalg.norm(np.roll(output, -1, axis=0) - output, axis=1)
        perimeter = max(float(lengths.sum()), 1.0)
        shortest = int(np.argmin(lengths))
        if lengths[shortest] >= 0.045 * perimeter:
            break
        first = shortest
        second = (shortest + 1) % len(output)
        choices = []
        for remove_index in (first, second):
            candidate = np.delete(output, remove_index, axis=0)
            candidate_mask = _polygon_mask(mask.shape, candidate)
            intersection = cv2.countNonZero(cv2.bitwise_and(mask, candidate_mask))
            union = cv2.countNonZero(cv2.bitwise_or(mask, candidate_mask))
            choices.append((intersection / max(union, 1), candidate))
        output = max(choices, key=lambda item: item[0])[1]
    return output


def _nearest_contour_point(point: np.ndarray, contour_points: np.ndarray) -> tuple[np.ndarray, float]:
    distances = np.linalg.norm(contour_points - point[None, :], axis=1)
    index = int(np.argmin(distances))
    return contour_points[index].copy(), float(distances[index])


def _nearest_junction(point: np.ndarray, locations: np.ndarray, maximum: float) -> tuple[np.ndarray, bool]:
    if len(locations) == 0:
        return point, False
    distances = np.linalg.norm(locations - point[None, :], axis=1)
    index = int(np.argmin(distances))
    return (locations[index], True) if distances[index] <= maximum else (point, False)


def _stable_junction_locations(junctions, radius: float) -> np.ndarray:
    clusters: list[list[np.ndarray]] = []
    for item in junctions:
        point = np.asarray([item.x, item.y], np.float32)
        for cluster in clusters:
            if np.linalg.norm(point - np.mean(cluster, axis=0)) <= radius:
                cluster.append(point)
                break
        else:
            clusters.append([point])
    if not clusters:
        return np.empty((0, 2), np.float32)
    return np.asarray([np.mean(cluster, axis=0) for cluster in clusters], np.float32)


def _point_to_segment_distance(point: np.ndarray, line: EdgeLine) -> float:
    first, second = line.points()
    vector = second - first
    position = float(np.dot(point - first, vector) / max(np.dot(vector, vector), 1e-6))
    closest = first + np.clip(position, 0.0, 1.0) * vector
    return float(np.linalg.norm(point - closest))


def _infinite_intersection(first: EdgeLine, second: EdgeLine) -> np.ndarray | None:
    p, p2 = first.points()
    q, q2 = second.points()
    matrix = np.column_stack((p2 - p, -(q2 - q)))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-5:
        return None
    parameters = np.linalg.solve(matrix, q - p)
    return p + parameters[0] * (p2 - p)


def _build_line_hypotheses(lines: tuple[EdgeLine, ...], scale: float) -> list[EdgeLine]:
    groups: list[list[EdgeLine]] = []
    for line in sorted(lines, key=lambda item: item.length, reverse=True):
        center = line.points().mean(axis=0)
        for group in groups:
            reference = group[0]
            direction = reference.points()[1] - reference.points()[0]
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            normal = np.asarray([-direction[1], direction[0]], np.float32)
            reference_center = reference.points().mean(axis=0)
            offset = abs(float(np.dot(center - reference_center, normal)))
            if _angle_difference(line.angle_deg, reference.angle_deg) <= 12.0 and offset <= 0.035 * scale:
                group.append(line)
                break
        else:
            groups.append([line])
    output = []
    for group in groups:
        points = np.vstack([line.points() for line in group])
        center = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - center, full_matrices=False)
        direction = vh[0]
        projections = (points - center) @ direction
        endpoints = np.vstack((
            center + direction * projections.min(),
            center + direction * projections.max(),
        ))
        total = max(sum(line.length for line in group), 1e-6)
        support = sum(line.support * line.length for line in group) / total
        contrast = sum(line.contrast * line.length for line in group) / total
        output.append(_line_from_points(endpoints, support, contrast))
    return sorted(output, key=lambda item: item.length, reverse=True)[:16]


def _extended_junction_locations(lines: list[EdgeLine], mask: np.ndarray, scale: float) -> np.ndarray:
    candidates = []
    strong_lines = [line for line in lines if line.length >= 0.15 * scale]
    for index, first in enumerate(strong_lines):
        for second in strong_lines[index + 1:]:
            if _angle_difference(first.angle_deg, second.angle_deg) < 15.0:
                continue
            point = _infinite_intersection(first, second)
            if point is None:
                continue
            x, y = np.rint(point).astype(int)
            if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]) or mask[y, x] == 0:
                continue
            if max(_point_to_segment_distance(point, first),
                   _point_to_segment_distance(point, second)) > 0.34 * scale:
                continue
            candidates.append(point)
    pseudo = [type("Junction", (), {"x": float(p[0]), "y": float(p[1])}) for p in candidates]
    return _stable_junction_locations(pseudo, 0.07 * scale)


def _boundary_anchors(line: EdgeLine, contour_points: np.ndarray, scale: float) -> np.ndarray:
    points = line.points()
    direction = points[1] - points[0]
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    normal = np.asarray([-direction[1], direction[0]], np.float32)
    center = points.mean(axis=0)
    offsets = np.abs((contour_points - center) @ normal)
    nearby = contour_points[offsets <= max(2.5, 0.018 * scale)]
    if len(nearby) < 2:
        return np.empty((0, 2), np.float32)
    projections = (nearby - center) @ direction
    return np.asarray((nearby[int(np.argmin(projections))],
                       nearby[int(np.argmax(projections))]), np.float32)


def _segment_edge_support(points: np.ndarray, edge_map: np.ndarray) -> float:
    length = float(np.linalg.norm(points[1] - points[0]))
    count = max(12, int(round(length)))
    weights = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    samples = points[0] * (1.0 - weights) + points[1] * weights
    xy = np.rint(samples).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, edge_map.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, edge_map.shape[0] - 1)
    support = cv2.dilate(edge_map, np.ones((5, 5), np.uint8))
    return float(np.mean(support[xy[:, 1], xy[:, 0]] > 0))


def _segment_boundary_overlap(points: np.ndarray, mask: np.ndarray, scale: float) -> float:
    length = float(np.linalg.norm(points[1] - points[0]))
    count = max(12, int(round(length)))
    weights = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    samples = points[0] * (1.0 - weights) + points[1] * weights
    xy = np.rint(samples).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, mask.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, mask.shape[0] - 1)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    return float(np.mean(distance[xy[:, 1], xy[:, 0]] <= 0.035 * scale))


def _clean_internal_lines(topology, clean_mask: np.ndarray, contour: np.ndarray, scale: float):
    boundary_points = contour.reshape(-1, 2).astype(np.float32)
    hypotheses = _build_line_hypotheses(topology.raw_lines, scale)
    # Rebuild junctions only from substantial line hypotheses. Junctions made by
    # a long ridge crossing a tiny texture fragment are a common source of noise.
    stable_junctions = _extended_junction_locations(hypotheses, clean_mask, scale)
    candidates: list[tuple[StructuralLine, tuple[bool, bool]]] = []
    for line in hypotheses:
        contrast_score = min(line.contrast / 18.0, 1.0)
        length_score = min(line.length / max(0.30 * scale, 1.0), 1.0)
        confidence = 0.55 * line.support + 0.25 * contrast_score + 0.20 * length_score
        if line.length < 0.12 * scale or line.support < 0.42 or confidence < 0.50:
            continue
        original_endpoints = line.points().copy()
        endpoints = original_endpoints.copy()
        connected = [False, False]
        connection_kind = ["none", "none"]
        boundary_anchors = _boundary_anchors(line, boundary_points, scale)
        for index in range(2):
            options: list[tuple[float, np.ndarray, str]] = []
            snapped, at_junction = _nearest_junction(endpoints[index], stable_junctions, 0.40 * scale)
            if at_junction:
                options.append((float(np.linalg.norm(snapped - endpoints[index])), snapped, "junction"))
            for anchor in boundary_anchors:
                distance = float(np.linalg.norm(anchor - endpoints[index]))
                if distance <= 0.55 * scale:
                    options.append((distance, anchor, "boundary"))
            if options:
                _, snapped, kind = min(options, key=lambda item: item[0])
                endpoints[index] = snapped
                connected[index] = True
                connection_kind[index] = kind
        # Both fragment ends can be closer to the same apex than to the distant
        # silhouette. Preserve the nearer apex connection and extend the other
        # end to its matching boundary instead of collapsing the ridge to a point.
        if np.linalg.norm(endpoints[1] - endpoints[0]) < 0.12 * scale and len(boundary_anchors):
            junction_end = int(np.argmin([
                np.linalg.norm(original_endpoints[index] - endpoints[index])
                if connection_kind[index] == "junction" else np.inf
                for index in range(2)
            ]))
            boundary_end = 1 - junction_end
            anchor = min(
                boundary_anchors,
                key=lambda point: float(np.linalg.norm(point - original_endpoints[boundary_end])),
            )
            endpoints[boundary_end] = anchor
            connected[boundary_end] = True
            connection_kind[boundary_end] = "boundary"
        length = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        if length < 0.12 * scale:
            continue
        # Texture strokes tend to float inside a face. Keep only graph-connected
        # ridges; an uncertain open endpoint is deliberately rejected.
        if not all(connected):
            continue
        reconstructed_support = _segment_edge_support(endpoints, topology.edge_map)
        if reconstructed_support < 0.09:
            continue
        if _segment_boundary_overlap(endpoints, clean_mask, scale) > 0.55:
            continue
        if connection_kind.count("boundary") == 2 and (
            reconstructed_support < 0.20 or line.support < 0.58
        ):
            continue
        if length > line.length + 0.90 * scale:
            continue
        candidates.append((StructuralLine(
            float(endpoints[0, 0]), float(endpoints[0, 1]),
            float(endpoints[1, 0]), float(endpoints[1, 1]),
            "internal", float(confidence), float(line.support),
        ), (bool(connected[0]), bool(connected[1]))))

    # Remove duplicate fitted ridges after endpoint snapping.
    kept: list[tuple[StructuralLine, tuple[bool, bool]]] = []
    for candidate in sorted(candidates, key=lambda item: -np.linalg.norm(
        item[0].points()[1] - item[0].points()[0]
    )):
        points = candidate[0].points()
        duplicate = False
        for existing, _ in kept:
            other = existing.points()
            direct = np.linalg.norm(points[0] - other[0]) + np.linalg.norm(points[1] - other[1])
            reverse = np.linalg.norm(points[0] - other[1]) + np.linalg.norm(points[1] - other[0])
            if min(direct, reverse) <= 0.11 * scale:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept[:12]


def _prune_final_dangling_lines(cleaned, clean_mask: np.ndarray, contour: np.ndarray, scale: float):
    kept = list(cleaned)
    boundary_points = contour.reshape(-1, 2).astype(np.float32)
    while kept:
        edge_lines = [
            _line_from_points(item[0].points(), item[0].support, item[0].confidence * 18.0)
            for item in kept
        ]
        junctions = _junctions(edge_lines, clean_mask, scale)
        junction_locations = np.asarray(
            [[item.x, item.y] for item in junctions], np.float32
        ).reshape(-1, 2)
        updated = []
        for line, _ in kept:
            connections = []
            for point in line.points():
                _, boundary_distance = _nearest_contour_point(point, boundary_points)
                _, at_junction = _nearest_junction(
                    point, junction_locations, 0.045 * scale
                )
                connections.append(boundary_distance <= 0.025 * scale or at_junction)
            if all(connections):
                updated.append((line, (bool(connections[0]), bool(connections[1]))))
        if len(updated) == len(kept):
            return updated
        kept = updated
    return []


def _cluster_vertices(vertices: list[StructuralVertex], radius: float) -> tuple[StructuralVertex, ...]:
    clusters: list[list[StructuralVertex]] = []
    for vertex in vertices:
        for cluster in clusters:
            center = np.mean([[item.x, item.y] for item in cluster], axis=0)
            if np.linalg.norm(np.asarray([vertex.x, vertex.y]) - center) <= radius:
                cluster.append(vertex)
                break
        else:
            clusters.append([vertex])
    output = []
    priority = {"junction": 3, "boundary_attachment": 2, "contour": 1}
    for cluster in clusters:
        weights = np.asarray([max(item.confidence, 0.1) for item in cluster])
        coordinates = np.asarray([[item.x, item.y] for item in cluster])
        center = np.average(coordinates, axis=0, weights=weights)
        strongest = max(cluster, key=lambda item: (priority[item.kind], item.degree))
        output.append(StructuralVertex(
            float(center[0]), float(center[1]), strongest.kind,
            max(item.degree for item in cluster), max(item.confidence for item in cluster),
        ))
    return tuple(output)


def extract_structural_contour(image_bgr: np.ndarray, mask: np.ndarray) -> StructuralContour:
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[:2] != np.asarray(mask).shape:
        raise ValueError("image and mask shapes do not match")
    clean_mask = _largest_clean_mask(mask)
    area = cv2.countNonZero(clean_mask)
    if area < 100:
        raise ValueError("structural contour requires at least 100 object pixels")
    scale = float(np.sqrt(area))
    polygon, outer_iou, fit_error = _fit_outer_polygon(clean_mask)
    topology = extract_edge_topology(image, clean_mask)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    cleaned = _clean_internal_lines(topology, clean_mask, contour, scale)
    cleaned = _prune_final_dangling_lines(cleaned, clean_mask, contour, scale)
    perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
    circularity = float(4.0 * np.pi * cv2.contourArea(contour) / (perimeter * perimeter))
    if len(polygon) >= 9 and circularity >= 0.78:
        # A smooth circular/elliptic silhouette (for example a cone seen from
        # above) has continuous shading, not discrete polyhedral ridges.
        cleaned = []

    outer_lines = []
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        outer_lines.append(StructuralLine(
            float(first[0]), float(first[1]), float(second[0]), float(second[1]),
            "outer", outer_iou, 1.0,
        ))
    internal_lines = tuple(item[0] for item in cleaned)
    edge_lines = [
        _line_from_points(item.points(), item.support, item.confidence * 18.0)
        for item in internal_lines
    ]
    junctions = _junctions(edge_lines, clean_mask, scale) if edge_lines else []

    vertices = [
        StructuralVertex(float(point[0]), float(point[1]), "contour", 2, outer_iou)
        for point in polygon
    ]
    contour_points = contour.reshape(-1, 2).astype(np.float32)
    dangling = 0
    for line, connections in cleaned:
        for point, connected in zip(line.points(), connections):
            nearest, boundary_distance = _nearest_contour_point(point, contour_points)
            junction_locations = np.asarray(
                [[item.x, item.y] for item in junctions], np.float32
            ).reshape(-1, 2)
            junction_point, at_junction = _nearest_junction(
                point, junction_locations, 0.04 * scale
            )
            if at_junction:
                degree = max(
                    (item.degree for item in junctions
                     if np.linalg.norm(junction_point - [item.x, item.y]) <= 1.0),
                    default=2,
                )
                vertices.append(StructuralVertex(
                    float(junction_point[0]), float(junction_point[1]),
                    "junction", degree, line.confidence,
                ))
            elif boundary_distance <= 0.025 * scale:
                vertices.append(StructuralVertex(
                    float(nearest[0]), float(nearest[1]),
                    "boundary_attachment", 3, line.confidence,
                ))
            elif not connected:
                dangling += 1
    vertices_tuple = _cluster_vertices(vertices, 0.025 * scale)

    line_map = np.zeros(clean_mask.shape, np.uint8)
    for line in outer_lines:
        cv2.line(line_map, tuple(np.rint(line.points()[0]).astype(int)),
                 tuple(np.rint(line.points()[1]).astype(int)), 255, 2, cv2.LINE_AA)
    for line in internal_lines:
        cv2.line(line_map, tuple(np.rint(line.points()[0]).astype(int)),
                 tuple(np.rint(line.points()[1]).astype(int)), 190, 2, cv2.LINE_AA)
    heatmap = np.zeros(clean_mask.shape, np.float32)
    for vertex in vertices_tuple:
        x, y = np.clip(np.rint([vertex.x, vertex.y]).astype(int),
                       [0, 0], [heatmap.shape[1] - 1, heatmap.shape[0] - 1])
        heatmap[y, x] = max(heatmap[y, x], vertex.confidence)
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), 3.0)
    heatmap = np.uint8(np.clip(heatmap / max(float(heatmap.max()), 1e-6) * 255, 0, 255))

    rejected_ratio = 1.0 - len(internal_lines) / max(len(topology.raw_lines), 1)
    internal_confidence = (
        float(np.mean([line.confidence for line in internal_lines]))
        if internal_lines else 1.0
    )
    quality = float(np.clip(0.60 * outer_iou + 0.30 * internal_confidence
                            + 0.10 * (dangling == 0), 0.0, 1.0))
    reason = "accepted" if outer_iou >= 0.92 and dangling == 0 else "structure_uncertain"
    return StructuralContour(
        clean_mask, topology.edge_map, line_map, heatmap, polygon,
        tuple(outer_lines), internal_lines, vertices_tuple, outer_iou, fit_error,
        float(np.clip(rejected_ratio, 0.0, 1.0)), dangling, quality, reason,
    )


def render_structural_contour(image_bgr: np.ndarray, result: StructuralContour) -> np.ndarray:
    canvas = np.asarray(image_bgr).copy()
    for line in result.outer_lines:
        points = np.rint(line.points()).astype(int)
        cv2.line(canvas, tuple(points[0]), tuple(points[1]), (0, 220, 0), 2, cv2.LINE_AA)
    for line in result.internal_lines:
        points = np.rint(line.points()).astype(int)
        cv2.line(canvas, tuple(points[0]), tuple(points[1]), (0, 180, 255), 2, cv2.LINE_AA)
    colours = {"contour": (255, 80, 0), "boundary_attachment": (255, 0, 255),
               "junction": (0, 0, 255)}
    for vertex in result.vertices:
        center = tuple(np.rint([vertex.x, vertex.y]).astype(int))
        cv2.circle(canvas, center, 4, colours[vertex.kind], -1, cv2.LINE_AA)
    return canvas


def structural_contour_vector(result: StructuralContour) -> np.ndarray:
    scale = max(float(np.sqrt(cv2.countNonZero(result.clean_mask))), 1.0)
    outer_count = np.zeros(10, np.float32)
    outer_count[int(np.clip(len(result.outer_polygon), 3, 12)) - 3] = 1.0
    outer_lengths = sorted((
        float(np.linalg.norm(line.points()[1] - line.points()[0])) / scale
        for line in result.outer_lines
    ), reverse=True)[:12]
    outer_lengths += [0.0] * (12 - len(outer_lengths))
    internal_lengths = sorted((
        float(np.linalg.norm(line.points()[1] - line.points()[0])) / scale
        for line in result.internal_lines
    ), reverse=True)[:8]
    internal_lengths += [0.0] * (8 - len(internal_lengths))
    angles = []
    for line in result.internal_lines:
        direction = line.points()[1] - line.points()[0]
        angles.append(float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0))
    pair_angles = [
        min(abs(first - second) % 180.0, 180.0 - abs(first - second) % 180.0)
        for index, first in enumerate(angles) for second in angles[index + 1:]
    ]
    angle_histogram, _ = np.histogram(pair_angles, bins=9, range=(0.0, 90.0))
    angle_histogram = angle_histogram.astype(np.float32)
    angle_histogram /= max(float(angle_histogram.sum()), 1.0)
    junctions = [item for item in result.vertices if item.kind == "junction"]
    degree_counts = np.asarray([
        sum(item.degree == 2 for item in junctions),
        sum(item.degree == 3 for item in junctions),
        sum(item.degree == 4 for item in junctions),
        sum(item.degree >= 5 for item in junctions),
    ], np.float32) / 4.0
    kind_counts = np.asarray([
        sum(item.kind == "contour" for item in result.vertices) / 12.0,
        len(junctions) / 4.0,
        sum(item.kind == "boundary_attachment" for item in result.vertices) / 12.0,
    ], np.float32)
    center = np.asarray(result.clean_mask.shape[::-1], np.float32) / 2.0
    radii = sorted((
        float(np.linalg.norm(np.asarray([item.x, item.y]) - center)) / scale
        for item in junctions
    ))[:4]
    radii += [0.0] * (4 - len(radii))
    metrics = np.asarray([
        result.outer_iou,
        result.contour_fit_error_px / scale,
        result.rejected_candidate_ratio,
        result.quality,
        result.dangling_endpoint_count / 4.0,
    ], np.float32)
    vector = np.concatenate((
        outer_count,
        np.asarray(outer_lengths, np.float32),
        np.asarray([len(result.internal_lines) / 8.0], np.float32),
        np.asarray(internal_lengths, np.float32),
        angle_histogram,
        degree_counts,
        kind_counts,
        np.asarray(radii, np.float32),
        metrics,
    )).astype(np.float32)
    if len(vector) != STRUCTURAL_VECTOR_LENGTH:
        raise AssertionError("unexpected structural feature vector length")
    return vector
