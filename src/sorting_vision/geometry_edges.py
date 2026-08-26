from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# These OpenCV objects are relatively expensive to construct. The live pipeline is
# single threaded, so sharing them also keeps latency measurements representative.
_CLAHE = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
_LSD = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
EDGE_PARAMETERS: dict[str, float | int | str] = {
    "input_size": 256,
    "clahe_clip_limit": 2.2,
    "bilateral_diameter": 7,
    "lsd_refine": "standard",
    "minimum_length_scale": 0.055,
    "minimum_inside_ratio": 0.82,
    "maximum_boundary_overlap": 0.32,
    "minimum_edge_support": 0.25,
    "merge_angle_deg": 8.0,
    "merge_line_distance_scale": 0.025,
    "merge_gap_scale": 0.08,
    "face_gap_close_scale": 0.14,
    "face_vertex_feature": 1,
    "minimum_topology_quality": 0.42,
}


@dataclass(frozen=True)
class EdgeLine:
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle_deg: float
    support: float
    contrast: float

    def points(self) -> np.ndarray:
        return np.asarray([[self.x1, self.y1], [self.x2, self.y2]], np.float32)


@dataclass(frozen=True)
class EdgeJunction:
    x: float
    y: float
    degree: int
    line_indices: tuple[int, ...]


@dataclass(frozen=True)
class EdgeTopology:
    enhanced_gray: np.ndarray
    edge_map: np.ndarray
    raw_lines: tuple[EdgeLine, ...]
    merged_lines: tuple[EdgeLine, ...]
    junctions: tuple[EdgeJunction, ...]
    parallel_pairs: int
    converging_pairs: int
    face_areas: tuple[float, ...]
    boundary_endpoint_ratio: float
    quality: float
    reason: str
    object_scale_px: float
    face_vertices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_lines": [asdict(line) for line in self.raw_lines],
            "merged_lines": [asdict(line) for line in self.merged_lines],
            "junctions": [asdict(node) for node in self.junctions],
            "parallel_pairs": self.parallel_pairs,
            "converging_pairs": self.converging_pairs,
            "face_areas": list(self.face_areas),
            "face_vertices": list(self.face_vertices),
            "boundary_endpoint_ratio": self.boundary_endpoint_ratio,
            "quality": self.quality,
            "reason": self.reason,
            "object_scale_px": self.object_scale_px,
        }


def _angle_difference(first: float, second: float) -> float:
    difference = abs(first - second) % 180.0
    return min(difference, 180.0 - difference)


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _line_from_points(
    points: np.ndarray, support: float, contrast: float
) -> EdgeLine:
    first, second = np.asarray(points, np.float32)
    dx, dy = second - first
    length = float(np.hypot(dx, dy))
    angle = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
    return EdgeLine(
        float(first[0]),
        float(first[1]),
        float(second[0]),
        float(second[1]),
        length,
        angle,
        float(support),
        float(contrast),
    )


def _sample_segment(line: np.ndarray, count: int | None = None) -> np.ndarray:
    first, second = line
    length = float(np.linalg.norm(second - first))
    # Dense pixel-by-pixel sampling did not improve rejection on the 256 px crop,
    # but dominated runtime when LSD returned many long fragments.
    count = count or min(64, max(8, int(round(length / 2.0))))
    weights = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    return first[None, :] * (1.0 - weights) + second[None, :] * weights


def _line_metrics(
    points: np.ndarray,
    mask: np.ndarray,
    boundary_band: np.ndarray,
    edge_support: np.ndarray,
    gradient: np.ndarray,
) -> tuple[float, float, float, float]:
    samples = _sample_segment(points)
    xs = np.clip(np.rint(samples[:, 0]).astype(int), 0, mask.shape[1] - 1)
    ys = np.clip(np.rint(samples[:, 1]).astype(int), 0, mask.shape[0] - 1)
    inside = float(np.mean(mask[ys, xs] > 0))
    boundary = float(np.mean(boundary_band[ys, xs] > 0))
    support = float(np.mean(edge_support[ys, xs] > 0))
    contrast = float(np.mean(gradient[ys, xs]))
    return inside, boundary, support, contrast


def _point_line_distance(point: np.ndarray, line: EdgeLine) -> float:
    first, second = line.points()
    vector = second - first
    return float(
        abs(_cross_2d(vector, point - first)) / max(np.linalg.norm(vector), 1e-6)
    )


def _projection_gap(first: EdgeLine, second: EdgeLine) -> float:
    direction = first.points()[1] - first.points()[0]
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    origin = first.points()[0]
    a = np.sort((first.points() - origin) @ direction)
    b = np.sort((second.points() - origin) @ direction)
    return float(max(a[0] - b[1], b[0] - a[1], 0.0))


def _can_merge(first: EdgeLine, second: EdgeLine, scale: float) -> bool:
    if _angle_difference(first.angle_deg, second.angle_deg) > 8.0:
        return False
    distances = [
        _point_line_distance(point, first) for point in second.points()
    ] + [_point_line_distance(point, second) for point in first.points()]
    return min(distances) <= 0.025 * scale and _projection_gap(first, second) <= 0.08 * scale


def _merge_pair(first: EdgeLine, second: EdgeLine) -> EdgeLine:
    points = np.vstack((first.points(), second.points()))
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    direction = vh[0]
    projections = (points - center) @ direction
    endpoints = np.vstack(
        (center + direction * projections.min(), center + direction * projections.max())
    )
    total = max(first.length + second.length, 1e-6)
    support = (first.support * first.length + second.support * second.length) / total
    contrast = (first.contrast * first.length + second.contrast * second.length) / total
    return _line_from_points(endpoints, support, contrast)


def _merge_lines(lines: list[EdgeLine], scale: float) -> list[EdgeLine]:
    merged = sorted(lines, key=lambda line: line.length, reverse=True)
    changed = True
    while changed:
        changed = False
        for first_index in range(len(merged)):
            for second_index in range(first_index + 1, len(merged)):
                if _can_merge(merged[first_index], merged[second_index], scale):
                    combined = _merge_pair(merged[first_index], merged[second_index])
                    merged[first_index] = combined
                    del merged[second_index]
                    changed = True
                    break
            if changed:
                break
    return sorted(merged, key=lambda line: line.length, reverse=True)


def _intersection(
    first: EdgeLine, second: EdgeLine, extension: float
) -> np.ndarray | None:
    p, p2 = first.points()
    q, q2 = second.points()
    r, s = p2 - p, q2 - q
    denominator = _cross_2d(r, s)
    if abs(denominator) < 1e-5:
        return None
    t = _cross_2d(q - p, s) / denominator
    u = _cross_2d(q - p, r) / denominator
    first_allowance = extension / max(first.length, 1e-6)
    second_allowance = extension / max(second.length, 1e-6)
    if not (-first_allowance <= t <= 1.0 + first_allowance):
        return None
    if not (-second_allowance <= u <= 1.0 + second_allowance):
        return None
    return p + t * r


def _junctions(lines: list[EdgeLine], mask: np.ndarray, scale: float):
    candidates: list[tuple[np.ndarray, set[int]]] = []
    extension = 0.06 * scale
    for first_index, first in enumerate(lines):
        for second_index in range(first_index + 1, len(lines)):
            second = lines[second_index]
            if _angle_difference(first.angle_deg, second.angle_deg) < 12.0:
                continue
            point = _intersection(first, second, extension)
            if point is None:
                continue
            x, y = np.rint(point).astype(int)
            if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
                continue
            if mask[y, x] == 0:
                continue
            candidates.append((point, {first_index, second_index}))
    clusters: list[tuple[np.ndarray, set[int], int]] = []
    radius = 0.04 * scale
    for point, indices in candidates:
        for cluster_index, (center, line_indices, count) in enumerate(clusters):
            if float(np.linalg.norm(point - center)) <= radius:
                updated_count = count + 1
                updated_center = (center * count + point) / updated_count
                clusters[cluster_index] = (
                    updated_center,
                    line_indices | indices,
                    updated_count,
                )
                break
        else:
            clusters.append((point, set(indices), 1))
    return [
        EdgeJunction(
            float(center[0]),
            float(center[1]),
            len(line_indices),
            tuple(sorted(line_indices)),
        )
        for center, line_indices, _ in clusters
    ]


def _extended_barrier_line(
    line: EdgeLine, mask: np.ndarray, scale: float
) -> tuple[tuple[int, int], tuple[int, int]]:
    points = line.points()
    direction = points[1] - points[0]
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    result: list[np.ndarray] = []
    for index, sign in ((0, -1.0), (1, 1.0)):
        point = points[index].copy()
        x, y = np.clip(
            np.rint(point).astype(int),
            [0, 0],
            [mask.shape[1] - 1, mask.shape[0] - 1],
        )
        # Only close a short gap to the silhouette. Extending a pyramid ridge
        # backwards through an interior apex would create a fictitious face.
        if distance[y, x] <= 0.12 * scale:
            for _ in range(max(2, int(round(0.14 * scale)))):
                candidate = point + sign * direction
                cx, cy = np.rint(candidate).astype(int)
                if not (0 <= cx < mask.shape[1] and 0 <= cy < mask.shape[0]):
                    break
                if mask[cy, cx] == 0:
                    break
                point = candidate
        result.append(point)
    return tuple(
        tuple(int(value) for value in np.rint(point).astype(int))
        for point in result
    )


def _face_geometry(
    mask: np.ndarray, lines: list[EdgeLine], scale: float, close_gaps: bool
) -> tuple[list[float], list[int]]:
    barriers = np.zeros_like(mask)
    for line in lines:
        endpoints = (
            _extended_barrier_line(line, mask, scale)
            if close_gaps
            else (
                (int(round(line.x1)), int(round(line.y1))),
                (int(round(line.x2)), int(round(line.y2))),
            )
        )
        cv2.line(
            barriers,
            endpoints[0],
            endpoints[1],
            255,
            3,
            cv2.LINE_AA,
        )
    regions = cv2.bitwise_and(mask, cv2.bitwise_not(barriers))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(regions, 8)
    object_area = max(float(cv2.countNonZero(mask)), 1.0)
    faces: list[tuple[float, int]] = []
    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA])
        if area < 0.025 * object_area:
            continue
        component = (labels == index).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = max(contours, key=cv2.contourArea)
        perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
        vertices = len(cv2.approxPolyDP(contour, 0.035 * perimeter, True))
        faces.append((area / object_area, int(np.clip(vertices, 0, 12))))
    faces.sort(reverse=True)
    return [item[0] for item in faces], [item[1] for item in faces]


def extract_edge_topology(
    image_bgr: np.ndarray, mask: np.ndarray, enhanced_faces: bool = True
) -> EdgeTopology:
    image = np.asarray(image_bgr)
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if image.shape[:2] != binary.shape or cv2.countNonZero(binary) < 100:
        raise ValueError("edge topology requires a matching non-empty object mask")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = _CLAHE.apply(gray)
    enhanced = cv2.bilateralFilter(enhanced, 7, 28, 28)
    median_inside = int(np.median(enhanced[binary > 0]))
    enhanced[binary == 0] = median_inside

    gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    gradient_inside = gradient[binary > 0]
    low = max(8.0, float(np.percentile(gradient_inside, 62)) * 0.55)
    high = max(low + 6.0, float(np.percentile(gradient_inside, 82)))
    edges = cv2.Canny(enhanced, int(low), int(high))
    edges[binary == 0] = 0
    boundary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    boundary_band = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, boundary_kernel)
    edge_support = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    object_area = float(cv2.countNonZero(binary))
    scale = max(float(np.sqrt(object_area)), 1.0)
    detected = _LSD.detect(enhanced)[0]
    raw: list[EdgeLine] = []
    if detected is not None:
        for coordinates in detected.reshape(-1, 4):
            points = coordinates.reshape(2, 2).astype(np.float32)
            length = float(np.linalg.norm(points[1] - points[0]))
            if length < max(8.0, 0.055 * scale):
                continue
            inside, boundary, support, contrast = _line_metrics(
                points, binary, boundary_band, edge_support, gradient
            )
            # A genuine internal ridge may terminate on the silhouette, but it
            # should not run along the silhouette. The previous 0.58 allowance
            # admitted entire polygon sides and polluted junction/parallel counts.
            if inside < 0.82 or boundary > 0.32 or support < 0.25:
                continue
            if contrast < max(6.0, 0.35 * high):
                continue
            raw.append(_line_from_points(points, support, contrast))
    raw = sorted(raw, key=lambda line: line.length, reverse=True)[:32]
    merged = _merge_lines(raw, scale)[:16]
    junctions = _junctions(merged, binary, scale)

    parallel_pairs = 0
    converging_pairs = 0
    for first_index, first in enumerate(merged):
        for second in merged[first_index + 1 :]:
            if _angle_difference(first.angle_deg, second.angle_deg) <= 8.0:
                parallel_pairs += 1
            elif _intersection(first, second, 0.08 * scale) is not None:
                converging_pairs += 1

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    boundary_endpoints = 0
    for line in merged:
        for point in line.points():
            x, y = np.clip(np.rint(point).astype(int), [0, 0], [binary.shape[1] - 1, binary.shape[0] - 1])
            if distance[y, x] <= 0.055 * scale:
                boundary_endpoints += 1
    endpoint_ratio = boundary_endpoints / max(2 * len(merged), 1)
    faces, face_vertices = _face_geometry(binary, merged, scale, enhanced_faces)
    mean_support = float(np.mean([line.support for line in merged])) if merged else 0.0
    line_score = min(len(merged) / 4.0, 1.0)
    junction_score = min(len(junctions) / 2.0, 1.0)
    face_score = min(max(len(faces) - 1, 0) / 3.0, 1.0)
    quality = float(np.clip(0.45 * line_score + 0.25 * mean_support + 0.2 * junction_score + 0.1 * face_score, 0.0, 1.0))
    reason = "accepted" if len(merged) >= 2 and quality >= 0.42 else "edge_evidence_low"
    return EdgeTopology(
        enhanced,
        edges,
        tuple(raw),
        tuple(merged),
        tuple(junctions),
        parallel_pairs,
        converging_pairs,
        tuple(faces),
        float(endpoint_ratio),
        quality,
        reason,
        scale,
        tuple(face_vertices),
    )


def edge_topology_vector(
    topology: EdgeTopology, include_face_vertices: bool = False
) -> np.ndarray:
    lines = list(topology.merged_lines)
    scale = topology.object_scale_px
    lengths = sorted((line.length / scale for line in lines), reverse=True)[:8]
    lengths += [0.0] * (8 - len(lengths))
    pair_angles: list[float] = []
    pair_weights: list[float] = []
    for first_index, first in enumerate(lines):
        for second in lines[first_index + 1 :]:
            difference = _angle_difference(first.angle_deg, second.angle_deg)
            pair_angles.append(min(difference, 90.0))
            pair_weights.append(min(first.length, second.length) / scale)
    angle_histogram, _ = np.histogram(
        pair_angles,
        bins=9,
        range=(0.0, 90.0),
        weights=pair_weights if pair_weights else None,
    )
    angle_histogram = angle_histogram.astype(np.float32)
    angle_histogram /= max(float(angle_histogram.sum()), 1.0)
    degrees = [junction.degree for junction in topology.junctions]
    degree_counts = [degrees.count(value) / 4.0 for value in (2, 3, 4)]
    if topology.junctions:
        strongest = max(topology.junctions, key=lambda item: item.degree)
        center = np.asarray(topology.enhanced_gray.shape[::-1], np.float32) / 2.0
        apex_offset = float(np.linalg.norm(np.asarray([strongest.x, strongest.y]) - center) / scale)
        max_degree = strongest.degree / 6.0
    else:
        apex_offset = 0.0
        max_degree = 0.0
    faces = list(topology.face_areas[:6])
    faces += [0.0] * (6 - len(faces))
    supports = [line.support for line in lines]
    contrasts = [line.contrast for line in lines]
    pair_count = max(len(lines) * (len(lines) - 1) / 2.0, 1.0)
    values = [
        min(len(lines) / 12.0, 1.0),
        min(sum(line.length for line in lines) / (12.0 * scale), 1.0),
        *lengths,
        *angle_histogram.tolist(),
        topology.parallel_pairs / pair_count,
        topology.converging_pairs / pair_count,
        *degree_counts,
        max_degree,
        apex_offset,
        topology.boundary_endpoint_ratio,
        min(len(topology.face_areas) / 8.0, 1.0),
        *faces,
        float(np.mean(supports)) if supports else 0.0,
        float(np.std(supports)) if supports else 0.0,
        min(float(np.mean(contrasts)) / 255.0, 2.0) if contrasts else 0.0,
        min(float(np.std(contrasts)) / 255.0, 2.0) if contrasts else 0.0,
        topology.quality,
    ]
    if include_face_vertices:
        face_vertices = list(topology.face_vertices[:6])
        face_vertices += [0] * (6 - len(face_vertices))
        values.extend(min(value / 8.0, 1.5) for value in face_vertices)
    return np.asarray(values, np.float32)


def render_edge_lines(
    image_bgr: np.ndarray, topology: EdgeTopology, merged: bool = True
) -> np.ndarray:
    canvas = image_bgr.copy()
    lines = topology.merged_lines if merged else topology.raw_lines
    for index, line in enumerate(lines):
        colour = (0, 220, 0) if merged else (0, 150, 255)
        first = (int(round(line.x1)), int(round(line.y1)))
        second = (int(round(line.x2)), int(round(line.y2)))
        cv2.line(canvas, first, second, colour, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(index),
            first,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            colour,
            1,
            cv2.LINE_AA,
        )
    if merged:
        for node in topology.junctions:
            colour = (255, 0, 255) if node.degree >= 3 else (255, 180, 0)
            cv2.circle(canvas, (int(round(node.x)), int(round(node.y))), 5, colour, -1)
    return canvas
