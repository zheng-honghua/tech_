from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np

from .rgbd import CameraIntrinsics, Plane, backproject_pixels, fit_plane_ransac, fit_plane_svd


@dataclass(frozen=True)
class PlanarFace3D:
    face_id: int
    mask: np.ndarray
    plane: Plane
    area_px: int
    center_camera_mm: np.ndarray


@dataclass(frozen=True)
class FusedEdge3D:
    """One internal ridge supported by registered RGB and metric depth."""

    x1: float
    y1: float
    x2: float
    y2: float
    length_px: float
    rgb_support: float
    depth_support: float
    confidence: float
    source: str

    def points(self) -> np.ndarray:
        return np.asarray(((self.x1, self.y1), (self.x2, self.y2)), np.float32)


@dataclass(frozen=True)
class FaceTopology3D:
    faces: tuple[PlanarFace3D, ...]
    adjacency: tuple[tuple[int, int], ...]
    angles_deg: tuple[float, ...]
    triple_junctions: int
    evidence_ratio: float
    rgb_edge_support: float
    fused_edges: tuple[FusedEdge3D, ...] = ()
    rejected_rgb_edges: tuple[FusedEdge3D, ...] = ()
    rgb_candidate_map: np.ndarray | None = None
    depth_candidate_map: np.ndarray | None = None
    fused_edge_map: np.ndarray | None = None

    @property
    def quality(self) -> float:
        if not self.faces:
            return 0.0
        fit = np.mean([max(0.0, 1.0 - face.plane.rmse_mm / 2.0) for face in self.faces])
        coverage = min(1.0, self.evidence_ratio / 0.65)
        return float(np.clip(0.65 * coverage + 0.25 * fit + 0.1 * self.rgb_edge_support, 0, 1))


def _points_for_pixels(
    rows: np.ndarray,
    columns: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    origin_uv: tuple[int, int],
) -> np.ndarray:
    pixels = np.column_stack((columns + origin_uv[0], rows + origin_uv[1])).astype(np.float64)
    return backproject_pixels(pixels, depth_mm[rows, columns], intrinsics)


def _rgb_boundary_support(color_bgr: np.ndarray | None, faces: list[PlanarFace3D]) -> float:
    if color_bgr is None or len(faces) < 2:
        return 0.0
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    threshold = float(np.percentile(gradient, 75))
    strong = gradient >= max(8.0, threshold)
    boundary = np.zeros(gray.shape, np.uint8)
    union = np.zeros(gray.shape, np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for face in faces:
        ring = cv2.morphologyEx(face.mask, cv2.MORPH_GRADIENT, kernel)
        boundary |= ring
        union |= face.mask
    outer = cv2.morphologyEx(union, cv2.MORPH_GRADIENT, kernel)
    outer = cv2.dilate(outer, kernel, iterations=1)
    boundary[outer > 0] = 0
    pixels = boundary > 0
    return float(np.mean(strong[pixels])) if np.any(pixels) else 0.0


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    source = (np.asarray(mask) > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(source)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(source):
        opened = cv2.morphologyEx(source, cv2.MORPH_OPEN, element)
        skeleton |= cv2.subtract(source, opened)
        source = cv2.erode(source, element)
    return skeleton


def _remove_small_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )
    result = np.zeros(mask.shape, np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            result[labels == label] = 255
    return result


def _rgb_candidate_edges(color_bgr: np.ndarray, mask: np.ndarray, scale: float) -> np.ndarray:
    """Return permissive internal RGB edges; depth decides whether they are real."""
    lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (0, 0), 1.1)
    gx = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(np.sum(gx * gx + gy * gy, axis=2))
    inside = magnitude[mask > 0]
    threshold = max(3.0, float(np.percentile(inside, 58)) * 0.62)
    candidate = (magnitude >= threshold).astype(np.uint8) * 255
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    candidate[(mask == 0) | (distance < max(3.0, 0.025 * scale))] = 0
    close_width = max(3, int(round(0.018 * scale)) | 1)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_width, close_width)),
    )
    candidate = _skeletonize(candidate)
    return _remove_small_components(candidate, max(5, int(round(0.002 * scale * scale))))


def _line_from_pixels(
    pixels_xy: np.ndarray,
    rgb_support: float,
    depth_support: float,
    confidence: float,
    source: str,
) -> FusedEdge3D | None:
    points = np.asarray(pixels_xy, np.float32).reshape(-1, 2)
    if len(points) < 2:
        return None
    center = np.mean(points, axis=0)
    covariance = np.cov(points - center, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))].astype(np.float32)
    projection = (points - center) @ direction
    endpoints = np.vstack(
        (center + direction * float(np.min(projection)),
         center + direction * float(np.max(projection)))
    )
    length = float(np.linalg.norm(endpoints[1] - endpoints[0]))
    if not np.isfinite(length) or length <= 0:
        return None
    return FusedEdge3D(
        float(endpoints[0, 0]), float(endpoints[0, 1]),
        float(endpoints[1, 0]), float(endpoints[1, 1]), length,
        float(np.clip(rgb_support, 0, 1)), float(np.clip(depth_support, 0, 1)),
        float(np.clip(confidence, 0, 1)), source,
    )


def _pixel_linearity(pixels_xy: np.ndarray) -> float:
    points = np.asarray(pixels_xy, np.float32).reshape(-1, 2)
    if len(points) < 3:
        return 0.0
    eigenvalues = np.maximum(
        np.linalg.eigvalsh(np.cov(points - np.mean(points, axis=0), rowvar=False)),
        0.0,
    )
    return float(eigenvalues[-1] / max(float(eigenvalues.sum()), 1e-6))


def _sample_line_support(line: FusedEdge3D, support_map: np.ndarray) -> float:
    count = max(12, int(round(line.length_px)))
    weights = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    points = line.points()
    samples = points[0] * (1 - weights) + points[1] * weights
    xy = np.rint(samples).astype(np.int32)
    xy[:, 0] = np.clip(xy[:, 0], 0, support_map.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, support_map.shape[0] - 1)
    return float(np.mean(support_map[xy[:, 1], xy[:, 0]] > 0))


def _depth_boundary_lines(
    faces: list[PlanarFace3D],
    adjacency: list[tuple[int, int]],
    rgb_edges: np.ndarray,
    mask: np.ndarray,
    scale: float,
) -> tuple[list[FusedEdge3D], np.ndarray]:
    depth_map = np.zeros(mask.shape, np.uint8)
    lines: list[FusedEdge3D] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    outer_distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    rgb_support_map = cv2.dilate(rgb_edges, np.ones((5, 5), np.uint8))
    by_id = {face.face_id: face for face in faces}
    for left_id, right_id in adjacency:
        left, right = by_id[left_id], by_id[right_id]
        cosine = abs(float(left.plane.normal @ right.plane.normal))
        normal_angle = float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))
        plane_separation = max(
            abs(float(left.plane.normal @ right.center_camera_mm + left.plane.offset)),
            abs(float(right.plane.normal @ left.center_camera_mm + right.plane.offset)),
        )
        if normal_angle < 6.0 and plane_separation < 1.0:
            continue
        contact = (
            (cv2.dilate(left.mask, kernel) > 0)
            & (cv2.dilate(right.mask, kernel) > 0)
            & (outer_distance >= max(3.0, 0.025 * scale))
        ).astype(np.uint8) * 255
        count, labels, stats, _ = cv2.connectedComponentsWithStats(contact, 8)
        for component_id in range(1, count):
            if int(stats[component_id, cv2.CC_STAT_AREA]) < max(8, int(0.025 * scale)):
                continue
            rows, columns = np.nonzero(labels == component_id)
            pixels = np.column_stack((columns, rows))
            if _pixel_linearity(pixels) < 0.82:
                continue
            line = _line_from_pixels(
                pixels, 0.0, 1.0, 0.70, "depth"
            )
            if line is None or line.length_px < max(8.0, 0.10 * scale):
                continue
            line_count = max(12, int(line.length_px))
            line_weights = np.linspace(0, 1, line_count, dtype=np.float32)[:, None]
            line_samples = line.points()[0] * (1 - line_weights) + line.points()[1] * line_weights
            line_xy = np.rint(line_samples).astype(np.int32)
            line_xy[:, 0] = np.clip(line_xy[:, 0], 0, mask.shape[1] - 1)
            line_xy[:, 1] = np.clip(line_xy[:, 1], 0, mask.shape[0] - 1)
            if np.mean(outer_distance[line_xy[:, 1], line_xy[:, 0]] < max(4.0, 0.06 * scale)) > 0.30:
                continue
            rgb_support = _sample_line_support(line, rgb_support_map)
            line = FusedEdge3D(
                line.x1, line.y1, line.x2, line.y2, line.length_px,
                rgb_support, 1.0, min(1.0, 0.70 + 0.30 * rgb_support),
                "both" if rgb_support >= 0.18 else "depth",
            )
            if rgb_support < 0.06:
                continue
            lines.append(line)
            cv2.line(
                depth_map,
                tuple(np.rint(line.points()[0]).astype(int)),
                tuple(np.rint(line.points()[1]).astype(int)),
                255, 1, cv2.LINE_AA,
            )
    return lines, depth_map


def _line_angle(line: FusedEdge3D) -> float:
    vector = line.points()[1] - line.points()[0]
    return float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0)


def _angle_difference(first: FusedEdge3D, second: FusedEdge3D) -> float:
    difference = abs(_line_angle(first) - _line_angle(second)) % 180.0
    return min(difference, 180.0 - difference)


def _line_distance(first: FusedEdge3D, second: FusedEdge3D) -> float:
    return float(min(
        np.linalg.norm(point - other)
        for point in first.points() for other in second.points()
    ))


def _matching_depth_support(
    line: FusedEdge3D, depth_lines: list[FusedEdge3D], scale: float
) -> float:
    matches = [
        other for other in depth_lines
        if _angle_difference(line, other) <= 18.0
        and _line_distance(line, other) <= max(4.0, 0.07 * scale)
    ]
    if not matches:
        return 0.0
    distance = min(_line_distance(line, other) for other in matches)
    return float(np.clip(1.0 - distance / max(4.0, 0.07 * scale), 0, 1))


def _bilateral_depth_support(
    line: FusedEdge3D,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    origin_uv: tuple[int, int],
    scale: float,
) -> float:
    points = line.points()
    direction = points[1] - points[0]
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    normal = np.asarray((-direction[1], direction[0]), np.float32)
    offset = max(2.0, 0.025 * scale)
    weights = np.linspace(0.08, 0.92, max(18, int(line.length_px)), dtype=np.float32)[:, None]
    center = points[0] * (1 - weights) + points[1] * weights
    planes: list[Plane] = []
    point_sets: list[np.ndarray] = []
    for sign in (-1.0, 1.0):
        # A one-pixel-wide trace is collinear in 3-D and cannot determine a
        # plane normal.  Use a real band on each side of the proposed ridge.
        band_offsets = offset + np.arange(3, dtype=np.float32) * 2.0
        samples = np.vstack([
            center + sign * normal * band_offset
            for band_offset in band_offsets
        ])
        xy = np.rint(samples).astype(np.int32)
        valid_xy = (
            (xy[:, 0] >= 0) & (xy[:, 0] < depth.shape[1])
            & (xy[:, 1] >= 0) & (xy[:, 1] < depth.shape[0])
        )
        xy = xy[valid_xy]
        if len(xy) < 12:
            return 0.0
        values = depth[xy[:, 1], xy[:, 0]]
        valid = (mask[xy[:, 1], xy[:, 0]] > 0) & np.isfinite(values) & (values > 0)
        if float(np.mean(valid)) < 0.75 or int(np.count_nonzero(valid)) < 12:
            return 0.0
        xy = xy[valid]
        values = values[valid]
        camera_points = _points_for_pixels(
            xy[:, 1], xy[:, 0], depth, intrinsics, origin_uv
        )
        plane = fit_plane_svd(camera_points)
        if plane.rmse_mm > 1.8:
            return 0.0
        planes.append(plane)
        point_sets.append(camera_points)
    cosine = abs(float(planes[0].normal @ planes[1].normal))
    angle = float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))
    angle_score = np.clip((angle - 6.0) / 24.0, 0, 1)
    # Raw depth differs naturally across a tilted flat face.  Cross-evaluate
    # each fitted plane on the opposite band so only a crease/step remains.
    cross_residual = max(
        float(np.median(np.abs(point_sets[1] @ planes[0].normal + planes[0].offset))),
        float(np.median(np.abs(point_sets[0] @ planes[1].normal + planes[1].offset))),
    )
    separation_score = np.clip((cross_residual - 0.7) / 3.0, 0, 1)
    return float(max(angle_score, separation_score))


def _rgb_lines(
    rgb_edges: np.ndarray,
    depth_lines: list[FusedEdge3D],
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    origin_uv: tuple[int, int],
    scale: float,
) -> tuple[list[FusedEdge3D], list[FusedEdge3D]]:
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(rgb_edges)[0]
    accepted: list[FusedEdge3D] = []
    rejected: list[FusedEdge3D] = []
    if detected is None:
        return accepted, rejected
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    support_map = cv2.dilate(rgb_edges, np.ones((3, 3), np.uint8))
    coordinates_by_length = sorted(
        detected.reshape(-1, 4),
        key=lambda value: -float(np.linalg.norm(value[2:] - value[:2])),
    )[:32]
    for coordinates in coordinates_by_length:
        endpoints = coordinates.reshape(2, 2).astype(np.float32)
        provisional = _line_from_pixels(endpoints, 0.0, 0.0, 0.0, "rgb")
        if provisional is None or provisional.length_px < max(8.0, 0.10 * scale):
            continue
        count = max(12, int(provisional.length_px))
        weights = np.linspace(0, 1, count, dtype=np.float32)[:, None]
        samples = provisional.points()[0] * (1 - weights) + provisional.points()[1] * weights
        xy = np.rint(samples).astype(np.int32)
        xy[:, 0] = np.clip(xy[:, 0], 0, mask.shape[1] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, mask.shape[0] - 1)
        if np.mean(mask[xy[:, 1], xy[:, 0]] > 0) < 0.88:
            continue
        if np.mean(distance[xy[:, 1], xy[:, 0]] < max(3.0, 0.025 * scale)) > 0.30:
            continue
        center_depth = depth[xy[:, 1], xy[:, 0]]
        if np.mean(np.isfinite(center_depth) & (center_depth > 0)) < 0.70:
            continue
        rgb_support = _sample_line_support(provisional, support_map)
        matched = _matching_depth_support(provisional, depth_lines, scale)
        bilateral = (
            0.0 if matched >= 0.18
            else _bilateral_depth_support(
                provisional, depth, mask, intrinsics, origin_uv, scale
            )
        )
        depth_support = max(matched, bilateral)
        confidence = 0.35 * rgb_support + 0.65 * depth_support
        line = FusedEdge3D(
            provisional.x1, provisional.y1, provisional.x2, provisional.y2,
            provisional.length_px, rgb_support, depth_support, confidence,
            "both" if matched > 0 else "rgb_depth_planes",
        )
        if depth_support >= 0.30 and confidence >= 0.45:
            accepted.append(line)
        else:
            rejected.append(line)
    return accepted, rejected


def _merge_fused_edges(lines: list[FusedEdge3D], scale: float) -> list[FusedEdge3D]:
    kept: list[FusedEdge3D] = []
    for line in sorted(lines, key=lambda item: (-item.confidence, -item.length_px)):
        duplicate_index = next((
            index for index, other in enumerate(kept)
            if _angle_difference(line, other) <= 15.0
            and _line_distance(line, other) <= max(4.0, 0.08 * scale)
        ), None)
        if duplicate_index is None:
            kept.append(line)
            continue
        other = kept[duplicate_index]
        merged = _line_from_pixels(
            np.vstack((line.points(), other.points())),
            max(line.rgb_support, other.rgb_support),
            max(line.depth_support, other.depth_support),
            max(line.confidence, other.confidence),
            "both" if "both" in {line.source, other.source} else other.source,
        )
        if merged is not None:
            kept[duplicate_index] = merged
    return sorted(kept, key=lambda item: item.length_px, reverse=True)[:16]


def _extract_fused_edges(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    faces: list[PlanarFace3D],
    adjacency: list[tuple[int, int]],
    intrinsics: CameraIntrinsics,
    origin_uv: tuple[int, int],
) -> tuple[tuple[FusedEdge3D, ...], tuple[FusedEdge3D, ...], np.ndarray, np.ndarray, np.ndarray]:
    scale = max(float(np.sqrt(cv2.countNonZero(mask))), 1.0)
    rgb_map = _rgb_candidate_edges(color_bgr, mask, scale)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        perimeter = max(float(cv2.arcLength(hull, True)), 1.0)
        area = max(float(cv2.contourArea(hull)), 1.0)
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        vertices = len(cv2.approxPolyDP(hull, 0.025 * perimeter, True))
        if circularity >= 0.84 and vertices >= 7:
            empty = np.zeros(mask.shape, np.uint8)
            return (), (), rgb_map, empty, empty.copy()
    depth_lines, depth_map = _depth_boundary_lines(
        faces, adjacency, rgb_map, mask, scale
    )
    rgb_lines, rejected = _rgb_lines(
        rgb_map, depth_lines, depth, mask, intrinsics, origin_uv, scale
    )
    fused = _merge_fused_edges([*depth_lines, *rgb_lines], scale)
    fused_map = np.zeros(mask.shape, np.uint8)
    for line in fused:
        cv2.line(
            fused_map,
            tuple(np.rint(line.points()[0]).astype(int)),
            tuple(np.rint(line.points()[1]).astype(int)),
            255, 2, cv2.LINE_AA,
        )
    return tuple(fused), tuple(rejected), rgb_map, depth_map, fused_map


def extract_face_topology(
    depth_crop_mm: np.ndarray,
    crop_mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    crop_origin_uv: tuple[int, int] = (0, 0),
    color_crop_bgr: np.ndarray | None = None,
    plane_threshold_mm: float = 1.4,
    min_face_area_px: int = 90,
    max_faces: int = 8,
    extract_fused_edges: bool = False,
) -> FaceTopology3D:
    """Extract visible planar patches without Hough/line detection.

    RANSAC hypotheses are estimated from face interiors. Morphology joins small
    depth holes, while connected support prevents unrelated coplanar regions
    from becoming one face.
    """
    depth = np.asarray(depth_crop_mm, np.float32)
    mask = (np.asarray(crop_mask) > 0).astype(np.uint8) * 255
    if depth.shape != mask.shape:
        raise ValueError("depth crop and object mask dimensions differ")
    valid = (mask > 0) & np.isfinite(depth) & (depth > 0)
    interior = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1) > 0
    remaining = valid & interior
    original_count = max(1, int(np.count_nonzero(valid)))
    faces: list[PlanarFace3D] = []
    rejected_rounds = 0
    while len(faces) < max_faces and np.count_nonzero(remaining) >= min_face_area_px:
        rows, columns = np.nonzero(remaining)
        points = _points_for_pixels(rows, columns, depth, intrinsics, crop_origin_uv)
        stride = max(1, len(points) // 3000)
        hypothesis = fit_plane_ransac(
            points[::stride], threshold_mm=plane_threshold_mm, iterations=60,
            seed=17 + len(faces) + rejected_rounds,
        )
        distances = np.abs(points @ hypothesis.normal + hypothesis.offset)
        inlier_mask = np.zeros(depth.shape, np.uint8)
        inlier_mask[rows[distances <= plane_threshold_mm], columns[distances <= plane_threshold_mm]] = 255
        inlier_mask = cv2.morphologyEx(
            inlier_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1
        )
        inlier_mask[~remaining] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(inlier_mask, 8)
        candidates = [index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= min_face_area_px]
        if not candidates:
            remaining[inlier_mask > 0] = False
            rejected_rounds += 1
            if rejected_rounds >= 3:
                break
            continue
        component_id = max(candidates, key=lambda index: stats[index, cv2.CC_STAT_AREA])
        component = (labels == component_id).astype(np.uint8) * 255
        face_rows, face_columns = np.nonzero(component)
        face_points = _points_for_pixels(
            face_rows, face_columns, depth, intrinsics, crop_origin_uv
        )
        plane = fit_plane_svd(face_points)
        faces.append(
            PlanarFace3D(
                len(faces), component, plane, len(face_points),
                np.mean(face_points, axis=0),
            )
        )
        remaining[component > 0] = False

    adjacency: list[tuple[int, int]] = []
    angles: list[float] = []
    kernel = np.ones((5, 5), np.uint8)
    for left, right in itertools.combinations(faces, 2):
        touching = np.any(
            (cv2.dilate(left.mask, kernel, iterations=1) > 0) & (right.mask > 0)
        )
        if not touching:
            continue
        adjacency.append((left.face_id, right.face_id))
        cosine = abs(float(left.plane.normal @ right.plane.normal))
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))))

    adjacent_set = {tuple(sorted(pair)) for pair in adjacency}
    triples = 0
    for one, two, three in itertools.combinations(faces, 3):
        ids = (one.face_id, two.face_id, three.face_id)
        if not all(tuple(sorted(pair)) in adjacent_set for pair in itertools.combinations(ids, 2)):
            continue
        normals = np.vstack((one.plane.normal, two.plane.normal, three.plane.normal))
        if abs(float(np.linalg.det(normals))) >= 0.08:
            triples += 1
    evidence = sum(face.area_px for face in faces) / original_count
    fused_edges: tuple[FusedEdge3D, ...] = ()
    rejected_rgb_edges: tuple[FusedEdge3D, ...] = ()
    rgb_candidate_map = depth_candidate_map = fused_edge_map = None
    if extract_fused_edges and color_crop_bgr is not None:
        (
            fused_edges, rejected_rgb_edges, rgb_candidate_map,
            depth_candidate_map, fused_edge_map,
        ) = _extract_fused_edges(
            color_crop_bgr, depth, mask, faces, adjacency, intrinsics,
            crop_origin_uv,
        )
    return FaceTopology3D(
        tuple(faces), tuple(adjacency), tuple(angles), triples,
        float(np.clip(evidence, 0, 1)), _rgb_boundary_support(color_crop_bgr, faces),
        fused_edges, rejected_rgb_edges, rgb_candidate_map, depth_candidate_map,
        fused_edge_map,
    )


TOPOLOGY_FEATURE_NAMES = (
    "plane_face_count", "plane_evidence_ratio", "plane_quality",
    "plane_area_largest", "plane_area_second", "plane_area_third",
    "plane_rmse_mean", "plane_rmse_max", "face_adjacency_count",
    "face_degree_max", "angle_parallel_ratio", "angle_acute_ratio",
    "angle_oblique_ratio", "angle_orthogonal_ratio", "angle_mean_deg",
    "angle_std_deg", "triple_junction_count", "rgb_edge_support",
)

FUSED_EDGE_FEATURE_NAMES = (
    "fused_edge_count", "fused_edge_length_ratio",
    "depth_boundary_rgb_support_ratio", "rgb_line_depth_support_ratio",
    "weak_edge_recovery_ratio", "fused_edge_confidence_mean",
    "fused_junction_count", "fused_parallel_pair_ratio",
    "fused_converging_pair_ratio",
)


def face_topology_features(topology: FaceTopology3D) -> np.ndarray:
    areas = sorted((face.area_px for face in topology.faces), reverse=True)
    total = max(float(sum(areas)), 1.0)
    area_ratios = [(areas[index] / total) if index < len(areas) else 0.0 for index in range(3)]
    rmses = [face.plane.rmse_mm for face in topology.faces]
    angles = np.asarray(topology.angles_deg, np.float32)
    degrees = [0] * len(topology.faces)
    for left, right in topology.adjacency:
        degrees[left] += 1
        degrees[right] += 1
    if len(angles):
        bins = [
            float(np.mean(angles < 15)),
            float(np.mean((angles >= 15) & (angles < 45))),
            float(np.mean((angles >= 45) & (angles < 75))),
            float(np.mean(angles >= 75)),
        ]
        angle_mean, angle_std = float(np.mean(angles)), float(np.std(angles))
    else:
        bins = [0.0] * 4
        angle_mean = angle_std = 0.0
    return np.asarray(
        [
            float(len(topology.faces)), topology.evidence_ratio, topology.quality,
            *area_ratios,
            float(np.mean(rmses)) if rmses else 0.0,
            float(np.max(rmses)) if rmses else 0.0,
            float(len(topology.adjacency)), float(max(degrees, default=0)),
            *bins, angle_mean, angle_std, float(topology.triple_junctions),
            topology.rgb_edge_support,
        ],
        np.float32,
    )


def _segment_intersection(first: FusedEdge3D, second: FusedEdge3D) -> bool:
    p, p2 = first.points()
    q, q2 = second.points()
    r, s = p2 - p, q2 - q
    denominator = float(r[0] * s[1] - r[1] * s[0])
    if abs(denominator) < 1e-6:
        return False
    delta = q - p
    t = float((delta[0] * s[1] - delta[1] * s[0]) / denominator)
    u = float((delta[0] * r[1] - delta[1] * r[0]) / denominator)
    return -0.08 <= t <= 1.08 and -0.08 <= u <= 1.08


def fused_edge_features(topology: FaceTopology3D) -> np.ndarray:
    lines = list(topology.fused_edges)
    scale = max(
        float(np.sqrt(sum(face.area_px for face in topology.faces))), 1.0
    )
    depth_lines = [line for line in lines if line.source in {"depth", "both"}]
    rgb_lines = [line for line in lines if line.rgb_support > 0]
    weak = [line for line in depth_lines if 0 < line.rgb_support < 0.25]
    pairs = list(itertools.combinations(lines, 2))
    parallel = sum(_angle_difference(left, right) <= 8.0 for left, right in pairs)
    converging = sum(
        _angle_difference(left, right) > 8.0 and _segment_intersection(left, right)
        for left, right in pairs
    )
    junctions: list[np.ndarray] = []
    tolerance = max(3.0, 0.04 * scale)
    for left, right in pairs:
        if not _segment_intersection(left, right):
            continue
        for point in (*left.points(), *right.points()):
            if not any(np.linalg.norm(point - existing) <= tolerance for existing in junctions):
                junctions.append(point)
    pair_count = max(len(pairs), 1)
    return np.asarray(
        [
            float(len(lines)),
            sum(line.length_px for line in lines) / scale,
            float(np.mean([line.rgb_support for line in depth_lines])) if depth_lines else 0.0,
            float(np.mean([line.depth_support for line in rgb_lines])) if rgb_lines else 0.0,
            len(weak) / max(len(depth_lines), 1),
            float(np.mean([line.confidence for line in lines])) if lines else 0.0,
            float(len(junctions)), parallel / pair_count, converging / pair_count,
        ],
        np.float32,
    )
