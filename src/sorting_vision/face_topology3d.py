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
class FaceTopology3D:
    faces: tuple[PlanarFace3D, ...]
    adjacency: tuple[tuple[int, int], ...]
    angles_deg: tuple[float, ...]
    triple_junctions: int
    evidence_ratio: float
    rgb_edge_support: float

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


def extract_face_topology(
    depth_crop_mm: np.ndarray,
    crop_mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    crop_origin_uv: tuple[int, int] = (0, 0),
    color_crop_bgr: np.ndarray | None = None,
    plane_threshold_mm: float = 1.4,
    min_face_area_px: int = 90,
    max_faces: int = 8,
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
    return FaceTopology3D(
        tuple(faces), tuple(adjacency), tuple(angles), triples,
        float(np.clip(evidence, 0, 1)), _rgb_boundary_support(color_crop_bgr, faces),
    )


TOPOLOGY_FEATURE_NAMES = (
    "plane_face_count", "plane_evidence_ratio", "plane_quality",
    "plane_area_largest", "plane_area_second", "plane_area_third",
    "plane_rmse_mean", "plane_rmse_max", "face_adjacency_count",
    "face_degree_max", "angle_parallel_ratio", "angle_acute_ratio",
    "angle_oblique_ratio", "angle_orthogonal_ratio", "angle_mean_deg",
    "angle_std_deg", "triple_junction_count", "rgb_edge_support",
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
