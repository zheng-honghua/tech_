from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .shape_registry import ShapeClass, ShapeRegistry
from .side_geometry import SideFeatureResult, extract_side_features
from .joint_topology import GRAPH_FEATURE_NAMES, build_joint_topology


CROSS_VIEW_FEATURE_VERSION = 4
CROSS_VIEW_FEATURE_NAMES = (
    "segment_count",
    "internal_segment_ratio",
    "boundary_segment_ratio",
    "junction_density",
    "orientation_entropy",
    "parallel_pair_ratio",
    "converging_pair_ratio",
    "projected_point_mask_ratio",
    "projected_point_mask_distance",
    "line_point_support_mean",
    "line_point_support_max",
    "supported_line_ratio",
    "supplemental_line_ratio",
    "obb_edge_support_ratio",
    "obb_edge_residual",
    "lifted_line_ratio",
    "lifted_span_ratio",
    "spatial_quality",
)


@dataclass(frozen=True)
class CrossViewFeatureResult:
    vector: np.ndarray
    group_ids: np.ndarray
    diagnostics: dict[str, Any]
    segments_px: np.ndarray
    projected_obb_edges_px: np.ndarray


def _camera_matrix(intrinsics: Any) -> np.ndarray:
    return np.asarray(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        np.float64,
    )


def _line_distance(points: np.ndarray, segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = segment[:2].astype(np.float64)
    direction = segment[2:].astype(np.float64) - start
    length2 = max(float(direction @ direction), 1e-9)
    delta = np.asarray(points, np.float64) - start
    along = delta @ direction / length2
    projection = start + along[:, None] * direction
    distance = np.linalg.norm(np.asarray(points, np.float64) - projection, axis=1)
    return distance, along


def _segment_pair_features(segments: np.ndarray, diagonal: float) -> tuple[float, float, float]:
    if len(segments) < 2:
        return 0.0, 0.0, 0.0
    angles = np.degrees(
        np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0])
    ) % 180.0
    endpoints = segments.reshape(-1, 2, 2)
    parallel = converging = junctions = pairs = 0
    for first in range(len(segments)):
        for second in range(first + 1, len(segments)):
            delta = abs(float(angles[first] - angles[second]))
            delta = min(delta, 180.0 - delta)
            parallel += int(delta <= 8.0)
            converging += int(12.0 <= delta <= 70.0)
            endpoint_distance = np.linalg.norm(
                endpoints[first, :, None, :] - endpoints[second, None, :, :], axis=2
            )
            junctions += int(delta >= 12.0 and float(endpoint_distance.min()) <= 0.08 * diagonal)
            pairs += 1
    return parallel / pairs, converging / pairs, junctions / max(len(segments), 1)


def _oriented_box_edges(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cloud = np.asarray(points, np.float64).reshape(-1, 3)
    if len(cloud) < 8:
        return np.empty((0, 3), np.float64), np.empty((0, 2), np.int32)
    center = np.median(cloud, axis=0)
    _, _, axes = np.linalg.svd(cloud - center, full_matrices=False)
    local = (cloud - center) @ axes.T
    lower = np.percentile(local, 2.0, axis=0)
    upper = np.percentile(local, 98.0, axis=0)
    corners_local = np.asarray(
        [
            [upper[0] if code & 1 else lower[0],
             upper[1] if code & 2 else lower[1],
             upper[2] if code & 4 else lower[2]]
            for code in range(8)
        ],
        np.float64,
    )
    corners = corners_local @ axes + center
    edges = np.asarray(
        [(first, second) for first in range(8) for second in range(first + 1, 8)
         if bin(first ^ second).count("1") == 1],
        np.int32,
    )
    return corners, edges


def _project_box_edges(
    points: np.ndarray, calibration: Any,
    box: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    corners, edge_indices = _oriented_box_edges(points) if box is None else box
    if not len(corners):
        return np.empty((0, 4), np.float32)
    side = calibration.transform_primary_points(corners)
    if np.any(side[:, 2] <= 1e-6):
        return np.empty((0, 4), np.float32)
    projected, _ = cv2.projectPoints(
        side,
        np.zeros(3, np.float64),
        np.zeros(3, np.float64),
        _camera_matrix(calibration.side_intrinsics),
        calibration.side_distortion,
    )
    pixels = projected.reshape(-1, 2)
    return np.asarray(
        [np.concatenate((pixels[first], pixels[second])) for first, second in edge_indices],
        np.float32,
    )


def _segment_plane_intersections(
    segment: np.ndarray, points_primary_mm: np.ndarray, calibration: Any,
    box: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[int, float]:
    pixels = np.asarray([segment[:2], segment[2:]], np.float64).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(
        pixels,
        _camera_matrix(calibration.side_intrinsics),
        calibration.side_distortion,
    ).reshape(-1, 2)
    rays3 = np.column_stack((rays, np.ones(2, np.float64)))
    normal_side = np.cross(rays3[0], rays3[1])
    norm = float(np.linalg.norm(normal_side))
    if norm <= 1e-9:
        return 0, 0.0
    normal_side /= norm
    transform = np.asarray(calibration.side_from_primary, np.float64)
    normal_primary = transform[:3, :3].T @ normal_side
    offset = float(normal_side @ transform[:3, 3])
    corners, edges = _oriented_box_edges(points_primary_mm) if box is None else box
    intersections: list[np.ndarray] = []
    for first, second in edges:
        start, end = corners[first], corners[second]
        first_value = float(start @ normal_primary + offset)
        second_value = float(end @ normal_primary + offset)
        denominator = first_value - second_value
        if abs(denominator) <= 1e-9:
            continue
        fraction = first_value / denominator
        if -1e-5 <= fraction <= 1.0 + 1e-5:
            intersections.append(start + fraction * (end - start))
    if len(intersections) < 2:
        return len(intersections), 0.0
    values = np.asarray(intersections)
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    return len(intersections), float(distances.max())


def extract_cross_view_features(
    side_image_bgr: np.ndarray,
    side_mask: np.ndarray,
    primary_points_mm: np.ndarray,
    calibration: Any,
) -> CrossViewFeatureResult:
    """Retain side line topology and bind it to the calibrated RGB-D cloud.

    A side image line defines a ray plane, not an independent metric 3-D edge.
    The extractor therefore reports support against the primary point cloud and
    its robust oriented envelope; it never invents depth or grasp coordinates.
    """

    image = np.asarray(side_image_bgr)
    mask = (np.asarray(side_mask) > 0).astype(np.uint8) * 255
    points = np.asarray(primary_points_mm, np.float64).reshape(-1, 3)
    if image.ndim != 3 or image.shape[:2] != mask.shape:
        raise ValueError("side image and mask dimensions must match")
    if len(points) < 8:
        raise ValueError("cross-view topology needs at least eight primary points")
    side_features: SideFeatureResult = extract_side_features(image, mask)
    segments = side_features.segments_px
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("cross-view topology needs a non-empty side mask")
    diagonal = max(float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min())), 1.0)
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    boundary_distance = cv2.distanceTransform(cv2.bitwise_not(boundary), cv2.DIST_L2, 3)

    parallel, converging, junction_density = _segment_pair_features(segments, diagonal)
    angles = (
        np.degrees(np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0]))
        % 180.0
        if len(segments) else np.empty(0)
    )
    histogram, _ = np.histogram(angles, bins=12, range=(0.0, 180.0))
    probabilities = histogram / max(float(histogram.sum()), 1.0)
    orientation_entropy = float(
        -np.sum(probabilities * np.log(probabilities + 1e-12)) / np.log(12.0)
    )

    boundary_flags: list[bool] = []
    for segment in segments:
        samples = np.linspace(segment[:2], segment[2:], 12)
        uv = np.rint(samples).astype(np.int32)
        uv[:, 0] = np.clip(uv[:, 0], 0, mask.shape[1] - 1)
        uv[:, 1] = np.clip(uv[:, 1], 0, mask.shape[0] - 1)
        boundary_flags.append(float(np.median(boundary_distance[uv[:, 1], uv[:, 0]])) <= 0.025 * diagonal)

    stride = max(1, len(points) // 2000)
    projected = calibration.project_primary_points(points[::stride])
    finite = np.all(np.isfinite(projected), axis=1)
    projected = projected[finite]
    inside = (
        (projected[:, 0] >= 0) & (projected[:, 0] < mask.shape[1])
        & (projected[:, 1] >= 0) & (projected[:, 1] < mask.shape[0])
    )
    projected = projected[inside]
    if not len(projected):
        raise ValueError("primary point cloud does not project into the side image")
    rounded = np.rint(projected).astype(np.int32)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, mask.shape[1] - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, mask.shape[0] - 1)
    projected_mask_ratio = float(np.mean(mask[rounded[:, 1], rounded[:, 0]] > 0))
    mask_distance = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3)
    projected_mask_distance = float(
        np.median(mask_distance[rounded[:, 1], rounded[:, 0]]) / diagonal
    )

    line_supports: list[float] = []
    lifted: list[bool] = []
    lifted_spans: list[float] = []
    # The same object envelope constrains every ray plane. Do one PCA/SVD per
    # object, not one per measured line plus another for projection.
    box = _oriented_box_edges(points)
    for segment in segments:
        distance, along = _line_distance(projected, segment)
        support = float(np.mean((distance <= max(3.0, 0.018 * diagonal)) & (along >= -0.1) & (along <= 1.1)))
        line_supports.append(support)
        intersections, span = _segment_plane_intersections(segment, points, calibration, box)
        lifted.append(intersections >= 2)
        lifted_spans.append(span)

    projected_box_edges = _project_box_edges(points, calibration, box)
    obb_residuals: list[float] = []
    for edge in projected_box_edges:
        length = float(np.linalg.norm(edge[2:] - edge[:2]))
        if length < 3.0 or not len(segments):
            continue
        edge_angle = float(np.degrees(np.arctan2(edge[3] - edge[1], edge[2] - edge[0])) % 180.0)
        best = 1.0
        edge_midpoint = (edge[:2] + edge[2:]) * 0.5
        for segment, angle in zip(segments, angles):
            delta = abs(edge_angle - float(angle))
            delta = min(delta, 180.0 - delta) / 30.0
            distance, _ = _line_distance(edge_midpoint.reshape(1, 2), segment)
            best = min(best, 0.55 * min(delta, 1.0) + 0.45 * min(float(distance[0]) / diagonal, 1.0))
        obb_residuals.append(best)

    line_support = np.asarray(line_supports, np.float32)
    support_threshold = 0.015
    supported_ratio = float(np.mean(line_support >= support_threshold)) if len(line_support) else 0.0
    supplemental_ratio = float(
        np.mean((line_support < support_threshold) & (~np.asarray(boundary_flags, bool)))
    ) if len(line_support) else 0.0
    obb_edge_support = float(np.mean(np.asarray(obb_residuals) <= 0.28)) if obb_residuals else 0.0
    obb_residual = float(np.median(obb_residuals)) if obb_residuals else 1.0
    lifted_ratio = float(np.mean(lifted)) if lifted else 0.0
    object_extent = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
    lifted_span_ratio = float(np.median(lifted_spans) / object_extent) if lifted_spans else 0.0
    spatial_quality = float(np.clip(
        0.45 * projected_mask_ratio
        + 0.30 * (1.0 - min(projected_mask_distance / 0.10, 1.0))
        + 0.25 * obb_edge_support,
        0.0,
        1.0,
    ))
    cross = np.asarray(
        [
            min(len(segments) / 24.0, 1.5),
            float(np.mean(~np.asarray(boundary_flags, bool))) if boundary_flags else 0.0,
            float(np.mean(boundary_flags)) if boundary_flags else 0.0,
            min(junction_density / 2.0, 1.5),
            orientation_entropy,
            parallel,
            converging,
            projected_mask_ratio,
            min(projected_mask_distance / 0.10, 2.0),
            float(np.mean(line_support)) if len(line_support) else 0.0,
            float(np.max(line_support)) if len(line_support) else 0.0,
            supported_ratio,
            supplemental_ratio,
            obb_edge_support,
            obb_residual,
            lifted_ratio,
            min(lifted_span_ratio, 2.0),
            spatial_quality,
        ],
        np.float32,
    )
    graph = build_joint_topology(points, segments, calibration)
    graph_vector = np.asarray([graph["features"][name] for name in GRAPH_FEATURE_NAMES], np.float32)
    vector = np.concatenate((side_features.vector, cross, graph_vector)).astype(np.float32)
    group_ids = np.concatenate(
        (side_features.group_ids, np.full(len(cross), 4, np.int32), np.full(len(graph_vector), 5, np.int32))
    )
    diagnostics: dict[str, Any] = {
        "joint_topology_graph": graph,
        **side_features.diagnostics,
        **{name: float(value) for name, value in zip(CROSS_VIEW_FEATURE_NAMES, cross)},
        "segment_count_raw": int(len(segments)),
        "segments_px": np.rint(segments).astype(int).tolist(),
        "projected_obb_edges_px": np.rint(projected_box_edges).astype(int).tolist(),
        "feature_version": CROSS_VIEW_FEATURE_VERSION,
    }
    return CrossViewFeatureResult(vector, group_ids, diagnostics, segments, projected_box_edges)


class CrossViewTopologyModel:
    """Classifier over side silhouette plus calibrated cross-view topology."""

    backend = "opencv"

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray,
        group_ids: np.ndarray,
        registry: ShapeRegistry,
        *,
        method: str = "rtrees",
        distance_threshold: float = 5.0,
        margin_threshold: float = 0.08,
    ) -> None:
        if method not in {"knn", "rtrees"}:
            raise ValueError("cross-view method must be knn or rtrees")
        self.features = np.asarray(features, np.float32)
        self.labels = np.asarray(labels).astype(str)
        self.mean = np.asarray(mean, np.float32)
        self.scale = np.maximum(np.asarray(scale, np.float32), 1e-5)
        self.group_ids = np.asarray(group_ids, np.int32)
        self.feature_version = 4 if np.any(self.group_ids == 5) else 3
        self.registry = registry
        self.class_ids = registry.class_ids
        self.registry_hash = registry.registry_hash
        self.method = method
        self.distance_threshold = float(distance_threshold)
        self.margin_threshold = float(margin_threshold)
        self.last_class_scores: dict[str, float] = {}
        self.last_feature_diagnostics: dict[str, Any] = {}
        if self.features.ndim != 2 or not len(self.features) or len(self.features) != len(self.labels):
            raise ValueError("cross-view model needs matching non-empty features and labels")
        if self.features.shape[1] != len(self.mean) or len(self.mean) != len(self.scale):
            raise ValueError("cross-view feature dimensions do not match")
        if len(self.group_ids) != self.features.shape[1]:
            raise ValueError("cross-view feature groups do not match")
        unknown = sorted(set(self.labels) - set(self.class_ids))
        if unknown:
            raise ValueError(f"cross-view labels are absent from registry: {unknown}")
        self._forest = self._train_forest() if method == "rtrees" else None

    @classmethod
    def train(
        cls,
        samples: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, Any, str]],
        registry: ShapeRegistry,
        *,
        method: str = "rtrees",
    ) -> "CrossViewTopologyModel":
        vectors: list[np.ndarray] = []
        labels: list[str] = []
        groups: np.ndarray | None = None
        for image, mask, points, calibration, raw_label in samples:
            extracted = extract_cross_view_features(image, mask, points, calibration)
            vectors.append(extracted.vector)
            labels.append(registry.resolve(raw_label))
            groups = extracted.group_ids
        if not vectors or groups is None:
            raise ValueError("no usable cross-view training samples")
        raw = np.stack(vectors).astype(np.float32)
        mean = raw.mean(axis=0)
        scale = raw.std(axis=0)
        scale[scale < 1e-5] = 1.0
        normalized = (raw - mean) / scale
        missing = sorted(set(registry.class_ids) - set(labels))
        if missing:
            raise ValueError(f"cross-view training set does not cover every enabled class: {missing}")
        model = cls(normalized, np.asarray(labels), mean, scale, groups, registry, method=method)
        leave_one_out: list[float] = []
        for index, vector in enumerate(normalized):
            distances = model._distances(vector).astype(np.float64)
            distances[index] = np.inf
            same = distances[model.labels == model.labels[index]]
            finite = same[np.isfinite(same)]
            if len(finite):
                leave_one_out.append(float(finite.min()))
        model.distance_threshold = max(
            float(np.percentile(leave_one_out, 99)) * 1.35 if leave_one_out else 5.0,
            0.25,
        )
        return model

    def _distances(self, feature: np.ndarray) -> np.ndarray:
        squared = (self.features - feature) ** 2
        output = np.zeros(len(self.features), np.float32)
        weights = np.asarray(
            [0.15, 0.13, 0.15, 0.13, 0.24, 0.20] if self.feature_version == 4
            else [0.18, 0.16, 0.18, 0.16, 0.32], np.float32,
        )
        for group, weight in enumerate(weights):
            selected = self.group_ids == group
            if np.any(selected):
                output += float(weight) * squared[:, selected].mean(axis=1)
        return np.sqrt(output / float(weights.sum()))

    def _train_forest(self) -> Any:
        forest = cv2.ml.RTrees_create()
        forest.setMaxDepth(14)
        forest.setMinSampleCount(2)
        forest.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 220, 0.0))
        ids = np.asarray([self.class_ids.index(item) for item in self.labels], np.int32)
        cv2.setRNGSeed(20260917)
        forest.train(self.features, cv2.ml.ROW_SAMPLE, ids)
        return forest

    def _scores(self, feature: np.ndarray) -> dict[str, float]:
        if self.method == "rtrees":
            assert self._forest is not None
            votes = np.asarray(self._forest.getVotes(feature.reshape(1, -1), 0))
            scores = {item: 0.0 for item in self.class_ids}
            if votes.ndim == 2 and votes.shape[0] >= 2:
                for raw_class, count in zip(votes[0], votes[1]):
                    index = int(raw_class)
                    if 0 <= index < len(self.class_ids):
                        scores[self.class_ids[index]] = float(count)
            total = sum(scores.values())
            if total <= 0:
                _, predicted = self._forest.predict(feature.reshape(1, -1))
                scores[self.class_ids[int(predicted[0, 0])]] = 1.0
                total = 1.0
            return {key: value / total for key, value in scores.items()}
        distances = self._distances(feature)
        class_distances = []
        for label in self.class_ids:
            selected = np.sort(distances[self.labels == label])
            class_distances.append(float(np.mean(selected[: min(3, len(selected))])))
        values = np.asarray(class_distances, np.float64)
        logits = -(values - values.min()) / max(float(np.std(values)), 0.10)
        probabilities = np.exp(logits - logits.max())
        probabilities /= max(float(probabilities.sum()), 1e-12)
        return {label: float(probabilities[index]) for index, label in enumerate(self.class_ids)}

    def predict(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        primary_points_mm: np.ndarray,
        calibration: Any,
    ) -> tuple[str, float, dict[str, Any]]:
        try:
            extracted = extract_cross_view_features(image_bgr, mask, primary_points_mm, calibration)
        except ValueError as error:
            self.last_class_scores = {}
            self.last_feature_diagnostics = {}
            return "unknown", 0.0, {"reason": "cross_view_feature_invalid", "detail": str(error)}
        self.last_feature_diagnostics = extracted.diagnostics
        # v3 remains an explicit rollback path: its classifier never consumes
        # appended graph features, though the new graph can still be inspected.
        raw = extracted.vector if self.feature_version == 4 else extracted.vector[:-len(GRAPH_FEATURE_NAMES)]
        feature = (raw - self.mean) / self.scale
        scores = self._scores(feature)
        self.last_class_scores = scores
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        label, probability = ordered[0]
        margin = probability - (ordered[1][1] if len(ordered) > 1 else 0.0)
        distance = float(self._distances(feature).min())
        reason = "accepted"
        if distance > self.distance_threshold:
            reason = "distance_rejected"
        elif margin < self.margin_threshold:
            reason = "margin_rejected"
        return (label if reason == "accepted" else "unknown"), probability, {
            "nearest_label": label,
            "reason": reason,
            "distance": distance,
            "distance_threshold": self.distance_threshold,
            "margin": margin,
            "margin_threshold": self.margin_threshold,
            "feature_groups": extracted.diagnostics,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            model_type=np.asarray(["cross_view_topology"]),
            feature_version=np.asarray([self.feature_version], np.int32),
            method=np.asarray([self.method]),
            features=self.features,
            labels=self.labels,
            mean=self.mean,
            scale=self.scale,
            group_ids=self.group_ids,
            class_ids=np.asarray(self.class_ids),
            registry_hash=np.asarray([self.registry_hash]),
            registry_json=np.asarray([json.dumps(self.registry.to_dict(), ensure_ascii=False)]),
            distance_threshold=np.asarray([self.distance_threshold], np.float32),
            margin_threshold=np.asarray([self.margin_threshold], np.float32),
        )

    @classmethod
    def load(
        cls, path: str | Path, registry: ShapeRegistry | None = None
    ) -> "CrossViewTopologyModel":
        with np.load(path, allow_pickle=False) as data:
            if str(data["model_type"][0]) != "cross_view_topology":
                raise ValueError("not a cross-view topology model")
            if int(data["feature_version"][0]) not in (3, CROSS_VIEW_FEATURE_VERSION):
                raise ValueError("unsupported cross-view feature version")
            expected = 4 if np.any(data["group_ids"] == 5) else 3
            if int(data["feature_version"][0]) != expected:
                raise ValueError("cross-view feature version and groups disagree")
            value = json.loads(str(data["registry_json"][0]))
            stored = ShapeRegistry(
                version=int(value["version"]),
                classes=tuple(ShapeClass.from_dict(item) for item in value["classes"]),
            )
            active = stored if registry is None else registry
            active.validate_model_classes(data["class_ids"].astype(str), str(data["registry_hash"][0]))
            return cls(
                data["features"], data["labels"], data["mean"], data["scale"],
                data["group_ids"], active, method=str(data["method"][0]),
                distance_threshold=float(data["distance_threshold"][0]),
                margin_threshold=float(data["margin_threshold"][0]),
            )


def annotate_cross_view_topology(
    image_bgr: np.ndarray, result: CrossViewFeatureResult
) -> np.ndarray:
    canvas = np.asarray(image_bgr).copy()
    for edge in result.projected_obb_edges_px:
        start = tuple(np.rint(edge[:2]).astype(int))
        end = tuple(np.rint(edge[2:]).astype(int))
        cv2.line(canvas, start, end, (255, 160, 0), 1, cv2.LINE_AA)
    for segment in result.segments_px:
        start = tuple(np.rint(segment[:2]).astype(int))
        end = tuple(np.rint(segment[2:]).astype(int))
        cv2.line(canvas, start, end, (0, 255, 0), 2, cv2.LINE_AA)
    graph = result.diagnostics.get("joint_topology_graph", {})
    for raw in graph.get("edges_side_px", []):
        segment = np.rint(raw).astype(np.int32)
        cv2.line(canvas, tuple(segment[:2]), tuple(segment[2:]), (255, 0, 255), 2, cv2.LINE_AA)
    for index in (item["segment_id"] for item in graph.get("unresolved_side_segments", [])):
        segment = result.segments_px[index]
        cv2.line(canvas, tuple(np.rint(segment[:2]).astype(int)), tuple(np.rint(segment[2:]).astype(int)), (0, 140, 255), 1, cv2.LINE_AA)
    return canvas
