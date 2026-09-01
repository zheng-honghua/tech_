from __future__ import annotations

import hashlib
import json
import csv
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry_models import GeometryCandidate, GeometryPrediction
from .geometry_edges import (
    EDGE_PARAMETERS,
    EdgeTopology,
    edge_topology_vector,
    extract_edge_topology,
)
from .geometry_structure import (
    STRUCTURAL_VECTOR_LENGTH,
    StructuralContour,
    extract_structural_contour,
    structural_contour_vector,
)


GEOMETRY_LABELS: dict[str, tuple[str, str]] = {
    "三棱柱": ("triangular_prism", "三棱柱"),
    "三棱锥": ("triangular_pyramid", "三棱锥"),
    "四棱锥": ("square_pyramid", "四棱锥"),
    "五棱柱": ("pentagonal_prism", "五棱柱"),
    "五棱锥": ("pentagonal_pyramid", "五棱锥"),
    "六棱柱": ("hexagonal_prism", "六棱柱"),
    "六棱锥": ("hexagonal_pyramid", "六棱锥"),
    "正八面体": ("octahedron", "正八面体"),
    "圆锥": ("cone", "圆锥"),
}
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
LEGACY_FEATURE_VERSION = 1
PREVIOUS_EDGE_FEATURE_VERSION = 2
FACE_VERTEX_FEATURE_VERSION = 3
EDGE_FEATURE_VERSION = 4
STRUCTURE_FEATURE_VERSION = 5
MODEL_VERSION = 3
EDGE_GROUP_WEIGHTS = np.asarray([0.20, 0.20, 0.05, 0.55], np.float32)
EDGE_GROUP_WEIGHTS_V3 = np.asarray(
    [0.15, 0.15, 0.05, 0.45, 0.20], np.float32
)
STRUCTURE_GROUP_WEIGHTS = np.asarray([0.10, 0.15, 0.20, 0.55], np.float32)


@dataclass(frozen=True)
class GeometrySample:
    path: Path
    label_id: str
    label_name: str
    image_bgr: np.ndarray
    sha256: str


@dataclass(frozen=True)
class GeometryPreprocessed:
    image_bgr: np.ndarray
    mask: np.ndarray
    bbox_px: tuple[int, int, int, int]
    candidate_count: int = 1


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_geometry_samples(data_root: str | Path) -> tuple[list[GeometrySample], list[dict[str, str]]]:
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"geometry data root does not exist: {root}")
    samples: list[GeometrySample] = []
    errors: list[dict[str, str]] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        label = GEOMETRY_LABELS.get(directory.name)
        if label is None:
            errors.append({"path": str(directory), "reason": "unknown_label_directory"})
            continue
        label_id, label_name = label
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                errors.append({"path": str(path), "reason": "unreadable_image"})
                continue
            samples.append(
                GeometrySample(path, label_id, label_name, image, _file_hash(path))
            )
    return samples, errors


def _select_component(
    mask: np.ndarray, reject_border: bool, max_area_ratio: float = 0.8
) -> tuple[np.ndarray | None, int]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    height, width = mask.shape
    image_area = height * width
    center = np.array([width / 2.0, height / 2.0])
    candidates: list[tuple[float, int]] = []
    for label in range(1, count):
        x, y, item_width, item_height, area = stats[label]
        if area < max(180, image_area * 0.0005) or area > image_area * max_area_ratio:
            continue
        touches = x <= 3 or y <= 3 or x + item_width >= width - 3 or y + item_height >= height - 3
        if reject_border and touches:
            continue
        distance = float(np.linalg.norm(centroids[label] - center))
        distance /= max(float(np.linalg.norm(center)), 1.0)
        score = float(area) * max(0.25, 1.0 - 0.55 * distance)
        candidates.append((score, label))
    if not candidates:
        return None, 0
    selected = max(candidates)[1]
    return (labels == selected).astype(np.uint8) * 255, len(candidates)


def preprocess_geometry_object(
    image_bgr: np.ndarray,
    supplied_mask: np.ndarray | None = None,
    output_size: int = 128,
) -> GeometryPreprocessed | None:
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        return None
    if supplied_mask is None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturated = ((hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 35)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        candidate = cv2.morphologyEx(saturated, cv2.MORPH_OPEN, kernel)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
        component, candidate_count = _select_component(candidate, reject_border=True)
        if component is None:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
            border = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]), axis=0)
            delta = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
            candidate = (delta >= 24.0).astype(np.uint8) * 255
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
            component, candidate_count = _select_component(candidate, reject_border=True)
    else:
        candidate = (np.asarray(supplied_mask) > 0).astype(np.uint8) * 255
        if candidate.shape != image.shape[:2]:
            return None
        component, candidate_count = _select_component(
            candidate, reject_border=False, max_area_ratio=0.98
        )
    if component is None:
        return None

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    padding = max(4, int(round(max(width, height) * 0.08)))
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(image.shape[1], x + width + padding), min(image.shape[0], y + height + padding)
    crop = image[y0:y1, x0:x1]
    crop_mask = component[y0:y1, x0:x1]
    available = output_size - 16
    scale = min(available / max(crop.shape[1], 1), available / max(crop.shape[0], 1))
    resized_width = max(1, int(round(crop.shape[1] * scale)))
    resized_height = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(
        crop_mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST
    )
    canvas = np.full((output_size, output_size, 3), 245, np.uint8)
    mask_canvas = np.zeros((output_size, output_size), np.uint8)
    left = (output_size - resized_width) // 2
    top = (output_size - resized_height) // 2
    region = canvas[top : top + resized_height, left : left + resized_width]
    region_mask = resized_mask > 0
    region[region_mask] = resized[region_mask]
    mask_canvas[top : top + resized_height, left : left + resized_width] = resized_mask
    return GeometryPreprocessed(
        canvas, mask_canvas, (x, y, width, height), candidate_count
    )


def _legacy_geometry_features(preprocessed: GeometryPreprocessed) -> np.ndarray:
    image = preprocessed.image_bgr
    mask = preprocessed.mask
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inside = gray[mask > 0].astype(np.float32)
    if inside.size < 25:
        raise ValueError("geometry object mask is too small")
    mean = float(inside.mean())
    std = max(float(inside.std()), 8.0)
    normalized = np.full_like(gray, 245)
    values = np.clip((gray.astype(np.float32) - mean) * (32.0 / std) + 128.0, 0, 255)
    normalized[mask > 0] = values[mask > 0].astype(np.uint8)

    gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    cell_histograms = np.zeros((8, 8, 9), np.float32)
    for row in range(8):
        for column in range(8):
            y0, y1 = row * 16, (row + 1) * 16
            x0, x1 = column * 16, (column + 1) * 16
            histogram, _ = np.histogram(
                angle[y0:y1, x0:x1] % 180.0,
                bins=9,
                range=(0.0, 180.0),
                weights=magnitude[y0:y1, x0:x1],
            )
            cell_histograms[row, column] = histogram
    blocks: list[np.ndarray] = []
    for row in range(7):
        for column in range(7):
            block = cell_histograms[row : row + 2, column : column + 2].reshape(-1)
            blocks.append(block / np.sqrt(float(block @ block) + 1e-6))
    hog_values = np.concatenate(blocks).astype(np.float32)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = max(float(cv2.contourArea(contour)), 1.0)
    perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
    approximation = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    _, (rect_width, rect_height), _ = cv2.minAreaRect(contour)
    hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
    hu = cv2.HuMoments(cv2.moments(contour)).reshape(-1)
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    geometry = np.asarray(
        [
            len(approximation) / 12.0,
            4.0 * np.pi * area / (perimeter * perimeter),
            min(rect_width, rect_height) / max(rect_width, rect_height, 1.0),
            area / hull_area,
            area / float(mask.size),
            *hu.tolist(),
        ],
        dtype=np.float32,
    )

    valid = mask > 0
    orientation, _ = np.histogram(
        angle[valid] % 180.0,
        bins=12,
        range=(0.0, 180.0),
        weights=magnitude[valid],
    )
    orientation = orientation.astype(np.float32)
    orientation /= max(float(orientation.sum()), 1.0)
    brightness, _ = np.histogram(inside, bins=8, range=(0.0, 256.0))
    brightness = brightness.astype(np.float32) / max(float(inside.size), 1.0)
    pooled: list[float] = []
    for row in range(4):
        for column in range(4):
            y0, y1 = row * 32, (row + 1) * 32
            x0, x1 = column * 32, (column + 1) * 32
            cell_mask = mask[y0:y1, x0:x1] > 0
            cell = normalized[y0:y1, x0:x1]
            pooled.append(float(cell[cell_mask].mean() / 255.0) if np.any(cell_mask) else 0.0)
    return np.concatenate(
        (hog_values, geometry, orientation, brightness, np.asarray(pooled, np.float32))
    ).astype(np.float32)


def _legacy_input(preprocessed: GeometryPreprocessed) -> GeometryPreprocessed:
    if preprocessed.image_bgr.shape[:2] == (128, 128):
        return preprocessed
    return GeometryPreprocessed(
        cv2.resize(preprocessed.image_bgr, (128, 128), interpolation=cv2.INTER_AREA),
        cv2.resize(preprocessed.mask, (128, 128), interpolation=cv2.INTER_NEAREST),
        preprocessed.bbox_px,
        preprocessed.candidate_count,
    )


def geometry_feature_groups(
    feature_set: str, feature_version: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if feature_set == "legacy":
        return np.zeros(1812, np.int32), np.ones(1, np.float32)
    if feature_set == "structure-topology":
        sections = [
            np.zeros(1764, np.int32),
            np.ones(12, np.int32),
            np.full(36, 2, np.int32),
            np.full(STRUCTURAL_VECTOR_LENGTH, 3, np.int32),
        ]
        return np.concatenate(sections), STRUCTURE_GROUP_WEIGHTS.copy()
    if feature_set != "edge-topology":
        raise ValueError(f"unsupported geometry feature set: {feature_set}")
    version = EDGE_FEATURE_VERSION if feature_version is None else feature_version
    base_topology_count = len(
        edge_topology_vector(
            EdgeTopology(
                np.zeros((1, 1), np.uint8),
                np.zeros((1, 1), np.uint8),
                (),
                (),
                (),
                0,
                0,
                (),
                0.0,
                0.0,
                "edge_evidence_low",
                1.0,
            ),
            include_face_vertices=False,
        )
    )
    sections = [
        np.zeros(1764, np.int32),
        np.ones(12, np.int32),
        np.zeros(12, np.int32),
        np.full(24, 2, np.int32),
        np.full(base_topology_count, 3, np.int32),
    ]
    if version >= FACE_VERTEX_FEATURE_VERSION:
        sections.append(np.full(6, 4, np.int32))
        return np.concatenate(sections), EDGE_GROUP_WEIGHTS_V3.copy()
    return np.concatenate(sections), EDGE_GROUP_WEIGHTS.copy()


def extract_geometry_features(
    preprocessed: GeometryPreprocessed,
    feature_set: str = "legacy",
    topology: EdgeTopology | None = None,
    feature_version: int | None = None,
    structure: StructuralContour | None = None,
) -> np.ndarray:
    legacy = _legacy_geometry_features(_legacy_input(preprocessed))
    if feature_set == "legacy":
        return legacy
    if feature_set == "structure-topology":
        structure = structure or extract_structural_contour(
            preprocessed.image_bgr, preprocessed.mask
        )
        return np.concatenate((legacy, structural_contour_vector(structure))).astype(
            np.float32
        )
    if feature_set != "edge-topology":
        raise ValueError(f"unsupported geometry feature set: {feature_set}")
    version = EDGE_FEATURE_VERSION if feature_version is None else feature_version
    topology = topology or extract_edge_topology(
        preprocessed.image_bgr,
        preprocessed.mask,
        enhanced_faces=version >= FACE_VERTEX_FEATURE_VERSION,
        morph_color_assist=version >= EDGE_FEATURE_VERSION,
    )
    return np.concatenate(
        (
            legacy,
            edge_topology_vector(
                topology,
                include_face_vertices=version >= FACE_VERTEX_FEATURE_VERSION,
            ),
        )
    ).astype(np.float32)


class GeometryRGBModel:
    backend = "opencv"

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        class_names: dict[str, str],
        distance_threshold: float,
        margin_threshold: float = 0.04,
        source_hashes: list[str] | None = None,
        feature_version: int = LEGACY_FEATURE_VERSION,
        feature_set: str = "legacy",
        feature_group_ids: np.ndarray | None = None,
        feature_group_weights: np.ndarray | None = None,
        model_version: int = MODEL_VERSION,
        edge_parameters: dict[str, Any] | None = None,
        class_margin_thresholds: dict[str, float] | None = None,
        class_distance_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.features = np.asarray(features, np.float32)
        self.labels = np.asarray(labels).astype(str)
        self.feature_mean = np.asarray(feature_mean, np.float32)
        self.feature_scale = np.asarray(feature_scale, np.float32)
        self.class_names = dict(class_names)
        self.distance_threshold = float(max(distance_threshold, 1e-5))
        self.margin_threshold = float(margin_threshold)
        self.source_hashes = list(source_hashes or [])
        self.feature_version = int(feature_version)
        self.feature_set = str(feature_set)
        self.model_version = int(model_version)
        self.edge_parameters = dict(
            edge_parameters
            if edge_parameters is not None
            else (EDGE_PARAMETERS if self.feature_set == "edge-topology" else {})
        )
        self.class_margin_thresholds = {
            str(key): float(value)
            for key, value in (class_margin_thresholds or {}).items()
        }
        self.class_distance_thresholds = {
            str(key): float(value)
            for key, value in (class_distance_thresholds or {}).items()
        }
        self.feature_group_ids = (
            np.zeros(self.features.shape[1], np.int32)
            if feature_group_ids is None
            else np.asarray(feature_group_ids, np.int32)
        )
        self.feature_group_weights = (
            np.ones(1, np.float32)
            if feature_group_weights is None
            else np.asarray(feature_group_weights, np.float32)
        )
        if len(self.features) != len(self.labels) or not len(self.features):
            raise ValueError("geometry model requires matching non-empty features and labels")
        if len(self.feature_group_ids) != self.features.shape[1]:
            raise ValueError("geometry feature group ids do not match feature count")
        if int(self.feature_group_ids.max(initial=0)) >= len(self.feature_group_weights):
            raise ValueError("geometry feature group weight is missing")

    @classmethod
    def fit(
        cls,
        raw_features: np.ndarray,
        labels: list[str] | np.ndarray,
        class_names: dict[str, str],
        source_hashes: list[str] | None = None,
        feature_set: str = "legacy",
        feature_group_ids: np.ndarray | None = None,
        feature_group_weights: np.ndarray | None = None,
        margin_threshold: float | None = None,
        class_margin_thresholds: dict[str, float] | None = None,
        class_distance_thresholds: dict[str, float] | None = None,
        allow_singleton_classes: bool = False,
    ) -> "GeometryRGBModel":
        values = np.asarray(raw_features, np.float32)
        label_values = np.asarray(labels).astype(str)
        if values.ndim != 2 or len(values) != len(label_values):
            raise ValueError("training features must be a 2-D array matching labels")
        counts = Counter(label_values.tolist())
        if not allow_singleton_classes and any(
            count < 2 for count in counts.values()
        ):
            raise ValueError("each geometry class requires at least two samples")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        # A variance floor prevents tiny pixel-level mask changes from exploding
        # after standardisation in this very small training set.
        scale = np.maximum(scale, 0.05)
        standardized = (values - mean) / scale
        if feature_group_ids is None or feature_group_weights is None:
            feature_group_ids, feature_group_weights = geometry_feature_groups(
                feature_set, EDGE_FEATURE_VERSION
            )
        feature_group_ids = np.asarray(feature_group_ids, np.int32)
        feature_group_weights = np.asarray(feature_group_weights, np.float32)
        if len(feature_group_ids) != values.shape[1]:
            raise ValueError("feature group layout does not match training features")

        def distance_to_many(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
            squared = (candidates - reference) ** 2
            result = np.zeros(len(candidates), np.float32)
            total_weight = max(float(feature_group_weights.sum()), 1e-6)
            for group, weight in enumerate(feature_group_weights):
                selected = feature_group_ids == group
                if np.any(selected):
                    result += float(weight) * squared[:, selected].mean(axis=1)
            return np.sqrt(result / total_weight)

        same_distances: list[float] = []
        for index, label in enumerate(label_values):
            indices = np.flatnonzero(label_values == label)
            indices = indices[indices != index]
            if not len(indices):
                continue
            distances = distance_to_many(standardized[index], standardized[indices])
            same_distances.append(float(np.min(distances)))
        threshold = max(
            float(np.percentile(same_distances, 95)) * 1.35
            if same_distances
            else 0.25,
            0.25,
        )
        return cls(
            standardized,
            label_values,
            mean,
            scale,
            class_names,
            threshold,
            source_hashes=source_hashes,
            margin_threshold=(
                float(margin_threshold)
                if margin_threshold is not None
                else (0.06 if feature_set == "structure-topology" else (
                    0.075 if feature_set == "edge-topology" else 0.04
                ))
            ),
            feature_version=(
                STRUCTURE_FEATURE_VERSION
                if feature_set == "structure-topology"
                else (EDGE_FEATURE_VERSION if feature_set == "edge-topology"
                      else LEGACY_FEATURE_VERSION)
            ),
            feature_set=feature_set,
            feature_group_ids=feature_group_ids,
            feature_group_weights=feature_group_weights,
            class_margin_thresholds=(
                class_margin_thresholds
                if class_margin_thresholds is not None
                else (
                    {
                        "triangular_prism": 0.12,
                        "pentagonal_prism": 0.045,
                        "hexagonal_prism": 0.15,
                    }
                    if feature_set == "edge-topology"
                    else {}
                )
            ),
            class_distance_thresholds=(
                class_distance_thresholds
                if class_distance_thresholds is not None
                else (
                    {"hexagonal_pyramid": 1.05}
                    if feature_set == "edge-topology"
                    else {}
                )
            ),
        )

    def _distances(self, feature: np.ndarray) -> np.ndarray:
        squared = (self.features - feature) ** 2
        result = np.zeros(len(self.features), np.float32)
        total_weight = max(float(self.feature_group_weights.sum()), 1e-6)
        for group, weight in enumerate(self.feature_group_weights):
            selected = self.feature_group_ids == group
            if np.any(selected):
                result += float(weight) * squared[:, selected].mean(axis=1)
        return np.sqrt(result / total_weight)

    def predict_feature(self, raw_feature: np.ndarray) -> tuple[str, float, dict[str, float | str]]:
        feature = (np.asarray(raw_feature, np.float32) - self.feature_mean) / self.feature_scale
        distances = self._distances(feature)
        class_distances: dict[str, float] = {}
        for label in sorted(set(self.labels.tolist())):
            values = np.sort(distances[self.labels == label])
            class_distances[label] = float(np.mean(values[: min(2, len(values))]))
        ordered = sorted(class_distances.items(), key=lambda item: item[1])
        best_label, best_distance = ordered[0]
        second_distance = ordered[1][1] if len(ordered) > 1 else self.distance_threshold * 2.0
        margin = (second_distance - best_distance) / max(second_distance, 1e-6)
        distance_score = float(np.exp(-((best_distance / self.distance_threshold) ** 2)))
        margin_score = float(np.clip(margin / 0.35, 0.0, 1.0))
        confidence = float(np.clip(0.72 * distance_score + 0.28 * margin_score, 0.0, 1.0))
        required_margin = self.class_margin_thresholds.get(
            best_label, self.margin_threshold
        )
        required_distance = self.class_distance_thresholds.get(
            best_label, self.distance_threshold
        )
        accepted = best_distance <= required_distance and margin >= required_margin
        diagnostics: dict[str, float | str] = {
            "nearest_label": best_label,
            "distance": best_distance,
            "distance_threshold": required_distance,
            "margin": margin,
            "margin_threshold": required_margin,
            "reason": (
                "accepted"
                if accepted
                else (
                    "distance_rejected"
                    if best_distance > required_distance
                    else "margin_rejected"
                )
            ),
        }
        return (best_label if accepted else "unknown"), confidence, diagnostics

    def predict(self, image_bgr: np.ndarray, mask: np.ndarray | None = None) -> tuple[str, float, dict[str, float | str]]:
        # Re-segment the colour-normalised object inside the crop so training and
        # runtime use the same silhouette. The upstream mask may include shadows.
        output_size = 256 if self.feature_set in {"edge-topology", "structure-topology"} else 128
        prepared = preprocess_geometry_object(image_bgr, output_size=output_size)
        if prepared is None and mask is not None:
            prepared = preprocess_geometry_object(
                image_bgr, mask, output_size=output_size
            )
        if prepared is None:
            return "unknown", 0.0, {"reason": "object_not_found"}
        return self.predict_preprocessed(prepared)

    def predict_preprocessed(
        self,
        prepared: GeometryPreprocessed,
        topology: EdgeTopology | None = None,
    ) -> tuple[str, float, dict[str, float | str]]:
        """Predict a standardized crop without repeating full-image segmentation."""
        if prepared.candidate_count != 1:
            return "unknown", 0.0, {
                "reason": "multiple_objects",
                "candidate_count": float(prepared.candidate_count),
            }
        if self.feature_set == "edge-topology":
            topology = topology or extract_edge_topology(
                prepared.image_bgr,
                prepared.mask,
                enhanced_faces=(
                    self.feature_version >= FACE_VERTEX_FEATURE_VERSION
                ),
                morph_color_assist=self.feature_version >= EDGE_FEATURE_VERSION,
            )
            if topology.reason != "accepted":
                return "unknown", 0.0, {
                    "reason": topology.reason,
                    "edge_quality": topology.quality,
                    "edge_count": float(len(topology.merged_lines)),
                }
            if len(topology.merged_lines) >= 14 and not topology.junctions:
                return "unknown", 0.0, {
                    "reason": "topology_conflict",
                    "edge_quality": topology.quality,
                    "edge_count": float(len(topology.merged_lines)),
                }
        structure = None
        if self.feature_set == "structure-topology":
            structure = extract_structural_contour(
                prepared.image_bgr, prepared.mask
            )
        return self.predict_feature(
            extract_geometry_features(
                prepared, self.feature_set, topology, self.feature_version,
                structure,
            )
        )

    def _geometry_prediction(
        self,
        label: str,
        confidence: float,
        diagnostics: dict[str, float | str],
        inference_ms: float,
    ) -> GeometryPrediction:
        nearest = str(diagnostics.get("nearest_label", label))
        candidates = (
            (GeometryCandidate(nearest, confidence),)
            if nearest not in {"unknown", ""}
            else ()
        )
        accepted = label != "unknown" and diagnostics.get("reason") == "accepted"
        return GeometryPrediction(
            label_id=label,
            label_name=self.class_names.get(label, "未知形状"),
            confidence=confidence,
            accepted=accepted,
            backend=self.backend,
            reason=str(diagnostics.get("reason", "unknown")),
            top_candidates=candidates,
            inference_ms=inference_ms,
        )

    def predict_geometry(
        self, image_bgr: np.ndarray, mask: np.ndarray | None = None
    ) -> GeometryPrediction:
        started = time.perf_counter()
        label, confidence, diagnostics = self.predict(image_bgr, mask)
        return self._geometry_prediction(
            label,
            confidence,
            diagnostics,
            (time.perf_counter() - started) * 1000.0,
        )

    def predict_preprocessed_geometry(
        self,
        prepared: GeometryPreprocessed,
        topology: EdgeTopology | None = None,
    ) -> GeometryPrediction:
        started = time.perf_counter()
        label, confidence, diagnostics = self.predict_preprocessed(
            prepared, topology
        )
        return self._geometry_prediction(
            label,
            confidence,
            diagnostics,
            (time.perf_counter() - started) * 1000.0,
        )

    def __call__(self, crop: np.ndarray, crop_mask: np.ndarray) -> tuple[str, float]:
        prediction = self.predict_geometry(crop, crop_mask)
        return prediction.label_id, prediction.confidence

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            model_version=np.asarray([self.model_version], np.int32),
            feature_version=np.asarray([self.feature_version], np.int32),
            feature_set=np.asarray([self.feature_set]),
            feature_group_ids=self.feature_group_ids,
            feature_group_weights=self.feature_group_weights,
            edge_parameters_json=np.asarray([
                json.dumps(self.edge_parameters, ensure_ascii=False)
            ]),
            class_margin_thresholds_json=np.asarray([
                json.dumps(self.class_margin_thresholds, ensure_ascii=False)
            ]),
            class_distance_thresholds_json=np.asarray([
                json.dumps(self.class_distance_thresholds, ensure_ascii=False)
            ]),
            features=self.features,
            labels=self.labels,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            class_names_json=np.asarray([json.dumps(self.class_names, ensure_ascii=False)]),
            distance_threshold=np.asarray([self.distance_threshold], np.float32),
            margin_threshold=np.asarray([self.margin_threshold], np.float32),
            source_hashes=np.asarray(self.source_hashes),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GeometryRGBModel":
        with np.load(Path(path), allow_pickle=False) as value:
            model_version = int(value["model_version"][0])
            if model_version not in {1, 2, MODEL_VERSION}:
                raise ValueError("unsupported geometry RGB model version")
            feature_version = int(value["feature_version"][0])
            if feature_version not in {
                LEGACY_FEATURE_VERSION,
                PREVIOUS_EDGE_FEATURE_VERSION,
                FACE_VERTEX_FEATURE_VERSION,
                EDGE_FEATURE_VERSION,
                STRUCTURE_FEATURE_VERSION,
            }:
                raise ValueError("unsupported geometry feature version")
            feature_set = (
                str(value["feature_set"][0])
                if "feature_set" in value.files
                else "legacy"
            )
            group_ids = (
                value["feature_group_ids"]
                if "feature_group_ids" in value.files
                else np.zeros(value["features"].shape[1], np.int32)
            )
            group_weights = (
                value["feature_group_weights"]
                if "feature_group_weights" in value.files
                else np.ones(1, np.float32)
            )
            edge_parameters = (
                json.loads(str(value["edge_parameters_json"][0]))
                if "edge_parameters_json" in value.files
                else {}
            )
            class_margin_thresholds = (
                json.loads(str(value["class_margin_thresholds_json"][0]))
                if "class_margin_thresholds_json" in value.files
                else {}
            )
            class_distance_thresholds = (
                json.loads(str(value["class_distance_thresholds_json"][0]))
                if "class_distance_thresholds_json" in value.files
                else {}
            )
            return cls(
                value["features"],
                value["labels"],
                value["feature_mean"],
                value["feature_scale"],
                json.loads(str(value["class_names_json"][0])),
                float(value["distance_threshold"][0]),
                float(value["margin_threshold"][0]),
                value["source_hashes"].astype(str).tolist(),
                feature_version,
                feature_set,
                group_ids,
                group_weights,
                model_version,
                edge_parameters,
                class_margin_thresholds,
                class_distance_thresholds,
            )


def _features_from_samples(
    samples: list[GeometrySample], feature_set: str = "legacy"
) -> tuple[np.ndarray, list[GeometrySample], list[dict[str, str]]]:
    features: list[np.ndarray] = []
    valid: list[GeometrySample] = []
    errors: list[dict[str, str]] = []
    for sample in samples:
        output_size = 256 if feature_set in {"edge-topology", "structure-topology"} else 128
        prepared = preprocess_geometry_object(sample.image_bgr, output_size=output_size)
        if prepared is None:
            errors.append({"path": str(sample.path), "reason": "object_not_found"})
            continue
        features.append(
            extract_geometry_features(
                prepared,
                feature_set,
                feature_version=(
                    STRUCTURE_FEATURE_VERSION
                    if feature_set == "structure-topology"
                    else EDGE_FEATURE_VERSION
                ),
            )
        )
        valid.append(sample)
    if not features:
        return np.empty((0, 0), np.float32), valid, errors
    return np.stack(features), valid, errors


def audit_geometry_dataset(data_root: str | Path) -> dict[str, Any]:
    samples, errors = load_geometry_samples(data_root)
    counts = Counter(sample.label_id for sample in samples)
    dimensions = Counter(f"{sample.image_bgr.shape[1]}x{sample.image_bgr.shape[0]}" for sample in samples)
    hashes: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        hashes[sample.sha256].append(str(sample.path))
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    _, valid, preprocessing_errors = _features_from_samples(samples)
    warnings = []
    for label, count in sorted(counts.items()):
        if count < 5:
            warnings.append({"label": label, "reason": "fewer_than_5_samples", "count": count})
    return {
        "data_root": str(Path(data_root)),
        "total_images": len(samples),
        "class_counts": dict(sorted(counts.items())),
        "dimensions": dict(sorted(dimensions.items())),
        "preprocessing_success": len(valid),
        "duplicates": duplicates,
        "errors": [*errors, *preprocessing_errors],
        "warnings": warnings,
    }


def train_geometry_model(
    data_root: str | Path,
    feature_set: str = "legacy",
    additional_data_roots: list[str | Path] | None = None,
) -> tuple[GeometryRGBModel, dict[str, Any]]:
    roots = [Path(data_root), *(Path(item) for item in (additional_data_roots or []))]
    samples: list[GeometrySample] = []
    load_errors: list[dict[str, str]] = []
    duplicate_samples: list[str] = []
    seen_hashes: set[str] = set()
    for root in roots:
        batch_samples, batch_errors = load_geometry_samples(root)
        load_errors.extend(batch_errors)
        for sample in batch_samples:
            if sample.sha256 in seen_hashes:
                duplicate_samples.append(str(sample.path))
                continue
            seen_hashes.add(sample.sha256)
            samples.append(sample)
    features, valid, preprocessing_errors = _features_from_samples(
        samples, feature_set
    )
    labels = [sample.label_id for sample in valid]
    names = {sample.label_id: sample.label_name for sample in valid}
    model = GeometryRGBModel.fit(
        features,
        labels,
        names,
        [sample.sha256 for sample in valid],
        feature_set=feature_set,
    )
    report = {
        "training_samples": len(valid),
        "data_roots": [str(root) for root in roots],
        "duplicate_samples_skipped": duplicate_samples,
        "class_counts": dict(sorted(Counter(labels).items())),
        "feature_count": int(features.shape[1]),
        "feature_set": feature_set,
        "feature_group_weights": model.feature_group_weights.tolist(),
        "margin_threshold": model.margin_threshold,
        "class_margin_thresholds": model.class_margin_thresholds,
        "class_distance_thresholds": model.class_distance_thresholds,
        "distance_threshold": round(model.distance_threshold, 6),
        "errors": [*load_errors, *preprocessing_errors],
        "same_batch_only": len(roots) == 1,
    }
    return model, report


def evaluate_geometry_holdout(
    training_data_root: str | Path,
    test_data_root: str | Path,
    model_path: str | Path,
) -> dict[str, Any]:
    training_samples, training_errors = load_geometry_samples(training_data_root)
    test_samples, test_errors = load_geometry_samples(test_data_root)
    training_hashes = {sample.sha256 for sample in training_samples}
    duplicates = [
        str(sample.path) for sample in test_samples if sample.sha256 in training_hashes
    ]
    fresh = [sample for sample in test_samples if sample.sha256 not in training_hashes]
    model = GeometryRGBModel.load(model_path)
    rows: list[dict[str, Any]] = []
    for sample in fresh:
        predicted, confidence, diagnostics = model.predict(sample.image_bgr)
        rows.append(
            {
                "path": str(sample.path),
                "true_label": sample.label_id,
                "predicted_label": predicted,
                "confidence": round(confidence, 6),
                "reason": diagnostics.get("reason", "unknown"),
                "correct": predicted == sample.label_id,
            }
        )
    labels = sorted({sample.label_id for sample in fresh})
    matrix_labels = [*labels, "unknown"]
    index_by_label = {label: index for index, label in enumerate(matrix_labels)}
    matrix = np.zeros((len(labels), len(matrix_labels)), np.int32)
    for row in rows:
        truth = str(row["true_label"])
        predicted = str(row["predicted_label"])
        matrix[index_by_label[truth], index_by_label.get(predicted, index_by_label["unknown"])] += 1
    recalls = {
        label: float(matrix[index, index] / max(matrix[index].sum(), 1))
        for index, label in enumerate(labels)
    }
    accepted = [row for row in rows if row["predicted_label"] != "unknown"]
    correct = sum(bool(row["correct"]) for row in rows)
    correct_accepted = sum(bool(row["correct"]) for row in accepted)
    return {
        "evaluation": "hash_excluded_holdout",
        "training_data_root": str(Path(training_data_root)),
        "test_data_root": str(Path(test_data_root)),
        "model_path": str(Path(model_path)),
        "test_images_total": len(test_samples),
        "exact_duplicates_excluded": duplicates,
        "samples": len(rows),
        "labels": matrix_labels,
        "confusion_matrix": matrix.tolist(),
        "accuracy": round(correct / max(len(rows), 1), 6),
        "macro_recall": round(float(np.mean(list(recalls.values()))) if recalls else 0.0, 6),
        "per_class_recall": {key: round(value, 6) for key, value in recalls.items()},
        "accepted": len(accepted),
        "correct_accepted": correct_accepted,
        "wrong_accepted": len(accepted) - correct_accepted,
        "accepted_precision": round(correct_accepted / max(len(accepted), 1), 6),
        "rejected": len(rows) - len(accepted),
        "predictions": rows,
        "errors": [*training_errors, *test_errors],
    }


def evaluate_geometry_model(data_root: str | Path, model_path: str | Path) -> dict[str, Any]:
    reference = GeometryRGBModel.load(model_path)
    samples, load_errors = load_geometry_samples(data_root)
    features, valid, preprocessing_errors = _features_from_samples(
        samples, reference.feature_set
    )
    if features.shape[1] != len(reference.feature_mean):
        raise ValueError("dataset features do not match model feature version")
    true_labels = [sample.label_id for sample in valid]
    class_names = {sample.label_id: sample.label_name for sample in valid}
    predictions: list[str] = []
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(valid):
        keep = np.arange(len(valid)) != index
        fold = GeometryRGBModel.fit(
            features[keep],
            np.asarray(true_labels)[keep],
            class_names,
            feature_set=reference.feature_set,
            feature_group_ids=reference.feature_group_ids,
            feature_group_weights=reference.feature_group_weights,
            margin_threshold=reference.margin_threshold,
            class_margin_thresholds=reference.class_margin_thresholds,
            class_distance_thresholds=reference.class_distance_thresholds,
            allow_singleton_classes=True,
        )
        predicted, confidence, diagnostics = fold.predict_feature(features[index])
        predictions.append(predicted)
        rows.append(
            {
                "path": str(sample.path),
                "true_label": sample.label_id,
                "predicted_label": predicted,
                "confidence": round(confidence, 6),
                "reason": diagnostics["reason"],
            }
        )
    labels = sorted(set(true_labels))
    matrix_labels = [*labels, "unknown"]
    index_by_label = {label: index for index, label in enumerate(matrix_labels)}
    matrix = np.zeros((len(labels), len(matrix_labels)), np.int32)
    for truth, prediction in zip(true_labels, predictions):
        matrix[index_by_label[truth], index_by_label.get(prediction, index_by_label["unknown"])] += 1
    recalls = {
        label: float(matrix[row, row] / max(matrix[row].sum(), 1))
        for row, label in enumerate(labels)
    }
    accuracy = float(np.mean(np.asarray(true_labels) == np.asarray(predictions)))
    return {
        "same_batch_only": True,
        "evaluation": "leave_one_out",
        "feature_set": reference.feature_set,
        "samples": len(valid),
        "labels": matrix_labels,
        "confusion_matrix": matrix.tolist(),
        "accuracy": round(accuracy, 6),
        "per_class_recall": {key: round(value, 6) for key, value in recalls.items()},
        "macro_recall": round(float(np.mean(list(recalls.values()))), 6),
        "predictions": rows,
        "errors": [*load_errors, *preprocessing_errors],
    }


def compare_geometry_models(
    data_root: str | Path,
    legacy_model_path: str | Path,
    edge_model_path: str | Path,
) -> dict[str, Any]:
    legacy = evaluate_geometry_model(data_root, legacy_model_path)
    edge = evaluate_geometry_model(data_root, edge_model_path)
    legacy_rows = {row["path"]: row for row in legacy["predictions"]}
    edge_rows = {row["path"]: row for row in edge["predictions"]}
    paths = sorted(set(legacy_rows) | set(edge_rows))
    rows = []
    for path in paths:
        old = legacy_rows.get(path, {})
        new = edge_rows.get(path, {})
        rows.append(
            {
                "path": path,
                "true_label": new.get("true_label", old.get("true_label")),
                "legacy_prediction": old.get("predicted_label", "missing"),
                "legacy_confidence": old.get("confidence", 0.0),
                "legacy_reason": old.get("reason", "missing"),
                "edge_prediction": new.get("predicted_label", "missing"),
                "edge_confidence": new.get("confidence", 0.0),
                "edge_reason": new.get("reason", "missing"),
            }
        )
    accuracy_delta = round(edge["accuracy"] - legacy["accuracy"], 6)
    recall_delta = round(edge["macro_recall"] - legacy["macro_recall"], 6)
    return {
        "same_batch_only": True,
        "legacy_model": str(Path(legacy_model_path)),
        "edge_model": str(Path(edge_model_path)),
        "legacy": legacy,
        "edge_topology": edge,
        "accuracy_delta": accuracy_delta,
        "macro_recall_delta": recall_delta,
        "both_improved": accuracy_delta > 0 and recall_delta > 0,
        "recommendation": (
            "edge_topology_experimental"
            if accuracy_delta > 0 and recall_delta > 0
            else "keep_legacy_default"
        ),
        "predictions": rows,
    }


def export_geometry_results(
    data_root: str | Path,
    model_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    root = Path(data_root)
    target = Path(output_root)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"geometry export directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    model = GeometryRGBModel.load(model_path)
    audit = audit_geometry_dataset(root)
    evaluation = evaluate_geometry_model(root, model_path)
    loo_by_path = {item["path"]: item for item in evaluation["predictions"]}
    samples, load_errors = load_geometry_samples(root)
    manifest: list[dict[str, Any]] = []

    for sample in samples:
        output_size = 256 if model.feature_set in {"edge-topology", "structure-topology"} else 128
        prepared = preprocess_geometry_object(sample.image_bgr, output_size=output_size)
        if prepared is None:
            continue
        feature = extract_geometry_features(
            prepared, model.feature_set, feature_version=model.feature_version
        )
        predicted, confidence, diagnostics = model.predict_feature(feature)
        relative_source = sample.path.relative_to(root)
        item_dir = target / "按真实类别" / sample.label_name / sample.path.stem
        item_dir.mkdir(parents=True, exist_ok=True)
        original_path = item_dir / "original.jpg"
        normalized_path = item_dir / "normalized.png"
        mask_path = item_dir / "mask.png"
        annotated_path = item_dir / "annotated.jpg"
        shutil.copy2(sample.path, original_path)
        if not cv2.imwrite(str(normalized_path), prepared.image_bgr):
            raise OSError(f"failed to write {normalized_path}")
        if not cv2.imwrite(str(mask_path), prepared.mask):
            raise OSError(f"failed to write {mask_path}")

        annotated = sample.image_bgr.copy()
        x, y, width, height = prepared.bbox_px
        colour = (0, 180, 0) if predicted == sample.label_id else (
            (0, 190, 255) if predicted == "unknown" else (0, 0, 220)
        )
        cv2.rectangle(annotated, (x, y), (x + width, y + height), colour, 3)
        cv2.putText(
            annotated,
            f"true={sample.label_id} pred={predicted} conf={confidence:.2f}",
            (max(8, x), max(28, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            colour,
            2,
            cv2.LINE_AA,
        )
        if not cv2.imwrite(str(annotated_path), annotated):
            raise OSError(f"failed to write {annotated_path}")

        loo = loo_by_path.get(str(sample.path), {})
        record = {
            "source": str(relative_source),
            "sha256": sample.sha256,
            "true_label": sample.label_id,
            "true_name": sample.label_name,
            "training_replay_prediction": predicted,
            "training_replay_confidence": round(confidence, 6),
            "training_replay_reason": diagnostics["reason"],
            "training_replay_correct": predicted == sample.label_id,
            "leave_one_out_prediction": loo.get("predicted_label", "unknown"),
            "leave_one_out_confidence": loo.get("confidence", 0.0),
            "leave_one_out_reason": loo.get("reason", "missing"),
            "leave_one_out_correct": loo.get("predicted_label") == sample.label_id,
            "artifacts": {
                "original": str(original_path.relative_to(target)),
                "normalized": str(normalized_path.relative_to(target)),
                "mask": str(mask_path.relative_to(target)),
                "annotated": str(annotated_path.relative_to(target)),
            },
        }
        (item_dir / "result.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest.append(record)

    with (target / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for record in manifest:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    labels = evaluation["labels"]
    with (target / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\predicted", *labels])
        for label, row in zip(labels[:-1], evaluation["confusion_matrix"]):
            writer.writerow([label, *row])

    replay_accuracy = float(
        np.mean([record["training_replay_correct"] for record in manifest])
    ) if manifest else 0.0
    summary = {
        "data_root": str(root),
        "model_path": str(Path(model_path)),
        "output_root": str(target),
        "exported_images": len(manifest),
        "training_replay_accuracy": round(replay_accuracy, 6),
        "leave_one_out_accuracy": evaluation["accuracy"],
        "same_batch_only": True,
        "warning": "训练集回放准确率不能代表泛化能力，以留一评测和独立批次测试为准。",
        "audit": audit,
        "evaluation": evaluation,
        "load_errors": load_errors,
    }
    (target / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "说明.txt").write_text(
        "几何RGB测试结果\n"
        f"图片数量：{len(manifest)}\n"
        f"训练集回放准确率：{replay_accuracy:.2%}\n"
        f"留一评测准确率：{evaluation['accuracy']:.2%}\n"
        "注意：这些图片参与了模型训练，训练集回放结果不能作为比赛验收。\n"
        "每张图片目录包含原图、标准化裁剪、掩膜、标注图和JSON结果。\n",
        encoding="utf-8",
    )
    return summary
