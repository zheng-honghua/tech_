from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .shape_registry import ShapeClass, ShapeRegistry


SIDE_FEATURE_VERSION = 1
WIDTH_BINS = 16
ORIENTATION_BINS = 12
CURVATURE_BINS = 8
FOURIER_BINS = 10


@dataclass(frozen=True)
class SideFeatureResult:
    vector: np.ndarray
    group_ids: np.ndarray
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class SideTrainingSample:
    directory: Path
    image_bgr: np.ndarray
    mask: np.ndarray
    label_id: str
    split: str
    batch_id: str
    instance_id: str
    color_id: str
    sha256: str


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        (np.asarray(mask) > 0).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ValueError("side mask has no foreground contour")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 25:
        raise ValueError("side mask foreground is too small")
    return contour


def _width_profile(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = bbox
    cropped = mask[y : y + height, x : x + width] > 0
    values: list[float] = []
    for index in range(WIDTH_BINS):
        y0 = int(round(index * height / WIDTH_BINS))
        y1 = max(y0 + 1, int(round((index + 1) * height / WIDTH_BINS)))
        rows = cropped[y0:min(y1, height)]
        row_widths: list[int] = []
        for row in rows:
            columns = np.flatnonzero(row)
            if len(columns):
                row_widths.append(int(columns[-1] - columns[0] + 1))
        values.append(float(np.median(row_widths)) / max(width, 1) if row_widths else 0.0)
    return np.asarray(values, np.float32)


def _line_features(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 135)
    edges = cv2.bitwise_and(edges, (mask > 0).astype(np.uint8) * 255)
    minimum = max(8, int(round(min(mask.shape) * 0.06)))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=max(12, minimum),
        minLineLength=minimum, maxLineGap=max(3, minimum // 3),
    )
    histogram = np.zeros(ORIENTATION_BINS, np.float32)
    angles: list[float] = []
    if lines is not None:
        for raw in np.asarray(lines).reshape(-1, 4):
            x0, y0, x1, y1 = map(float, raw)
            length = float(np.hypot(x1 - x0, y1 - y0))
            angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180.0)
            histogram[min(int(angle / 180.0 * ORIENTATION_BINS), ORIENTATION_BINS - 1)] += length
            angles.append(angle)
    histogram /= max(float(histogram.sum()), 1.0)
    parallel = 0.0
    converging = 0.0
    if len(angles) >= 2:
        pairs = 0
        for index, first in enumerate(angles):
            for second in angles[index + 1 :]:
                delta = abs(first - second)
                delta = min(delta, 180.0 - delta)
                parallel += float(delta <= 8.0)
                converging += float(12.0 <= delta <= 65.0)
                pairs += 1
        parallel /= max(pairs, 1)
        converging /= max(pairs, 1)
    return np.concatenate((histogram, np.asarray([parallel, converging], np.float32))), {
        "line_count": float(len(angles)),
        "parallel_edge_ratio": parallel,
        "converging_edge_ratio": converging,
    }


def extract_side_features(image_bgr: np.ndarray, mask: np.ndarray) -> SideFeatureResult:
    image = np.asarray(image_bgr)
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if image.ndim != 3 or image.shape[:2] != binary.shape:
        raise ValueError("side image and mask dimensions must match")
    contour = _largest_contour(binary)
    x, y, width, height = cv2.boundingRect(contour)
    area = max(float(cv2.contourArea(contour)), 1.0)
    perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
    hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
    profile = _width_profile(binary, (x, y, width, height))
    top = float(np.mean(profile[: WIDTH_BINS // 4]))
    middle = float(np.mean(profile[WIDTH_BINS * 3 // 8 : WIDTH_BINS * 5 // 8]))
    bottom = float(np.mean(profile[-WIDTH_BINS // 4 :]))
    reversed_error = float(np.mean(np.abs(profile - profile[::-1])))
    vertical_stability = float(np.std(profile))
    taper = float((bottom - top) / max(bottom + top, 1e-5))

    approximation = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
    _, (rect_width, rect_height), _ = cv2.minAreaRect(contour)
    ellipse_residual = 1.0
    ellipse_aspect = 0.0
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        axes = ellipse[1]
        ellipse_area = np.pi * axes[0] * axes[1] / 4.0
        ellipse_residual = abs(area - ellipse_area) / max(ellipse_area, 1.0)
        ellipse_aspect = min(axes) / max(max(axes), 1.0)
    hu = cv2.HuMoments(cv2.moments(contour)).reshape(-1)
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    hu = np.clip(hu / 12.0, -1.0, 1.0).astype(np.float32)

    points = contour.reshape(-1, 2).astype(np.float32)
    center = points.mean(axis=0)
    scale = max(float(np.linalg.norm(points - center, axis=1).mean()), 1e-5)
    complex_points = ((points[:, 0] - center[0]) + 1j * (points[:, 1] - center[1])) / scale
    sampled_index = np.linspace(0, len(complex_points) - 1, 128).astype(np.int32)
    spectrum = np.abs(np.fft.fft(complex_points[sampled_index]))
    fourier = spectrum[1 : FOURIER_BINS + 1].astype(np.float32)
    fourier /= max(float(fourier[0]) if len(fourier) else 0.0, 1e-5)

    previous = np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0)
    first_angles = np.arctan2(points[:, 1] - previous[:, 1], points[:, 0] - previous[:, 0])
    next_angles = np.arctan2(following[:, 1] - points[:, 1], following[:, 0] - points[:, 0])
    curvature = np.abs(np.arctan2(np.sin(next_angles - first_angles), np.cos(next_angles - first_angles)))
    curvature_hist, _ = np.histogram(curvature, bins=CURVATURE_BINS, range=(0.0, np.pi))
    curvature_hist = curvature_hist.astype(np.float32)
    curvature_hist /= max(float(curvature_hist.sum()), 1.0)

    line_vector, line_diagnostics = _line_features(image, binary)
    scalar = np.asarray(
        [
            top, middle, bottom,
            top / max(bottom, 1e-5), bottom / max(top, 1e-5),
            taper, reversed_error, vertical_stability,
            len(approximation) / 16.0,
            4.0 * np.pi * area / (perimeter * perimeter),
            area / hull_area,
            min(rect_width, rect_height) / max(rect_width, rect_height, 1.0),
            area / max(float(width * height), 1.0),
            ellipse_residual, ellipse_aspect,
        ],
        np.float32,
    )
    vector = np.concatenate((profile, scalar, line_vector, hu, fourier, curvature_hist)).astype(np.float32)
    groups = np.concatenate(
        (
            np.zeros(len(profile), np.int32),
            np.ones(len(scalar), np.int32),
            np.full(len(line_vector), 2, np.int32),
            np.full(len(hu) + len(fourier) + len(curvature_hist), 3, np.int32),
        )
    )
    diagnostics = {
        "top_width_ratio": top,
        "middle_width_ratio": middle,
        "bottom_width_ratio": bottom,
        "taper": taper,
        "vertical_width_stability": vertical_stability,
        "vertical_symmetry_error": reversed_error,
        "vertex_count": float(len(approximation)),
        "solidity": area / hull_area,
        "ellipse_residual": ellipse_residual,
        "ellipse_aspect": ellipse_aspect,
        **line_diagnostics,
    }
    return SideFeatureResult(vector, groups, diagnostics)


def side_foreground_mask(image_bgr: np.ndarray, background_bgr: np.ndarray) -> np.ndarray:
    image = np.asarray(image_bgr)
    background = np.asarray(background_bgr)
    if image.shape != background.shape:
        raise ValueError("side training image and background dimensions differ")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    reference = cv2.cvtColor(background, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.linalg.norm(lab[:, :, 1:] - reference[:, :, 1:], axis=2)
    luminance = np.abs(lab[:, :, 0] - reference[:, :, 0])
    absolute = cv2.absdiff(image, background).max(axis=2)
    mask = ((chroma >= 7.0) | (luminance >= 20.0) | (absolute >= 25.0)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        raise ValueError("no foreground found in side training image")
    center = np.asarray([image.shape[1] / 2.0, image.shape[0] / 2.0])
    ranked: list[tuple[float, int]] = []
    for item in range(1, count):
        area = int(stats[item, cv2.CC_STAT_AREA])
        if area < max(80, image.shape[0] * image.shape[1] * 0.0002):
            continue
        distance = np.linalg.norm(centroids[item] - center) / max(np.linalg.norm(center), 1.0)
        ranked.append((area * max(0.35, 1.0 - 0.45 * distance), item))
    if not ranked:
        raise ValueError("side foreground components are too small")
    selected = max(ranked)[1]
    return (labels == selected).astype(np.uint8) * 255


def load_side_training_samples(
    data_root: str | Path,
    registry: ShapeRegistry,
    background_bgr: np.ndarray,
    *,
    require_reviewed: bool = True,
) -> tuple[list[SideTrainingSample], list[dict[str, str]]]:
    root = Path(data_root)
    samples: list[SideTrainingSample] = []
    errors: list[dict[str, str]] = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        directory = metadata_path.parent
        image_path = directory / "side-color.png"
        if not image_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if require_reviewed and not bool(metadata.get("human_reviewed", False)):
                raise ValueError("sample_not_human_reviewed")
            raw_label = str(metadata.get("label_id") or metadata.get("label") or directory.parent.name)
            if raw_label == "empty_tray":
                continue
            label = registry.resolve(raw_label)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("side_image_unreadable")
            mask_path = directory / "side-mask.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
            if mask is None:
                mask = side_foreground_mask(image, background_bgr)
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            samples.append(
                SideTrainingSample(
                    directory, image, mask, label,
                    str(metadata.get("split", "")), str(metadata.get("batch_id", "")),
                    str(metadata.get("instance_id", "")), str(metadata.get("color_id", "")), digest,
                )
            )
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            errors.append({"path": str(directory), "reason": str(error)})
    return samples, errors


def audit_side_dataset(samples: Iterable[SideTrainingSample], registry: ShapeRegistry) -> dict[str, Any]:
    items = list(samples)
    hash_splits: dict[str, set[str]] = {}
    for item in items:
        hash_splits.setdefault(item.sha256, set()).add(item.split)
    duplicate_leaks = sorted(key for key, splits in hash_splits.items() if len(splits) > 1)
    final_instances = {item.instance_id for item in items if item.split == "final_holdout" and item.instance_id}
    development_instances = {
        item.instance_id for item in items
        if item.split in {"train", "probability_calibration"} and item.instance_id
    }
    instance_leaks = sorted(final_instances & development_instances)
    counts = {
        class_id: {
            split: sum(item.label_id == class_id and item.split == split for item in items)
            for split in ("train", "probability_calibration", "final_holdout")
        }
        for class_id in registry.class_ids
    }
    coverage = {
        class_id: {
            split: {
                "color_counts": {
                    color: sum(
                        item.label_id == class_id and item.split == split and item.color_id == color
                        for item in items
                    )
                    for color in sorted(
                        {
                            item.color_id for item in items
                            if item.label_id == class_id and item.split == split and item.color_id
                        }
                    )
                },
                "instances": sorted(
                    {
                        item.instance_id for item in items
                        if item.label_id == class_id and item.split == split and item.instance_id
                    }
                ),
            }
            for split in ("train", "probability_calibration", "final_holdout")
        }
        for class_id in registry.class_ids
    }
    per_class_targets = {}
    for class_id in registry.class_ids:
        train = coverage[class_id]["train"]
        calibration = coverage[class_id]["probability_calibration"]
        final = coverage[class_id]["final_holdout"]
        development_instances = set(train["instances"]) | set(calibration["instances"])
        final_instances_for_class = set(final["instances"])
        per_class_targets[class_id] = {
            "counts_50_15_15": (
                counts[class_id]["train"] >= 50
                and counts[class_id]["probability_calibration"] >= 15
                and counts[class_id]["final_holdout"] >= 15
            ),
            "five_colors_each_split": (
                sum(value >= 10 for value in train["color_counts"].values()) >= 5
                and sum(value >= 3 for value in calibration["color_counts"].values()) >= 5
                and sum(value >= 3 for value in final["color_counts"].values()) >= 5
            ),
            "development_has_two_instances": (
                len(train["instances"]) >= 2 and len(calibration["instances"]) >= 2
            ),
            "final_has_new_instance": (
                bool(final_instances_for_class)
                and not (development_instances & final_instances_for_class)
            ),
            "three_instances_total": (
                len(development_instances | final_instances_for_class) >= 3
            ),
        }
    targets_met = all(
        all(per_class_targets[class_id].values()) for class_id in registry.class_ids
    )
    return {
        "schema_version": 1,
        "registry_hash": registry.registry_hash,
        "sample_count": len(items),
        "counts": counts,
        "coverage": coverage,
        "per_class_targets": per_class_targets,
        "collection_targets_met": targets_met,
        "cross_split_duplicate_hashes": duplicate_leaks,
        "final_instance_leaks": instance_leaks,
        "split_integrity_valid": not duplicate_leaks and not instance_leaks,
        "ready_for_acceptance": not duplicate_leaks and not instance_leaks and targets_met,
    }


def calibrate_side_acceptance(
    model: "SideGeometryModel", samples: Iterable[SideTrainingSample]
) -> dict[str, Any]:
    """Choose distance/margin rejection thresholds on the dedicated calibration split."""
    items = list(samples)
    if not items:
        raise ValueError("no side calibration samples")
    observations: list[tuple[str, str, float, float]] = []
    for item in items:
        _, _, diagnostics = model.predict(item.image_bgr, item.mask)
        observations.append(
            (
                item.label_id,
                str(diagnostics.get("nearest_label", "unknown")),
                float(diagnostics.get("distance", float("inf"))),
                float(diagnostics.get("margin", 0.0)),
            )
        )
    finite_distances = np.asarray(
        [item[2] for item in observations if np.isfinite(item[2])], np.float64
    )
    if not len(finite_distances):
        raise ValueError("side calibration produced no finite distances")
    distance_candidates = sorted(
        set(
            [0.0, *map(float, finite_distances)]
            + list(map(float, np.quantile(finite_distances, np.linspace(0.05, 1.0, 20))))
        )
    )
    observed_margins = np.asarray([item[3] for item in observations], np.float64)
    margin_candidates = sorted(
        set([0.0, *map(float, observed_margins), *map(float, np.linspace(0.02, 0.40, 20))])
    )
    labels = sorted({item[0] for item in observations})
    candidates: list[dict[str, Any]] = []
    for distance_threshold in distance_candidates:
        for margin_threshold in margin_candidates:
            predictions = [
                predicted if distance <= distance_threshold and margin >= margin_threshold else None
                for _, predicted, distance, margin in observations
            ]
            accepted = sum(item is not None for item in predictions)
            wrong = sum(
                prediction is not None and prediction != observation[0]
                for prediction, observation in zip(predictions, observations)
            )
            recalls = []
            for label in labels:
                indices = [index for index, value in enumerate(observations) if value[0] == label]
                recalls.append(
                    sum(predictions[index] == label for index in indices) / max(len(indices), 1)
                )
            candidates.append(
                {
                    "distance_threshold": float(distance_threshold),
                    "margin_threshold": float(margin_threshold),
                    "wrong_accept_count": int(wrong),
                    "macro_recall": float(np.mean(recalls)),
                    "coverage": accepted / len(observations),
                }
            )
    candidates.sort(
        key=lambda item: (
            item["wrong_accept_count"] == 0,
            -item["wrong_accept_count"],
            item["macro_recall"],
            item["coverage"],
            -item["distance_threshold"],
            item["margin_threshold"],
        ),
        reverse=True,
    )
    selected = candidates[0]
    model.distance_threshold = selected["distance_threshold"]
    model.margin_threshold = selected["margin_threshold"]
    return {
        "schema_version": 1,
        "split": "probability_calibration",
        "sample_count": len(observations),
        "selection_priority": ["zero_wrong_accept", "macro_recall", "coverage"],
        "selected": selected,
    }


class SideGeometryModel:
    """Scale-normalised side-view KNN or OpenCV RTrees classifier."""

    backend = "opencv"

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray,
        registry: ShapeRegistry,
        *,
        method: str = "knn",
        group_ids: np.ndarray | None = None,
        group_weights: np.ndarray | None = None,
        distance_threshold: float = 4.0,
        margin_threshold: float = 0.08,
    ) -> None:
        if method not in {"knn", "rtrees"}:
            raise ValueError("side method must be knn or rtrees")
        self.features = np.asarray(features, np.float32)
        self.labels = np.asarray(labels).astype(str)
        self.mean = np.asarray(mean, np.float32)
        self.scale = np.maximum(np.asarray(scale, np.float32), 1e-5)
        self.registry = registry
        self.class_ids = registry.class_ids
        self.registry_hash = registry.registry_hash
        self.method = method
        self.group_ids = (
            np.zeros(self.features.shape[1], np.int32)
            if group_ids is None else np.asarray(group_ids, np.int32)
        )
        self.group_weights = (
            np.asarray([0.25, 0.20, 0.30, 0.25], np.float32)
            if group_weights is None else np.asarray(group_weights, np.float32)
        )
        self.distance_threshold = float(distance_threshold)
        self.margin_threshold = float(margin_threshold)
        self.last_class_scores: dict[str, float] = {}
        self.last_feature_diagnostics: dict[str, float] = {}
        if self.features.ndim != 2 or len(self.features) != len(self.labels) or not len(self.labels):
            raise ValueError("side model needs matching non-empty features and labels")
        if self.features.shape[1] != len(self.mean) or len(self.mean) != len(self.scale):
            raise ValueError("side feature normalization dimensions do not match")
        if len(self.group_ids) != self.features.shape[1]:
            raise ValueError("side feature group dimensions do not match")
        unknown = sorted(set(self.labels) - set(self.class_ids))
        if unknown:
            raise ValueError(f"side model labels are absent from registry: {unknown}")
        self._forest = self._train_forest() if method == "rtrees" else None

    @classmethod
    def train(
        cls,
        images_and_masks: Iterable[tuple[np.ndarray, np.ndarray, str]],
        registry: ShapeRegistry,
        *,
        method: str = "knn",
    ) -> "SideGeometryModel":
        vectors: list[np.ndarray] = []
        labels: list[str] = []
        group_ids: np.ndarray | None = None
        for image, mask, raw_label in images_and_masks:
            label = registry.resolve(raw_label)
            result = extract_side_features(image, mask)
            vectors.append(result.vector)
            labels.append(label)
            group_ids = result.group_ids
        if not vectors:
            raise ValueError("no usable side-view training samples")
        raw = np.stack(vectors).astype(np.float32)
        mean = raw.mean(axis=0)
        scale = raw.std(axis=0)
        scale[scale < 1e-5] = 1.0
        normalized = (raw - mean) / scale
        if set(labels) != set(registry.class_ids):
            missing = sorted(set(registry.class_ids) - set(labels))
            raise ValueError(f"side training set does not cover every enabled class: {missing}")
        model = cls(normalized, np.asarray(labels), mean, scale, registry, method=method, group_ids=group_ids)
        # Conservative in-sample scale; independent validation must replace the
        # acceptance thresholds before promotion.
        leave_one_out: list[float] = []
        for index, item in enumerate(normalized):
            distances = model._distances(item).astype(np.float64)
            distances[index] = np.inf
            same_class = distances[model.labels == model.labels[index]]
            finite = same_class[np.isfinite(same_class)]
            if len(finite):
                leave_one_out.append(float(finite.min()))
        model.distance_threshold = max(
            float(np.percentile(leave_one_out, 99)) * 1.35 if leave_one_out else 4.0,
            0.25,
        )
        return model

    def _distances(self, feature: np.ndarray) -> np.ndarray:
        squared = (self.features - feature) ** 2
        output = np.zeros(len(self.features), np.float32)
        for group, weight in enumerate(self.group_weights):
            selected = self.group_ids == group
            if np.any(selected):
                output += float(weight) * squared[:, selected].mean(axis=1)
        return np.sqrt(output / max(float(self.group_weights.sum()), 1e-6))

    def _train_forest(self) -> Any:
        forest = cv2.ml.RTrees_create()
        forest.setMaxDepth(12)
        forest.setMinSampleCount(2)
        forest.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 160, 0.0))
        ids = np.asarray([self.class_ids.index(item) for item in self.labels], np.int32)
        forest.train(self.features, cv2.ml.ROW_SAMPLE, ids)
        return forest

    def _rtrees_scores(self, feature: np.ndarray) -> dict[str, float]:
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

    def predict(self, image_bgr: np.ndarray, mask: np.ndarray | None = None) -> tuple[str, float, dict[str, Any]]:
        if mask is None:
            self.last_class_scores = {}
            return "unknown", 0.0, {"reason": "side_mask_required"}
        try:
            extracted = extract_side_features(image_bgr, mask)
        except ValueError as error:
            self.last_class_scores = {}
            return "unknown", 0.0, {"reason": "side_feature_invalid", "detail": str(error)}
        self.last_feature_diagnostics = extracted.diagnostics
        feature = (extracted.vector - self.mean) / self.scale
        if self.method == "rtrees":
            scores = self._rtrees_scores(feature)
            best_distance = float(self._distances(feature).min())
        else:
            distances = self._distances(feature)
            class_distances: dict[str, float] = {}
            for item in self.class_ids:
                selected = np.sort(distances[self.labels == item])
                class_distances[item] = (
                    float(np.mean(selected[: min(3, len(selected))])) if len(selected) else float("inf")
                )
            finite = np.asarray([class_distances[item] for item in self.class_ids], np.float64)
            replacement = float(np.max(finite[np.isfinite(finite)]) + 5.0)
            finite[~np.isfinite(finite)] = replacement
            logits = -(finite - finite.min()) / max(float(np.std(finite)), 0.10)
            probabilities = np.exp(logits - logits.max())
            probabilities /= max(float(probabilities.sum()), 1e-12)
            scores = {item: float(probabilities[index]) for index, item in enumerate(self.class_ids)}
            best_distance = min(class_distances.values())
        self.last_class_scores = scores
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        label, probability = ordered[0]
        margin = probability - (ordered[1][1] if len(ordered) > 1 else 0.0)
        reason = "accepted"
        if best_distance > self.distance_threshold:
            reason = "distance_rejected"
        elif margin < self.margin_threshold:
            reason = "margin_rejected"
        return (label if reason == "accepted" else "unknown"), probability, {
            "nearest_label": label,
            "reason": reason,
            "distance": best_distance,
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
            model_type=np.asarray(["side_geometry"]),
            feature_version=np.asarray([SIDE_FEATURE_VERSION], np.int32),
            method=np.asarray([self.method]),
            features=self.features,
            labels=self.labels,
            mean=self.mean,
            scale=self.scale,
            group_ids=self.group_ids,
            group_weights=self.group_weights,
            class_ids=np.asarray(self.class_ids),
            registry_hash=np.asarray([self.registry_hash]),
            registry_json=np.asarray([json.dumps(self.registry.to_dict(), ensure_ascii=False)]),
            distance_threshold=np.asarray([self.distance_threshold], np.float32),
            margin_threshold=np.asarray([self.margin_threshold], np.float32),
        )

    @classmethod
    def load(cls, path: str | Path, registry: ShapeRegistry | None = None) -> "SideGeometryModel":
        with np.load(path, allow_pickle=False) as data:
            registry_value = json.loads(str(data["registry_json"][0]))
            stored_registry = ShapeRegistry(
                version=int(registry_value["version"]),
                classes=tuple(ShapeClass.from_dict(item) for item in registry_value["classes"]),
            )
            active = stored_registry if registry is None else registry
            active.validate_model_classes(data["class_ids"].astype(str), str(data["registry_hash"][0]))
            return cls(
                data["features"], data["labels"], data["mean"], data["scale"], active,
                method=str(data["method"][0]), group_ids=data["group_ids"],
                group_weights=data["group_weights"],
                distance_threshold=float(data["distance_threshold"][0]),
                margin_threshold=float(data["margin_threshold"][0]),
            )
