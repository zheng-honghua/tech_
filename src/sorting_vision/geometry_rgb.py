from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


GEOMETRY_LABELS: dict[str, tuple[str, str]] = {
    "三棱柱": ("triangular_prism", "三棱柱"),
    "三棱锥": ("triangular_pyramid", "三棱锥"),
    "四棱锥": ("square_pyramid", "四棱锥"),
    "五棱柱": ("pentagonal_prism", "五棱柱"),
    "六棱柱": ("hexagonal_prism", "六棱柱"),
    "六棱锥": ("hexagonal_pyramid", "六棱锥"),
    "正八面体": ("octahedron", "正八面体"),
}
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
FEATURE_VERSION = 1
MODEL_VERSION = 1


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


def extract_geometry_features(preprocessed: GeometryPreprocessed) -> np.ndarray:
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


class GeometryRGBModel:
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
    ) -> None:
        self.features = np.asarray(features, np.float32)
        self.labels = np.asarray(labels).astype(str)
        self.feature_mean = np.asarray(feature_mean, np.float32)
        self.feature_scale = np.asarray(feature_scale, np.float32)
        self.class_names = dict(class_names)
        self.distance_threshold = float(max(distance_threshold, 1e-5))
        self.margin_threshold = float(margin_threshold)
        self.source_hashes = list(source_hashes or [])
        if len(self.features) != len(self.labels) or not len(self.features):
            raise ValueError("geometry model requires matching non-empty features and labels")

    @classmethod
    def fit(
        cls,
        raw_features: np.ndarray,
        labels: list[str] | np.ndarray,
        class_names: dict[str, str],
        source_hashes: list[str] | None = None,
    ) -> "GeometryRGBModel":
        values = np.asarray(raw_features, np.float32)
        label_values = np.asarray(labels).astype(str)
        if values.ndim != 2 or len(values) != len(label_values):
            raise ValueError("training features must be a 2-D array matching labels")
        counts = Counter(label_values.tolist())
        if any(count < 2 for count in counts.values()):
            raise ValueError("each geometry class requires at least two samples")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        # A variance floor prevents tiny pixel-level mask changes from exploding
        # after standardisation in this very small training set.
        scale = np.maximum(scale, 0.05)
        standardized = (values - mean) / scale
        same_distances: list[float] = []
        for index, label in enumerate(label_values):
            indices = np.flatnonzero(label_values == label)
            indices = indices[indices != index]
            distances = np.sqrt(np.mean((standardized[indices] - standardized[index]) ** 2, axis=1))
            same_distances.append(float(np.min(distances)))
        threshold = max(float(np.percentile(same_distances, 95)) * 1.35, 0.25)
        return cls(
            standardized,
            label_values,
            mean,
            scale,
            class_names,
            threshold,
            source_hashes=source_hashes,
        )

    def predict_feature(self, raw_feature: np.ndarray) -> tuple[str, float, dict[str, float | str]]:
        feature = (np.asarray(raw_feature, np.float32) - self.feature_mean) / self.feature_scale
        distances = np.sqrt(np.mean((self.features - feature) ** 2, axis=1))
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
        accepted = best_distance <= self.distance_threshold and margin >= self.margin_threshold
        diagnostics: dict[str, float | str] = {
            "nearest_label": best_label,
            "distance": best_distance,
            "distance_threshold": self.distance_threshold,
            "margin": margin,
            "margin_threshold": self.margin_threshold,
            "reason": "accepted" if accepted else ("distance_rejected" if best_distance > self.distance_threshold else "margin_rejected"),
        }
        return (best_label if accepted else "unknown"), confidence, diagnostics

    def predict(self, image_bgr: np.ndarray, mask: np.ndarray | None = None) -> tuple[str, float, dict[str, float | str]]:
        # Re-segment the colour-normalised object inside the crop so training and
        # runtime use the same silhouette. The upstream mask may include shadows.
        prepared = preprocess_geometry_object(image_bgr)
        if prepared is None and mask is not None:
            prepared = preprocess_geometry_object(image_bgr, mask)
        if prepared is None:
            return "unknown", 0.0, {"reason": "object_not_found"}
        if prepared.candidate_count != 1:
            return "unknown", 0.0, {
                "reason": "multiple_objects",
                "candidate_count": float(prepared.candidate_count),
            }
        return self.predict_feature(extract_geometry_features(prepared))

    def __call__(self, crop: np.ndarray, crop_mask: np.ndarray) -> tuple[str, float]:
        label, confidence, _ = self.predict(crop, crop_mask)
        return label, confidence

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            model_version=np.asarray([MODEL_VERSION], np.int32),
            feature_version=np.asarray([FEATURE_VERSION], np.int32),
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
            if int(value["model_version"][0]) != MODEL_VERSION:
                raise ValueError("unsupported geometry RGB model version")
            if int(value["feature_version"][0]) != FEATURE_VERSION:
                raise ValueError("unsupported geometry feature version")
            return cls(
                value["features"],
                value["labels"],
                value["feature_mean"],
                value["feature_scale"],
                json.loads(str(value["class_names_json"][0])),
                float(value["distance_threshold"][0]),
                float(value["margin_threshold"][0]),
                value["source_hashes"].astype(str).tolist(),
            )


def _features_from_samples(samples: list[GeometrySample]) -> tuple[np.ndarray, list[GeometrySample], list[dict[str, str]]]:
    features: list[np.ndarray] = []
    valid: list[GeometrySample] = []
    errors: list[dict[str, str]] = []
    for sample in samples:
        prepared = preprocess_geometry_object(sample.image_bgr)
        if prepared is None:
            errors.append({"path": str(sample.path), "reason": "object_not_found"})
            continue
        features.append(extract_geometry_features(prepared))
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


def train_geometry_model(data_root: str | Path) -> tuple[GeometryRGBModel, dict[str, Any]]:
    samples, load_errors = load_geometry_samples(data_root)
    features, valid, preprocessing_errors = _features_from_samples(samples)
    labels = [sample.label_id for sample in valid]
    names = {sample.label_id: sample.label_name for sample in valid}
    model = GeometryRGBModel.fit(features, labels, names, [sample.sha256 for sample in valid])
    report = {
        "training_samples": len(valid),
        "class_counts": dict(sorted(Counter(labels).items())),
        "feature_count": int(features.shape[1]),
        "distance_threshold": round(model.distance_threshold, 6),
        "errors": [*load_errors, *preprocessing_errors],
        "same_batch_only": True,
    }
    return model, report


def evaluate_geometry_model(data_root: str | Path, model_path: str | Path) -> dict[str, Any]:
    reference = GeometryRGBModel.load(model_path)
    samples, load_errors = load_geometry_samples(data_root)
    features, valid, preprocessing_errors = _features_from_samples(samples)
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
        "samples": len(valid),
        "labels": matrix_labels,
        "confusion_matrix": matrix.tolist(),
        "accuracy": round(accuracy, 6),
        "per_class_recall": {key: round(value, 6) for key, value in recalls.items()},
        "macro_recall": round(float(np.mean(list(recalls.values()))), 6),
        "predictions": rows,
        "errors": [*load_errors, *preprocessing_errors],
    }
