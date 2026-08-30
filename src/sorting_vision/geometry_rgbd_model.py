from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import VisionConfig, load_config
from .geometry3d import object_point_cloud, segment_depth_objects, valid_depth_mask
from .rgbd import Plane, RGBDFrame, depth_to_points, fit_plane_ransac, fit_plane_svd
from .rgbd_dataset import EMPTY_TRAY_LABEL, load_rgbd_dataset_entries
from .camera import load_rgbd_frame


FEATURE_NAMES = (
    "extent_major_mm", "extent_middle_mm", "extent_minor_mm",
    "extent_middle_over_major", "extent_minor_over_major",
    "eigen_middle_over_major", "eigen_minor_over_major",
    "surface_rmse_mm", "depth_range_mm", "depth_iqr_mm",
    "silhouette_aspect", "circularity", "solidity", "vertices",
    "height_hist_0", "height_hist_1", "height_hist_2", "height_hist_3",
    "height_hist_4", "valid_depth_ratio",
)


def extract_rgbd_geometry_features(
    points_camera_mm: np.ndarray,
    depth_crop_mm: np.ndarray,
    crop_mask: np.ndarray,
) -> np.ndarray:
    """Extract pose-tolerant metric and visible-surface features from one object."""
    points = np.asarray(points_camera_mm, np.float64)
    mask = np.asarray(crop_mask, np.uint8)
    if len(points) < 40 or cv2.countNonZero(mask) < 40:
        raise ValueError("insufficient_object_depth")
    centered = points - np.mean(points, axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, axes = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 1e-9)
    local = centered @ axes[:, order]
    extents = np.percentile(local, 97.5, axis=0) - np.percentile(local, 2.5, axis=0)
    extents = np.maximum(np.sort(extents)[::-1], 1e-3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("missing_object_contour")
    contour = max(contours, key=cv2.contourArea)
    area = max(float(cv2.contourArea(contour)), 1.0)
    perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
    hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
    _, (width, height), _ = cv2.minAreaRect(contour)
    aspect = min(width, height) / max(width, height, 1.0)
    vertices = len(cv2.approxPolyDP(contour, 0.035 * perimeter, True))

    plane = fit_plane_svd(points)
    depth = np.asarray(depth_crop_mm, np.float32)
    valid = (mask > 0) & np.isfinite(depth) & (depth > 0)
    values = depth[valid]
    if len(values) < 40:
        raise ValueError("insufficient_valid_depth")
    low, high = np.percentile(values, [2, 98])
    normalized = np.clip((values - low) / max(float(high - low), 1.0), 0, 1)
    histogram = np.histogram(normalized, bins=5, range=(0, 1))[0].astype(np.float64)
    histogram /= max(float(histogram.sum()), 1.0)
    features = np.asarray(
        [
            *extents.tolist(),
            extents[1] / extents[0], extents[2] / extents[0],
            eigenvalues[1] / eigenvalues[0], eigenvalues[2] / eigenvalues[0],
            plane.rmse_mm, float(high - low),
            float(np.percentile(values, 75) - np.percentile(values, 25)),
            aspect, 4.0 * np.pi * area / (perimeter * perimeter),
            area / hull_area, float(vertices), *histogram.tolist(),
            float(np.mean(valid[mask > 0])),
        ],
        np.float32,
    )
    if not np.all(np.isfinite(features)):
        raise ValueError("non_finite_rgbd_features")
    return features


class DepthGeometryModel:
    """Standardised nearest-class model implementing ShapeModel3D."""

    def __init__(
        self,
        labels: list[str],
        mean: np.ndarray,
        scale: np.ndarray,
        centroids: np.ndarray,
        thresholds: np.ndarray,
        min_margin: float = 0.08,
    ) -> None:
        self.labels = labels
        self.mean = np.asarray(mean, np.float32)
        self.scale = np.asarray(scale, np.float32)
        self.centroids = np.asarray(centroids, np.float32)
        self.thresholds = np.asarray(thresholds, np.float32)
        self.min_margin = float(min_margin)
        if self.centroids.shape != (len(labels), len(FEATURE_NAMES)):
            raise ValueError("invalid RGB-D model dimensions")

    @classmethod
    def fit(cls, features: np.ndarray, labels: list[str]) -> "DepthGeometryModel":
        values = np.asarray(features, np.float32)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError("invalid RGB-D training features")
        unique = sorted(set(labels))
        if len(unique) < 2:
            raise ValueError("at least two object classes are required")
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale[scale < 1e-4] = 1.0
        standard = (values - mean) / scale
        centroids = np.vstack([np.mean(standard[np.asarray(labels) == label], axis=0) for label in unique])
        thresholds = []
        for index, label in enumerate(unique):
            distances = np.mean(np.abs(standard[np.asarray(labels) == label] - centroids[index]), axis=1)
            thresholds.append(max(0.45, float(np.percentile(distances, 95)) * 1.35 + 0.05))
        return cls(unique, mean, scale, centroids, np.asarray(thresholds, np.float32))

    def predict_features(self, features: np.ndarray) -> tuple[str, float, str]:
        standard = (np.asarray(features, np.float32) - self.mean) / self.scale
        distances = np.mean(np.abs(self.centroids - standard), axis=1)
        order = np.argsort(distances)
        best = int(order[0])
        distance = float(distances[best])
        gap = float(distances[order[1]] - distance) if len(order) > 1 else float("inf")
        relative = float(np.clip(1.0 - distance / max(float(self.thresholds[best]), 1e-6), 0, 1))
        confidence = 0.78 + 0.21 * relative
        if distance > float(self.thresholds[best]):
            return "unknown", 0.0, "distance_rejected"
        if gap < self.min_margin:
            return "unknown", min(confidence, 0.5), "margin_rejected"
        return self.labels[best], confidence, "accepted"

    def classify(
        self,
        points_camera_mm: np.ndarray,
        color_crop_bgr: np.ndarray,
        depth_crop_mm: np.ndarray,
        crop_mask: np.ndarray,
    ) -> tuple[str, float]:
        del color_crop_bgr
        try:
            features = extract_rgbd_geometry_features(
                points_camera_mm, depth_crop_mm, crop_mask
            )
        except ValueError:
            return "unknown", 0.0
        label, confidence, _ = self.predict_features(features)
        return label, confidence

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            model_type=np.asarray("rgbd_geometry_v1"),
            feature_names=np.asarray(FEATURE_NAMES), labels=np.asarray(self.labels),
            mean=self.mean, scale=self.scale, centroids=self.centroids,
            thresholds=self.thresholds, min_margin=np.asarray(self.min_margin),
            metadata_json=np.asarray(json.dumps(metadata or {}, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DepthGeometryModel":
        with np.load(Path(path), allow_pickle=False) as data:
            if str(data["model_type"]) != "rgbd_geometry_v1":
                raise ValueError("unsupported RGB-D geometry model")
            if tuple(data["feature_names"].tolist()) != FEATURE_NAMES:
                raise ValueError("RGB-D feature version mismatch")
            return cls(
                data["labels"].tolist(), data["mean"], data["scale"],
                data["centroids"], data["thresholds"], float(data["min_margin"]),
            )


def _fit_background_plane(frame: RGBDFrame, cfg: VisionConfig) -> Plane:
    mask = valid_depth_mask(frame.depth_mm, cfg.rgbd).astype(np.uint8) * 255
    points, _ = depth_to_points(frame.depth_mm, frame.intrinsics, mask, stride=8)
    return fit_plane_ransac(points, cfg.rgbd.plane_ransac_threshold_mm)


def train_rgbd_geometry_model(
    data_root: str | Path,
    output: str | Path,
    config: VisionConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_config()
    entries = load_rgbd_dataset_entries(data_root)
    backgrounds: dict[str, RGBDFrame] = {}
    for entry in entries:
        if entry["label_id"] == EMPTY_TRAY_LABEL:
            backgrounds[str(entry["batch_id"])] = load_rgbd_frame(entry["absolute_sample_dir"])
    features: list[np.ndarray] = []
    labels: list[str] = []
    rejected: list[dict[str, str]] = []
    counts: defaultdict[str, int] = defaultdict(int)
    for entry in entries:
        label = str(entry["label_id"])
        if label == EMPTY_TRAY_LABEL:
            continue
        batch = str(entry["batch_id"])
        if batch not in backgrounds:
            rejected.append({"sample_dir": entry["sample_dir"], "reason": "missing_batch_empty_tray"})
            continue
        try:
            frame = load_rgbd_frame(entry["absolute_sample_dir"])
            background = backgrounds[batch]
            if frame.intrinsics != background.intrinsics:
                raise ValueError("intrinsics_mismatch_with_empty_tray")
            plane = _fit_background_plane(background, cfg)
            objects, _ = segment_depth_objects(
                frame.color_bgr, frame.depth_mm, frame.intrinsics, plane, cfg.rgbd
            )
            if len(objects) != 1:
                raise ValueError(f"expected_one_object_got_{len(objects)}")
            item = objects[0]
            points, _ = object_point_cloud(item, frame.depth_mm, frame.intrinsics, stride=2)
            x, y, width, height = item.bbox
            feature = extract_rgbd_geometry_features(
                points,
                frame.depth_mm[y:y + height, x:x + width],
                item.mask[y:y + height, x:x + width],
            )
            features.append(feature)
            labels.append(label)
            counts[label] += 1
        except (ValueError, OSError, FileNotFoundError) as error:
            rejected.append({"sample_dir": entry["sample_dir"], "reason": str(error)})
    if len(set(labels)) < 2:
        raise ValueError("training needs at least two valid object classes")
    model = DepthGeometryModel.fit(np.vstack(features), labels)
    report = {
        "model_type": "rgbd_geometry_v1",
        "data_root": str(data_root),
        "accepted_samples": len(features),
        "class_counts": dict(sorted(counts.items())),
        "empty_tray_batches": sorted(backgrounds),
        "rejected": rejected,
        "training_replay_only": True,
    }
    model.save(output, report)
    return report
