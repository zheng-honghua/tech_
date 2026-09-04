from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import VisionConfig, load_config
from .geometry3d import object_point_cloud, segment_depth_objects, valid_depth_mask
from .rgbd import (
    Plane,
    RGBDFrame,
    depth_to_points,
    fit_plane_ransac,
    fit_plane_svd,
    resize_rgbd_frame,
)
from .rgbd_dataset import EMPTY_TRAY_LABEL, load_rgbd_dataset_entries
from .camera import load_rgbd_frame
from .face_topology3d import (
    TOPOLOGY_FEATURE_NAMES,
    extract_face_topology,
    face_topology_features,
)
from .rgbd import CameraIntrinsics


BASE_FEATURE_NAMES = (
    "extent_major_mm", "extent_middle_mm", "extent_minor_mm",
    "extent_middle_over_major", "extent_minor_over_major",
    "eigen_middle_over_major", "eigen_minor_over_major",
    "surface_rmse_mm", "depth_range_mm", "depth_iqr_mm",
    "silhouette_aspect", "circularity", "solidity", "vertices",
    "height_hist_0", "height_hist_1", "height_hist_2", "height_hist_3",
    "height_hist_4", "valid_depth_ratio",
)
FEATURE_NAMES = (*BASE_FEATURE_NAMES, *TOPOLOGY_FEATURE_NAMES)


def extract_rgbd_geometry_features(
    points_camera_mm: np.ndarray,
    depth_crop_mm: np.ndarray,
    crop_mask: np.ndarray,
    intrinsics: CameraIntrinsics | None = None,
    crop_origin_uv: tuple[int, int] = (0, 0),
    color_crop_bgr: np.ndarray | None = None,
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
    base_features = np.asarray(
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
    if intrinsics is None:
        height, width = depth.shape
        focal = float(max(width, height))
        intrinsics = CameraIntrinsics(width, height, focal, focal, width / 2, height / 2)
        crop_origin_uv = (0, 0)
    topology = extract_face_topology(
        depth, mask, intrinsics, crop_origin_uv, color_crop_bgr=color_crop_bgr
    )
    features = np.concatenate((base_features, face_topology_features(topology)))
    if not np.all(np.isfinite(features)):
        raise ValueError("non_finite_rgbd_features")
    return features


class DepthGeometryModel:
    """Standardised multi-pose nearest-neighbour model implementing ShapeModel3D."""

    def __init__(
        self,
        labels: list[str],
        mean: np.ndarray,
        scale: np.ndarray,
        centroids: np.ndarray,
        thresholds: np.ndarray,
        min_margin: float = 0.08,
        feature_names: tuple[str, ...] = FEATURE_NAMES,
        exemplars: np.ndarray | None = None,
        exemplar_label_indices: np.ndarray | None = None,
        neighbors: int = 3,
    ) -> None:
        self.labels = labels
        self.mean = np.asarray(mean, np.float32)
        self.scale = np.asarray(scale, np.float32)
        self.centroids = np.asarray(centroids, np.float32)
        self.thresholds = np.asarray(thresholds, np.float32)
        self.min_margin = float(min_margin)
        self.feature_names = tuple(feature_names)
        self.exemplars = None if exemplars is None else np.asarray(exemplars, np.float32)
        self.exemplar_label_indices = (
            None if exemplar_label_indices is None
            else np.asarray(exemplar_label_indices, np.int32)
        )
        self.neighbors = max(1, int(neighbors))
        self.last_diagnostics: dict[str, float] = {}
        if self.centroids.shape != (len(labels), len(self.feature_names)):
            raise ValueError("invalid RGB-D model dimensions")
        if self.exemplars is not None:
            if self.exemplars.ndim != 2 or self.exemplars.shape[1] != len(self.feature_names):
                raise ValueError("invalid RGB-D exemplar dimensions")
            if self.exemplar_label_indices is None or len(self.exemplar_label_indices) != len(self.exemplars):
                raise ValueError("invalid RGB-D exemplar labels")

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
        label_indices = np.asarray([unique.index(label) for label in labels], np.int32)
        neighbors = 3
        own_scores: list[list[float]] = [[] for _ in unique]
        for sample_index, sample in enumerate(standard):
            distances = np.mean(np.abs(standard - sample), axis=1)
            distances[sample_index] = np.inf
            own = np.sort(distances[label_indices == label_indices[sample_index]])
            finite = own[np.isfinite(own)]
            if len(finite):
                own_scores[label_indices[sample_index]].append(
                    float(np.mean(finite[:min(neighbors, len(finite))]))
                )
        thresholds = [
            max(0.35, float(np.percentile(scores, 98)) * 1.35 + 0.04)
            if scores else 0.75
            for scores in own_scores
        ]
        return cls(
            unique, mean, scale, centroids, np.asarray(thresholds, np.float32),
            exemplars=standard, exemplar_label_indices=label_indices, neighbors=neighbors,
        )

    def _class_distances(self, standard: np.ndarray) -> np.ndarray:
        if self.exemplars is None or self.exemplar_label_indices is None:
            return np.mean(np.abs(self.centroids - standard), axis=1)
        sample_distances = np.mean(np.abs(self.exemplars - standard), axis=1)
        scores = np.full(len(self.labels), np.inf, np.float32)
        for class_index in range(len(self.labels)):
            values = np.sort(sample_distances[self.exemplar_label_indices == class_index])
            if len(values):
                scores[class_index] = float(np.mean(values[:min(self.neighbors, len(values))]))
        return scores

    def training_data(self) -> tuple[np.ndarray, list[str]]:
        """Recover raw features and labels stored by a v3 exemplar model."""
        if self.exemplars is None or self.exemplar_label_indices is None:
            raise ValueError("RGB-D model does not contain training exemplars")
        raw = self.exemplars * self.scale + self.mean
        labels = [self.labels[int(index)] for index in self.exemplar_label_indices]
        return raw.astype(np.float32), labels

    def predict_features(self, features: np.ndarray) -> tuple[str, float, str]:
        supplied = np.asarray(features, np.float32)
        if supplied.shape == (len(FEATURE_NAMES),) and self.feature_names != FEATURE_NAMES:
            indices = [FEATURE_NAMES.index(name) for name in self.feature_names]
            supplied = supplied[indices]
        standard = (supplied - self.mean) / self.scale
        distances = self._class_distances(standard)
        order = np.argsort(distances)
        best = int(order[0])
        distance = float(distances[best])
        gap = float(distances[order[1]] - distance) if len(order) > 1 else float("inf")
        relative_gap = gap / max(distance, 0.05)
        relative = float(np.clip(1.0 - distance / max(float(self.thresholds[best]), 1e-6), 0, 1))
        confidence = 0.78 + 0.21 * relative
        if distance > float(self.thresholds[best]):
            return "unknown", 0.0, "distance_rejected"
        margin_value = relative_gap if self.exemplars is not None else gap
        if margin_value < self.min_margin:
            return "unknown", min(confidence, 0.5), "margin_rejected"
        return self.labels[best], confidence, "accepted"

    def classify(
        self,
        points_camera_mm: np.ndarray,
        color_crop_bgr: np.ndarray,
        depth_crop_mm: np.ndarray,
        crop_mask: np.ndarray,
        intrinsics: CameraIntrinsics | None = None,
        crop_origin_uv: tuple[int, int] = (0, 0),
    ) -> tuple[str, float]:
        try:
            features = extract_rgbd_geometry_features(
                points_camera_mm, depth_crop_mm, crop_mask, intrinsics,
                crop_origin_uv, color_crop_bgr,
            )
        except ValueError:
            self.last_diagnostics = {"plane_topology_quality": 0.0}
            return "unknown", 0.0
        start = len(FEATURE_NAMES) - len(TOPOLOGY_FEATURE_NAMES)
        self.last_diagnostics = {
            f"topology_{name}": float(features[start + index])
            for index, name in enumerate(TOPOLOGY_FEATURE_NAMES)
        }
        label, confidence, _ = self.predict_features(features)
        return label, confidence

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            model_type=np.asarray(
                "rgbd_geometry_v3_multipose_knn" if self.exemplars is not None
                else ("rgbd_geometry_v2_face_topology"
                      if self.feature_names == FEATURE_NAMES else "rgbd_geometry_v1")
            ),
            feature_names=np.asarray(self.feature_names), labels=np.asarray(self.labels),
            mean=self.mean, scale=self.scale, centroids=self.centroids,
            thresholds=self.thresholds, min_margin=np.asarray(self.min_margin),
            exemplars=(self.exemplars if self.exemplars is not None
                       else np.empty((0, len(self.feature_names)), np.float32)),
            exemplar_label_indices=(self.exemplar_label_indices
                                    if self.exemplar_label_indices is not None
                                    else np.empty(0, np.int32)),
            neighbors=np.asarray(self.neighbors),
            metadata_json=np.asarray(json.dumps(metadata or {}, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DepthGeometryModel":
        with np.load(Path(path), allow_pickle=False) as data:
            model_type = str(data["model_type"])
            if model_type not in {
                "rgbd_geometry_v1", "rgbd_geometry_v2_face_topology",
                "rgbd_geometry_v3_multipose_knn",
            }:
                raise ValueError("unsupported RGB-D geometry model")
            feature_names = tuple(data["feature_names"].tolist())
            expected = BASE_FEATURE_NAMES if model_type == "rgbd_geometry_v1" else FEATURE_NAMES
            if feature_names != expected:
                raise ValueError("RGB-D feature version mismatch")
            exemplars = data["exemplars"] if model_type == "rgbd_geometry_v3_multipose_knn" else None
            exemplar_labels = (
                data["exemplar_label_indices"]
                if model_type == "rgbd_geometry_v3_multipose_knn" else None
            )
            neighbors = int(data["neighbors"]) if "neighbors" in data.files else 3
            return cls(
                data["labels"].tolist(), data["mean"], data["scale"],
                data["centroids"], data["thresholds"], float(data["min_margin"]),
                feature_names, exemplars, exemplar_labels, neighbors,
            )


def detect_tray_roi_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Locate the cool white tray used by the fixed overhead capture setup."""
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("tray ROI detection requires a BGR image")
    image_area = image.shape[0] * image.shape[1]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value_floor = max(90.0, float(np.percentile(hsv[:, :, 2], 60)))
    neutral_white = (
        (hsv[:, :, 1] <= 35) & (hsv[:, :, 2] >= value_floor)
    ).astype(np.uint8) * 255
    neutral_corners = (
        neutral_white[0, 0], neutral_white[0, -1],
        neutral_white[-1, 0], neutral_white[-1, -1],
    )
    if (
        cv2.countNonZero(neutral_white) >= image_area * 0.75
        and all(neutral_corners)
    ):
        return np.full(image.shape[:2], 255, np.uint8)
    cool_white = (
        (hsv[:, :, 0] >= 70)
        & (hsv[:, :, 0] <= 135)
        & (hsv[:, :, 1] <= 110)
    )
    candidate = (
        cool_white & (hsv[:, :, 2] >= value_floor)
    ).astype(np.uint8) * 255
    scale = max(1.0, min(image.shape[:2]) / 720.0)
    close_size = max(9, int(round(31 * scale)) | 1)
    open_size = max(3, int(round(7 * scale)) | 1)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        np.ones((close_size, close_size), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        np.ones((open_size, open_size), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    candidate_corners = (
        candidate[0, 0], candidate[0, -1], candidate[-1, 0], candidate[-1, -1]
    )
    if cv2.countNonZero(candidate) >= image_area * 0.75 and all(candidate_corners):
        return np.full(image.shape[:2], 255, np.uint8)
    border_margin = max(2, int(round(min(image.shape[:2]) * 0.01)))
    def component_choices(mask: np.ndarray) -> tuple[
        list[tuple[float, int]], np.ndarray
    ]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        choices: list[tuple[float, int]] = []
        for label in range(1, count):
            x, y, width, height, area = stats[label]
            ratio = float(area) / image_area
            rectangularity = float(area) / max(width * height, 1)
            touches_image_border = (
                x <= border_margin
                or y <= border_margin
                or x + width >= image.shape[1] - border_margin
                or y + height >= image.shape[0] - border_margin
            )
            if (
                not touches_image_border
                and 0.05 <= ratio <= 0.65
                and rectangularity >= 0.45
            ):
                choices.append((float(area) * rectangularity, label))
        return choices, labels

    choices, labels = component_choices(candidate)
    if not choices:
        neutral_candidate = cv2.morphologyEx(
            neutral_white,
            cv2.MORPH_CLOSE,
            np.ones((close_size, close_size), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        neutral_candidate = cv2.morphologyEx(
            neutral_candidate,
            cv2.MORPH_OPEN,
            np.ones((open_size, open_size), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        choices, labels = component_choices(neutral_candidate)
    if not choices:
        raise ValueError("tray_roi_not_found")
    selected = max(choices)[1]
    component = (labels == selected).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    rectangle = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32)
    roi = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(roi, rectangle, 255)
    x, y, width, height = cv2.boundingRect(rectangle)
    # Stop at the tray's inner wall. An edge object still contributes its portion
    # on the tray floor, while the raised white rim itself remains outside.
    inset = max(5, int(round(min(width, height) * 0.045)))
    roi = cv2.erode(
        roi,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inset * 2 + 1, inset * 2 + 1)),
    )
    if cv2.countNonZero(roi) < image_area * 0.03:
        raise ValueError("tray_roi_too_small")
    return roi


def detect_rgb_object_support(image_bgr: np.ndarray, tray_roi: np.ndarray) -> np.ndarray:
    """Find coloured/dark object regions against the nearly white tray surface."""
    image = np.asarray(image_bgr)
    roi = np.asarray(tray_roi) > 0
    if image.ndim != 3 or image.shape[:2] != roi.shape:
        raise ValueError("RGB object support requires matching image and tray ROI")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tray_reference_pixels = roi & (hsv[:, :, 1] <= 70) & (hsv[:, :, 2] >= 80)
    if int(tray_reference_pixels.sum()) < 500:
        tray_reference_pixels = roi
    reference = np.median(lab[tray_reference_pixels], axis=0)
    colour_distance = np.linalg.norm(lab - reference, axis=2)
    candidate = (
        roi
        & (hsv[:, :, 2] >= 20)
        & ((colour_distance >= 20.0) | (hsv[:, :, 1] >= 65))
    ).astype(np.uint8) * 255
    scale = max(1.0, min(image.shape[:2]) / 720.0)
    open_size = max(3, int(round(5 * scale)) | 1)
    close_size = max(5, int(round(11 * scale)) | 1)
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8)
    )
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8), iterations=2
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    support = np.zeros_like(candidate)
    min_area = max(120, int(round(image.shape[0] * image.shape[1] * 0.0002)))
    min_thickness = max(open_size * 2, int(round(min(image.shape[:2]) * 0.025)))
    for label in range(1, count):
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        short_side = max(1, min(width, height))
        elongated_rim = max(width, height) / short_side > 6.0 and short_side < min_thickness
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area and not elongated_rim:
            support[labels == label] = 255
    return cv2.dilate(
        support,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
        iterations=1,
    )


def _fit_background_plane(
    frame: RGBDFrame, cfg: VisionConfig, roi_mask: np.ndarray | None = None
) -> Plane:
    mask = valid_depth_mask(frame.depth_mm, cfg.rgbd)
    if roi_mask is not None:
        mask &= np.asarray(roi_mask) > 0
    mask = mask.astype(np.uint8) * 255
    points, _ = depth_to_points(frame.depth_mm, frame.intrinsics, mask, stride=8)
    return fit_plane_ransac(points, cfg.rgbd.plane_ransac_threshold_mm)


def train_rgbd_geometry_model(
    data_root: str | Path,
    output: str | Path,
    config: VisionConfig | None = None,
    batch_ids: set[str] | None = None,
    base_model_path: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or load_config()
    entries = load_rgbd_dataset_entries(data_root)
    selected_batches = None if batch_ids is None else {str(value) for value in batch_ids}
    if selected_batches is not None:
        entries = [entry for entry in entries if str(entry["batch_id"]) in selected_batches]
    backgrounds: dict[str, tuple[RGBDFrame, np.ndarray, Plane]] = {}
    processing_scale = float(cfg.rgbd.processing_scale)
    for entry in entries:
        if entry["label_id"] == EMPTY_TRAY_LABEL:
            frame = resize_rgbd_frame(
                load_rgbd_frame(entry["absolute_sample_dir"]), processing_scale
            )
            roi = detect_tray_roi_mask(frame.color_bgr)
            backgrounds[str(entry["batch_id"])] = (
                frame,
                roi,
                _fit_background_plane(frame, cfg, roi),
            )
    features: list[np.ndarray] = []
    labels: list[str] = []
    rejected: list[dict[str, str]] = []
    counts: defaultdict[str, int] = defaultdict(int)
    samples_with_extra_components = 0
    for entry in entries:
        label = str(entry["label_id"])
        if label == EMPTY_TRAY_LABEL:
            continue
        batch = str(entry["batch_id"])
        if batch not in backgrounds:
            rejected.append({"sample_dir": entry["sample_dir"], "reason": "missing_batch_empty_tray"})
            continue
        try:
            frame = resize_rgbd_frame(
                load_rgbd_frame(entry["absolute_sample_dir"]), processing_scale
            )
            background, _tray_roi, _reference_plane = backgrounds[batch]
            if frame.intrinsics != background.intrinsics:
                raise ValueError("intrinsics_mismatch_with_empty_tray")
            frame_tray_roi = detect_tray_roi_mask(frame.color_bgr)
            plane = _fit_background_plane(frame, cfg, frame_tray_roi)
            support = detect_rgb_object_support(frame.color_bgr, frame_tray_roi)
            objects, _ = segment_depth_objects(
                frame.color_bgr,
                frame.depth_mm,
                frame.intrinsics,
                plane,
                cfg.rgbd,
                roi_mask=frame_tray_roi,
                support_mask=support,
                split_touching_objects=False,
            )
            if not objects:
                raise ValueError(f"expected_one_object_got_{len(objects)}")
            if len(objects) > 1:
                samples_with_extra_components += 1
            # Capture protocol guarantees one labelled object inside the tray.
            # Keep the dominant RGB/depth component and ignore small depth speckles.
            item = max(objects, key=lambda candidate: candidate.area)
            points, _ = object_point_cloud(item, frame.depth_mm, frame.intrinsics, stride=2)
            x, y, width, height = item.bbox
            feature = extract_rgbd_geometry_features(
                points,
                frame.depth_mm[y:y + height, x:x + width],
                item.mask[y:y + height, x:x + width],
                frame.intrinsics,
                (x, y),
                frame.color_bgr[y:y + height, x:x + width],
            )
            features.append(feature)
            labels.append(label)
            counts[label] += 1
        except (ValueError, OSError, FileNotFoundError) as error:
            rejected.append({"sample_dir": entry["sample_dir"], "reason": str(error)})
    added_samples = len(features)
    base_samples = 0
    if base_model_path is not None:
        base_features, base_labels = DepthGeometryModel.load(base_model_path).training_data()
        base_samples = len(base_labels)
        features = [*base_features, *features]
        labels = [*base_labels, *labels]
    if len(set(labels)) < 2:
        raise ValueError("training needs at least two valid object classes")
    model = DepthGeometryModel.fit(np.vstack(features), labels)
    report = {
        "model_type": "rgbd_geometry_v3_multipose_knn",
        "data_root": str(data_root),
        "accepted_samples": len(features),
        "added_samples": added_samples,
        "base_samples": base_samples,
        "base_model": str(base_model_path) if base_model_path is not None else None,
        "class_counts": dict(sorted(counts.items())),
        "empty_tray_batches": sorted(backgrounds),
        "selected_batches": sorted(selected_batches) if selected_batches is not None else "all",
        "rejected": rejected,
        "samples_with_extra_components": samples_with_extra_components,
        "training_object_selection": "largest_rgb_depth_component",
        "processing_scale": processing_scale,
        "training_replay_only": True,
    }
    model.save(output, report)
    return report
