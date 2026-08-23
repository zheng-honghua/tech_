from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from .classification import LabelPrediction
from .config import ClassificationConfig
from .rgbd import fit_plane_svd


class ShapeModel3D(Protocol):
    """Interface for a learned RGB-D or point-cloud classifier."""

    def classify(
        self,
        points_camera_mm: np.ndarray,
        color_crop_bgr: np.ndarray,
        depth_crop_mm: np.ndarray,
        crop_mask: np.ndarray,
    ) -> tuple[str, float]: ...


def _surface_features(points: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    contour_values, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contour_values, key=cv2.contourArea)
    area = max(float(cv2.contourArea(contour)), 1.0)
    perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
    approximation = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
    circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
    _, (rect_width, rect_height), _ = cv2.minAreaRect(contour)
    silhouette_ratio = min(rect_width, rect_height) / max(rect_width, rect_height, 1.0)

    centered = points - np.mean(points, axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    eigenvalues = np.sort(eigenvalues)[::-1]
    total = max(float(eigenvalues.sum()), 1e-9)
    plane = fit_plane_svd(points)
    extents = np.ptp(points, axis=0)
    return {
        "visible_vertices": float(len(approximation)),
        "circularity": circularity,
        "silhouette_ratio": float(silhouette_ratio),
        "surface_planarity": float(1.0 - eigenvalues[-1] / total),
        "surface_curvature": float(eigenvalues[-1] / total),
        "surface_rmse_mm": plane.rmse_mm,
        "extent_x_mm": float(extents[0]),
        "extent_y_mm": float(extents[1]),
        "extent_z_mm": float(extents[2]),
        "point_count": float(len(points)),
    }


class PrimitiveShapeClassifier3D:
    """Geometry baseline for common solids using valid depth plus silhouette support.

    It deliberately rejects inputs without a point cloud. A learned model should
    replace this baseline once real competition samples are available.
    """

    def __init__(self, cfg: ClassificationConfig):
        self.cfg = cfg

    def classify(self, points_camera_mm: np.ndarray, mask: np.ndarray) -> LabelPrediction:
        points = np.asarray(points_camera_mm, dtype=np.float64)
        if len(points) < 40:
            return self._prediction("unknown", 0.0, {"point_count": float(len(points))})
        features = _surface_features(points, mask)
        vertices = int(round(features["visible_vertices"]))
        circularity = features["circularity"]
        rmse = features["surface_rmse_mm"]
        curvature = features["surface_curvature"]
        ratio = features["silhouette_ratio"]

        if circularity >= 0.78 and vertices >= 7:
            curved = rmse >= 1.0 or curvature >= 0.0015
            label = "sphere" if curved else "cylinder"
            confidence = 0.82 + 0.16 * min(1.0, abs(rmse - 1.0) / 2.0)
            return self._prediction(label, confidence, features)
        if vertices == 3:
            return self._prediction("triangular_prism", 0.86, features)
        if vertices == 4:
            # Downsampling can leave one tiny extra corner on a triangular face.
            # Its low circularity and near-square envelope distinguish it from
            # a long cuboid without treating a 2-D contour as sufficient input.
            if circularity < 0.68 and ratio > 0.75:
                return self._prediction("triangular_prism", 0.78, features)
            label = "cube" if ratio >= 0.82 else "cuboid"
            confidence = 0.82 + 0.15 * min(1.0, abs(ratio - 0.82) / 0.3)
            return self._prediction(label, confidence, features)
        if vertices == 5:
            return self._prediction("pentagonal_prism", 0.84, features)
        if vertices == 6:
            return self._prediction("hexagonal_prism", 0.84, features)
        return self._prediction("unknown", 0.35, features)

    def _prediction(
        self, label_id: str, confidence: float, features: dict[str, float]
    ) -> LabelPrediction:
        return LabelPrediction(
            label_id,
            self.cfg.shapes.get(label_id, label_id),
            float(np.clip(confidence, 0.0, 1.0)),
            features,
        )


class HybridShapeClassifier3D:
    def __init__(
        self,
        cfg: ClassificationConfig,
        model: ShapeModel3D | None = None,
        model_weight: float = 0.8,
    ) -> None:
        self.cfg = cfg
        self.baseline = PrimitiveShapeClassifier3D(cfg)
        self.model = model
        self.model_weight = model_weight

    def classify(
        self,
        points_camera_mm: np.ndarray,
        color_crop_bgr: np.ndarray,
        depth_crop_mm: np.ndarray,
        crop_mask: np.ndarray,
    ) -> LabelPrediction:
        baseline = self.baseline.classify(points_camera_mm, crop_mask)
        if self.model is None:
            return baseline
        label, model_confidence = self.model.classify(
            points_camera_mm, color_crop_bgr, depth_crop_mm, crop_mask
        )
        model_confidence = float(np.clip(model_confidence, 0.0, 1.0))
        agreement = label == baseline.label_id
        fused = self.model_weight * model_confidence
        if agreement:
            fused += (1.0 - self.model_weight) * baseline.confidence
        elif model_confidence < 0.8:
            return self.baseline._prediction("unknown", fused, baseline.features)
        return LabelPrediction(
            label,
            self.cfg.shapes.get(label, label),
            float(np.clip(fused, 0.0, 1.0)),
            {**baseline.features, "model_agrees_with_geometry": float(agreement)},
        )
