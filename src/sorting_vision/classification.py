from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .config import ClassificationConfig


@dataclass(frozen=True)
class LabelPrediction:
    label_id: str
    label_name: str
    confidence: float
    features: dict[str, float]


def _hex_to_lab(value: str) -> np.ndarray:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"invalid colour hex value: {value!r}")
    rgb = np.array([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)
    bgr = rgb[::-1].reshape(1, 1, 3)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


class LabColorClassifier:
    def __init__(self, cfg: ClassificationConfig):
        self.cfg = cfg
        self.prototypes = {
            color_id: _hex_to_lab(details["hex"])
            for color_id, details in cfg.colors.items()
        }

    def classify(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        lab_image: np.ndarray | None = None,
    ) -> LabelPrediction:
        kernel = np.ones((7, 7), np.uint8)
        interior = cv2.erode(mask, kernel, iterations=2)
        if cv2.countNonZero(interior) < 25:
            interior = mask
        lab = (
            cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
            if lab_image is None
            else lab_image
        )
        pixels = lab[interior > 0]
        if len(pixels) == 0:
            return LabelPrediction("unknown", "未知颜色", 0.0, {})

        # Trim highlights and deep shadows using the lightness channel.
        lower, upper = np.percentile(pixels[:, 0], [10, 90])
        trimmed = pixels[(pixels[:, 0] >= lower) & (pixels[:, 0] <= upper)]
        sample = np.median(trimmed if len(trimmed) else pixels, axis=0)
        distances = {
            key: float(np.linalg.norm(sample - prototype))
            for key, prototype in self.prototypes.items()
        }
        color_id, distance = min(distances.items(), key=lambda pair: pair[1])
        ordered = sorted(distances.values())
        separation = ordered[1] - ordered[0] if len(ordered) > 1 else self.cfg.max_color_distance
        absolute = max(0.0, 1.0 - distance / self.cfg.max_color_distance)
        relative = float(np.clip(separation / 25.0, 0.0, 1.0))
        confidence = float(np.clip(0.75 * absolute + 0.25 * relative, 0.0, 1.0))
        if distance > self.cfg.max_color_distance:
            return LabelPrediction(
                "unknown", "未知颜色", confidence, {"distance": distance}
            )
        return LabelPrediction(
            color_id,
            self.cfg.colors[color_id]["name"],
            confidence,
            {"distance": distance, "separation": separation},
        )


class GeometricShapeClassifier:
    def __init__(self, cfg: ClassificationConfig):
        self.cfg = cfg

    def classify(self, mask: np.ndarray) -> LabelPrediction:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._prediction("unknown", 0.0, {})
        contour = max(contours, key=cv2.contourArea)
        area = max(float(cv2.contourArea(contour)), 1.0)
        perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
        # A slightly stronger simplification removes tiny anti-aliased corners
        # introduced by perspective rectification without collapsing real sides.
        approximation = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        vertices = len(approximation)
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        _, (rect_width, rect_height), _ = cv2.minAreaRect(contour)
        aspect_ratio = min(rect_width, rect_height) / max(rect_width, rect_height, 1.0)
        hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
        solidity = min(1.0, area / hull_area)
        features = {
            "vertices": float(vertices),
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
        }

        if circularity >= 0.82 and vertices >= 7:
            return self._prediction("circle", min(1.0, circularity), features)
        if vertices == 3:
            return self._prediction("triangle", 0.9 * solidity, features)
        if vertices == 4:
            if aspect_ratio >= 0.84:
                confidence = 0.78 + 0.22 * min(1.0, (aspect_ratio - 0.84) / 0.16)
                return self._prediction("square", confidence * solidity, features)
            confidence = 0.8 + 0.2 * min(1.0, (0.84 - aspect_ratio) / 0.35)
            return self._prediction("rectangle", confidence * solidity, features)
        if vertices == 5:
            return self._prediction("pentagon", 0.88 * solidity, features)
        if vertices == 6:
            return self._prediction("hexagon", 0.88 * solidity, features)
        return self._prediction("unknown", 0.35 * solidity, features)

    def _prediction(
        self, label_id: str, confidence: float, features: dict[str, float]
    ) -> LabelPrediction:
        return LabelPrediction(
            label_id,
            self.cfg.shapes.get(label_id, label_id),
            float(np.clip(confidence, 0.0, 1.0)),
            features,
        )


class HybridShapeClassifier:
    """Fuses an optional trained model with deterministic contour features.

    The model callable receives a BGR crop and binary crop mask and returns
    ``(label_id, confidence)``. With no model, the geometry result is used.
    """

    def __init__(
        self,
        cfg: ClassificationConfig,
        model: Callable[[np.ndarray, np.ndarray], tuple[str, float]] | None = None,
        model_weight: float = 0.7,
    ):
        self.cfg = cfg
        self.geometry = GeometricShapeClassifier(cfg)
        self.model = model
        self.model_weight = model_weight

    def classify(
        self, crop: np.ndarray, crop_mask: np.ndarray
    ) -> LabelPrediction:
        geometric = self.geometry.classify(crop_mask)
        if self.model is None:
            return geometric
        model_label, model_confidence = self.model(crop, crop_mask)
        model_confidence = float(np.clip(model_confidence, 0.0, 1.0))
        if model_label == geometric.label_id:
            confidence = self.model_weight * model_confidence + (1 - self.model_weight) * geometric.confidence
        else:
            confidence = self.model_weight * model_confidence
            if confidence < self.cfg.min_shape_confidence:
                return LabelPrediction("unknown", self.cfg.shapes.get("unknown", "未知形状"), confidence, geometric.features)
        return LabelPrediction(
            model_label,
            self.cfg.shapes.get(model_label, model_label),
            confidence,
            geometric.features,
        )
