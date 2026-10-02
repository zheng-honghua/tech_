"""Camera-specific colour statistics and bounded HSV/Lab seed recovery."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .classification import LabelPrediction
from .visual_contract import V7


def canonical_color(label: str) -> str:
    return "blue" if label == "bule" else label


@dataclass
class ColorFrameContext:
    original: np.ndarray
    corrected: np.ndarray
    hsv: np.ndarray
    lab: np.ndarray
    correction: dict[str, Any]
    cache: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, image: np.ndarray, correction: dict[str, Any]):
        gains = np.asarray(correction["gains_bgr"], np.float32)
        if gains.shape != (3,) or not np.isfinite(gains).all() or np.any(gains <= 0):
            raise ValueError("invalid fixed white balance gains")
        corrected = np.clip(image.astype(np.float32) * gains, 0, 255).astype(np.uint8)
        return cls(image, corrected, cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV),
                   cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB), dict(correction))


def neutral_correction(background: np.ndarray | None, roi: np.ndarray | None = None) -> dict[str, Any]:
    fallback = {"gains_bgr": [1., 1., 1.], "reason": "neutral_reference_missing", "pixels": 0}
    if background is None:
        return fallback
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    valid = (hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 60) & (hsv[:, :, 2] < 245)
    if roi is not None:
        valid &= cv2.erode((roi > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    count = int(valid.sum())
    if count < 100:
        return {**fallback, "reason": "insufficient_neutral_reference", "pixels": count}
    median = np.median(background[valid], axis=0)
    gains = median.mean() / np.maximum(median, 1.)
    if np.any(gains < .75) or np.any(gains > 1.33):
        return {**fallback, "reason": "neutral_gains_out_of_range", "pixels": count}
    return {"gains_bgr": gains.tolist(), "reason": "batch_empty_tray", "pixels": count}


def _hue_distance(values: np.ndarray, center: float) -> np.ndarray:
    delta = np.abs(values.astype(np.float32) - center)
    return np.minimum(delta, 180. - delta)


def recover_components(context: ColorFrameContext, min_saturation: int, min_value: int,
                       min_area: int, split_hue_gap: int = 18,
                       geometric_support: np.ndarray | None = None) -> list[np.ndarray]:
    """Grow chromatic seeds locally; no convex hull or unsupported gap filling."""
    from .config import RGBDConfig
    from .geometry3d import _hsv_component_masks

    hsv, lab = context.hsv, context.lab
    strong = ((hsv[:, :, 1] >= min_saturation) & (hsv[:, :, 2] >= min_value)).astype(np.uint8) * 255
    # The old hue splitter is used only for seeds. Geodesic growth retains
    # all lit/shaded pixels consistent with that seed's circular distribution.
    cfg = RGBDConfig(min_area_px=min_area, hsv_split_hue_gap=split_hue_gap)
    seeds = list(_hsv_component_masks(strong, hsv, cfg, minimum_pixels=max(20, min_area // 5)))
    output = []
    for seed in seeds:
        ys, xs = np.nonzero(seed)
        if len(xs) < 20:
            continue
        padding = min(160, max(12, int(.75 * np.sqrt(len(xs)))))
        x0, x1 = max(0, int(xs.min()) - padding), min(seed.shape[1], int(xs.max()) + padding + 1)
        y0, y1 = max(0, int(ys.min()) - padding), min(seed.shape[0], int(ys.max()) + padding + 1)
        sample = hsv[seed > 0]
        angles = sample[:, 0].astype(float) * np.pi / 90
        center = float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()) * 90 / np.pi % 180)
        tolerance = float(np.clip(np.percentile(_hue_distance(sample[:, 0], center), 90) + 8, 12, 25))
        local = hsv[y0:y1, x0:x1]
        chroma = lab[y0:y1, x0:x1, 1:].astype(float) - 128
        reference = np.median(lab[seed > 0, 1:].astype(float) - 128, axis=0)
        cosine = (chroma @ reference) / np.maximum(np.linalg.norm(chroma, axis=2) * np.linalg.norm(reference), 1.)
        weak = ((local[:, :, 1] >= max(20, .35 * np.percentile(sample[:, 1], 15)))
                & (local[:, :, 2] >= 8) & (_hue_distance(local[:, :, 0], center) <= tolerance)
                & (cosine >= .75))
        local_seed = seed[y0:y1, x0:x1] > 0
        # High saturation seeds from a different object are not growth territory.
        weak &= ~((strong[y0:y1, x0:x1] > 0) & ~local_seed
                  & (_hue_distance(local[:, :, 0], center) > tolerance))
        if geometric_support is not None:
            # Neutral highlights require independently observed elevated depth.
            highlight = ((local[:, :, 1] < min_saturation) & (local[:, :, 2] >= 180)
                         & (geometric_support[y0:y1, x0:x1] > 0))
            weak |= highlight
        weak |= local_seed
        count, labels = cv2.connectedComponents(weak.astype(np.uint8))
        owned = np.unique(labels[local_seed])
        grown = np.isin(labels, owned[owned > 0])
        full = np.zeros_like(seed)
        full[y0:y1, x0:x1] = grown.astype(np.uint8) * 255
        if cv2.countNonZero(full) >= min_area:
            output.append(full)
    # Multiple seeds may converge on the same shaded face: unite overlapping
    # masks, while keeping white gaps open. Depth later resolves ownership.
    result: list[np.ndarray] = []
    for mask in output:
        # Hue partitions describe surfaces, not independent physical objects.
        # Touching colour regions remain one component until depth geometry
        # supplies an independent instance boundary.
        nearby = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        overlapping = [i for i, old in enumerate(result) if np.any((old > 0) & (nearby > 0))]
        for i in reversed(overlapping):
            mask = cv2.bitwise_or(mask, result.pop(i))
        result.append(mask)
    return result


def color_vector(context: ColorFrameContext, mask: np.ndarray) -> np.ndarray:
    inside = cv2.erode((mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    valid = inside & (context.hsv[:, :, 1] >= 45) & (context.hsv[:, :, 2] >= 18) & (context.hsv[:, :, 2] < 250)
    if valid.sum() < 20:
        raise ValueError("insufficient_chromatic_interior")
    hue = context.hsv[:, :, 0][valid].astype(float) * np.pi / 90
    if np.hypot(np.cos(hue).mean(), np.sin(hue).mean()) < .75:
        raise ValueError("ambiguous_multiple_hues")
    lab = context.lab[valid].astype(float)
    return np.array([np.cos(hue).mean(), np.sin(hue).mean(),
                     np.median(lab[:, 1] - 128) / 128, np.median(lab[:, 2] - 128) / 128,
                     np.median(lab[:, 0]) / 255 * .15], np.float64)


class CameraColorProfile:
    def __init__(self, camera: str, prototypes: dict[str, list[list[float]]],
                 distance_threshold: float = .65, margin_threshold: float = .12,
                 provenance: dict[str, Any] | None = None):
        if camera not in {"primary", "side"}:
            raise ValueError("unknown colour camera")
        self.camera, self.prototypes = camera, prototypes
        self.distance_threshold, self.margin_threshold = distance_threshold, margin_threshold
        self.provenance = provenance or {}
        if not prototypes or not np.isfinite([distance_threshold, margin_threshold]).all() or distance_threshold <= 0 or margin_threshold < 0:
            raise ValueError("invalid colour profile thresholds")
        for values in prototypes.values():
            array = np.asarray(values)
            if array.ndim != 2 or array.shape[1] != 5 or not np.isfinite(array).all():
                raise ValueError("invalid colour prototypes")

    @classmethod
    def fit(cls, camera: str, samples: list[tuple[str, np.ndarray]], provenance: dict[str, Any]):
        prototypes = {}
        for label in sorted({canonical_color(label) for label, _ in samples}):
            values = np.stack([v for key, v in samples if canonical_color(key) == label])
            if len(values) < 6:
                raise ValueError("at least six reviewed samples per colour required")
            # Three brightness quantiles model different visible-face mixtures.
            order = np.argsort(values[:, -1])
            prototypes[label] = [np.median(values[chunk], axis=0).tolist()
                                 for chunk in np.array_split(order, min(3, len(order))) if len(chunk)]
        return cls(camera, prototypes, provenance=provenance)

    def predict_vector(self, vector: np.ndarray) -> LabelPrediction:
        if np.asarray(vector).shape != (5,) or not np.isfinite(vector).all():
            raise ValueError("invalid colour vector")
        distances = {key: float(np.linalg.norm(np.asarray(values) - vector, axis=1).min())
                     for key, values in self.prototypes.items()}
        order = sorted(distances, key=distances.get)
        label, distance = order[0], distances[order[0]]
        margin = distances[order[1]] - distance if len(order) > 1 else 1.
        confidence = float(np.clip(.8 + .19 * (1 - distance / self.distance_threshold), 0, .99))
        accepted = distance <= self.distance_threshold and margin >= self.margin_threshold
        scores = {key: float(np.exp(-value / .15)) for key, value in distances.items()}
        total = sum(scores.values())
        return LabelPrediction(label if accepted else "unknown", label if accepted else "未知颜色",
            confidence if accepted else min(.5, confidence),
            {"camera": self.camera, "distance": distance, "margin": margin, "nearest_label": label,
             "profile_hash": self.content_hash, "input_contract": V7},
            {key: value / max(total, 1e-12) for key, value in scores.items()},
            "accepted" if accepted else "distance_rejected" if distance > self.distance_threshold else "margin_rejected")

    def classify(self, context: ColorFrameContext, mask: np.ndarray) -> LabelPrediction:
        try:
            return self.predict_vector(color_vector(context, mask))
        except ValueError as error:
            return LabelPrediction("unknown", "未知颜色", 0., {"reason": str(error)}, {}, str(error))

    def to_dict(self):
        return {"version": 7, "contract": V7, "camera": self.camera, "prototypes": self.prototypes,
                "distance_threshold": self.distance_threshold, "margin_threshold": self.margin_threshold,
                "provenance": self.provenance}

    @property
    def content_hash(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps({**self.to_dict(), "hash": self.content_hash}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, camera: str):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("contract") != V7 or value.get("camera") != camera:
            raise ValueError("colour profile camera/contract mismatch")
        profile = cls(camera, value["prototypes"], value["distance_threshold"], value["margin_threshold"], value["provenance"])
        if profile.content_hash != value.get("hash"):
            raise ValueError("colour profile hash mismatch")
        return profile
