from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class GeometryCandidate:
    label_id: str
    confidence: float


@dataclass(frozen=True)
class GeometryPrediction:
    label_id: str
    label_name: str
    confidence: float
    accepted: bool
    backend: str
    reason: str
    top_candidates: tuple[GeometryCandidate, ...] = ()
    inference_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class GeometryShapeModel(Protocol):
    backend: str

    def predict_geometry(
        self, image_bgr: np.ndarray, mask: np.ndarray | None = None
    ) -> GeometryPrediction:
        ...

    def __call__(
        self, crop: np.ndarray, crop_mask: np.ndarray
    ) -> tuple[str, float]:
        ...


class EnsembleGeometryModel:
    """Reserved composition point for a future explicitly enabled fusion policy."""

    backend = "ensemble"

    def __init__(self, primary: GeometryShapeModel, verifier: GeometryShapeModel) -> None:
        self.primary = primary
        self.verifier = verifier

    def predict_geometry(
        self, image_bgr: np.ndarray, mask: np.ndarray | None = None
    ) -> GeometryPrediction:
        raise RuntimeError(
            "geometry ensemble policy is intentionally disabled; select one backend"
        )

    def __call__(self, crop: np.ndarray, crop_mask: np.ndarray) -> tuple[str, float]:
        result = self.predict_geometry(crop, crop_mask)
        return result.label_id, result.confidence
