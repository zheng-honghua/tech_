from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class DetectionStatus(str, Enum):
    PICKABLE = "PICKABLE"
    UNCERTAIN = "UNCERTAIN"
    OCCLUDED = "OCCLUDED"


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"x": round(self.x, 3), "y": round(self.y, 3)}


@dataclass(frozen=True)
class Confidence:
    segmentation: float
    color: float
    shape: float
    pose: float

    @property
    def combined(self) -> float:
        values = np.clip(
            [self.segmentation, self.color, self.shape, self.pose], 1e-6, 1.0
        )
        return float(np.prod(values) ** (1.0 / len(values)))

    def to_dict(self) -> dict[str, float]:
        return {
            "segmentation": round(self.segmentation, 4),
            "color": round(self.color, 4),
            "shape": round(self.shape, 4),
            "pose": round(self.pose, 4),
            "combined": round(self.combined, 4),
        }


@dataclass
class VisionResult:
    frame_id: str
    object_id: str
    color_id: str
    color_name: str
    shape_id: str
    shape_name: str
    class_key: str
    center_mm: Point2D
    angle_deg: float | None
    confidence: Confidence
    status: DetectionStatus
    bbox_px: tuple[int, int, int, int]
    clearance_px: float = float("inf")
    selected: bool = False
    crop_image: np.ndarray | None = field(default=None, repr=False)
    extensions: dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return f"{self.color_name}{self.shape_name}"

    def to_dict(self, crop_path: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "frame_id": self.frame_id,
            "object_id": self.object_id,
            "color_id": self.color_id,
            "color_name": self.color_name,
            "shape_id": self.shape_id,
            "shape_name": self.shape_name,
            "class_key": self.class_key,
            "display_name": self.display_name,
            "center_mm": self.center_mm.to_dict(),
            "angle_deg": None if self.angle_deg is None else round(self.angle_deg, 3),
            "confidence": self.confidence.to_dict(),
            "status": self.status.value,
            "bbox_px": list(self.bbox_px),
            "selected": self.selected,
            "extensions": self.extensions,
        }
        if crop_path is not None:
            result["crop_image"] = crop_path
        return result


@dataclass
class SegmentedObject:
    mask: np.ndarray
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    area: float
    segmentation_confidence: float
    touches_border: bool = False
    clearance_px: float = float("inf")
