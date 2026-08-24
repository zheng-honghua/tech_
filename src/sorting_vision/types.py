from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class DetectionStatus(str, Enum):
    PICKABLE = "PICKABLE"
    UNCERTAIN = "UNCERTAIN"
    OCCLUDED = "OCCLUDED"
    DEPTH_INVALID = "DEPTH_INVALID"
    NO_GRASP_SURFACE = "NO_GRASP_SURFACE"
    DEPTH_REQUIRED = "DEPTH_REQUIRED"


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"x": round(self.x, 3), "y": round(self.y, 3)}


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float

    def to_dict(self) -> dict[str, float]:
        return {"x": round(self.x, 3), "y": round(self.y, 3), "z": round(self.z, 3)}

    @classmethod
    def from_array(cls, value: np.ndarray) -> "Point3D":
        return cls(float(value[0]), float(value[1]), float(value[2]))


@dataclass(frozen=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float

    def to_dict(self) -> dict[str, float]:
        return {
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "z": round(self.z, 6),
            "w": round(self.w, 6),
        }


@dataclass(frozen=True)
class Pose3D:
    position_mm: Point3D
    quaternion_xyzw: Quaternion
    surface_normal: Point3D
    approach_vector: Point3D

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_mm": self.position_mm.to_dict(),
            "quaternion_xyzw": self.quaternion_xyzw.to_dict(),
            "surface_normal": self.surface_normal.to_dict(),
            "approach_vector": self.approach_vector.to_dict(),
        }


@dataclass(frozen=True)
class GraspInfo:
    cup_diameter_mm: float
    flatness_rmse_mm: float
    edge_clearance_mm: float
    valid_depth_ratio: float
    score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "cup_diameter_mm": round(self.cup_diameter_mm, 3),
            "flatness_rmse_mm": round(self.flatness_rmse_mm, 4),
            "edge_clearance_mm": round(self.edge_clearance_mm, 3),
            "valid_depth_ratio": round(self.valid_depth_ratio, 4),
            "score": round(self.score, 4),
        }


@dataclass(frozen=True)
class Confidence3D:
    segmentation: float
    color: float
    shape: float
    pose: float
    grasp: float

    @property
    def combined(self) -> float:
        values = np.clip(
            [self.segmentation, self.color, self.shape, self.pose, self.grasp],
            1e-6,
            1.0,
        )
        return float(np.prod(values) ** (1.0 / len(values)))

    def to_dict(self) -> dict[str, float]:
        return {
            "segmentation": round(self.segmentation, 4),
            "color": round(self.color, 4),
            "shape": round(self.shape, 4),
            "pose": round(self.pose, 4),
            "grasp": round(self.grasp, 4),
            "combined": round(self.combined, 4),
        }


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
class VisionResult3D:
    frame_id: str
    object_id: str
    color_id: str
    color_name: str
    shape_id: str
    shape_name: str
    class_key: str
    pose_3d: Pose3D | None
    grasp: GraspInfo | None
    confidence: Confidence3D
    status: DetectionStatus
    bbox_px: tuple[int, int, int, int]
    center_mm: Point2D | None = None
    angle_deg: float | None = None
    selected: bool = False
    crop_image: np.ndarray | None = field(default=None, repr=False)
    depth_crop: np.ndarray | None = field(default=None, repr=False)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    schema_version: int = field(default=2, init=False)

    @property
    def display_name(self) -> str:
        return f"{self.color_name}{self.shape_name}"

    def to_dict(
        self,
        crop_path: str | None = None,
        depth_crop_path: str | None = None,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "frame_id": self.frame_id,
            "object_id": self.object_id,
            "color_id": self.color_id,
            "color_name": self.color_name,
            "shape_id": self.shape_id,
            "shape_name": self.shape_name,
            "class_key": self.class_key,
            "display_name": self.display_name,
            "pose_3d": None if self.pose_3d is None else self.pose_3d.to_dict(),
            "grasp": None if self.grasp is None else self.grasp.to_dict(),
            "confidence": self.confidence.to_dict(),
            "status": self.status.value,
            "bbox_px": list(self.bbox_px),
            "center_mm": None if self.center_mm is None else self.center_mm.to_dict(),
            "angle_deg": None if self.angle_deg is None else round(self.angle_deg, 3),
            "selected": self.selected,
            "diagnostics": self.diagnostics,
        }
        if crop_path is not None:
            value["crop_image"] = crop_path
        if depth_crop_path is not None:
            value["depth_crop"] = depth_crop_path
        return value


@dataclass
class SegmentedObject:
    mask: np.ndarray
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    area: float
    segmentation_confidence: float
    touches_border: bool = False
    clearance_px: float = float("inf")
