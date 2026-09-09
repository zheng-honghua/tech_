from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .camera import RGBFrame, SynchronizedFramePair
from .config import DualViewConfig
from .geometry_rgb import GeometryRGBModel
from .rgbd import CameraIntrinsics, Plane, RGBDFrame
from .types import Confidence3D, DetectionStatus, VisionResult3D


class FusionState(str, Enum):
    FUSED = "FUSED"
    TOP_ONLY = "TOP_ONLY"
    SIDE_MISSING = "SIDE_MISSING"
    SIDE_OCCLUDED = "SIDE_OCCLUDED"
    UNSYNCED = "UNSYNCED"
    CALIBRATION_INVALID = "CALIBRATION_INVALID"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class DualCalibrationMetrics:
    primary_rms_px: float
    side_rms_px: float
    joint_projection_p95_px: float
    scale_error_ratio: float

    def __post_init__(self) -> None:
        values = (
            self.primary_rms_px,
            self.side_rms_px,
            self.joint_projection_p95_px,
            self.scale_error_ratio,
        )
        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError("dual calibration metrics must be finite and non-negative")

    @property
    def valid(self) -> bool:
        return (
            self.primary_rms_px <= 0.8
            and self.side_rms_px <= 0.8
            and self.joint_projection_p95_px <= 3.0
            and self.scale_error_ratio <= 0.01
        )

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "primary_rms_px": self.primary_rms_px,
            "side_rms_px": self.side_rms_px,
            "joint_projection_p95_px": self.joint_projection_p95_px,
            "scale_error_ratio": self.scale_error_ratio,
            "valid": self.valid,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DualCalibrationMetrics":
        return cls(
            float(value["primary_rms_px"]),
            float(value["side_rms_px"]),
            float(value["joint_projection_p95_px"]),
            float(value["scale_error_ratio"]),
        )


@dataclass(frozen=True)
class DualViewCalibration:
    """Cross-camera geometry in millimetres.

    ``side_from_primary`` maps homogeneous primary-camera points into the side
    camera.  The RGB-D calibration remains the sole source of tray depth and
    robot coordinates.
    """

    primary_intrinsics: CameraIntrinsics
    primary_distortion: np.ndarray
    side_intrinsics: CameraIntrinsics
    side_distortion: np.ndarray
    side_from_primary: np.ndarray
    metrics: DualCalibrationMetrics
    platform_id: str
    version: str
    board: dict[str, Any]
    primary_image_size: tuple[int, int]
    tray_plane_primary: Plane | None = None
    tray_from_primary: np.ndarray | None = None

    def __post_init__(self) -> None:
        primary_distortion = np.asarray(self.primary_distortion, np.float64).reshape(-1)
        distortion = np.asarray(self.side_distortion, np.float64).reshape(-1)
        transform = np.asarray(self.side_from_primary, np.float64)
        if transform.shape != (4, 4) or not np.allclose(
            transform[3], [0, 0, 0, 1], atol=1e-6
        ):
            raise ValueError("side_from_primary must be a homogeneous 4x4 transform")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
            raise ValueError("side_from_primary rotation must be orthonormal")
        if np.linalg.det(rotation) < 0.99:
            raise ValueError("side_from_primary rotation must be right-handed")
        if not self.platform_id.strip() or not self.version.strip():
            raise ValueError("dual calibration platform and version are required")
        if tuple(self.primary_image_size) != (
            self.primary_intrinsics.width,
            self.primary_intrinsics.height,
        ):
            raise ValueError("primary calibration image size is inconsistent")
        object.__setattr__(self, "primary_distortion", primary_distortion)
        object.__setattr__(self, "side_distortion", distortion)
        object.__setattr__(self, "side_from_primary", transform)
        if self.tray_from_primary is not None:
            tray_transform = np.asarray(self.tray_from_primary, np.float64)
            if tray_transform.shape != (4, 4) or not np.allclose(
                tray_transform[3], [0, 0, 0, 1], atol=1e-6
            ):
                raise ValueError("tray_from_primary must be a homogeneous 4x4 transform")
            tray_rotation = tray_transform[:3, :3]
            if not np.allclose(tray_rotation.T @ tray_rotation, np.eye(3), atol=2e-3):
                raise ValueError("tray_from_primary rotation must be orthonormal")
            object.__setattr__(self, "tray_from_primary", tray_transform)

    @property
    def valid(self) -> bool:
        return self.metrics.valid and self.tray_from_primary is not None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": 1,
            "platform_id": self.platform_id,
            "version": self.version,
            "primary_image_size": list(self.primary_image_size),
            "primary_intrinsics": self.primary_intrinsics.to_dict(),
            "primary_distortion": self.primary_distortion.tolist(),
            "side_intrinsics": self.side_intrinsics.to_dict(),
            "side_distortion": self.side_distortion.tolist(),
            "T_side_from_primary": self.side_from_primary.tolist(),
            "metrics": self.metrics.to_dict(),
            "board": self.board,
        }
        if self.tray_plane_primary is not None:
            value["tray_plane_primary"] = self.tray_plane_primary.to_dict()
        if self.tray_from_primary is not None:
            value["T_tray_from_primary"] = self.tray_from_primary.tolist()
        value["valid"] = self.valid
        value["calibration_hash"] = self.content_hash(value)
        return value

    @staticmethod
    def content_hash(value: dict[str, Any]) -> str:
        unsigned = {key: item for key, item in value.items() if key != "calibration_hash"}
        payload = json.dumps(unsigned, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def calibration_hash(self) -> str:
        return str(self.to_dict()["calibration_hash"])

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "DualViewCalibration":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        expected_hash = str(value.get("calibration_hash", ""))
        actual_hash = cls.content_hash(value)
        if expected_hash and expected_hash != actual_hash:
            raise ValueError("dual calibration hash mismatch")
        return cls(
            primary_intrinsics=CameraIntrinsics(**value["primary_intrinsics"]),
            primary_distortion=np.asarray(value["primary_distortion"], np.float64),
            side_intrinsics=CameraIntrinsics(**value["side_intrinsics"]),
            side_distortion=np.asarray(value["side_distortion"], np.float64),
            side_from_primary=np.asarray(value["T_side_from_primary"], np.float64),
            metrics=DualCalibrationMetrics.from_dict(value["metrics"]),
            platform_id=str(value["platform_id"]),
            version=str(value["version"]),
            board=dict(value.get("board", {})),
            primary_image_size=tuple(map(int, value["primary_image_size"])),
            tray_plane_primary=(
                None
                if value.get("tray_plane_primary") is None
                else Plane.from_dict(value["tray_plane_primary"])
            ),
            tray_from_primary=(
                None
                if value.get("T_tray_from_primary") is None
                else np.asarray(value["T_tray_from_primary"], np.float64)
            ),
        )

    def transform_primary_points(self, points_primary_mm: np.ndarray) -> np.ndarray:
        points = np.asarray(points_primary_mm, np.float64).reshape(-1, 3)
        return points @ self.side_from_primary[:3, :3].T + self.side_from_primary[:3, 3]

    def project_primary_points(self, points_primary_mm: np.ndarray) -> np.ndarray:
        side_points = self.transform_primary_points(points_primary_mm)
        valid = side_points[:, 2] > 1e-6
        if not np.any(valid):
            return np.empty((0, 2), np.float64)
        projected, _ = cv2.projectPoints(
            side_points[valid],
            np.zeros(3, np.float64),
            np.zeros(3, np.float64),
            _camera_matrix(self.side_intrinsics),
            self.side_distortion,
        )
        return projected.reshape(-1, 2)


@dataclass(frozen=True)
class ProjectedROI:
    bbox: tuple[int, int, int, int]
    polygon: np.ndarray
    visible_ratio: float

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


@dataclass(frozen=True)
class SideEvidence:
    scores: dict[str, float]
    label: str
    probability: float
    reason: str
    quality: float
    roi: tuple[int, int, int, int]
    mask_pixels: int
    blur_variance: float


def _camera_matrix(intrinsics: CameraIntrinsics) -> np.ndarray:
    return np.asarray(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        np.float64,
    )


def calibrated_probabilities(
    class_distances: dict[str, float], temperature: float = 1.0
) -> dict[str, float]:
    """Convert per-class distances to a complete, temperature-scaled vector."""
    if not class_distances:
        return {}
    labels = sorted(class_distances)
    values = np.asarray([class_distances[label] for label in labels], np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return {}
    ceiling = float(np.max(values[finite]) + max(np.std(values[finite]), 1.0) * 4.0)
    values[~finite] = ceiling
    scale = max(float(np.median(np.abs(values - np.median(values)))), 0.05)
    logits = -(values - float(values.min())) / (scale * max(float(temperature), 1e-3))
    logits -= float(logits.max())
    exp_values = np.exp(logits)
    probabilities = exp_values / max(float(exp_values.sum()), 1e-12)
    return {label: float(probabilities[index]) for index, label in enumerate(labels)}


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    positive = {str(key): max(0.0, float(value)) for key, value in scores.items()}
    total = sum(positive.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in positive.items()}


def temperature_scale_scores(
    scores: dict[str, float], temperature: float
) -> dict[str, float]:
    values = normalize_scores(scores)
    if not values:
        return {}
    power = 1.0 / max(float(temperature), 1e-3)
    return normalize_scores({key: max(value, 1e-9) ** power for key, value in values.items()})


def build_object_envelope(points_primary_mm: np.ndarray, tray_plane: Plane) -> np.ndarray:
    points = np.asarray(points_primary_mm, np.float64).reshape(-1, 3)
    if not len(points):
        return points
    distance = tray_plane.signed_distance(points)
    bases = points - distance[:, None] * tray_plane.normal[None, :]
    # Extremal points are sufficient and bound the projected object more robustly
    # than a noisy dense cloud.
    all_points = np.vstack((points, bases))
    indices: set[int] = set()
    for axis in range(3):
        indices.add(int(np.argmin(all_points[:, axis])))
        indices.add(int(np.argmax(all_points[:, axis])))
    indices.update(range(0, len(all_points), max(1, len(all_points) // 80)))
    return all_points[sorted(indices)]


def projected_roi(
    calibration: DualViewCalibration,
    points_primary_mm: np.ndarray,
    tray_plane: Plane,
    image_shape: tuple[int, int] | tuple[int, int, int],
    padding_ratio: float = 0.15,
) -> ProjectedROI | None:
    height, width = image_shape[:2]
    pixels = calibration.project_primary_points(
        build_object_envelope(points_primary_mm, tray_plane)
    )
    if len(pixels) < 3:
        return None
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    visible_ratio = float(np.mean(inside))
    hull = cv2.convexHull(np.rint(pixels).astype(np.int32)).reshape(-1, 2)
    x, y, roi_width, roi_height = cv2.boundingRect(hull)
    pad_x = int(round(roi_width * padding_ratio))
    pad_y = int(round(roi_height * padding_ratio))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + roi_width + pad_x)
    y1 = min(height, y + roi_height + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None
    return ProjectedROI((x0, y0, x1 - x0, y1 - y0), hull, visible_ratio)


def roi_overlap_ratio(first: ProjectedROI, second: ProjectedROI) -> float:
    ax, ay, aw, ah = first.bbox
    bx, by, bw, bh = second.bbox
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    return intersection / max(1, min(first.area, second.area))


def _side_mask(
    image: np.ndarray,
    background: np.ndarray,
    roi: ProjectedROI,
    cfg: DualViewConfig,
) -> tuple[np.ndarray, int, float, float]:
    x, y, width, height = roi.bbox
    crop = image[y : y + height, x : x + width]
    reference = background[y : y + height, x : x + width]
    if crop.shape != reference.shape or not crop.size:
        return np.zeros((height, width), np.uint8), 0, 0.0, 0.0
    difference = cv2.absdiff(crop, reference).max(axis=2)
    mask = (difference >= cfg.side_background_delta).astype(np.uint8) * 255
    projected_support = np.zeros_like(mask)
    local_polygon = np.rint(roi.polygon - np.asarray([x, y])).astype(np.int32)
    cv2.fillPoly(projected_support, [local_polygon], 255)
    projected_support = cv2.dilate(
        projected_support,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    mask = cv2.bitwise_and(mask, projected_support)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return np.zeros_like(mask), 0, 0.0, 0.0
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == component).astype(np.uint8) * 255
    pixels = int(stats[component, cv2.CC_STAT_AREA])
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    area_quality = float(np.clip(pixels / max(cfg.side_min_area_px, 1), 0.0, 1.0))
    blur_quality = float(np.clip(blur / max(cfg.side_min_blur_variance, 1e-6), 0.0, 1.0))
    fill_ratio = pixels / max(width * height, 1)
    fill_quality = float(np.clip((0.92 - fill_ratio) / 0.22, 0.0, 1.0))
    quality = min(roi.visible_ratio, area_quality, blur_quality, fill_quality)
    return mask, pixels, blur, quality


class DualViewFusion:
    def __init__(
        self,
        calibration: DualViewCalibration | None,
        side_background_bgr: np.ndarray | None,
        side_model: GeometryRGBModel | None,
        config: DualViewConfig,
        tray_plane_primary: Plane,
    ) -> None:
        self.calibration = calibration
        self.side_background = (
            None if side_background_bgr is None else np.asarray(side_background_bgr)
        )
        self.side_model = side_model
        self.config = config
        self.tray_plane = tray_plane_primary

    def apply(
        self,
        results: list[VisionResult3D],
        point_clouds: dict[str, np.ndarray],
        pair: SynchronizedFramePair,
    ) -> None:
        started = time.perf_counter()
        base_state = self._base_state(pair)
        if base_state is not None:
            for result in results:
                self._apply_top_only(result, pair, base_state, started)
            return

        assert pair.side is not None
        assert self.calibration is not None
        rois: dict[str, ProjectedROI] = {}
        for result in results:
            points = point_clouds.get(result.object_id, np.empty((0, 3)))
            roi = projected_roi(
                self.calibration,
                points,
                self.tray_plane,
                pair.side.color_bgr.shape,
                self.config.roi_padding_ratio,
            )
            if roi is not None:
                rois[result.object_id] = roi

        occluded: set[str] = set()
        identifiers = list(rois)
        for first_index, first_id in enumerate(identifiers):
            for second_id in identifiers[first_index + 1 :]:
                if roi_overlap_ratio(rois[first_id], rois[second_id]) > self.config.roi_overlap_threshold:
                    occluded.update((first_id, second_id))

        for result in results:
            roi = rois.get(result.object_id)
            if roi is None:
                self._apply_top_only(result, pair, FusionState.TOP_ONLY, started)
            elif result.object_id in occluded:
                self._apply_top_only(
                    result, pair, FusionState.SIDE_OCCLUDED, started, roi=roi
                )
            else:
                evidence = self._classify_side(pair.side.color_bgr, roi)
                if (
                    evidence.quality < self.config.min_side_quality
                    or not evidence.scores
                    or evidence.reason != "accepted"
                ):
                    self._apply_top_only(
                        result,
                        pair,
                        FusionState.TOP_ONLY,
                        started,
                        roi=roi,
                        evidence=evidence,
                    )
                else:
                    self._fuse(result, pair, evidence, started)

    @staticmethod
    def annotate_side(
        side_bgr: np.ndarray, results: list[VisionResult3D]
    ) -> np.ndarray:
        canvas = np.asarray(side_bgr).copy()
        colours = {
            FusionState.FUSED.value: (0, 200, 0),
            FusionState.TOP_ONLY.value: (0, 190, 255),
            FusionState.SIDE_MISSING.value: (0, 0, 220),
            FusionState.SIDE_OCCLUDED.value: (180, 0, 180),
            FusionState.UNSYNCED.value: (0, 0, 220),
            FusionState.CALIBRATION_INVALID.value: (0, 0, 220),
            FusionState.CONFLICT.value: (0, 0, 255),
        }
        for result in results:
            diagnostics = result.diagnostics.get("dual_view", {})
            roi = diagnostics.get("side_roi")
            if roi is None:
                continue
            x, y, width, height = map(int, roi)
            state = str(diagnostics.get("fusion_state", FusionState.TOP_ONLY.value))
            colour = colours.get(state, (255, 255, 255))
            cv2.rectangle(canvas, (x, y), (x + width, y + height), colour, 2)
            cv2.putText(
                canvas,
                f"{result.object_id} {state}",
                (x, max(18, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                colour,
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _base_state(self, pair: SynchronizedFramePair) -> FusionState | None:
        if self.calibration is None or not self.calibration.valid:
            return FusionState.CALIBRATION_INVALID
        if pair.side is None or self.side_background is None or self.side_model is None:
            return FusionState.SIDE_MISSING
        if not pair.synchronized:
            return FusionState.UNSYNCED
        if pair.side.color_bgr.shape != self.side_background.shape:
            return FusionState.CALIBRATION_INVALID
        expected = self.calibration.side_intrinsics
        if pair.side.color_bgr.shape[:2] != (expected.height, expected.width):
            return FusionState.CALIBRATION_INVALID
        primary_expected = self.calibration.primary_intrinsics
        if pair.primary.color_bgr.shape[:2] != (
            primary_expected.height,
            primary_expected.width,
        ):
            return FusionState.CALIBRATION_INVALID
        primary_actual = pair.primary.intrinsics
        if not (
            np.isclose(primary_actual.fx, primary_expected.fx, rtol=0.02)
            and np.isclose(primary_actual.fy, primary_expected.fy, rtol=0.02)
            and np.isclose(primary_actual.cx, primary_expected.cx, atol=3.0)
            and np.isclose(primary_actual.cy, primary_expected.cy, atol=3.0)
        ):
            return FusionState.CALIBRATION_INVALID
        if self.calibration.platform_id != self.config.platform_id:
            return FusionState.CALIBRATION_INVALID
        calibrated_plane = self.calibration.tray_plane_primary
        if calibrated_plane is not None:
            alignment = float(
                np.clip(np.dot(calibrated_plane.normal, self.tray_plane.normal), -1.0, 1.0)
            )
            angle_deg = math.degrees(math.acos(alignment))
            calibrated_origin = -calibrated_plane.offset * calibrated_plane.normal
            plane_shift = abs(float(self.tray_plane.signed_distance(calibrated_origin)))
            if angle_deg > 3.0 or plane_shift > 3.0:
                return FusionState.CALIBRATION_INVALID
        return None

    def _classify_side(self, image: np.ndarray, roi: ProjectedROI) -> SideEvidence:
        assert self.side_background is not None and self.side_model is not None
        mask, pixels, blur, quality = _side_mask(
            image, self.side_background, roi, self.config
        )
        x, y, width, height = roi.bbox
        crop = image[y : y + height, x : x + width].copy()
        model_crop = np.full_like(crop, 245)
        model_crop[mask > 0] = crop[mask > 0]
        label, confidence, diagnostics = self.side_model.predict(model_crop, mask)
        raw_scores = getattr(self.side_model, "last_class_scores", {})
        scores = temperature_scale_scores(raw_scores, self.config.side_temperature)
        probability = scores.get(
            str(diagnostics.get("nearest_label", label)), float(confidence)
        )
        return SideEvidence(
            scores=scores,
            label=str(diagnostics.get("nearest_label", label)),
            probability=float(probability),
            reason=str(diagnostics.get("reason", "unknown")),
            quality=float(quality),
            roi=roi.bbox,
            mask_pixels=pixels,
            blur_variance=blur,
        )

    def _top_scores(self, result: VisionResult3D) -> dict[str, float]:
        diagnostics = result.diagnostics
        scores = diagnostics.get("top_shape_scores", {})
        if isinstance(scores, dict) and scores:
            return temperature_scale_scores(
                {str(key): float(value) for key, value in scores.items()},
                self.config.top_temperature,
            )
        if result.shape_id == "unknown":
            return {}
        confidence = float(np.clip(result.confidence.shape, 0.0, 1.0))
        return {result.shape_id: confidence, "unknown": 1.0 - confidence}

    def _apply_top_only(
        self,
        result: VisionResult3D,
        pair: SynchronizedFramePair,
        state: FusionState,
        started: float,
        roi: ProjectedROI | None = None,
        evidence: SideEvidence | None = None,
    ) -> None:
        scores = self._top_scores(result)
        top_probability = max(scores.values(), default=result.confidence.shape)
        # Explicit dual mode has a stricter degraded-mode gate.  It never
        # changes a non-shape safety failure into PICKABLE.
        if result.status == DetectionStatus.PICKABLE and top_probability < self.config.top_only_probability:
            result.status = DetectionStatus.UNCERTAIN
            result.selected = False
        self._diagnostics(
            result, pair, state, scores, None, scores, started, roi, evidence
        )

    def _fuse(
        self,
        result: VisionResult3D,
        pair: SynchronizedFramePair,
        evidence: SideEvidence,
        started: float,
    ) -> None:
        top_scores = self._top_scores(result)
        if not top_scores:
            self._apply_top_only(result, pair, FusionState.TOP_ONLY, started, evidence=evidence)
            return
        top_label, top_probability = max(top_scores.items(), key=lambda item: item[1])
        side_label, side_probability = max(
            evidence.scores.items(), key=lambda item: item[1]
        )
        if (
            top_probability >= self.config.conflict_probability
            and side_probability >= self.config.conflict_probability
            and top_label != side_label
        ):
            if result.status in {DetectionStatus.PICKABLE, DetectionStatus.UNCERTAIN}:
                result.status = DetectionStatus.UNCERTAIN
            result.selected = False
            self._diagnostics(
                result,
                pair,
                FusionState.CONFLICT,
                top_scores,
                evidence.scores,
                {},
                started,
                evidence=evidence,
            )
            return

        side_weight = self.config.side_weight * evidence.quality
        top_weight = 1.0 - side_weight
        labels = sorted(set(top_scores) | set(evidence.scores))
        log_scores = {
            label: top_weight * math.log(max(top_scores.get(label, 1e-9), 1e-9))
            + side_weight * math.log(max(evidence.scores.get(label, 1e-9), 1e-9))
            for label in labels
        }
        maximum = max(log_scores.values())
        fused = normalize_scores(
            {label: math.exp(value - maximum) for label, value in log_scores.items()}
        )
        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        winner, probability = ordered[0]
        margin = probability - (ordered[1][1] if len(ordered) > 1 else 0.0)
        top_reason = str(result.diagnostics.get("top_shape_rejection_reason", "accepted"))
        top_two = {label for label, _ in sorted(top_scores.items(), key=lambda item: item[1], reverse=True)[:2]}
        accepted = (
            probability >= self.config.fused_probability
            and margin >= self.config.fused_margin
            and top_reason != "distance_rejected"
            and (top_reason != "margin_rejected" or winner in top_two)
        )
        if not accepted:
            if result.status in {DetectionStatus.PICKABLE, DetectionStatus.UNCERTAIN}:
                result.status = DetectionStatus.UNCERTAIN
            result.selected = False
        else:
            result.shape_id = winner
            result.shape_name = str(
                result.diagnostics.get("shape_names", {}).get(winner, winner)
            )
            result.class_key = f"{result.color_id}:{winner}"
            result.confidence = Confidence3D(
                result.confidence.segmentation,
                result.confidence.color,
                probability,
                result.confidence.pose,
                result.confidence.grasp,
            )
            if (
                result.status == DetectionStatus.UNCERTAIN
                and top_reason == "margin_rejected"
                and bool(result.diagnostics.get("dual_shape_upgrade_allowed", False))
            ):
                result.status = DetectionStatus.PICKABLE
        self._diagnostics(
            result,
            pair,
            FusionState.FUSED if accepted else FusionState.TOP_ONLY,
            top_scores,
            evidence.scores,
            fused,
            started,
            evidence=evidence,
            fused_margin=margin,
        )

    def _diagnostics(
        self,
        result: VisionResult3D,
        pair: SynchronizedFramePair,
        state: FusionState,
        top_scores: dict[str, float],
        side_scores: dict[str, float] | None,
        fused_scores: dict[str, float],
        started: float,
        roi: ProjectedROI | None = None,
        evidence: SideEvidence | None = None,
        fused_margin: float | None = None,
    ) -> None:
        roi_value = evidence.roi if evidence is not None else (None if roi is None else roi.bbox)
        result.diagnostics["dual_view"] = {
            "fusion_state": state.value,
            "primary_frame_id": pair.primary.frame_id,
            "side_frame_id": None if pair.side is None else pair.side.frame_id,
            "primary_timestamp_ns": pair.primary.timestamp_ns,
            "side_timestamp_ns": None if pair.side is None else pair.side.timestamp_ns,
            "primary_host_timestamp_ns": pair.primary_host_timestamp_ns,
            "side_host_timestamp_ns": pair.side_host_timestamp_ns,
            "pair_delta_ms": pair.pair_delta_ms,
            "max_pair_delta_ms": pair.max_pair_delta_ms,
            "side_error": pair.side_error,
            "side_roi": None if roi_value is None else list(roi_value),
            "side_quality": None if evidence is None else round(evidence.quality, 5),
            "side_blur_variance": None if evidence is None else round(evidence.blur_variance, 4),
            "side_mask_pixels": None if evidence is None else evidence.mask_pixels,
            "side_reason": None if evidence is None else evidence.reason,
            "top_class_scores": {key: round(value, 7) for key, value in top_scores.items()},
            "side_class_scores": None if side_scores is None else {
                key: round(value, 7) for key, value in side_scores.items()
            },
            "fused_class_scores": {key: round(value, 7) for key, value in fused_scores.items()},
            "fused_margin": None if fused_margin is None else round(fused_margin, 7),
            "calibration_version": None if self.calibration is None else self.calibration.version,
            "calibration_hash": None if self.calibration is None else self.calibration.calibration_hash,
            "platform_id": self.config.platform_id,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 4),
        }


def save_synchronized_pair(
    pair: SynchronizedFramePair,
    directory: str | Path,
    platform_id: str,
    calibration_hash: str,
    label: str | None = None,
    reviewed: bool = False,
) -> None:
    """Save a self-contained paired sample without implying it is calibrated."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    if pair.side is None:
        raise ValueError("cannot save a dual sample without a side frame")
    if not cv2.imwrite(str(target / "primary-color.png"), pair.primary.color_bgr):
        raise OSError("failed to save primary colour image")
    np.save(target / "primary-depth.npy", pair.primary.depth)
    if not cv2.imwrite(str(target / "side-color.png"), pair.side.color_bgr):
        raise OSError("failed to save side colour image")
    metadata = {
        "schema_version": 1,
        "platform_id": platform_id,
        "calibration_hash": calibration_hash,
        "label": label,
        "human_reviewed": bool(reviewed),
        "primary_frame_id": pair.primary.frame_id,
        "side_frame_id": pair.side.frame_id,
        "primary_timestamp_ns": pair.primary.timestamp_ns,
        "side_timestamp_ns": pair.side.timestamp_ns,
        "primary_host_timestamp_ns": pair.primary_host_timestamp_ns,
        "side_host_timestamp_ns": pair.side_host_timestamp_ns,
        "pair_delta_ms": pair.pair_delta_ms,
        "max_pair_delta_ms": pair.max_pair_delta_ms,
        "synchronized": pair.synchronized,
        "primary_color_timestamp_ns": pair.primary.color_timestamp_ns,
        "primary_depth_timestamp_ns": pair.primary.depth_timestamp_ns,
        "primary_intrinsics": pair.primary.intrinsics.to_dict(),
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_synchronized_pair(directory: str | Path) -> SynchronizedFramePair:
    source = Path(directory)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    primary_color = cv2.imread(str(source / "primary-color.png"), cv2.IMREAD_COLOR)
    side_color = cv2.imread(str(source / "side-color.png"), cv2.IMREAD_COLOR)
    if primary_color is None or side_color is None:
        raise FileNotFoundError(f"incomplete dual colour images: {source}")
    primary = RGBDFrame(
        primary_color,
        np.load(source / "primary-depth.npy", allow_pickle=False),
        CameraIntrinsics(**metadata["primary_intrinsics"]),
        int(metadata["primary_timestamp_ns"]),
        str(metadata["primary_frame_id"]),
        metadata.get("primary_color_timestamp_ns"),
        metadata.get("primary_depth_timestamp_ns"),
    )
    side = RGBFrame(
        side_color,
        int(metadata["side_timestamp_ns"]),
        str(metadata["side_frame_id"]),
        int(metadata["side_timestamp_ns"]),
    )
    max_delta = float(metadata.get("max_pair_delta_ms", 50.0))
    return SynchronizedFramePair(
        primary,
        side,
        int(metadata["primary_host_timestamp_ns"]),
        int(metadata["side_host_timestamp_ns"]),
        max_delta,
    )


def _require_charuco() -> Any:
    aruco = getattr(cv2, "aruco", None)
    if aruco is None or not hasattr(aruco, "CharucoBoard"):
        raise RuntimeError(
            "ChArUco calibration requires OpenCV contrib; install sorting-vision[dual]"
        )
    return aruco


def calibrate_charuco_pairs(
    primary_images: Iterable[np.ndarray],
    side_images: Iterable[np.ndarray],
    *,
    platform_id: str,
    squares_x: int = 7,
    squares_y: int = 5,
    square_length_mm: float = 20.0,
    marker_length_mm: float = 14.0,
    dictionary_name: str = "DICT_5X5_100",
    tray_pose_index: int | None = None,
) -> DualViewCalibration:
    """Calibrate independent intrinsics and the side-from-primary transform.

    At least 20 detected poses are required, of which at least 15 must be paired
    common-board observations.  The saved ``valid`` flag is derived only from
    measured RMS, symmetric epipolar P95 and triangulated board scale.
    """
    aruco = _require_charuco()
    if not hasattr(aruco, dictionary_name):
        raise ValueError(f"unknown ArUco dictionary: {dictionary_name}")
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
    board = aruco.CharucoBoard(
        (int(squares_x), int(squares_y)),
        float(square_length_mm),
        float(marker_length_mm),
        dictionary,
    )
    detector = aruco.CharucoDetector(board)
    primary_values = list(primary_images)
    side_values = list(side_images)
    if len(primary_values) != len(side_values):
        raise ValueError("primary and side calibration image counts differ")
    if len(primary_values) < 20:
        raise ValueError("at least 20 paired ChArUco poses are required")
    primary_size = (primary_values[0].shape[1], primary_values[0].shape[0])
    side_size = (side_values[0].shape[1], side_values[0].shape[0])
    board_points = np.asarray(board.getChessboardCorners(), np.float32)

    def detect(image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        corners, identifiers, _, _ = detector.detectBoard(image)
        if identifiers is None or corners is None or len(identifiers) < 6:
            return None
        return corners.reshape(-1, 2).astype(np.float32), identifiers.reshape(-1).astype(int)

    primary_detected = [detect(image) for image in primary_values]
    side_detected = [detect(image) for image in side_values]
    if sum(item is not None for item in primary_detected) < 20:
        raise ValueError("fewer than 20 usable primary ChArUco poses")
    if sum(item is not None for item in side_detected) < 20:
        raise ValueError("fewer than 20 usable side ChArUco poses")

    def observations(detected: list[tuple[np.ndarray, np.ndarray] | None]):
        object_points, image_points = [], []
        for item in detected:
            if item is None:
                continue
            corners, identifiers = item
            object_points.append(board_points[identifiers].reshape(-1, 1, 3))
            image_points.append(corners.reshape(-1, 1, 2))
        return object_points, image_points

    primary_object, primary_points = observations(primary_detected)
    side_object, side_points = observations(side_detected)
    primary_rms, primary_matrix, primary_distortion, _, _ = cv2.calibrateCamera(
        primary_object, primary_points, primary_size, None, None
    )
    side_rms, side_matrix, side_distortion, _, _ = cv2.calibrateCamera(
        side_object, side_points, side_size, None, None
    )
    tray_from_primary = None
    tray_plane_primary = None
    if tray_pose_index is not None:
        if not 0 <= tray_pose_index < len(primary_detected):
            raise ValueError("tray_pose_index is outside the calibration image list")
        tray_observation = primary_detected[tray_pose_index]
        if tray_observation is None:
            raise ValueError("the selected flat-tray ChArUco pose was not detected")
        tray_corners, tray_ids = tray_observation
        solved, tray_rotation_vector, tray_translation = cv2.solvePnP(
            board_points[tray_ids],
            tray_corners,
            primary_matrix,
            primary_distortion,
        )
        if not solved:
            raise ValueError("cannot solve the flat-board tray pose")
        tray_rotation, _ = cv2.Rodrigues(tray_rotation_vector)
        primary_from_tray = np.eye(4, dtype=np.float64)
        primary_from_tray[:3, :3] = tray_rotation
        primary_from_tray[:3, 3] = tray_translation.reshape(3)
        tray_from_primary = np.linalg.inv(primary_from_tray)
        tray_normal_primary = tray_rotation[:, 2]
        tray_origin_primary = tray_translation.reshape(3)
        tray_plane_primary = Plane(
            tray_normal_primary,
            -float(np.dot(tray_normal_primary, tray_origin_primary)),
        )

    common_object: list[np.ndarray] = []
    common_primary: list[np.ndarray] = []
    common_side: list[np.ndarray] = []
    for top_item, side_item in zip(primary_detected, side_detected):
        if top_item is None or side_item is None:
            continue
        top_corners, top_ids = top_item
        side_corners, side_ids = side_item
        common = sorted(set(top_ids.tolist()) & set(side_ids.tolist()))
        if len(common) < 6:
            continue
        top_lookup = {identifier: top_corners[index] for index, identifier in enumerate(top_ids)}
        side_lookup = {identifier: side_corners[index] for index, identifier in enumerate(side_ids)}
        common_object.append(board_points[common].reshape(-1, 1, 3))
        common_primary.append(np.asarray([top_lookup[item] for item in common], np.float32).reshape(-1, 1, 2))
        common_side.append(np.asarray([side_lookup[item] for item in common], np.float32).reshape(-1, 1, 2))
    if len(common_object) < 15:
        raise ValueError("at least 15 jointly visible ChArUco poses are required")
    _, primary_matrix, primary_distortion, side_matrix, side_distortion, rotation, translation, _, fundamental = cv2.stereoCalibrate(
        common_object,
        common_primary,
        common_side,
        primary_matrix,
        primary_distortion,
        side_matrix,
        side_distortion,
        primary_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    epipolar_errors: list[float] = []
    scale_errors: list[float] = []
    projection_primary = np.hstack((np.eye(3), np.zeros((3, 1))))
    projection_side = np.hstack((rotation, translation.reshape(3, 1)))
    for top_points, view_points, objects in zip(common_primary, common_side, common_object):
        top_flat = top_points.reshape(-1, 2)
        side_flat = view_points.reshape(-1, 2)
        top_h = np.column_stack((top_flat, np.ones(len(top_flat))))
        side_h = np.column_stack((side_flat, np.ones(len(side_flat))))
        side_lines = (fundamental @ top_h.T).T
        top_lines = (fundamental.T @ side_h.T).T
        epipolar_errors.extend(
            np.abs(np.sum(side_lines * side_h, axis=1))
            / np.maximum(np.linalg.norm(side_lines[:, :2], axis=1), 1e-9)
        )
        epipolar_errors.extend(
            np.abs(np.sum(top_lines * top_h, axis=1))
            / np.maximum(np.linalg.norm(top_lines[:, :2], axis=1), 1e-9)
        )
        top_normalized = cv2.undistortPoints(top_points, primary_matrix, primary_distortion).reshape(-1, 2).T
        side_normalized = cv2.undistortPoints(view_points, side_matrix, side_distortion).reshape(-1, 2).T
        homogeneous = cv2.triangulatePoints(
            projection_primary, projection_side, top_normalized, side_normalized
        )
        triangulated = (homogeneous[:3] / homogeneous[3]).T
        object_flat = objects.reshape(-1, 3)
        if len(triangulated) >= 2:
            measured = np.linalg.norm(np.diff(triangulated, axis=0), axis=1)
            expected = np.linalg.norm(np.diff(object_flat, axis=0), axis=1)
            valid = expected > 0
            scale_errors.extend(np.abs(measured[valid] / expected[valid] - 1.0))
    joint_p95 = float(np.percentile(epipolar_errors, 95))
    scale_error = float(np.median(scale_errors)) if scale_errors else float("inf")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.reshape(3)
    side_intrinsics = CameraIntrinsics(
        side_size[0], side_size[1], float(side_matrix[0, 0]), float(side_matrix[1, 1]),
        float(side_matrix[0, 2]), float(side_matrix[1, 2]), 1.0,
    )
    primary_intrinsics = CameraIntrinsics(
        primary_size[0], primary_size[1], float(primary_matrix[0, 0]),
        float(primary_matrix[1, 1]), float(primary_matrix[0, 2]),
        float(primary_matrix[1, 2]), 1.0,
    )
    version_payload = np.concatenate((rotation.reshape(-1), translation.reshape(-1)))
    version = time.strftime("%Y%m%d-%H%M%S") + "-" + hashlib.sha256(
        version_payload.tobytes()
    ).hexdigest()[:8]
    return DualViewCalibration(
        primary_intrinsics,
        primary_distortion,
        side_intrinsics,
        side_distortion,
        transform,
        DualCalibrationMetrics(float(primary_rms), float(side_rms), joint_p95, scale_error),
        platform_id,
        version,
        {
            "type": "ChArUco",
            "dictionary": dictionary_name,
            "squares_x": squares_x,
            "squares_y": squares_y,
            "square_length_mm": square_length_mm,
            "marker_length_mm": marker_length_mm,
            "paired_pose_count": len(common_object),
        },
        primary_size,
        tray_plane_primary,
        tray_from_primary,
    )
