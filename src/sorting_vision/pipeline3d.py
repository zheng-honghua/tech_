from __future__ import annotations

import time

import cv2
import numpy as np

from .classification import LabColorClassifier
from .classification3d import HybridShapeClassifier3D, ShapeModel3D
from .config import VisionConfig, load_config
from .geometry3d import (
    DepthSegmentedObject,
    estimate_plane_shift_mm,
    height_map_from_plane,
    object_point_cloud,
    segment_depth_objects,
    valid_depth_mask,
)
from .grasp3d import find_suction_grasp
from .pose import principal_angle_deg
from .rgbd import (
    CameraIntrinsics,
    RGBDCalibration,
    RGBDFrame,
    depth_to_points,
    fit_plane_ransac,
)
from .types import (
    Confidence3D,
    DetectionStatus,
    Point2D,
    VisionResult3D,
)


class VisionPipeline3D:
    def __init__(
        self,
        config: VisionConfig | None = None,
        calibration: RGBDCalibration | None = None,
        background_frame: RGBDFrame | None = None,
        shape_model: ShapeModel3D | None = None,
    ) -> None:
        self.config = config or load_config()
        if calibration is None:
            if background_frame is None:
                raise ValueError("calibration or an empty-tray RGB-D frame is required")
            depth = background_frame.depth_mm
            points, _ = depth_to_points(
                depth,
                background_frame.intrinsics,
                valid_depth_mask(depth, self.config.rgbd).astype(np.uint8) * 255,
                stride=8,
            )
            plane = fit_plane_ransac(
                points,
                threshold_mm=self.config.rgbd.plane_ransac_threshold_mm,
            )
            calibration = RGBDCalibration(
                background_frame.intrinsics, np.eye(4, dtype=np.float64), plane
            )
        self.calibration = calibration
        self.color_classifier = LabColorClassifier(self.config.classification)
        self.shape_classifier = HybridShapeClassifier3D(
            self.config.classification, model=shape_model
        )
        self._last_signature: tuple[str, float, float, float] | None = None
        self._stable_count = 0
        self._last_health: dict[str, object] = {
            "ok": False,
            "reason": "no_frame_processed",
        }

    def process(self, frame: RGBDFrame) -> list[VisionResult3D]:
        self._validate_frame(frame)
        working_frame, processing_scale = self._prepare_frame(frame)
        active_calibration = RGBDCalibration(
            working_frame.intrinsics,
            self.calibration.camera_to_robot,
            self.calibration.tray_plane_camera,
        )
        depth_mm = working_frame.depth_mm
        valid = valid_depth_mask(depth_mm, self.config.rgbd)
        global_valid_ratio = float(np.mean(valid))
        heights = height_map_from_plane(
            depth_mm, working_frame.intrinsics, self.calibration.tray_plane_camera
        )
        plane_shift = estimate_plane_shift_mm(
            depth_mm,
            working_frame.intrinsics,
            self.calibration.tray_plane_camera,
            self.config.rgbd,
            heights=heights,
        )
        healthy = (
            global_valid_ratio >= 0.5
            and np.isfinite(plane_shift)
            and abs(plane_shift) <= self.config.rgbd.max_plane_shift_mm
            and frame.sync_delta_ms <= self.config.rgbd.max_rgb_depth_sync_ms
        )
        reason = "ok"
        if global_valid_ratio < 0.5:
            reason = "insufficient_global_depth"
        elif not np.isfinite(plane_shift):
            reason = "tray_plane_not_visible"
        elif abs(plane_shift) > self.config.rgbd.max_plane_shift_mm:
            reason = "tray_plane_shift_exceeded"
        elif frame.sync_delta_ms > self.config.rgbd.max_rgb_depth_sync_ms:
            reason = "rgb_depth_out_of_sync"
        self._last_health = {
            "ok": healthy,
            "reason": reason,
            "frame_id": frame.frame_id,
            "timestamp_ns": frame.timestamp_ns,
            "global_valid_depth_ratio": round(global_valid_ratio, 5),
            "tray_plane_shift_mm": None if not np.isfinite(plane_shift) else round(plane_shift, 4),
            "rgb_depth_sync_delta_ms": round(frame.sync_delta_ms, 4),
            "calibration_valid": True,
        }

        objects, _ = segment_depth_objects(
            working_frame.color_bgr,
            depth_mm,
            working_frame.intrinsics,
            self.calibration.tray_plane_camera,
            self.config.rgbd,
            heights=heights,
        )
        lab = cv2.cvtColor(working_frame.color_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        results = [
            self._analyze_object(
                working_frame, active_calibration, depth_mm, lab, item, index
            )
            for index, item in enumerate(objects, start=1)
        ]
        if processing_scale != 1.0:
            self._restore_image_coordinates(results, processing_scale)
        if healthy:
            self._select_target(results)
            self._apply_temporal_stability(results)
        else:
            for result in results:
                result.status = DetectionStatus.DEPTH_INVALID
                result.selected = False
            self.reset_tracking()
        return results

    def _prepare_frame(self, frame: RGBDFrame) -> tuple[RGBDFrame, float]:
        scale = float(self.config.rgbd.processing_scale)
        if not 0 < scale <= 1.0:
            raise ValueError("rgbd.processing_scale must be in the interval (0, 1]")
        if np.isclose(scale, 1.0):
            return frame, 1.0
        width = max(1, int(round(frame.intrinsics.width * scale)))
        height = max(1, int(round(frame.intrinsics.height * scale)))
        intrinsics = CameraIntrinsics(
            width=width,
            height=height,
            fx=frame.intrinsics.fx * scale,
            fy=frame.intrinsics.fy * scale,
            cx=(frame.intrinsics.cx + 0.5) * scale - 0.5,
            cy=(frame.intrinsics.cy + 0.5) * scale - 0.5,
            depth_scale_to_mm=frame.intrinsics.depth_scale_to_mm,
        )
        return (
            RGBDFrame(
                cv2.resize(frame.color_bgr, (width, height), interpolation=cv2.INTER_AREA),
                cv2.resize(frame.depth, (width, height), interpolation=cv2.INTER_NEAREST),
                intrinsics,
                frame.timestamp_ns,
                frame.frame_id,
                frame.color_timestamp_ns,
                frame.depth_timestamp_ns,
            ),
            scale,
        )

    @staticmethod
    def _restore_image_coordinates(
        results: list[VisionResult3D], scale: float
    ) -> None:
        for result in results:
            result.bbox_px = tuple(int(round(value / scale)) for value in result.bbox_px)
            pixel = result.diagnostics.get("grasp_pixel_uv")
            if pixel is not None:
                result.diagnostics["grasp_pixel_uv"] = [
                    int(round(pixel[0] / scale)),
                    int(round(pixel[1] / scale)),
                ]
            result.diagnostics["processing_scale"] = scale

    def _validate_frame(self, frame: RGBDFrame) -> None:
        expected = self.calibration.intrinsics
        actual = frame.intrinsics
        numeric = ("fx", "fy", "cx", "cy", "depth_scale_to_mm")
        if actual.width != expected.width or actual.height != expected.height:
            raise ValueError("frame dimensions do not match RGB-D calibration")
        if any(not np.isclose(getattr(actual, name), getattr(expected, name)) for name in numeric):
            raise ValueError("frame intrinsics do not match RGB-D calibration")

    def _analyze_object(
        self,
        frame: RGBDFrame,
        active_calibration: RGBDCalibration,
        depth_mm: np.ndarray,
        lab_frame: np.ndarray,
        item: DepthSegmentedObject,
        index: int,
    ) -> VisionResult3D:
        x, y, width, height = item.bbox
        padding = 8
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1 = min(frame.intrinsics.width, x + width + padding)
        y1 = min(frame.intrinsics.height, y + height + padding)
        crop = frame.color_bgr[y0:y1, x0:x1].copy()
        crop_mask = item.mask[y0:y1, x0:x1].copy()
        neutral = np.full_like(crop, 245)
        neutral[crop_mask > 0] = crop[crop_mask > 0]
        crop = neutral
        depth_crop = depth_mm[y0:y1, x0:x1].copy()

        color = self.color_classifier.classify(
            frame.color_bgr, item.mask, lab_image=lab_frame
        )
        points, _ = object_point_cloud(
            item,
            depth_mm,
            frame.intrinsics,
            stride=max(1, self.config.rgbd.point_sample_stride),
        )
        shape = self.shape_classifier.classify(
            points, crop, depth_crop, crop_mask,
            frame.intrinsics, (x0, y0),
        )
        grasp = find_suction_grasp(
            item, depth_mm, active_calibration, self.config.grasp
        )
        grasp_score = 0.0 if grasp is None else grasp.info.score
        pose_confidence = 0.0 if grasp is None else grasp.pose_confidence
        confidence = Confidence3D(
            segmentation=item.segmentation_confidence,
            color=color.confidence,
            shape=shape.confidence,
            pose=pose_confidence,
            grasp=grasp_score,
        )

        classification_cfg = self.config.classification
        if item.valid_depth_ratio < self.config.rgbd.min_valid_depth_ratio:
            status = DetectionStatus.DEPTH_INVALID
        elif item.touches_border or item.clearance_px < self.config.rgbd.min_clearance_px:
            status = DetectionStatus.OCCLUDED
        elif (
            color.label_id == "unknown"
            or shape.label_id == "unknown"
            or color.confidence < classification_cfg.min_color_confidence
            or shape.confidence < classification_cfg.min_shape_confidence
        ):
            status = DetectionStatus.UNCERTAIN
        elif grasp is None or grasp.info.score < self.config.grasp.min_grasp_score:
            status = DetectionStatus.NO_GRASP_SURFACE
        else:
            status = DetectionStatus.PICKABLE

        pose = None if grasp is None else grasp.pose
        center_mm = (
            None
            if pose is None
            else Point2D(pose.position_mm.x, pose.position_mm.y)
        )
        angle = principal_angle_deg(item.contour)
        diagnostics = {
            "height_min_mm": round(item.height_min_mm, 3),
            "height_max_mm": round(item.height_max_mm, 3),
            "object_valid_depth_ratio": round(item.valid_depth_ratio, 5),
            "clearance_px": None if not np.isfinite(item.clearance_px) else round(item.clearance_px, 3),
            "shape_features": {key: round(float(value), 5) for key, value in shape.features.items()},
            "grasp_pixel_uv": None if grasp is None else list(grasp.pixel_uv),
        }
        return VisionResult3D(
            frame_id=frame.frame_id,
            object_id=f"{frame.frame_id}-{index:02d}",
            color_id=color.label_id,
            color_name=color.label_name,
            shape_id=shape.label_id,
            shape_name=shape.label_name,
            class_key=f"{color.label_id}:{shape.label_id}",
            pose_3d=pose,
            grasp=None if grasp is None else grasp.info,
            confidence=confidence,
            status=status,
            bbox_px=item.bbox,
            center_mm=center_mm,
            angle_deg=angle,
            crop_image=crop,
            depth_crop=depth_crop,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _select_target(results: list[VisionResult3D]) -> None:
        candidates = [item for item in results if item.status == DetectionStatus.PICKABLE]
        if candidates:
            max(candidates, key=lambda item: item.confidence.combined).selected = True

    def _apply_temporal_stability(self, results: list[VisionResult3D]) -> None:
        selected = next((item for item in results if item.selected), None)
        if selected is None or selected.pose_3d is None:
            self.reset_tracking()
            return
        point = selected.pose_3d.position_mm
        signature = (selected.class_key, point.x, point.y, point.z)
        same = False
        if self._last_signature is not None:
            last_key, x, y, z = self._last_signature
            same = last_key == signature[0] and np.linalg.norm(
                np.array([x - point.x, y - point.y, z - point.z])
            ) <= 3.0
        self._stable_count = self._stable_count + 1 if same else 1
        self._last_signature = signature
        if self._stable_count < max(1, self.config.selection.stable_frames):
            selected.selected = False

    def acknowledge_pick(self) -> None:
        self.reset_tracking()

    def reset_tracking(self) -> None:
        self._last_signature = None
        self._stable_count = 0

    def health(self) -> dict[str, object]:
        return dict(self._last_health)

    @staticmethod
    def annotate(frame: RGBDFrame, results: list[VisionResult3D]) -> np.ndarray:
        canvas = frame.color_bgr.copy()
        colours = {
            DetectionStatus.PICKABLE: (0, 200, 0),
            DetectionStatus.UNCERTAIN: (0, 190, 255),
            DetectionStatus.OCCLUDED: (0, 0, 220),
            DetectionStatus.DEPTH_INVALID: (180, 0, 180),
            DetectionStatus.NO_GRASP_SURFACE: (0, 120, 255),
        }
        shape_aliases = {
            "triangular_prism": "tri",
            "pentagonal_prism": "pent",
            "hexagonal_prism": "hex",
            "cylinder": "cyl",
            "cuboid": "box",
        }
        for result in results:
            x, y, width, height = result.bbox_px
            colour = (255, 80, 0) if result.selected else colours[result.status]
            cv2.rectangle(canvas, (x, y), (x + width, y + height), colour, 2)
            cv2.putText(
                canvas,
                f"{result.color_id}/{shape_aliases.get(result.shape_id, result.shape_id)} {result.confidence.combined:.2f}",
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                1,
                cv2.LINE_AA,
            )
            pixel = result.diagnostics.get("grasp_pixel_uv")
            if pixel is not None:
                cv2.drawMarker(canvas, tuple(pixel), colour, cv2.MARKER_CROSS, 14, 2)
        return canvas
