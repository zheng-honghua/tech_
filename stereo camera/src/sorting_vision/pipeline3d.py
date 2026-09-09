from __future__ import annotations

import time

import cv2
import numpy as np

from .classification import LabColorClassifier
from .camera import SynchronizedFramePair
from .rgb_instance import recover_rgb_masks
from .classification3d import HybridShapeClassifier3D, ShapeModel3D
from .config import VisionConfig, load_config
from .dual_view import DualViewFusion
from .geometry3d import (
    DepthSegmentedObject,
    estimate_plane_shift_mm,
    height_map_from_plane,
    object_point_cloud,
    segment_depth_objects,
    valid_depth_mask,
)
from .grasp3d import find_suction_grasp
from .geometry_rgbd_model import constrain_tray_roi, detect_rgb_object_support, detect_tray_roi_mask
from .pose import principal_angle_deg
from .rgbd import (
    RGBDCalibration,
    RGBDFrame,
    depth_to_points,
    fit_plane_ransac,
    resize_rgbd_frame,
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
        dual_view_fusion: DualViewFusion | None = None,
    ) -> None:
        self.config = config or load_config()
        if calibration is None:
            if background_frame is None:
                raise ValueError("calibration or an empty-tray RGB-D frame is required")
            depth = background_frame.depth_mm
            tray_roi = detect_tray_roi_mask(background_frame.color_bgr)
            calibration_mask = (
                valid_depth_mask(depth, self.config.rgbd) & (tray_roi > 0)
            ).astype(np.uint8) * 255
            points, _ = depth_to_points(
                depth,
                background_frame.intrinsics,
                calibration_mask,
                stride=8,
            )
            plane = fit_plane_ransac(
                points,
                threshold_mm=self.config.rgbd.plane_ransac_threshold_mm,
            )
            scale = self.config.rgbd.processing_scale
            reference = detect_tray_roi_mask(resize_rgbd_frame(background_frame, scale).color_bgr)
            contours, _ = cv2.findContours(reference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygon = max(contours, key=cv2.contourArea).reshape(-1, 2) / scale
            polygon = np.minimum(polygon, [background_frame.intrinsics.width - 1,
                                           background_frame.intrinsics.height - 1])
            calibration = RGBDCalibration(
                background_frame.intrinsics, np.eye(4, dtype=np.float64), plane,
                tuple(map(tuple, polygon.tolist())),
            )
        self.calibration = calibration
        self.color_classifier = LabColorClassifier(self.config.classification)
        self.shape_classifier = HybridShapeClassifier3D(
            self.config.classification, model=shape_model
        )
        self.dual_view_fusion = dual_view_fusion
        self._last_object_points: dict[str, np.ndarray] = {}
        self._last_signature: tuple[str, float, float, float] | None = None
        self._stable_count = 0
        self._last_tray_roi: np.ndarray | None = None
        self._last_health: dict[str, object] = {
            "ok": False,
            "reason": "no_frame_processed",
        }

    def process(self, frame: RGBDFrame) -> list[VisionResult3D]:
        return self._process(frame, None)

    def process_pair(self, pair: SynchronizedFramePair) -> list[VisionResult3D]:
        """Process dual input while keeping primary RGB-D as the authority."""
        return self._process(pair.primary, pair)

    def _process(
        self, frame: RGBDFrame, pair: SynchronizedFramePair | None
    ) -> list[VisionResult3D]:
        self._validate_frame(frame)
        self._last_object_points = {}
        working_frame, processing_scale = self._prepare_frame(frame)
        active_calibration = RGBDCalibration(
            working_frame.intrinsics,
            self.calibration.camera_to_robot,
            self.calibration.tray_plane_camera,
        )
        depth_mm = working_frame.depth_mm
        valid = valid_depth_mask(depth_mm, self.config.rgbd)
        try:
            tray_roi = detect_tray_roi_mask(working_frame.color_bgr)
            if self.calibration.tray_roi_polygon is not None:
                reference = np.zeros_like(tray_roi)
                polygon = np.rint(np.asarray(self.calibration.tray_roi_polygon) * processing_scale).astype(np.int32)
                cv2.fillPoly(reference, [polygon], 255)
                tray_roi = constrain_tray_roi(tray_roi, reference)
            object_support = detect_rgb_object_support(
                working_frame.color_bgr, tray_roi
            )
        except ValueError as error:
            self._last_tray_roi = None
            self._last_health = {
                "ok": False,
                "reason": str(error),
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                "calibration_valid": True,
                "tray_roi_valid": False,
            }
            self.reset_tracking()
            return []
        self._last_tray_roi = cv2.resize(
            tray_roi,
            (frame.intrinsics.width, frame.intrinsics.height),
            interpolation=cv2.INTER_NEAREST,
        )
        roi_pixels = tray_roi > 0
        tray_valid_ratio = float(np.mean(valid[roi_pixels]))
        heights = height_map_from_plane(
            depth_mm, working_frame.intrinsics, self.calibration.tray_plane_camera
        )
        plane_shift = estimate_plane_shift_mm(
            depth_mm,
            working_frame.intrinsics,
            self.calibration.tray_plane_camera,
            self.config.rgbd,
            heights=heights,
            roi_mask=tray_roi,
        )
        healthy = (
            tray_valid_ratio >= 0.5
            and np.isfinite(plane_shift)
            and abs(plane_shift) <= self.config.rgbd.max_plane_shift_mm
            and frame.sync_delta_ms <= self.config.rgbd.max_rgb_depth_sync_ms
        )
        reason = "ok"
        if tray_valid_ratio < 0.5:
            reason = "insufficient_tray_depth"
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
            "global_valid_depth_ratio": round(float(np.mean(valid)), 5),
            "tray_valid_depth_ratio": round(tray_valid_ratio, 5),
            "tray_roi_area_ratio": round(float(np.mean(roi_pixels)), 5),
            "tray_roi_valid": True,
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
            roi_mask=tray_roi,
            support_mask=object_support,
        )
        lab = cv2.cvtColor(working_frame.color_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        rgb_masks = recover_rgb_masks(
            working_frame.color_bgr, [item.mask for item in objects], tray_roi,
        )
        results = [
            self._analyze_object(
                working_frame, active_calibration, depth_mm, lab, item, index, rgb_masks[index - 1]
            )
            for index, item in enumerate(objects, start=1)
        ]
        if processing_scale != 1.0:
            self._restore_image_coordinates(results, processing_scale)
        if pair is not None and self.dual_view_fusion is not None:
            self.dual_view_fusion.apply(results, self._last_object_points, pair)
            states = sorted(
                {
                    str(
                        item.diagnostics.get("dual_view", {}).get(
                            "fusion_state", "TOP_ONLY"
                        )
                    )
                    for item in results
                }
            )
            self._last_health["dual_view"] = {
                "pair_delta_ms": pair.pair_delta_ms,
                "synchronized": pair.synchronized,
                "primary_frame_id": pair.primary.frame_id,
                "side_frame_id": None if pair.side is None else pair.side.frame_id,
                "side_error": pair.side_error,
                "fusion_states": states,
            }
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
        return resize_rgbd_frame(frame, scale), scale

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
            for key in ("rgb_bbox_px", "rgb_crop_origin_uv"):
                if key in result.diagnostics:
                    result.diagnostics[key] = [int(round(value / scale)) for value in result.diagnostics[key]]

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
        rgb_mask: np.ndarray | None = None,
    ) -> VisionResult3D:
        rgb_mask = item.mask if rgb_mask is None else rgb_mask
        x, y, width, height = cv2.boundingRect(rgb_mask)
        padding = 8
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1 = min(frame.intrinsics.width, x + width + padding)
        y1 = min(frame.intrinsics.height, y + height + padding)
        crop = frame.color_bgr[y0:y1, x0:x1].copy()
        crop_mask = item.mask[y0:y1, x0:x1].copy()
        neutral = np.full_like(crop, 245)
        visible = rgb_mask[y0:y1, x0:x1] > 0
        neutral[visible] = crop[visible]
        crop = neutral
        depth_crop = depth_mm[y0:y1, x0:x1].copy()

        color = self.color_classifier.classify(
            frame.color_bgr, rgb_mask, lab_image=lab_frame
        )
        points, _ = object_point_cloud(
            item,
            depth_mm,
            frame.intrinsics,
            stride=max(1, self.config.rgbd.point_sample_stride),
        )
        # Existing NPZ features depend on the depth-owned crop and its local
        # sampling grid. Keep that input stable while exporting a complete RGB
        # silhouette separately; a new RGB model can consume the latter later.
        sx, sy, sw, sh = item.bbox
        sx0, sy0 = max(0, sx - padding), max(0, sy - padding)
        sx1 = min(frame.intrinsics.width, sx + sw + padding)
        sy1 = min(frame.intrinsics.height, sy + sh + padding)
        shape_mask = item.mask[sy0:sy1, sx0:sx1]
        shape_rgb = frame.color_bgr[sy0:sy1, sx0:sx1].copy()
        shape_rgb[shape_mask == 0] = 245
        shape = self.shape_classifier.classify(
            points, shape_rgb, depth_mm[sy0:sy1, sx0:sx1], shape_mask,
            frame.intrinsics, (sx0, sy0),
        )
        top_shape_scores = dict(
            shape.class_scores or self.shape_classifier.last_class_scores
        )
        top_shape_reason = str(
            shape.rejection_reason or self.shape_classifier.last_rejection_reason
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

        shape_upgrade_allowed = (
            status == DetectionStatus.UNCERTAIN
            and top_shape_reason == "margin_rejected"
            and item.valid_depth_ratio >= self.config.rgbd.min_valid_depth_ratio
            and not item.touches_border
            and item.clearance_px >= self.config.rgbd.min_clearance_px
            and color.label_id != "unknown"
            and color.confidence >= classification_cfg.min_color_confidence
            and grasp is not None
            and grasp.info.score >= self.config.grasp.min_grasp_score
        )

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
            "rgb_mask_pixels": int(cv2.countNonZero(rgb_mask)),
            "depth_mask_pixels": int(cv2.countNonZero(item.mask)),
            "rgb_crop_origin_uv": [x0, y0],
            "rgb_bbox_px": [x, y, width, height],
            "top_shape_scores": {
                key: round(float(value), 7) for key, value in top_shape_scores.items()
            },
            "top_shape_rejection_reason": top_shape_reason,
            "shape_names": dict(self.config.classification.shapes),
            "dual_shape_upgrade_allowed": shape_upgrade_allowed,
        }
        result = VisionResult3D(
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
            bbox_px=(x, y, width, height),
            center_mm=center_mm,
            angle_deg=angle,
            crop_image=crop,
            depth_crop=depth_crop,
            rgb_crop_mask=visible.astype(np.uint8) * 255,
            depth_valid_crop_mask=(
                (crop_mask > 0) & valid_depth_mask(depth_crop, self.config.rgbd)
            ).astype(np.uint8) * 255,
            diagnostics=diagnostics,
        )
        self._last_object_points[result.object_id] = points
        return result

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

    def annotate(self, frame: RGBDFrame, results: list[VisionResult3D]) -> np.ndarray:
        canvas = frame.color_bgr.copy()
        if self._last_tray_roi is not None:
            outside = self._last_tray_roi == 0
            canvas[outside] = (canvas[outside].astype(np.float32) * 0.22).astype(np.uint8)
            contours, _ = cv2.findContours(
                self._last_tray_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, (0, 255, 255), 3)
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
                f"{result.color_id}/{shape_aliases.get(result.shape_id, result.shape_id)} shape={result.confidence.shape:.2f}",
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"{result.status.value} color={result.confidence.color:.2f} grasp={result.confidence.grasp:.2f}",
                (x, min(canvas.shape[0] - 5, y + height + 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA,
            )
            pixel = result.diagnostics.get("grasp_pixel_uv")
            if pixel is not None:
                cv2.drawMarker(canvas, tuple(pixel), colour, cv2.MARKER_CROSS, 14, 2)
        return canvas
