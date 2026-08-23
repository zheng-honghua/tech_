from __future__ import annotations

import time
from collections.abc import Iterable

import cv2
import numpy as np

from .calibration import PerspectiveCalibration
from .classification import HybridShapeClassifier, LabColorClassifier
from .config import VisionConfig, load_config
from .extensions import VisionExtension
from .pose import estimate_pose, masked_crop
from .segmentation import InstanceSegmenter, segment_objects
from .types import Confidence, DetectionStatus, SegmentedObject, VisionResult


class VisionPipeline:
    def __init__(
        self,
        config: VisionConfig | None = None,
        calibration: PerspectiveCalibration | None = None,
        background: np.ndarray | None = None,
        instance_model: InstanceSegmenter | None = None,
        shape_model=None,
        extensions: Iterable[VisionExtension] = (),
    ) -> None:
        self.config = config or load_config()
        tray = self.config.tray
        self.calibration = calibration or PerspectiveCalibration.identity(
            tray.rectified_width_px,
            tray.rectified_height_px,
            tray.width_mm,
            tray.height_mm,
        )
        self.background = None if background is None else self._rectify_if_needed(background)
        self.instance_model = instance_model
        self.color_classifier = LabColorClassifier(self.config.classification)
        self.shape_classifier = HybridShapeClassifier(
            self.config.classification, model=shape_model
        )
        self.extensions = list(extensions)
        self._last_signature: tuple[str, float, float] | None = None
        self._stable_count = 0

    def _rectify_if_needed(self, image: np.ndarray) -> np.ndarray:
        expected = (
            self.calibration.output_height_px,
            self.calibration.output_width_px,
        )
        if image.shape[:2] == expected and np.allclose(
            self.calibration.source_points, self.calibration.destination_points
        ):
            return image.copy()
        return self.calibration.rectify(image)

    def process(self, frame: np.ndarray, frame_id: str | None = None) -> list[VisionResult]:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with three channels")
        rectified = self._rectify_if_needed(frame)
        frame_id = frame_id or str(time.time_ns())
        results = self._run_pass(rectified, frame_id, threshold_scale=1.0)
        self._select_target(results)

        if not any(result.selected for result in results):
            retry = self._run_pass(
                rectified,
                frame_id,
                threshold_scale=self.config.segmentation.retry_threshold_scale,
            )
            self._select_target(retry)
            if self._best_score(retry) > self._best_score(results):
                results = retry

        self._apply_temporal_stability(results)
        return results

    def rectify_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return the top-down image used by the detector."""
        return self._rectify_if_needed(frame)

    def _run_pass(
        self, frame: np.ndarray, frame_id: str, threshold_scale: float
    ) -> list[VisionResult]:
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        objects = segment_objects(
            frame,
            self.background,
            self.config.segmentation,
            threshold_scale=threshold_scale,
            model=self.instance_model,
        )
        return [
            self._analyze_object(frame, lab_frame, item, frame_id, index)
            for index, item in enumerate(objects, start=1)
        ]

    def _analyze_object(
        self,
        frame: np.ndarray,
        lab_frame: np.ndarray,
        item: SegmentedObject,
        frame_id: str,
        index: int,
    ) -> VisionResult:
        crop, crop_mask = masked_crop(frame, item.mask, item.bbox)
        color = self.color_classifier.classify(frame, item.mask, lab_image=lab_frame)
        shape = self.shape_classifier.classify(crop, crop_mask)
        center, angle, pose_confidence, _ = estimate_pose(
            item, shape.label_id, self.calibration
        )
        confidence = Confidence(
            segmentation=item.segmentation_confidence,
            color=color.confidence,
            shape=shape.confidence,
            pose=pose_confidence,
        )
        cfg = self.config.classification
        is_uncertain = (
            color.label_id == "unknown"
            or shape.label_id == "unknown"
            or color.confidence < cfg.min_color_confidence
            or shape.confidence < cfg.min_shape_confidence
            or confidence.combined < cfg.min_pick_confidence
        )
        is_occluded = (
            item.touches_border
            or item.clearance_px < self.config.segmentation.min_clearance_px
        )
        status = (
            DetectionStatus.OCCLUDED
            if is_occluded
            else DetectionStatus.UNCERTAIN
            if is_uncertain
            else DetectionStatus.PICKABLE
        )
        extension_values = {}
        for extension in self.extensions:
            try:
                extension_values[extension.name] = extension.analyze(crop, crop_mask)
            except Exception as error:  # An optional plugin must not stop sorting.
                extension_values[extension.name] = {
                    "error": f"{type(error).__name__}: {error}",
                    "confidence": 0.0,
                }
        return VisionResult(
            frame_id=frame_id,
            object_id=f"{frame_id}-{index:02d}",
            color_id=color.label_id,
            color_name=color.label_name,
            shape_id=shape.label_id,
            shape_name=shape.label_name,
            class_key=f"{color.label_id}:{shape.label_id}",
            center_mm=center,
            angle_deg=angle,
            confidence=confidence,
            status=status,
            bbox_px=item.bbox,
            clearance_px=item.clearance_px,
            crop_image=crop,
            extensions=extension_values,
        )

    def _selection_score(self, result: VisionResult) -> float:
        clearance_score = float(
            np.clip(
                result.clearance_px
                / max(1.0, 4.0 * self.config.segmentation.min_clearance_px),
                0.0,
                1.0,
            )
        )
        weight = self.config.selection.prefer_clearance_weight
        return (1.0 - weight) * result.confidence.combined + weight * clearance_score

    def _select_target(self, results: list[VisionResult]) -> None:
        candidates = [r for r in results if r.status == DetectionStatus.PICKABLE]
        if not candidates:
            return
        best = max(candidates, key=self._selection_score)
        best.selected = True

    @staticmethod
    def _best_score(results: list[VisionResult]) -> float:
        selected = [result.confidence.combined for result in results if result.selected]
        return max(selected, default=-1.0)

    def _apply_temporal_stability(self, results: list[VisionResult]) -> None:
        selected = next((result for result in results if result.selected), None)
        if selected is None:
            self._last_signature = None
            self._stable_count = 0
            return
        signature = (selected.class_key, selected.center_mm.x, selected.center_mm.y)
        same = False
        if self._last_signature is not None:
            last_key, last_x, last_y = self._last_signature
            same = last_key == signature[0] and np.hypot(
                last_x - signature[1], last_y - signature[2]
            ) <= 3.0
        self._stable_count = self._stable_count + 1 if same else 1
        self._last_signature = signature
        if self._stable_count < max(1, self.config.selection.stable_frames):
            selected.selected = False

    def reset_tracking(self) -> None:
        """Call after the controller confirms that a selected object was removed."""
        self._last_signature = None
        self._stable_count = 0

    @staticmethod
    def annotate(frame: np.ndarray, results: list[VisionResult]) -> np.ndarray:
        canvas = frame.copy()
        colours = {
            DetectionStatus.PICKABLE: (0, 200, 0),
            DetectionStatus.UNCERTAIN: (0, 190, 255),
            DetectionStatus.OCCLUDED: (0, 0, 220),
        }
        for result in results:
            x, y, width, height = result.bbox_px
            colour = (255, 80, 0) if result.selected else colours[result.status]
            cv2.rectangle(canvas, (x, y), (x + width, y + height), colour, 2)
            text = f"{result.class_key} {result.confidence.combined:.2f}"
            cv2.putText(
                canvas,
                text,
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                1,
                cv2.LINE_AA,
            )
        return canvas
