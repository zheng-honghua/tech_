from __future__ import annotations

import numpy as np

from .pipeline import VisionPipeline
from .types import Confidence3D, DetectionStatus, VisionResult3D


class RGBDevelopmentPipeline:
    """2-D development pipeline that can never authorize a robot pick."""

    def __init__(self, pipeline: VisionPipeline) -> None:
        self.pipeline = pipeline
        self._last_health: dict[str, object] = {
            "ok": True,
            "reason": "depth_required",
            "mode": "RGB_ONLY",
        }

    def process(self, frame) -> list[VisionResult3D]:
        rgb_frame_id = getattr(frame, "frame_id", None)
        image = getattr(frame, "color_bgr", frame)
        timestamp_ns = getattr(frame, "timestamp_ns", None)
        results_2d = self.pipeline.process(image, frame_id=rgb_frame_id)
        results: list[VisionResult3D] = []
        for item in results_2d:
            confidence = Confidence3D(
                segmentation=item.confidence.segmentation,
                color=item.confidence.color,
                shape=item.confidence.shape,
                pose=0.0,
                grasp=0.0,
            )
            results.append(
                VisionResult3D(
                    frame_id=item.frame_id,
                    object_id=item.object_id,
                    color_id=item.color_id,
                    color_name=item.color_name,
                    shape_id=item.shape_id,
                    shape_name=item.shape_name,
                    class_key=item.class_key,
                    pose_3d=None,
                    grasp=None,
                    confidence=confidence,
                    status=DetectionStatus.DEPTH_REQUIRED,
                    bbox_px=item.bbox_px,
                    center_mm=item.center_mm,
                    angle_deg=item.angle_deg,
                    selected=False,
                    crop_image=item.crop_image,
                    diagnostics={"mode": "RGB_ONLY", "provisional": True},
                )
            )
        self._last_health = {
            "ok": True,
            "reason": "depth_required",
            "mode": "RGB_ONLY",
            "frame_id": rgb_frame_id,
            "timestamp_ns": timestamp_ns,
        }
        return results

    def reset_tracking(self) -> None:
        self.pipeline.reset_tracking()

    def acknowledge_pick(self) -> None:
        self.reset_tracking()

    def health(self) -> dict[str, object]:
        return dict(self._last_health)

    def annotate(self, frame, results: list[VisionResult3D]) -> np.ndarray:
        image = getattr(frame, "color_bgr", frame)
        canvas = image.copy()
        import cv2

        for result in results:
            x, y, width, height = result.bbox_px
            cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 190, 255), 2)
            cv2.putText(
                canvas,
                f"{result.color_id}/{result.shape_id} RGB only",
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 190, 255),
                1,
                cv2.LINE_AA,
            )
        return canvas
