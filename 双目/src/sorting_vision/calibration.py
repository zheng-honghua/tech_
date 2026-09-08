from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .types import Point2D


@dataclass(frozen=True)
class PerspectiveCalibration:
    """Maps a camera image to a top-down tray view.

    Source corner order is top-left, top-right, bottom-right, bottom-left.
    Millimetre coordinates use the rectified tray's bottom-left as the origin.
    """

    source_points: np.ndarray
    output_width_px: int
    output_height_px: int
    tray_width_mm: float
    tray_height_mm: float

    def __post_init__(self) -> None:
        points = np.asarray(self.source_points, dtype=np.float32)
        if points.shape != (4, 2):
            raise ValueError("source_points must contain four (x, y) corners")
        object.__setattr__(self, "source_points", points)

    @property
    def destination_points(self) -> np.ndarray:
        return np.array(
            [
                [0, 0],
                [self.output_width_px - 1, 0],
                [self.output_width_px - 1, self.output_height_px - 1],
                [0, self.output_height_px - 1],
            ],
            dtype=np.float32,
        )

    @property
    def matrix(self) -> np.ndarray:
        return cv2.getPerspectiveTransform(self.source_points, self.destination_points)

    def rectify(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.ndim not in (2, 3):
            raise ValueError("image must be a valid grayscale or colour array")
        return cv2.warpPerspective(
            image,
            self.matrix,
            (self.output_width_px, self.output_height_px),
            flags=cv2.INTER_LINEAR,
        )

    def pixel_to_mm(self, x_px: float, y_px: float) -> Point2D:
        x_mm = x_px * self.tray_width_mm / max(1, self.output_width_px - 1)
        y_mm = (self.output_height_px - 1 - y_px) * self.tray_height_mm / max(
            1, self.output_height_px - 1
        )
        return Point2D(float(x_mm), float(y_mm))

    @classmethod
    def identity(
        cls,
        width_px: int,
        height_px: int,
        tray_width_mm: float,
        tray_height_mm: float,
    ) -> "PerspectiveCalibration":
        return cls(
            source_points=np.array(
                [[0, 0], [width_px - 1, 0], [width_px - 1, height_px - 1], [0, height_px - 1]],
                dtype=np.float32,
            ),
            output_width_px=width_px,
            output_height_px=height_px,
            tray_width_mm=tray_width_mm,
            tray_height_mm=tray_height_mm,
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "source_points": self.source_points.tolist(),
            "output_width_px": self.output_width_px,
            "output_height_px": self.output_height_px,
            "tray_width_mm": self.tray_width_mm,
            "tray_height_mm": self.tray_height_mm,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PerspectiveCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)

