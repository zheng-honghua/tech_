from __future__ import annotations

import math

import cv2
import numpy as np

from .calibration import PerspectiveCalibration
from .types import Point2D, SegmentedObject


SYMMETRIC_SHAPES = {"circle", "square", "hexagon"}


def safe_grasp_point(mask: np.ndarray) -> tuple[tuple[int, int], float]:
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, maximum, _, location = cv2.minMaxLoc(distance)
    return (int(location[0]), int(location[1])), float(maximum)


def principal_angle_deg(contour: np.ndarray) -> float:
    points = contour[:, 0, :].astype(np.float64)
    if len(points) < 3:
        return 0.0
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    # Image Y points down; tray Y points up.
    angle = math.degrees(math.atan2(-axis[1], axis[0]))
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return float(angle)


def estimate_pose(
    item: SegmentedObject,
    shape_id: str,
    calibration: PerspectiveCalibration,
) -> tuple[Point2D, float | None, float, tuple[int, int]]:
    point_px, radius_px = safe_grasp_point(item.mask)
    center_mm = calibration.pixel_to_mm(*point_px)
    _, _, width, height = item.bbox
    expected_radius = max(1.0, 0.5 * min(width, height))
    centeredness = float(np.clip(radius_px / expected_radius, 0.0, 1.0))
    pose_confidence = 0.65 + 0.35 * centeredness
    angle = None if shape_id in SYMMETRIC_SHAPES else principal_angle_deg(item.contour)
    return center_mm, angle, pose_confidence, point_px


def masked_crop(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    x, y, width, height = bbox
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1 = min(image.shape[1], x + width + padding)
    y1 = min(image.shape[0], y + height + padding)
    crop = image[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1].copy()
    neutral = np.full_like(crop, 245)
    neutral[crop_mask > 0] = crop[crop_mask > 0]
    return neutral, crop_mask

