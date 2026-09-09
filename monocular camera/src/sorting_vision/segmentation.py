from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from .config import SegmentationConfig
from .types import SegmentedObject


class InstanceSegmenter(Protocol):
    """Injection point for a trained lightweight instance segmenter."""

    def segment(self, image: np.ndarray) -> list[np.ndarray]: ...


def foreground_mask(
    image: np.ndarray,
    background: np.ndarray | None,
    cfg: SegmentationConfig,
    threshold_scale: float = 1.0,
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    if background is not None:
        if background.shape[:2] != image.shape[:2]:
            raise ValueError("background and image dimensions must match after rectification")
        bg_lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB).astype(np.float32)
        delta = np.linalg.norm(lab - bg_lab, axis=2)
    else:
        border = np.concatenate(
            [lab[0], lab[-1], lab[:, 0], lab[:, -1]], axis=0
        )
        reference = np.median(border, axis=0)
        delta = np.linalg.norm(lab - reference, axis=2)

    mask = (delta >= cfg.background_delta * threshold_scale).astype(np.uint8) * 255
    kernel_size = max(3, cfg.morphology_kernel | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _split_component(
    image: np.ndarray,
    component_mask: np.ndarray,
    cfg: SegmentationConfig,
) -> list[np.ndarray]:
    if not cfg.split_touching:
        return [component_mask]

    distance = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    maximum = float(distance.max())
    if maximum <= 0:
        return [component_mask]
    peaks = (distance >= maximum * cfg.watershed_peak_ratio).astype(np.uint8)
    count, markers = cv2.connectedComponents(peaks)
    if count <= 2:
        return [component_mask]

    sure_background = cv2.dilate(component_mask, np.ones((3, 3), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_background, peaks * 255)
    markers = markers.astype(np.int32) + 1
    markers[unknown > 0] = 0
    cv2.watershed(image.copy(), markers)

    parts: list[np.ndarray] = []
    for marker_id in range(2, int(markers.max()) + 1):
        part = (markers == marker_id).astype(np.uint8) * 255
        part = cv2.bitwise_and(part, component_mask)
        if cv2.countNonZero(part) >= cfg.min_area_px:
            parts.append(part)
    return parts if len(parts) >= 2 else [component_mask]


def _to_object(mask: np.ndarray, cfg: SegmentationConfig) -> SegmentedObject | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if not cfg.min_area_px <= area <= cfg.max_area_px:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    height, width = mask.shape
    margin = cfg.border_margin_px
    touches_border = x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin
    hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
    solidity = min(1.0, area / hull_area)
    confidence = float(np.clip(0.65 + 0.35 * solidity, 0.0, 1.0))
    return SegmentedObject(
        mask=mask,
        contour=contour,
        bbox=(x, y, w, h),
        area=area,
        segmentation_confidence=confidence,
        touches_border=touches_border,
    )


def segment_objects(
    image: np.ndarray,
    background: np.ndarray | None,
    cfg: SegmentationConfig,
    threshold_scale: float = 1.0,
    model: InstanceSegmenter | None = None,
) -> list[SegmentedObject]:
    if model is not None:
        masks = [
            (np.asarray(mask) > 0).astype(np.uint8) * 255
            for mask in model.segment(image)
        ]
    else:
        combined = foreground_mask(image, background, cfg, threshold_scale)
        component_count, labels = cv2.connectedComponents(combined)
        masks = []
        for label in range(1, component_count):
            component = (labels == label).astype(np.uint8) * 255
            if cv2.countNonZero(component) < cfg.min_area_px:
                continue
            masks.extend(_split_component(image, component, cfg))

    objects = [item for mask in masks if (item := _to_object(mask, cfg)) is not None]
    _assign_clearance(objects)
    return sorted(objects, key=lambda item: (item.bbox[1], item.bbox[0]))


def _assign_clearance(objects: list[SegmentedObject]) -> None:
    if len(objects) <= 1:
        return
    for index, current in enumerate(objects):
        x1, y1, w1, h1 = current.bbox
        nearest = float("inf")
        for other_index, other in enumerate(objects):
            if index == other_index:
                continue
            x2, y2, w2, h2 = other.bbox
            horizontal_gap = max(x1 - (x2 + w2), x2 - (x1 + w1), 0)
            vertical_gap = max(y1 - (y2 + h2), y2 - (y1 + h1), 0)
            nearest = min(nearest, float(np.hypot(horizontal_gap, vertical_gap)))
        current.clearance_px = nearest
