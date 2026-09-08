from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import RGBDConfig
from .rgbd import CameraIntrinsics, Plane, backproject_pixels, depth_to_points, fit_plane_svd


@dataclass
class DepthSegmentedObject:
    mask: np.ndarray
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    area: float
    height_min_mm: float
    height_max_mm: float
    valid_depth_ratio: float
    segmentation_confidence: float
    touches_border: bool = False
    clearance_px: float = float("inf")


def height_map_from_plane(
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    plane: Plane,
) -> np.ndarray:
    depth = np.asarray(depth_mm, dtype=np.float32)
    rows, columns = np.indices(depth.shape, dtype=np.float32)
    x = (columns - intrinsics.cx) * depth / intrinsics.fx
    y = (rows - intrinsics.cy) * depth / intrinsics.fy
    return (
        x * plane.normal[0]
        + y * plane.normal[1]
        + depth * plane.normal[2]
        + plane.offset
    ).astype(np.float32)


def valid_depth_mask(depth_mm: np.ndarray, cfg: RGBDConfig) -> np.ndarray:
    depth = np.asarray(depth_mm)
    return (
        np.isfinite(depth)
        & (depth >= cfg.min_depth_mm)
        & (depth <= cfg.max_depth_mm)
    )


def estimate_plane_shift_mm(
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    plane: Plane,
    cfg: RGBDConfig,
    heights: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
) -> float:
    if heights is None:
        heights = height_map_from_plane(depth_mm, intrinsics, plane)
    sample_stride = 4
    sampled_depth = depth_mm[::sample_stride, ::sample_stride]
    sampled_heights = heights[::sample_stride, ::sample_stride]
    valid = valid_depth_mask(sampled_depth, cfg)
    if roi_mask is not None:
        roi = np.asarray(roi_mask) > 0
        if roi.shape != depth_mm.shape:
            raise ValueError("plane-shift ROI mask must match the depth image")
        valid &= roi[::sample_stride, ::sample_stride]
    near_plane = valid & (
        np.abs(sampled_heights) <= max(10.0, cfg.foreground_height_mm * 2)
    )
    if int(near_plane.sum()) < 100:
        return float("inf")
    return float(np.median(sampled_heights[near_plane]))


def _split_touching(
    color: np.ndarray,
    component: np.ndarray,
    min_area: int,
) -> list[np.ndarray]:
    distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    maximum = float(distance.max())
    if maximum <= 0:
        return [component]
    peaks = (distance >= maximum * 0.65).astype(np.uint8)
    count, markers = cv2.connectedComponents(peaks)
    if count <= 2:
        return [component]
    sure_background = cv2.dilate(component, np.ones((3, 3), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_background, peaks * 255)
    markers = markers.astype(np.int32) + 1
    markers[unknown > 0] = 0
    cv2.watershed(color.copy(), markers)
    parts: list[np.ndarray] = []
    for marker_id in range(2, int(markers.max()) + 1):
        part = ((markers == marker_id) & (component > 0)).astype(np.uint8) * 255
        if cv2.countNonZero(part) >= min_area:
            parts.append(part)
    return parts if len(parts) >= 2 else [component]


def _merge_fragmented_components(
    masks: list[np.ndarray],
    color_bgr: np.ndarray,
    heights: np.ndarray,
    maximum_gap_px: int,
) -> list[np.ndarray]:
    """Join a small detached face fragment back to its main solid.

    RealSense grazing surfaces can fall below the height threshold and split one
    coloured solid.  The merge is deliberately asymmetric and requires close,
    strongly aligned bounding boxes, similar hue, and overlapping height ranges;
    similarly sized neighbouring solids are therefore left separate.
    """
    merged = [mask.copy() for mask in masks]
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    changed = True
    while changed:
        changed = False
        for first_index in range(len(merged)):
            if changed:
                break
            first = merged[first_index]
            first_area = cv2.countNonZero(first)
            first_box = cv2.boundingRect(first)
            for second_index in range(first_index + 1, len(merged)):
                second = merged[second_index]
                second_area = cv2.countNonZero(second)
                area_ratio = min(first_area, second_area) / max(first_area, second_area)
                if area_ratio > 0.4:
                    continue
                x1, y1, w1, h1 = first_box
                x2, y2, w2, h2 = cv2.boundingRect(second)
                gap_x = max(x1 - (x2 + w2), x2 - (x1 + w1), 0)
                gap_y = max(y1 - (y2 + h2), y2 - (y1 + h1), 0)
                if float(np.hypot(gap_x, gap_y)) > maximum_gap_px:
                    continue
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                aligned = (
                    overlap_x / max(1, min(w1, w2)) >= 0.45
                    or overlap_y / max(1, min(h1, h2)) >= 0.45
                )
                if not aligned:
                    continue
                first_pixels = first > 0
                second_pixels = second > 0
                first_heights = heights[first_pixels]
                second_heights = heights[second_pixels]
                if not len(first_heights) or not len(second_heights):
                    continue
                first_range = np.percentile(first_heights, [10, 90])
                second_range = np.percentile(second_heights, [10, 90])
                if min(first_range[1], second_range[1]) + 2.0 < max(
                    first_range[0], second_range[0]
                ):
                    continue
                hue_first = float(np.median(hsv[:, :, 0][first_pixels]))
                hue_second = float(np.median(hsv[:, :, 0][second_pixels]))
                hue_delta = abs(hue_first - hue_second)
                hue_delta = min(hue_delta, 180.0 - hue_delta)
                if hue_delta > 12.0:
                    continue
                union = cv2.bitwise_or(first, second)
                size = max(3, maximum_gap_px * 2 + 1)
                bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                closed = cv2.morphologyEx(union, cv2.MORPH_CLOSE, bridge)
                added = (closed > 0) & (union == 0)
                # Nearby faces are not sufficient evidence of a single solid:
                # the connecting pixels must also look like that solid in RGB.
                # A white tray gap (including a cool-tinted one) must stay open.
                if not np.any(added):
                    continue
                reference_saturation = min(
                    float(np.median(hsv[:, :, 1][first_pixels])),
                    float(np.median(hsv[:, :, 1][second_pixels])),
                )
                if reference_saturation < 60:
                    continue  # achromatic fragments need other evidence
                added_hue = hsv[:, :, 0][added].astype(np.float32)
                hue_distance = np.abs(added_hue - hue_first)
                hue_distance = np.minimum(hue_distance, 180 - hue_distance)
                supported = (
                    (hue_distance <= 12)
                    & (hsv[:, :, 1][added] >= max(60, reference_saturation * 0.6))
                    & (hsv[:, :, 2][added] >= 20)
                )
                if float(np.mean(supported)) < 0.85:
                    continue
                # Preserve only RGB-supported added pixels, and require that
                # they actually connect the two pieces. Depth is never filled.
                allowed_added = np.zeros_like(union)
                allowed_added[added] = supported.astype(np.uint8) * 255
                joined = cv2.bitwise_or(union, allowed_added)
                if cv2.connectedComponents(joined)[0] != 2:
                    continue
                merged[first_index] = joined
                del merged[second_index]
                changed = True
                break
    return merged


def _make_object(
    component: np.ndarray,
    depth_valid: np.ndarray,
    heights: np.ndarray,
    cfg: RGBDConfig,
) -> DepthSegmentedObject | None:
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if not cfg.min_area_px <= area <= cfg.max_area_px:
        return None
    x, y, width, height = cv2.boundingRect(contour)
    filled = np.zeros_like(component)
    cv2.drawContours(filled, [contour], -1, 255, -1)
    pixels = filled > 0
    valid_ratio = float(np.mean(depth_valid[pixels])) if np.any(pixels) else 0.0
    object_heights = heights[(component > 0) & depth_valid]
    if len(object_heights) == 0:
        return None
    hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
    solidity = min(1.0, area / hull_area)
    confidence = float(np.clip(0.55 + 0.3 * solidity + 0.15 * valid_ratio, 0.0, 1.0))
    margin = cfg.border_margin_px
    touches = (
        x <= margin
        or y <= margin
        or x + width >= component.shape[1] - margin
        or y + height >= component.shape[0] - margin
    )
    return DepthSegmentedObject(
        mask=component,
        contour=contour,
        bbox=(x, y, width, height),
        area=area,
        height_min_mm=float(np.percentile(object_heights, 5)),
        height_max_mm=float(np.percentile(object_heights, 95)),
        valid_depth_ratio=valid_ratio,
        segmentation_confidence=confidence,
        touches_border=touches,
    )


def _assign_clearance(objects: list[DepthSegmentedObject]) -> None:
    if len(objects) <= 1:
        return
    for index, current in enumerate(objects):
        x1, y1, width1, height1 = current.bbox
        nearest = float("inf")
        for other_index, other in enumerate(objects):
            if index == other_index:
                continue
            x2, y2, width2, height2 = other.bbox
            dx = max(x1 - (x2 + width2), x2 - (x1 + width1), 0)
            dy = max(y1 - (y2 + height2), y2 - (y1 + height1), 0)
            nearest = min(nearest, float(np.hypot(dx, dy)))
        current.clearance_px = nearest


def segment_depth_objects(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    tray_plane: Plane,
    cfg: RGBDConfig,
    heights: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
    split_touching_objects: bool = True,
) -> tuple[list[DepthSegmentedObject], np.ndarray]:
    valid = valid_depth_mask(depth_mm, cfg)
    if heights is None:
        heights = height_map_from_plane(depth_mm, intrinsics, tray_plane)
    foreground = (
        valid
        & (heights >= cfg.foreground_height_mm)
        & (heights <= cfg.max_object_height_mm)
    ).astype(np.uint8) * 255
    if roi_mask is not None:
        roi = np.asarray(roi_mask) > 0
        if roi.shape != foreground.shape:
            raise ValueError("RGB-D ROI mask must match the depth image")
        foreground[~roi] = 0
    if support_mask is not None:
        support = np.asarray(support_mask) > 0
        if support.shape != foreground.shape:
            raise ValueError("RGB-D support mask must match the depth image")
        foreground[~support] = 0
    kernel_size = max(3, cfg.morphology_kernel | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)

    count, labels = cv2.connectedComponents(foreground)
    masks: list[np.ndarray] = []
    components: list[np.ndarray] = []
    for label in range(1, count):
        component = (labels == label).astype(np.uint8) * 255
        if cv2.countNonZero(component) < cfg.min_area_px:
            continue
        components.append(component)
    components = _merge_fragmented_components(
        components, color_bgr, heights, maximum_gap_px=kernel_size * 2,
    )
    for component in components:
        masks.extend(
            _split_touching(color_bgr, component, cfg.min_area_px)
            if split_touching_objects else [component]
        )
    objects = [
        item
        for mask in masks
        if (item := _make_object(mask, valid, heights, cfg)) is not None
    ]
    _assign_clearance(objects)
    return sorted(objects, key=lambda item: (item.bbox[1], item.bbox[0])), heights


def object_point_cloud(
    item: DepthSegmentedObject,
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    return depth_to_points(depth_mm, intrinsics, item.mask, stride=stride)


def local_surface_plane(
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    center_uv: tuple[int, int],
    radius_px: int,
    allowed_mask: np.ndarray,
) -> tuple[Plane, np.ndarray, float] | None:
    center_u, center_v = center_uv
    u0, u1 = max(0, center_u - radius_px), min(depth_mm.shape[1], center_u + radius_px + 1)
    v0, v1 = max(0, center_v - radius_px), min(depth_mm.shape[0], center_v + radius_px + 1)
    local_depth = depth_mm[v0:v1, u0:u1]
    local_allowed = allowed_mask[v0:v1, u0:u1] > 0
    local_rows, local_columns = np.indices(local_depth.shape)
    global_rows = local_rows + v0
    global_columns = local_columns + u0
    disk = (
        (global_columns - center_u) ** 2 + (global_rows - center_v) ** 2
        <= radius_px * radius_px
    )
    requested = disk & local_allowed
    valid = requested & np.isfinite(local_depth) & (local_depth > 0)
    requested_count = int(requested.sum())
    if requested_count < 3 or int(valid.sum()) < 3:
        return None
    rows = global_rows[valid]
    columns = global_columns[valid]
    pixels = np.column_stack((columns, rows)).astype(np.float64)
    points = backproject_pixels(pixels, local_depth[valid], intrinsics)
    plane = fit_plane_svd(points)
    valid_ratio = float(valid.sum() / requested_count)
    return plane, points, valid_ratio
