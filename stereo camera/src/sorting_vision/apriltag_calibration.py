from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .dual_view import DualCalibrationMetrics, DualViewCalibration
from .rgbd import CameraIntrinsics, RGBDFrame, depth_to_points, fit_plane_ransac


@dataclass(frozen=True)
class AprilTagObservation:
    corners_by_id: dict[int, np.ndarray]

    def has(self, *tag_ids: int) -> bool:
        return all(tag_id in self.corners_by_id for tag_id in tag_ids)


def _aruco_dictionary(dictionary_name: str) -> Any:
    aruco = getattr(cv2, "aruco", None)
    if aruco is None or not hasattr(aruco, dictionary_name):
        raise RuntimeError(
            f"OpenCV contrib with {dictionary_name} is required for AprilTag calibration"
        )
    return aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))


@lru_cache(maxsize=8)
def _apriltag_detector(dictionary_name: str) -> Any:
    """Reuse the relatively expensive detector configuration between frames."""
    aruco = cv2.aruco
    return aruco.ArucoDetector(
        _aruco_dictionary(dictionary_name), aruco.DetectorParameters()
    )


def detect_apriltags(
    image_bgr: np.ndarray,
    dictionary_name: str = "DICT_APRILTAG_36h11",
    maximum_detection_width: int | None = None,
) -> AprilTagObservation:
    image = np.asarray(image_bgr)
    scale = 1.0
    detection_image = image
    if maximum_detection_width is not None and image.shape[1] > maximum_detection_width:
        scale = float(maximum_detection_width) / float(image.shape[1])
        detection_image = cv2.resize(
            image,
            (maximum_detection_width, max(1, int(round(image.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    corners, identifiers, _ = _apriltag_detector(dictionary_name).detectMarkers(
        detection_image
    )
    if identifiers is None:
        return AprilTagObservation({})
    full_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    refined: list[np.ndarray] = []
    for corner in corners:
        values = np.asarray(corner, np.float32).reshape(4, 2) / scale
        if (
            np.all(values[:, 0] >= 6)
            and np.all(values[:, 0] < image.shape[1] - 6)
            and np.all(values[:, 1] >= 6)
            and np.all(values[:, 1] < image.shape[0] - 6)
        ):
            cv2.cornerSubPix(
                full_gray,
                values.reshape(-1, 1, 2),
                (5, 5),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01),
            )
        refined.append(values)
    return AprilTagObservation(
        {
            int(identifier): corner
            for corner, identifier in zip(refined, identifiers.reshape(-1))
        }
    )


def draw_apriltags(
    image_bgr: np.ndarray,
    observation: AprilTagObservation,
) -> np.ndarray:
    canvas = np.asarray(image_bgr).copy()
    if observation.corners_by_id:
        identifiers = np.asarray(sorted(observation.corners_by_id), np.int32).reshape(-1, 1)
        corners = [
            observation.corners_by_id[int(identifier)].reshape(1, 4, 2)
            for identifier in identifiers.reshape(-1)
        ]
        cv2.aruco.drawDetectedMarkers(canvas, corners, identifiers)
    return canvas


def generate_three_tag_assets(
    output_dir: str | Path,
    *,
    fixed_tag_ids: tuple[int, int] = (0, 1),
    free_tag_id: int = 2,
    tag_size_mm: float,
    dictionary_name: str = "DICT_APRILTAG_36h11",
    marker_pixels: int = 1000,
) -> list[Path]:
    """Generate the three marker bitmaps and a non-scale placement preview.

    ``tag_size_mm`` is the required measured width of the black marker square
    after printing. The layout image is explanatory and must not be used as a
    dimensionally accurate print sheet.
    """
    if free_tag_id in fixed_tag_ids or fixed_tag_ids[0] == fixed_tag_ids[1]:
        raise ValueError("the three AprilTag IDs must be distinct")
    if tag_size_mm <= 0 or marker_pixels < 100:
        raise ValueError("tag size and marker resolution must be positive")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    dictionary = _aruco_dictionary(dictionary_name)
    generated: list[Path] = []
    marker_images: dict[int, np.ndarray] = {}
    for tag_id in (*fixed_tag_ids, free_tag_id):
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, marker_pixels)
        marker_images[tag_id] = marker
        quiet_zone = max(20, marker_pixels // 4)
        printable = cv2.copyMakeBorder(
            marker,
            quiet_zone,
            quiet_zone,
            quiet_zone,
            quiet_zone,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        path = target / f"apriltag-{tag_id}-{dictionary_name.lower()}.png"
        if not cv2.imwrite(str(path), printable):
            raise OSError(f"failed to write AprilTag image: {path}")
        generated.append(path)

    layout = np.full((900, 1200, 3), 245, np.uint8)
    cv2.rectangle(layout, (100, 90), (1100, 810), (70, 70, 70), 8)
    positions = {
        fixed_tag_ids[0]: (145, 135),
        fixed_tag_ids[1]: (855, 565),
    }
    display_size = 200
    for tag_id, (x, y) in positions.items():
        marker = cv2.resize(
            marker_images[tag_id], (display_size, display_size),
            interpolation=cv2.INTER_NEAREST,
        )
        layout[y : y + display_size, x : x + display_size] = cv2.cvtColor(
            marker, cv2.COLOR_GRAY2BGR
        )
        cv2.putText(
            layout, f"FIXED {tag_id}", (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (30, 30, 220), 2, cv2.LINE_AA,
        )
    free = cv2.resize(
        marker_images[free_tag_id], (display_size, display_size),
        interpolation=cv2.INTER_NEAREST,
    )
    rotation = cv2.getRotationMatrix2D((display_size / 2, display_size / 2), 22, 1)
    free = cv2.warpAffine(free, rotation, (display_size, display_size), borderValue=255)
    layout[350:550, 500:700] = cv2.cvtColor(free, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        layout, f"FREE {free_tag_id}: move / rotate / tilt", (430, 330),
        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 120, 20), 2, cv2.LINE_AA,
    )
    cv2.arrowedLine(layout, (350, 150), (790, 150), (230, 80, 20), 4)
    cv2.putText(
        layout, "+X: printed top edges of both fixed tags", (410, 135),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 80, 20), 2, cv2.LINE_AA,
    )
    cv2.putText(
        layout, "PLACEMENT PREVIEW ONLY - NOT TO SCALE", (325, 865),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 180), 2, cv2.LINE_AA,
    )
    layout_path = target / "three-apriltag-layout-preview.png"
    if not cv2.imwrite(str(layout_path), layout):
        raise OSError(f"failed to write AprilTag layout: {layout_path}")
    generated.append(layout_path)
    metadata_path = target / "three-apriltag-printing.json"
    metadata_path.write_text(
        json.dumps(
            {
                "dictionary": dictionary_name,
                "fixed_tag_ids": list(fixed_tag_ids),
                "free_tag_id": free_tag_id,
                "black_square_width_mm": tag_size_mm,
                "marker_pixels": marker_pixels,
                "quiet_zone_pixels_each_side": max(20, marker_pixels // 4),
                "printing_rule": "print without rescaling; verify the black square with calipers",
                "layout_is_to_scale": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    generated.append(metadata_path)
    return generated


def tag_object_points(tag_size_mm: float) -> np.ndarray:
    half = float(tag_size_mm) * 0.5
    return np.asarray(
        [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]],
        np.float32,
    )


def _camera_matrix(intrinsics: CameraIntrinsics) -> np.ndarray:
    return np.asarray(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        np.float64,
    )


def _intrinsics_from_matrix(
    matrix: np.ndarray, image_size: tuple[int, int]
) -> CameraIntrinsics:
    return CameraIntrinsics(
        image_size[0], image_size[1],
        float(matrix[0, 0]), float(matrix[1, 1]),
        float(matrix[0, 2]), float(matrix[1, 2]), 1.0,
    )


def _tag_pose(
    corners: np.ndarray,
    tag_size_mm: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    solved, rotation_vector, translation = cv2.solvePnP(
        tag_object_points(tag_size_mm),
        np.asarray(corners, np.float32),
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        raise ValueError("AprilTag pose solve failed")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    return rotation, translation.reshape(3)


def estimate_tray_frame_from_diagonal_tags(
    observation: AprilTagObservation,
    primary_intrinsics: CameraIntrinsics,
    primary_distortion: np.ndarray,
    *,
    fixed_tag_ids: tuple[int, int],
    tag_size_mm: float,
    tray_width_mm: float,
    tray_height_mm: float,
    fixed_tag_inset_mm: float,
) -> tuple[np.ndarray, float]:
    """Build tray coordinates from two aligned tags fixed at opposite corners.

    Both printed tags must have the same orientation: their top edges point
    along tray +X. Tag A is near (inset, inset), tag B near the opposite corner.
    """
    if not observation.has(*fixed_tag_ids):
        raise ValueError("both fixed diagonal AprilTags must be visible")
    matrix = _camera_matrix(primary_intrinsics)
    first_rotation, first_center = _tag_pose(
        observation.corners_by_id[fixed_tag_ids[0]],
        tag_size_mm,
        matrix,
        primary_distortion,
    )
    second_rotation, second_center = _tag_pose(
        observation.corners_by_id[fixed_tag_ids[1]],
        tag_size_mm,
        matrix,
        primary_distortion,
    )
    if abs(float(np.dot(first_rotation[:, 2], second_rotation[:, 2]))) < np.cos(np.deg2rad(10)):
        raise ValueError("fixed AprilTag planes disagree by more than 10 degrees")
    x_axis = first_rotation[:, 0] / np.linalg.norm(first_rotation[:, 0])
    y_axis = first_rotation[:, 1] - np.dot(first_rotation[:, 1], x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)
    diagonal = second_center - first_center
    if float(np.dot(diagonal, x_axis)) < 0:
        x_axis = -x_axis
    if float(np.dot(diagonal, y_axis)) < 0:
        y_axis = -y_axis
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    expected_x = tray_width_mm - 2.0 * fixed_tag_inset_mm
    expected_y = tray_height_mm - 2.0 * fixed_tag_inset_mm
    if expected_x <= 0 or expected_y <= 0:
        raise ValueError("fixed tag inset leaves no valid tray diagonal")
    predicted_second = first_center + expected_x * x_axis + expected_y * y_axis
    expected_diagonal = float(np.hypot(expected_x, expected_y))
    scale_error = float(
        np.linalg.norm(predicted_second - second_center) / max(expected_diagonal, 1e-6)
    )
    origin = (
        first_center
        - fixed_tag_inset_mm * x_axis
        - fixed_tag_inset_mm * y_axis
    )
    primary_from_tray = np.eye(4, dtype=np.float64)
    primary_from_tray[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    primary_from_tray[:3, 3] = origin
    return np.linalg.inv(primary_from_tray), scale_error


def _epipolar_p95(
    primary_points: list[np.ndarray],
    side_points: list[np.ndarray],
    fundamental: np.ndarray,
) -> float:
    errors: list[float] = []
    for primary, side in zip(primary_points, side_points):
        first = primary.reshape(-1, 2)
        second = side.reshape(-1, 2)
        first_h = np.column_stack((first, np.ones(len(first))))
        second_h = np.column_stack((second, np.ones(len(second))))
        second_lines = (fundamental @ first_h.T).T
        first_lines = (fundamental.T @ second_h.T).T
        errors.extend(
            np.abs(np.sum(second_lines * second_h, axis=1))
            / np.maximum(np.linalg.norm(second_lines[:, :2], axis=1), 1e-9)
        )
        errors.extend(
            np.abs(np.sum(first_lines * first_h, axis=1))
            / np.maximum(np.linalg.norm(first_lines[:, :2], axis=1), 1e-9)
        )
    return float(np.percentile(errors, 95))


def _stereo_tag_scale_error(
    primary_points: list[np.ndarray],
    side_points: list[np.ndarray],
    primary_matrix: np.ndarray,
    primary_distortion: np.ndarray,
    side_matrix: np.ndarray,
    side_distortion: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    tag_size_mm: float,
) -> float:
    errors: list[float] = []
    first_projection = np.hstack((np.eye(3), np.zeros((3, 1))))
    second_projection = np.hstack((rotation, translation.reshape(3, 1)))
    for primary, side in zip(primary_points, side_points):
        first = cv2.undistortPoints(primary, primary_matrix, primary_distortion).reshape(-1, 2).T
        second = cv2.undistortPoints(side, side_matrix, side_distortion).reshape(-1, 2).T
        homogeneous = cv2.triangulatePoints(first_projection, second_projection, first, second)
        points = (homogeneous[:3] / homogeneous[3]).T
        for first_index, second_index in ((0, 1), (1, 2), (2, 3), (3, 0)):
            measured = float(np.linalg.norm(points[first_index] - points[second_index]))
            errors.append(abs(measured / tag_size_mm - 1.0))
    return float(np.median(errors))


def calibrate_apriltag_pairs(
    primary_images: Iterable[np.ndarray],
    side_images: Iterable[np.ndarray],
    reference_primary_frame: RGBDFrame,
    fixed_reference_image: np.ndarray,
    *,
    platform_id: str,
    tag_size_mm: float,
    fixed_tag_ids: tuple[int, int] = (0, 1),
    free_tag_id: int = 2,
    tray_width_mm: float = 160.0,
    tray_height_mm: float = 160.0,
    fixed_tag_inset_mm: float | None = None,
    dictionary_name: str = "DICT_APRILTAG_36h11",
) -> DualViewCalibration:
    """Calibrate two cameras with two fixed diagonal tags and one moving tag."""
    if free_tag_id in fixed_tag_ids or fixed_tag_ids[0] == fixed_tag_ids[1]:
        raise ValueError("the three AprilTag IDs must be distinct")
    if tag_size_mm <= 0:
        raise ValueError("tag_size_mm must be positive")
    primary_values = list(primary_images)
    side_values = list(side_images)
    if len(primary_values) != len(side_values) or len(primary_values) < 20:
        raise ValueError("at least 20 paired free-AprilTag poses are required")
    primary_size = (primary_values[0].shape[1], primary_values[0].shape[0])
    side_size = (side_values[0].shape[1], side_values[0].shape[0])
    if primary_size != (
        reference_primary_frame.intrinsics.width,
        reference_primary_frame.intrinsics.height,
    ):
        raise ValueError("reference RGB-D dimensions differ from primary calibration images")
    primary_points: list[np.ndarray] = []
    side_points: list[np.ndarray] = []
    for primary_image, side_image in zip(primary_values, side_values):
        primary_observation = detect_apriltags(primary_image, dictionary_name)
        side_observation = detect_apriltags(side_image, dictionary_name)
        if primary_observation.has(free_tag_id) and side_observation.has(free_tag_id):
            primary_points.append(
                primary_observation.corners_by_id[free_tag_id].reshape(-1, 1, 2)
            )
            side_points.append(
                side_observation.corners_by_id[free_tag_id].reshape(-1, 1, 2)
            )
    if len(primary_points) < 20:
        raise ValueError("fewer than 20 usable paired free-AprilTag observations")
    object_template = tag_object_points(tag_size_mm).reshape(-1, 1, 3)
    object_points = [object_template.copy() for _ in primary_points]
    initial_primary = _camera_matrix(reference_primary_frame.intrinsics)
    primary_rms, primary_matrix, primary_distortion, _, _ = cv2.calibrateCamera(
        object_points,
        primary_points,
        primary_size,
        initial_primary,
        np.zeros(5, np.float64),
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )
    side_rms, side_matrix, side_distortion, _, _ = cv2.calibrateCamera(
        object_points, side_points, side_size, None, None
    )
    _, primary_matrix, primary_distortion, side_matrix, side_distortion, rotation, translation, _, fundamental = cv2.stereoCalibrate(
        object_points,
        primary_points,
        side_points,
        primary_matrix,
        primary_distortion,
        side_matrix,
        side_distortion,
        primary_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    primary_intrinsics = _intrinsics_from_matrix(primary_matrix, primary_size)
    side_intrinsics = _intrinsics_from_matrix(side_matrix, side_size)
    fixed_observation = detect_apriltags(fixed_reference_image, dictionary_name)
    inset = tag_size_mm * 0.5 if fixed_tag_inset_mm is None else fixed_tag_inset_mm
    tray_from_primary, diagonal_scale_error = estimate_tray_frame_from_diagonal_tags(
        fixed_observation,
        primary_intrinsics,
        primary_distortion,
        fixed_tag_ids=fixed_tag_ids,
        tag_size_mm=tag_size_mm,
        tray_width_mm=tray_width_mm,
        tray_height_mm=tray_height_mm,
        fixed_tag_inset_mm=inset,
    )
    primary_from_tray = np.linalg.inv(tray_from_primary)
    tray_corners = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [tray_width_mm, 0.0, 0.0],
            [tray_width_mm, tray_height_mm, 0.0],
            [0.0, tray_height_mm, 0.0],
        ],
        np.float64,
    )
    projected_tray, _ = cv2.projectPoints(
        tray_corners,
        cv2.Rodrigues(primary_from_tray[:3, :3])[0],
        primary_from_tray[:3, 3],
        primary_matrix,
        primary_distortion,
    )
    tray_mask = np.zeros(reference_primary_frame.depth_mm.shape, np.uint8)
    tray_polygon = np.rint(projected_tray.reshape(-1, 2)).astype(np.int32)
    cv2.fillConvexPoly(tray_mask, tray_polygon, 255)
    points, _ = depth_to_points(
        reference_primary_frame.depth_mm,
        reference_primary_frame.intrinsics,
        mask=tray_mask,
        stride=8,
    )
    tray_plane = fit_plane_ransac(points, threshold_mm=1.5)
    stereo_scale_error = _stereo_tag_scale_error(
        primary_points,
        side_points,
        primary_matrix,
        primary_distortion,
        side_matrix,
        side_distortion,
        rotation,
        translation,
        tag_size_mm,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.reshape(3)
    version_data = np.concatenate((rotation.reshape(-1), translation.reshape(-1)))
    version = time.strftime("%Y%m%d-%H%M%S") + "-apriltag-" + hashlib.sha256(
        version_data.tobytes()
    ).hexdigest()[:8]
    return DualViewCalibration(
        primary_intrinsics=primary_intrinsics,
        primary_distortion=primary_distortion,
        side_intrinsics=side_intrinsics,
        side_distortion=side_distortion,
        side_from_primary=transform,
        metrics=DualCalibrationMetrics(
            float(primary_rms),
            float(side_rms),
            _epipolar_p95(primary_points, side_points, fundamental),
            max(stereo_scale_error, diagonal_scale_error),
        ),
        platform_id=platform_id,
        version=version,
        board={
            "type": "AprilTag three-tag stereo",
            "dictionary": dictionary_name,
            "tag_size_mm": tag_size_mm,
            "fixed_tag_ids": list(fixed_tag_ids),
            "free_tag_id": free_tag_id,
            "fixed_tag_inset_mm": inset,
            "tray_width_mm": tray_width_mm,
            "tray_height_mm": tray_height_mm,
            "paired_pose_count": len(primary_points),
            "placement_rule": "fixed tags aligned at opposite tray corners; free tag moved/rotated",
        },
        primary_image_size=primary_size,
        tray_plane_primary=tray_plane,
        tray_from_primary=tray_from_primary,
    )


def free_tag_pose_signature(corners: np.ndarray) -> np.ndarray:
    values = np.asarray(corners, np.float64).reshape(4, 2)
    center = values.mean(axis=0)
    edge = values[1] - values[0]
    angle = np.arctan2(edge[1], edge[0])
    area = abs(float(cv2.contourArea(values.astype(np.float32))))
    return np.asarray([center[0], center[1], np.log(max(area, 1.0)), angle])


def pose_is_diverse(
    existing_signatures: list[np.ndarray],
    candidate: np.ndarray,
    minimum_center_px: float = 28.0,
    minimum_log_area: float = 0.12,
    minimum_angle_deg: float = 10.0,
) -> bool:
    candidate = np.asarray(candidate, np.float64).reshape(-1)
    if candidate.size == 0 or candidate.size % 4:
        raise ValueError("pose signature must contain one or more four-value views")
    if not existing_signatures:
        return True
    for existing in existing_signatures:
        existing = np.asarray(existing, np.float64).reshape(-1)
        if existing.size != candidate.size:
            raise ValueError("all pose signatures must have the same number of views")
        view_is_similar: list[bool] = []
        for offset in range(0, candidate.size, 4):
            current_view = candidate[offset : offset + 4]
            previous_view = existing[offset : offset + 4]
            center_change = float(
                np.linalg.norm(current_view[:2] - previous_view[:2])
            )
            area_change = abs(float(current_view[2] - previous_view[2]))
            angle_change = abs(float(np.arctan2(
                np.sin(current_view[3] - previous_view[3]),
                np.cos(current_view[3] - previous_view[3]),
            )))
            view_is_similar.append(
                center_change < minimum_center_px
                and area_change < minimum_log_area
                and angle_change < np.deg2rad(minimum_angle_deg)
            )
        if all(view_is_similar):
            return False
    return True
