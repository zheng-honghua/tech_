from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import GraspConfig
from .geometry3d import DepthSegmentedObject, local_surface_plane
from .rgbd import RGBDCalibration
from .types import GraspInfo, Point3D, Pose3D, Quaternion


@dataclass(frozen=True)
class GraspCandidate:
    pose: Pose3D
    info: GraspInfo
    pixel_uv: tuple[int, int]
    pose_confidence: float


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> Quaternion:
    value = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(value))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (value[2, 1] - value[1, 2]) / scale
        y = (value[0, 2] - value[2, 0]) / scale
        z = (value[1, 0] - value[0, 1]) / scale
    else:
        diagonal = np.diag(value)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + value[0, 0] - value[1, 1] - value[2, 2]) * 2.0
            w = (value[2, 1] - value[1, 2]) / scale
            x = 0.25 * scale
            y = (value[0, 1] + value[1, 0]) / scale
            z = (value[0, 2] + value[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + value[1, 1] - value[0, 0] - value[2, 2]) * 2.0
            w = (value[0, 2] - value[2, 0]) / scale
            x = (value[0, 1] + value[1, 0]) / scale
            y = 0.25 * scale
            z = (value[1, 2] + value[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + value[2, 2] - value[0, 0] - value[1, 1]) * 2.0
            w = (value[1, 0] - value[0, 1]) / scale
            x = (value[0, 2] + value[2, 0]) / scale
            y = (value[1, 2] + value[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    quaternion /= max(np.linalg.norm(quaternion), 1e-12)
    return Quaternion(*map(float, quaternion))


def _tool_orientation(approach_robot: np.ndarray, calibration: RGBDCalibration) -> Quaternion:
    z_axis = np.asarray(approach_robot, dtype=np.float64)
    z_axis /= max(np.linalg.norm(z_axis), 1e-12)
    reference = calibration.transform_vectors(np.array([[1.0, 0.0, 0.0]]))[0]
    x_axis = reference - z_axis * float(reference @ z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        reference = calibration.transform_vectors(np.array([[0.0, 1.0, 0.0]]))[0]
        x_axis = reference - z_axis * float(reference @ z_axis)
    x_axis /= max(np.linalg.norm(x_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-12)
    x_axis = np.cross(y_axis, z_axis)
    return rotation_matrix_to_quaternion(np.column_stack((x_axis, y_axis, z_axis)))


def _surface_point(
    center_uv: tuple[int, int], normal: np.ndarray, offset: float, calibration: RGBDCalibration
) -> np.ndarray | None:
    u, v = center_uv
    intrinsics = calibration.intrinsics
    ray = np.array(
        [(u - intrinsics.cx) / intrinsics.fx, (v - intrinsics.cy) / intrinsics.fy, 1.0],
        dtype=np.float64,
    )
    denominator = float(normal @ ray)
    if abs(denominator) < 1e-9:
        return None
    scale = -offset / denominator
    if scale <= 0:
        return None
    return ray * scale


def find_suction_grasp(
    item: DepthSegmentedObject,
    depth_mm: np.ndarray,
    calibration: RGBDCalibration,
    cfg: GraspConfig,
) -> GraspCandidate | None:
    intrinsics = calibration.intrinsics
    object_depths = depth_mm[(item.mask > 0) & np.isfinite(depth_mm) & (depth_mm > 0)]
    if len(object_depths) < 20:
        return None
    nominal_depth = float(np.mean(object_depths))
    focal = 0.5 * (intrinsics.fx + intrinsics.fy)
    cup_radius_px = max(2, int(math.ceil((cfg.cup_diameter_mm / 2.0) * focal / nominal_depth)))
    margin_px = max(1, int(math.ceil(cfg.edge_margin_mm * focal / nominal_depth)))
    required_clearance_px = cup_radius_px + margin_px
    distance = cv2.distanceTransform(item.mask, cv2.DIST_L2, 5)

    rows, columns = np.nonzero(distance >= required_clearance_px)
    if len(rows) == 0:
        return None
    order = np.argsort(distance[rows, columns])[::-1]
    candidates: list[tuple[int, int]] = []
    suppression_radius = max(2, cup_radius_px // 2)
    for ordered_index in order:
        candidate = (int(columns[ordered_index]), int(rows[ordered_index]))
        if any(np.hypot(candidate[0] - u, candidate[1] - v) < suppression_radius for u, v in candidates):
            continue
        candidates.append(candidate)
        if len(candidates) >= cfg.max_candidates:
            break

    best: GraspCandidate | None = None
    for center_uv in candidates:
        surface = local_surface_plane(
            depth_mm, intrinsics, center_uv, cup_radius_px, item.mask
        )
        if surface is None:
            continue
        plane, _, valid_ratio = surface
        normal_camera = plane.normal
        tilt_deg = math.degrees(
            math.acos(float(np.clip(normal_camera @ np.array([0.0, 0.0, -1.0]), -1.0, 1.0)))
        )
        if (
            plane.rmse_mm > cfg.max_flatness_rmse_mm
            or tilt_deg > cfg.max_tilt_deg
            or valid_ratio < cfg.min_patch_valid_ratio
        ):
            continue
        point_camera = _surface_point(center_uv, normal_camera, plane.offset, calibration)
        if point_camera is None:
            continue
        point_robot = calibration.transform_points(point_camera[None, :])[0]
        normal_robot = calibration.transform_vectors(normal_camera[None, :])[0]
        normal_robot /= max(np.linalg.norm(normal_robot), 1e-12)
        approach_robot = -normal_robot
        quaternion = _tool_orientation(approach_robot, calibration)

        edge_clearance_mm = float(distance[center_uv[1], center_uv[0]] * nominal_depth / focal)
        flatness_score = max(0.0, 1.0 - plane.rmse_mm / cfg.max_flatness_rmse_mm)
        tilt_score = max(0.0, 1.0 - tilt_deg / cfg.max_tilt_deg)
        edge_score = float(
            np.clip(
                (edge_clearance_mm - cfg.cup_diameter_mm / 2.0)
                / max(1.0, cfg.edge_margin_mm * 3.0),
                0.0,
                1.0,
            )
        )
        score = float(
            0.35 * flatness_score
            + 0.25 * tilt_score
            + 0.2 * valid_ratio
            + 0.2 * edge_score
        )
        info = GraspInfo(
            cup_diameter_mm=cfg.cup_diameter_mm,
            flatness_rmse_mm=plane.rmse_mm,
            edge_clearance_mm=edge_clearance_mm,
            valid_depth_ratio=valid_ratio,
            score=score,
        )
        pose = Pose3D(
            position_mm=Point3D.from_array(point_robot),
            quaternion_xyzw=quaternion,
            surface_normal=Point3D.from_array(normal_robot),
            approach_vector=Point3D.from_array(approach_robot),
        )
        pose_confidence = float(np.clip(0.5 * flatness_score + 0.5 * tilt_score, 0.0, 1.0))
        candidate = GraspCandidate(pose, info, center_uv, pose_confidence)
        if best is None or candidate.info.score > best.info.score:
            best = candidate
        if best.info.score >= 0.9:
            break
    return best
