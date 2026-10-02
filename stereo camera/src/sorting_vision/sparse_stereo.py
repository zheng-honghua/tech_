"""Calibrated sparse RGB reconstruction. Evidence only, never grasp geometry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .side_geometry import _side_lsd_segments


@dataclass(frozen=True)
class SparseStereoLimits:
    epipolar_px: float = 3.0
    reprojection_px: float = 2.0
    minimum_ray_angle_deg: float = 2.0
    depth_consistency_mm: float = 5.0
    prior_side_distance_px: float = 8.0
    ambiguity_gap: float = 0.25
    maximum_features: int = 48

    def __post_init__(self) -> None:
        if any(not np.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise ValueError("sparse stereo limits must be finite and positive")
        if not isinstance(self.maximum_features, int):
            raise ValueError("maximum_features must be an integer")


def camera_matrix(intrinsics: Any) -> np.ndarray:
    return np.array([[intrinsics.fx, 0., intrinsics.cx],
                     [0., intrinsics.fy, intrinsics.cy], [0., 0., 1.]], np.float64)


def camera_poses(calibration: Any) -> dict[str, Any]:
    transform = np.asarray(calibration.side_from_primary, np.float64)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    center = -rotation.T @ translation
    result = {"coordinate_frame": "PRIMARY_CAMERA_MM", "primary_center_mm": [0., 0., 0.],
              "side_center_mm": center.tolist(), "side_orientation_in_primary": rotation.T.tolist(),
              "baseline_mm": float(np.linalg.norm(center)), "calibration_hash": calibration.calibration_hash}
    tray = calibration.tray_from_primary
    if tray is not None:
        result["primary_center_tray_mm"] = np.asarray(tray)[:3, 3].tolist()
        result["side_center_tray_mm"] = (np.asarray(tray) @ np.r_[center, 1.])[:3].tolist()
    return result


def normalized_pixels(pixels: np.ndarray, intrinsics: Any, distortion: np.ndarray) -> np.ndarray:
    return cv2.undistortPoints(np.asarray(pixels, np.float64).reshape(-1, 1, 2),
                               camera_matrix(intrinsics), np.asarray(distortion, np.float64)).reshape(-1, 2)


def project_primary(points: np.ndarray, calibration: Any) -> np.ndarray:
    pixels, _ = cv2.projectPoints(np.asarray(points, np.float64).reshape(-1, 3), np.zeros(3), np.zeros(3),
                                  camera_matrix(calibration.primary_intrinsics), calibration.primary_distortion)
    return pixels.reshape(-1, 2)


def _refine_position(point, primary_uv, side_uv, calibration):
    """Finite-difference Gauss Newton, retaining only residual improvements."""
    target = np.r_[primary_uv, side_uv]

    def residual(value):
        return np.r_[project_primary(value[None], calibration)[0],
                     calibration.project_primary_points(value[None])[0]] - target

    current = np.asarray(point, np.float64).copy()
    for _ in range(5):
        error = residual(current)
        step_mm = .01
        jacobian = np.column_stack([(residual(current + np.eye(3)[axis] * step_mm) - error) / step_mm
                                    for axis in range(3)])
        step = np.linalg.lstsq(jacobian, error, rcond=None)[0]
        if not np.isfinite(step).all() or np.linalg.norm(step) > 10.:
            break
        candidate = current - step
        side_z = (calibration.side_from_primary @ np.r_[candidate, 1.])[2]
        if candidate[2] <= 0 or side_z <= 0 or np.linalg.norm(residual(candidate)) > np.linalg.norm(error):
            break
        current = candidate
        if np.linalg.norm(step) < 1e-6:
            break
    return current


def _perturbation_sensitivity(primary_uv, side_uv, calibration, point):
    shifts = []
    pixels = np.r_[primary_uv, side_uv]
    for axis in range(4):
        for sign in (-1, 1):
            perturbed = pixels.copy()
            perturbed[axis] += sign * .5
            a = normalized_pixels(perturbed[:2], calibration.primary_intrinsics, calibration.primary_distortion)[0]
            b = normalized_pixels(perturbed[2:], calibration.side_intrinsics, calibration.side_distortion)[0]
            raw = cv2.triangulatePoints(np.eye(3, 4), calibration.side_from_primary[:3], a[:, None], b[:, None])[:, 0]
            if abs(raw[3]) > 1e-10:
                candidate = _refine_position(raw[:3] / raw[3], perturbed[:2], perturbed[2:], calibration)
                shifts.append(float(np.linalg.norm(candidate - point)))
    return {"pixel_perturbation_px": .5, "maximum_shift_mm": max(shifts, default=None),
            "median_shift_mm": float(np.median(shifts)) if shifts else None,
            "is_measured_accuracy": False, "method": "independent_pixel_axis_perturbation"}


def triangulate_corner(primary_uv: np.ndarray, side_uv: np.ndarray, calibration: Any,
                       limits: SparseStereoLimits, depth_prior_mm: np.ndarray | None = None,
                       *, refine_position: bool = False) -> dict[str, Any]:
    """Input pixels are native camera coordinates, not resized/cropped pixels."""
    if not np.all(np.isfinite(primary_uv)) or not np.all(np.isfinite(side_uv)):
        return {"accepted": False, "reason": "NONFINITE_PIXEL"}
    first = normalized_pixels(primary_uv, calibration.primary_intrinsics, calibration.primary_distortion)[0]
    second = normalized_pixels(side_uv, calibration.side_intrinsics, calibration.side_distortion)[0]
    transform = np.asarray(calibration.side_from_primary, np.float64)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    center = -rotation.T @ translation
    if np.linalg.norm(center) < 1.:
        return {"accepted": False, "reason": "DEGENERATE_BASELINE"}
    ray_a, ray_b = np.r_[first, 1.], rotation.T @ np.r_[second, 1.]
    angle = np.degrees(np.arccos(np.clip(ray_a @ ray_b / np.linalg.norm(ray_a) / np.linalg.norm(ray_b), -1., 1.)))
    if min(angle, 180. - angle) < limits.minimum_ray_angle_deg:
        return {"accepted": False, "reason": "SMALL_RAY_ANGLE", "ray_angle_deg": float(angle)}
    homogeneous = cv2.triangulatePoints(np.eye(3, 4), transform[:3], first.reshape(2, 1), second.reshape(2, 1))[:, 0]
    if abs(homogeneous[3]) < 1e-10:
        return {"accepted": False, "reason": "POINT_AT_INFINITY"}
    point = homogeneous[:3] / homogeneous[3]
    if not np.all(np.isfinite(point)) or point[2] <= 0 or (rotation @ point + translation)[2] <= 0:
        return {"accepted": False, "reason": "CHEIRALITY_FAILED"}
    from .visual_contract import active_contract, V7
    refine = active_contract() == V7 or refine_position
    refinement_applied = False
    if refine:
        refined = _refine_position(point, primary_uv, side_uv, calibration)
        if refine_position and active_contract() != V7:
            # Retain the raw solution if refinement would worsen maximum
            # reprojection or fail depth consistency. The object envelope
            # still applies independently in match_corners.
            raw_error = max(np.linalg.norm(project_primary(point[None],calibration)[0]-primary_uv),
                            np.linalg.norm(calibration.project_primary_points(point[None])[0]-side_uv))
            new_error = max(np.linalg.norm(project_primary(refined[None],calibration)[0]-primary_uv),
                            np.linalg.norm(calibration.project_primary_points(refined[None])[0]-side_uv))
            depth_ok = depth_prior_mm is None or np.linalg.norm(refined-depth_prior_mm) <= limits.depth_consistency_mm
            if new_error <= min(raw_error,limits.reprojection_px) and depth_ok:
                point = refined
                refinement_applied = True
        else:
            point = refined
            refinement_applied = True
    top_error = float(np.linalg.norm(project_primary(point[None], calibration)[0] - primary_uv))
    side_error = float(np.linalg.norm(calibration.project_primary_points(point[None])[0] - side_uv))
    if max(top_error, side_error) > limits.reprojection_px:
        return {"accepted": False, "reason": "REPROJECTION_FAILED",
                "primary_error_px": top_error, "side_error_px": side_error}
    depth_error = None if depth_prior_mm is None else float(np.linalg.norm(point - depth_prior_mm))
    if depth_error is not None and depth_error > limits.depth_consistency_mm:
        return {"accepted": False, "reason": "DEPTH_CONFLICT", "depth_error_mm": depth_error}
    result = {"accepted": True, "reason": "ACCEPTED", "position_primary_mm": point.tolist(),
              "primary_uv": np.asarray(primary_uv).tolist(), "side_uv": np.asarray(side_uv).tolist(),
              "primary_error_px": top_error, "side_error_px": side_error,
              "ray_angle_deg": float(angle), "depth_error_mm": depth_error,
              "sources": ["TWO_VIEW_TRIANGULATED"] + ([] if depth_prior_mm is None else ["PRIMARY_DEPTH_CHECKED"])}
    if calibration.tray_from_primary is not None:
        result["position_tray_mm"] = (calibration.tray_from_primary @ np.r_[point, 1.])[:3].tolist()
    if refine:
        result["position_sensitivity"] = _perturbation_sensitivity(primary_uv, side_uv, calibration, point)
        result["refinement"] = "two_camera_reprojection_gauss_newton"
        result["refinement_applied"] = refinement_applied
        if not refinement_applied:
            result["refinement_fallback_reason"] = "retain_valid_dlt_under_max_reprojection_and_depth_gates"
    return result


def observations(image: np.ndarray, mask: np.ndarray, maximum: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Contour turns and nonparallel line junctions; never isolated texture endpoints."""
    segments = _side_lsd_segments(image, mask)
    segments = sorted(segments, key=lambda line: -np.linalg.norm(line[2:] - line[:2]))[:maximum]
    segments = np.asarray(segments, np.float64).reshape(-1, 4)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    from .visual_contract import active_contract, V7
    v7 = active_contract() == V7
    if contours:
        contour = max(contours, key=cv2.contourArea)
        polygon = cv2.approxPolyDP(contour, .012 * cv2.arcLength(contour, True), True).reshape(-1, 2).astype(float)
        for i, vertex in enumerate(polygon):
            a, b = polygon[i - 1] - vertex, polygon[(i + 1) % len(polygon)] - vertex
            angle = np.degrees(np.arccos(np.clip(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6), -1, 1)))
            if 25 <= angle <= 150:
                incident = sum(np.min(np.linalg.norm(line.reshape(2, 2) - vertex, axis=1)) <= 4.
                               for line in segments)
                if not v7 or incident >= 2:
                    candidates.append(vertex)
    for i, first in enumerate(segments):
        a = first[2:] - first[:2]
        for second in segments[i + 1:]:
            b = second[2:] - second[:2]
            angle = np.degrees(np.arccos(np.clip(abs(a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6), 0, 1)))
            if angle < 25:
                continue
            distances = np.linalg.norm(first.reshape(2, 2)[:, None] - second.reshape(2, 2)[None], axis=2)
            endpoint_a, endpoint_b = np.unravel_index(np.argmin(distances), distances.shape)
            if distances[endpoint_a, endpoint_b] <= 3:
                candidates.append((first.reshape(2, 2)[endpoint_a] + second.reshape(2, 2)[endpoint_b]) / 2)
    corners = []
    for vertex in candidates:
        if not corners or min(np.linalg.norm(vertex - old) for old in corners) > 4.:
            corners.append(vertex)
        if len(corners) >= maximum:
            break
    corners = np.asarray(corners, np.float64).reshape(-1, 2)
    if v7 and len(corners):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for index, corner in enumerate(corners):
            if np.all(corner >= 5) and corner[0] < gray.shape[1] - 5 and corner[1] < gray.shape[0] - 5:
                refined = cv2.cornerSubPix(gray, corner.astype(np.float32).reshape(1, 1, 2), (3, 3), (-1, -1),
                                          (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, .01)).reshape(2)
                if np.linalg.norm(refined - corner) <= 1.5:
                    corners[index] = refined
    return corners, segments


def match_corners(primary: np.ndarray, side: np.ndarray, cloud: np.ndarray, calibration: Any,
                  limits: SparseStereoLimits, auxiliary_cost: np.ndarray | None = None,
                  *, refine_positions: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not len(primary) or not len(side):
        return [], [{"reason": "NO_CORNERS"}]
    top_normal = normalized_pixels(primary, calibration.primary_intrinsics, calibration.primary_distortion)
    side_normal = normalized_pixels(side, calibration.side_intrinsics, calibration.side_distortion)
    transform = calibration.side_from_primary
    r, t = transform[:3, :3], transform[:3, 3]
    skew = np.array([[0., -t[2], t[1]], [t[2], 0., -t[0]], [-t[1], t[0], 0.]])
    essential = skew @ r
    top_h, side_h = np.c_[top_normal, np.ones(len(primary))], np.c_[side_normal, np.ones(len(side))]
    a_lines, b_lines = top_h @ essential.T, side_h @ essential
    error = np.abs(a_lines @ side_h.T)
    epi_side = error / np.maximum(np.linalg.norm(a_lines[:, :2], axis=1)[:, None], 1e-9)
    epi_top = error / np.maximum(np.linalg.norm(b_lines[:, :2], axis=1)[None], 1e-9)
    focal_top = (calibration.primary_intrinsics.fx + calibration.primary_intrinsics.fy) / 2
    focal_side = (calibration.side_intrinsics.fx + calibration.side_intrinsics.fy) / 2
    epipolar = np.maximum(epi_side * focal_side, epi_top * focal_top)
    cost = epipolar / limits.epipolar_px
    cost[epipolar > limits.epipolar_px] = np.inf
    if auxiliary_cost is not None:
        if auxiliary_cost.shape != cost.shape or not np.isfinite(auxiliary_cost).all() or np.any(auxiliary_cost < 0):
            raise ValueError("invalid corner auxiliary evidence")
        cost += auxiliary_cost
    projected = project_primary(cloud, calibration) if len(cloud) else np.empty((0, 2))
    priors = []
    for i, uv in enumerate(primary):
        near = np.linalg.norm(projected - uv, axis=1) <= 4.
        prior = np.median(cloud[near], axis=0) if np.count_nonzero(near) >= 3 else None
        priors.append(prior)
        if prior is not None:
            residual = np.linalg.norm(side - calibration.project_primary_points(prior[None])[0], axis=1)
            cost[i] += residual / limits.prior_side_distance_px
            cost[i, residual > limits.prior_side_distance_px] = np.inf

    def unique(values):
        finite = np.flatnonzero(np.isfinite(values))
        if not len(finite):
            return None
        ordered = finite[np.argsort(values[finite], kind="stable")]
        if len(ordered) > 1 and values[ordered[1]] - values[ordered[0]] < limits.ambiguity_gap:
            return None
        return int(ordered[0])

    side_best = [unique(cost[:, j]) for j in range(len(side))]
    matches, rejected = [], []
    for i in range(len(primary)):
        j = unique(cost[i])
        if j is None or side_best[j] != i:
            rejected.append({"primary_corner_id": i, "reason": "AMBIGUOUS_OR_NO_MUTUAL_MATCH"})
            continue
        result = triangulate_corner(primary[i], side[j], calibration, limits, priors[i], refine_position=refine_positions)
        result.update(primary_corner_id=i, side_corner_id=j, epipolar_error_px=float(epipolar[i, j]))
        if auxiliary_cost is not None:
            result["ranking_evidence"] = {"local_color_and_incident_topology_cost": float(auxiliary_cost[i,j]),
                "geometric_gates_unchanged": True, "one_to_one_mutual": True}
        if result["accepted"] and len(cloud):
            point = np.asarray(result["position_primary_mm"])
            if np.any(point < cloud.min(axis=0) - 15) or np.any(point > cloud.max(axis=0) + 15):
                result.update(accepted=False, reason="OUTSIDE_OBJECT_ENVELOPE")
        (matches if result["accepted"] else rejected).append(result)
    return matches, rejected


def _line_contains(line: np.ndarray, pixels: np.ndarray, tolerance: float = 3.) -> bool:
    direction = line[2:] - line[:2]
    length2 = float(direction @ direction)
    if length2 < 64:
        return False
    delta = pixels - line[:2]
    along = delta @ direction / length2
    residual = np.linalg.norm(delta - along[:, None] * direction, axis=1)
    return bool(np.all(residual <= tolerance) and np.all(along >= -.1) and np.all(along <= 1.1)
                and np.ptp(along) >= .5)


def reconstruct(primary_image: np.ndarray, primary_mask: np.ndarray, side_image: np.ndarray,
                 side_mask: np.ndarray, cloud: np.ndarray, calibration: Any,
                 limits: SparseStereoLimits = SparseStereoLimits(), *, refine_positions: bool = False) -> dict[str, Any]:
    for image, mask, intrinsics in ((primary_image, primary_mask, calibration.primary_intrinsics),
                                    (side_image, side_mask, calibration.side_intrinsics)):
        if image.shape[:2] != mask.shape or mask.shape != (intrinsics.height, intrinsics.width):
            raise ValueError("sparse stereo requires native calibrated image/mask sizes")
    cloud = np.asarray(cloud, np.float64).reshape(-1, 3)
    cloud = cloud[np.all(np.isfinite(cloud), axis=1) & (cloud[:, 2] > 0)]
    top, top_lines = observations(primary_image, primary_mask, limits.maximum_features)
    side, side_lines = observations(side_image, side_mask, limits.maximum_features)
    from .visual_contract import active_contract, V7
    auxiliary = None
    if active_contract() == V7 and len(top) and len(side):
        def descriptor(image, corners, lines):
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            values = []
            for uv in corners:
                x,y = np.rint(uv).astype(int)
                patch = lab[max(0,y-3):y+4, max(0,x-3):x+4,1:]
                chroma = np.median(patch.reshape(-1,2),axis=0)/128 if patch.size else np.ones(2)
                degree = sum(np.linalg.norm(uv-line.reshape(2,2),axis=1).min() <= 6 for line in lines)
                values.append(np.r_[chroma, degree])
            return np.asarray(values)
        first, second = descriptor(primary_image, top, top_lines), descriptor(side_image, side, side_lines)
        auxiliary = .05 * np.linalg.norm(first[:,None,:2]-second[None,:,:2],axis=2) + .1 * np.minimum(
            np.abs(first[:,None,2]-second[None,:,2]), 2)
    nodes, rejected = match_corners(top, side, cloud, calibration, limits, auxiliary, refine_positions=refine_positions)
    edges = []
    for i, a in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            uv_top = np.asarray([a["primary_uv"], b["primary_uv"]])
            uv_side = np.asarray([a["side_uv"], b["side_uv"]])
            top_support = [k for k, line in enumerate(top_lines) if _line_contains(line, uv_top)]
            side_support = [k for k, line in enumerate(side_lines) if _line_contains(line, uv_side)]
            if len(top_support) != 1 or len(side_support) != 1:
                continue
            # A line in each view defines two ray planes; their intersection
            # must agree with both independently reconstructed endpoints.
            first = normalized_pixels(top_lines[top_support[0]].reshape(2, 2), calibration.primary_intrinsics,
                                      calibration.primary_distortion)
            second = normalized_pixels(side_lines[side_support[0]].reshape(2, 2), calibration.side_intrinsics,
                                       calibration.side_distortion)
            normal_a = np.cross(np.r_[first[0], 1.], np.r_[first[1], 1.])
            normal_b_side = np.cross(np.r_[second[0], 1.], np.r_[second[1], 1.])
            normal_a /= max(np.linalg.norm(normal_a), 1e-9)
            normal_b_side /= max(np.linalg.norm(normal_b_side), 1e-9)
            rotation, translation = calibration.side_from_primary[:3, :3], calibration.side_from_primary[:3, 3]
            normal_b = rotation.T @ normal_b_side
            if np.linalg.norm(np.cross(normal_a, normal_b)) < .02:
                continue
            endpoints = np.asarray([a["position_primary_mm"], b["position_primary_mm"]])
            residual = max(float(np.max(np.abs(endpoints @ normal_a))),
                           float(np.max(np.abs(endpoints @ normal_b + normal_b_side @ translation))))
            length = float(np.linalg.norm(endpoints[1] - endpoints[0]))
            if residual > 2. or length < 4.:
                continue
            edges.append({"node_ids": [i, j], "endpoints_primary_mm": endpoints.tolist(),
                          "length_mm": length, "plane_residual_mm": residual,
                          "primary_segment_id": top_support[0], "side_segment_id": side_support[0],
                          "sources": ["TWO_VIEW_LINE_PLANES", "TWO_VIEW_TRIANGULATED_ENDPOINTS"]})
    nodes = [{**node, "node_id": i} for i, node in enumerate(nodes)]
    reprojections = [max(node["primary_error_px"], node["side_error_px"]) for node in nodes]
    return {"version": 1, "coordinate_frame": "PRIMARY_CAMERA_MM", "complete_mesh": False,
            "grasp_geometry_upgrade": False, "camera_poses": camera_poses(calibration),
            "nodes": nodes, "edges": edges, "rejected_corners": rejected,
            "primary_corners_px": top.tolist(), "side_corners_px": side.tolist(),
            "primary_segments_px": top_lines.tolist(), "side_segments_px": side_lines.tolist(),
            "reprojection_p95_px": float(np.percentile(reprojections, 95)) if reprojections else None,
            "limits": vars(limits),
            "position_refinement_enabled": refine_positions or active_contract() == V7,
            "limitations": ["PARTIAL_CO_VISIBLE_GEOMETRY", "RGB_LINES_MAY_BE_TEXTURE",
                            "NO_SIDE_ONLY_DEPTH", "NOT_ROBOT_COORDINATES"]}


def augment_graph(graph: dict[str, Any], sparse: dict[str, Any], merge_mm: float = 2.) -> dict[str, Any]:
    """Keep the old trained feature vector intact; expose a new provenance graph."""
    import copy

    result = copy.deepcopy(graph)
    result.update(version=2, feature_contract="SPARSE_GEOMETRY_NOT_V6_MODEL_INPUT", complete_mesh=False)
    nodes, edges = result.setdefault("nodes", []), result.setdefault("edges", [])
    mapping = {}
    for node in sparse["nodes"]:
        point = np.asarray(node["position_primary_mm"])
        distances = [np.linalg.norm(point - old["position_mm"]) for old in nodes]
        index = int(np.argmin(distances)) if distances and min(distances) <= merge_mm else len(nodes)
        if index == len(nodes):
            nodes.append({"node_id": index, "position_mm": point.tolist(), "degree": 0,
                          "sources": node["sources"], "position_tray_mm": node.get("position_tray_mm")})
        else:
            nodes[index]["sources"] = sorted(set(nodes[index].get("sources", ["PRIMARY_DEPTH_GRAPH"]) + node["sources"]))
        mapping[node["node_id"]] = index
    for edge in sparse["edges"]:
        ids = [mapping[index] for index in edge["node_ids"]]
        if ids[0] == ids[1]:
            continue
        old = next((candidate for candidate in edges if set(candidate["node_ids"]) == set(ids)), None)
        if old is not None:
            old["sources"] = sorted(set(old["sources"] + edge["sources"]))
        else:
            endpoints = np.asarray(edge["endpoints_primary_mm"])
            face_ids = [face["face_id"] for face in result.get("faces", []) if
                        np.max(np.abs(endpoints @ np.asarray(face["normal"]) + face["offset_mm"])) <= 2.]
            edges.append({**edge, "node_ids": ids, "face_ids": face_ids})
    for node in nodes:
        node["degree"] = sum(node["node_id"] in edge["node_ids"] for edge in edges)
    result["legacy_features"] = result.pop("features", {})
    result["sparse_geometry"] = sparse
    return result


def restore_native_mask(crop_mask: np.ndarray, origin_uv: list[int], scale: float,
                        image_shape: tuple[int, ...]) -> np.ndarray:
    if not 0 < scale <= 1:
        raise ValueError("invalid primary processing scale")
    height, width = image_shape[:2]
    output = np.zeros((height, width), np.uint8)
    restored = cv2.resize(crop_mask, (int(round(crop_mask.shape[1] / scale)),
                                      int(round(crop_mask.shape[0] / scale))), interpolation=cv2.INTER_NEAREST)
    x, y = map(int, origin_uv)
    left, top = max(0, x), max(0, y)
    right, bottom = min(width, x + restored.shape[1]), min(height, y + restored.shape[0])
    if right > left and bottom > top:
        output[top:bottom, left:right] = restored[top-y:bottom-y, left-x:right-x]
    return output
