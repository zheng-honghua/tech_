import cv2
import numpy as np

from sorting_vision.face_topology3d import (
    FUSED_EDGE_FEATURE_NAMES,
    TOPOLOGY_FEATURE_NAMES,
    extract_face_topology,
    face_topology_features,
    fused_edge_features,
)
from sorting_vision.rgbd_edge_audit import render_fused_edge_audit
from sorting_vision.rgbd import CameraIntrinsics


def test_extracts_multiple_connected_depth_planes_without_hough():
    height, width = 90, 120
    depth = np.full((height, width), 500, np.float32)
    depth[:, 40:80] = 520
    depth[:, 80:] = 540
    mask = np.zeros((height, width), np.uint8)
    mask[5:-5, 5:-5] = 255
    intrinsics = CameraIntrinsics(width, height, 160, 160, width / 2, height / 2)
    topology = extract_face_topology(
        depth, mask, intrinsics, min_face_area_px=150
    )
    assert len(topology.faces) == 3
    assert len(topology.adjacency) == 2
    assert topology.evidence_ratio > 0.75
    assert all(face.plane.rmse_mm < 0.1 for face in topology.faces)


def test_topology_feature_vector_is_fixed_and_finite():
    height, width = 80, 100
    depth = np.full((height, width), 600, np.float32)
    mask = np.zeros((height, width), np.uint8)
    mask[8:-8, 8:-8] = 255
    intrinsics = CameraIntrinsics(width, height, 140, 140, width / 2, height / 2)
    topology = extract_face_topology(depth, mask, intrinsics)
    features = face_topology_features(topology)
    assert features.shape == (len(TOPOLOGY_FEATURE_NAMES),)
    assert np.all(np.isfinite(features))
    assert features[0] == 1


def _fused_scene(flat_depth: bool = False):
    height = width = 120
    columns = np.arange(width, dtype=np.float32)[None, :]
    depth = np.repeat(500.0 + np.maximum(columns - 60, 0) * 0.35, height, axis=0)
    if flat_depth:
        depth[:] = 500
    mask = np.zeros((height, width), np.uint8)
    mask[8:-8, 8:-8] = 255
    color = np.full((height, width, 3), (105, 145, 180), np.uint8)
    color[:, 60:] = (108, 148, 183) if not flat_depth else (45, 75, 105)
    intrinsics = CameraIntrinsics(width, height, 180, 180, width / 2, height / 2)
    return color, depth.astype(np.float32), mask, intrinsics


def test_fused_edges_recover_weak_rgb_ridge_with_depth_support():
    color, depth, mask, intrinsics = _fused_scene()
    topology = extract_face_topology(
        depth, mask, intrinsics, color_crop_bgr=color,
        plane_threshold_mm=0.6, extract_fused_edges=True,
    )
    features = fused_edge_features(topology)
    assert features.shape == (len(FUSED_EDGE_FEATURE_NAMES),)
    assert np.all(np.isfinite(features))
    assert len(topology.fused_edges) >= 1
    assert any(line.depth_support > 0 for line in topology.fused_edges)


def test_rgb_shadow_on_one_flat_depth_plane_is_not_a_fused_ridge():
    color, depth, mask, intrinsics = _fused_scene(flat_depth=True)
    topology = extract_face_topology(
        depth, mask, intrinsics, color_crop_bgr=color, extract_fused_edges=True,
    )
    assert topology.fused_edges == ()
    assert len(topology.rejected_rgb_edges) >= 1


def test_depth_hole_does_not_create_a_fused_ridge():
    color, depth, mask, intrinsics = _fused_scene(flat_depth=True)
    color[:] = (105, 145, 180)
    depth[:, 58:62] = 0
    topology = extract_face_topology(
        depth, mask, intrinsics, color_crop_bgr=color, extract_fused_edges=True,
    )
    assert topology.fused_edges == ()


def test_fused_edge_extraction_is_quarter_turn_stable_and_renders():
    color, depth, mask, intrinsics = _fused_scene()
    first = extract_face_topology(
        depth, mask, intrinsics, color_crop_bgr=color,
        plane_threshold_mm=0.6, extract_fused_edges=True,
    )
    second = extract_face_topology(
        cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE), intrinsics,
        color_crop_bgr=cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE),
        plane_threshold_mm=0.6, extract_fused_edges=True,
    )
    assert abs(len(first.fused_edges) - len(second.fused_edges)) <= 1
    rendered = render_fused_edge_audit(color, first)
    assert set(rendered) == {
        "faces", "rgb-candidates", "depth-candidates", "fused", "rejected"
    }
    assert all(image.shape == color.shape for image in rendered.values())
