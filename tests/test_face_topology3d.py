import numpy as np

from sorting_vision.face_topology3d import (
    TOPOLOGY_FEATURE_NAMES,
    extract_face_topology,
    face_topology_features,
)
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
