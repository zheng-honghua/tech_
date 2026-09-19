from dataclasses import replace
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from sorting_vision.dual_view import DualViewFusion, SideEvidence
from sorting_vision.config import DualViewConfig
from sorting_vision.rgbd import Plane
from sorting_vision.sparse_stereo import (
    SparseStereoLimits, augment_graph, camera_poses, match_corners,
    observations, project_primary, reconstruct, restore_native_mask, triangulate_corner,
)
from test_cross_view_topology import _calibration
from test_dual_view import pair, result
from sorting_vision.types import DetectionStatus


def test_isolated_texture_endpoints_are_not_vertices(monkeypatch):
    monkeypatch.setattr("sorting_vision.sparse_stereo._side_lsd_segments",
                        lambda image, mask: [np.array([35., 40., 65., 40.])])
    mask = np.zeros((100, 100), np.uint8)
    mask[10:90, 10:90] = 255
    corners, _ = observations(np.zeros((100, 100, 3), np.uint8), mask)
    assert len(corners) == 4
    assert np.min(np.linalg.norm(corners - [35., 40.], axis=1)) > 4


def stereo():
    transform = np.eye(4)
    transform[0, 3] = -60.
    return replace(_calibration(), side_from_primary=transform, tray_from_primary=np.eye(4))


def pixels(point, calibration):
    return project_primary(np.asarray(point)[None], calibration)[0], calibration.project_primary_points(np.asarray(point)[None])[0]


def test_camera_center_is_inverse_transform_not_translation():
    poses = camera_poses(stereo())
    np.testing.assert_allclose(poses["side_center_mm"], [60, 0, 0])
    assert poses["baseline_mm"] == 60


def test_exact_corner_reconstruction_and_tray_coordinates():
    calibration = stereo()
    point = np.array([0., 10., 200.])
    a, b = pixels(point, calibration)
    output = triangulate_corner(a, b, calibration, SparseStereoLimits(), point)
    assert output["accepted"]
    np.testing.assert_allclose(output["position_primary_mm"], point, atol=1e-7)
    np.testing.assert_allclose(output["position_tray_mm"], point, atol=1e-7)
    assert "PRIMARY_DEPTH_CHECKED" in output["sources"]
    json.dumps(output, allow_nan=False)


def test_distorted_cameras_are_undistorted_before_triangulation():
    calibration = replace(stereo(), primary_distortion=np.array([.12, -.04, 0., 0., 0.]),
                          side_distortion=np.array([-.08, .02, 0., 0., 0.]))
    point = np.array([10., 10., 200.])
    a, b = pixels(point, calibration)
    output = triangulate_corner(a, b, calibration, SparseStereoLimits())
    assert output["accepted"]
    np.testing.assert_allclose(output["position_primary_mm"], point, atol=.001)


def test_degenerate_angle_depth_conflict_and_bad_pixels_rejected():
    calibration = stereo()
    a, b = pixels(np.array([0., 10., 200.]), calibration)
    assert triangulate_corner(a, b, _calibration(), SparseStereoLimits())["reason"] == "DEGENERATE_BASELINE"
    assert triangulate_corner(a, b, calibration, SparseStereoLimits(), np.array([0, 10, 240]))["reason"] == "DEPTH_CONFLICT"
    assert triangulate_corner(a, b, calibration, SparseStereoLimits(minimum_ray_angle_deg=40))["reason"] == "SMALL_RAY_ANGLE"
    assert triangulate_corner(np.array([np.nan, 1]), b, calibration, SparseStereoLimits())["reason"] == "NONFINITE_PIXEL"
    assert not triangulate_corner(a, b + [0, 30], calibration, SparseStereoLimits())["accepted"]


def test_ambiguous_same_epipolar_line_never_gets_arbitrary_match():
    a = np.array([[128., 128.]])
    b = np.array([[53., 128.], [63., 128.]])
    accepted, rejected = match_corners(a, b, np.empty((0, 3)), stereo(), SparseStereoLimits())
    assert not accepted
    assert rejected[0]["reason"] == "AMBIGUOUS_OR_NO_MUTUAL_MATCH"


def test_rgb_pair_can_reconstruct_without_fabricated_depth():
    point = np.array([0., 10., 200.])
    a, b = pixels(point, stereo())
    matches, _ = match_corners(a[None], b[None], np.empty((0, 3)), stereo(), SparseStereoLimits())
    assert len(matches) == 1
    assert matches[0]["depth_error_mm"] is None
    assert matches[0]["sources"] == ["TWO_VIEW_TRIANGULATED"]


def test_complete_reconstruction_native_sizes_and_finite_partial_graph():
    calibration = stereo()
    images, masks = [], []
    for box in ((103, 103, 50, 50), (28, 103, 50, 50)):
        image = np.full((256, 256, 3), 225, np.uint8)
        mask = np.zeros((256, 256), np.uint8)
        x, y, w, h = box
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
        image[mask > 0] = (10, 10, 180)
        images.append(image)
        masks.append(mask)
    x, y = np.meshgrid(np.linspace(-20, 20, 41), np.linspace(-20, 20, 41))
    cloud = np.c_[x.ravel(), y.ravel(), np.full(x.size, 200.)]
    before = cloud.copy()
    graph = reconstruct(images[0], masks[0], images[1], masks[1], cloud, calibration)
    assert len(graph["nodes"]) >= 2
    assert graph["edges"]
    assert graph["complete_mesh"] is False and graph["grasp_geometry_upgrade"] is False
    json.dumps(graph, allow_nan=False)
    np.testing.assert_array_equal(before, cloud)
    with pytest.raises(ValueError, match="native"):
        reconstruct(images[0][::2], masks[0][::2], images[1], masks[1], cloud, calibration)


def test_native_mask_restore_clips_out_of_frame_without_coordinate_wrap():
    crop = np.full((4, 4), 255, np.uint8)
    mask = restore_native_mask(crop, [-2, 5], .5, (12, 12, 3))
    assert np.count_nonzero(mask) == 6 * 7
    assert not mask[:, 11].any()
    with pytest.raises(ValueError):
        restore_native_mask(crop, [0, 0], 0, (12, 12, 3))


def test_augmented_graph_does_not_change_v6_features():
    old = {"nodes": [], "edges": [], "faces": [], "features": {"legacy": .25}}
    sparse = {"nodes": [{"node_id": 0, "position_primary_mm": [0, 0, 200], "sources": ["TWO_VIEW_TRIANGULATED"]}], "edges": []}
    augmented = augment_graph(old, sparse)
    assert old["features"] == {"legacy": .25} and not old["nodes"]
    assert augmented["legacy_features"] == old["features"]
    assert "features" not in augmented


def test_depth_veto_does_not_upgrade_or_change_primary_safety(monkeypatch):
    evidence = SideEvidence({"cube": .99}, "cube", .99, "accepted", .9, (0, 0, 100, 100),
                            100, 100., np.zeros((4, 2)), contours_px=[[[40, 40], [60, 40], [60, 60]]])
    item = result(status=DetectionStatus.DEPTH_INVALID)
    item.rgb_crop_mask = np.full((20, 20), 255, np.uint8)
    item.diagnostics["rgb_crop_origin_uv"] = [40, 40]
    monkeypatch.setattr("sorting_vision.sparse_stereo.reconstruct", lambda *args: {
        "nodes": [], "edges": [], "rejected_corners": [{"accepted": False, "reason": "DEPTH_CONFLICT"}] * 2})
    fusion = DualViewFusion(stereo(), np.zeros((100, 100, 3), np.uint8), None,
                            DualViewConfig(sparse_stereo_enabled=True), Plane(np.array([0., 0., -1.]), 200))
    output = fusion._sparse_geometry(item, pair(), np.empty((0, 3)), evidence)
    assert output.quality == 0
    assert item.status == DetectionStatus.DEPTH_INVALID and item.shape_id == "cube"
    assert output.topology_diagnostics["sparse_stereo"]["state"] == "DEPTH_CONFLICT"


def test_positive_limits_required():
    with pytest.raises(ValueError):
        SparseStereoLimits(epipolar_px=0)
