from __future__ import annotations

import numpy as np
import cv2

from sorting_vision.cross_view_topology import (
    CROSS_VIEW_FEATURE_NAMES,
    CrossViewTopologyModel,
    extract_cross_view_features,
    _oriented_box_edges,
    _segment_plane_intersections,
    _project_box_edges,
)
from sorting_vision.dual_view import DualViewCalibration, DualCalibrationMetrics
from sorting_vision.rgbd import CameraIntrinsics
from sorting_vision.shape_registry import ShapeClass, ShapeRegistry


def _calibration() -> DualViewCalibration:
    intrinsics = CameraIntrinsics(256, 256, 250.0, 250.0, 128.0, 128.0)
    return DualViewCalibration(
        intrinsics,
        np.zeros(5),
        intrinsics,
        np.zeros(5),
        np.eye(4),
        DualCalibrationMetrics(0.2, 0.2, 1.0, 0.005),
        "test",
        "test-v1",
        {},
        (256, 256),
    )


def _box_points(scale: float = 1.0) -> np.ndarray:
    x = np.linspace(-30.0 * scale, 30.0 * scale, 9)
    y = np.linspace(-40.0, 40.0, 11)
    z = np.linspace(190.0, 220.0, 5)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))


def _image(kind: str) -> tuple[np.ndarray, np.ndarray]:
    image = np.full((256, 256, 3), 225, np.uint8)
    mask = np.zeros((256, 256), np.uint8)
    if kind == "prism":
        polygon = np.asarray([[88, 76], [168, 76], [168, 181], [88, 181]], np.int32)
        cv2.fillConvexPoly(mask, polygon, 255)
        cv2.fillConvexPoly(image, polygon, (30, 40, 210))
        cv2.line(image, (110, 76), (110, 181), (10, 10, 80), 3)
    else:
        polygon = np.asarray([[128, 67], [177, 183], [79, 183]], np.int32)
        cv2.fillConvexPoly(mask, polygon, 255)
        cv2.fillConvexPoly(image, polygon, (30, 170, 30))
        cv2.line(image, (128, 67), (128, 183), (10, 70, 10), 3)
    return image, mask


def _registry() -> ShapeRegistry:
    return ShapeRegistry(
        1,
        (
            ShapeClass("prism", "棱柱", "prism", ()),
            ShapeClass("pyramid", "棱锥", "pyramid", ()),
        ),
    )


def test_cross_view_features_retain_segments_and_spatial_support():
    image, mask = _image("prism")
    result = extract_cross_view_features(image, mask, _box_points(), _calibration())
    assert np.all(np.isfinite(result.vector))
    assert len(result.vector) == len(result.group_ids)
    assert len(result.segments_px) >= 3
    assert len(result.projected_obb_edges_px) == 12
    assert result.diagnostics["projected_point_mask_ratio"] > 0.4
    assert result.diagnostics["spatial_quality"] > 0.25
    assert all(name in result.diagnostics for name in CROSS_VIEW_FEATURE_NAMES)


def test_cached_box_has_identical_projection_and_ray_intersections():
    points, calibration = _box_points(), _calibration()
    box = _oriented_box_edges(points)
    np.testing.assert_array_equal(_project_box_edges(points, calibration), _project_box_edges(points, calibration, box))
    line = np.asarray([100, 70, 140, 185], np.float32)
    assert _segment_plane_intersections(line, points, calibration) == _segment_plane_intersections(line, points, calibration, box)


def test_v3_rollback_ignores_appended_graph_features(tmp_path):
    image, mask = _image("prism")
    extracted = extract_cross_view_features(image, mask, _box_points(), _calibration())
    selected = extracted.group_ids < 5
    vector = extracted.vector[selected]
    registry = _registry()
    model = CrossViewTopologyModel(
        np.stack((vector, vector + 1)), np.asarray(["prism", "pyramid"]),
        np.zeros(len(vector)), np.ones(len(vector)), extracted.group_ids[selected],
        registry, method="knn",
    )
    assert model.feature_version == 3
    model.predict(image, mask, _box_points(), _calibration())
    expected = dict(model.last_class_scores)
    path = tmp_path / "rollback.npz"
    model.save(path)
    loaded = CrossViewTopologyModel.load(path, registry)
    loaded.predict(image, mask, _box_points(), _calibration())
    assert loaded.last_class_scores == expected
    assert "joint_topology_graph" in loaded.last_feature_diagnostics


def test_cross_view_model_round_trip_and_complete_scores(tmp_path):
    registry = _registry()
    calibration = _calibration()
    samples = []
    for label in registry.class_ids:
        for repeat in range(4):
            image, mask = _image("prism" if label == "prism" else "pyramid")
            image = np.roll(image, repeat - 1, axis=1)
            mask = np.roll(mask, repeat - 1, axis=1)
            samples.append((image, mask, _box_points(1.0 + 0.01 * repeat), calibration, label))
    model = CrossViewTopologyModel.train(samples, registry, method="rtrees")
    image, mask = _image("prism")
    model.predict(image, mask, _box_points(), calibration)
    expected = dict(model.last_class_scores)
    path = tmp_path / "cross-view.npz"
    model.save(path)
    cv2.setRNGSeed(99)
    loaded = CrossViewTopologyModel.load(path, registry)
    image, mask = _image("prism")
    loaded.predict(image, mask, _box_points(), calibration)
    assert set(loaded.last_class_scores) == set(registry.class_ids)
    assert sum(loaded.last_class_scores.values()) == 1.0
    assert loaded.last_class_scores == expected


def test_cross_view_model_rejects_wrong_artifact_type(tmp_path):
    path = tmp_path / "wrong.npz"
    np.savez_compressed(
        path,
        model_type=np.asarray(["side_geometry"]),
        feature_version=np.asarray([1], np.int32),
    )
    try:
        CrossViewTopologyModel.load(path, _registry())
    except ValueError as error:
        assert "not a cross-view" in str(error)
    else:
        raise AssertionError("wrong artifact type was accepted")
