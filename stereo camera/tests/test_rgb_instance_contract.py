from dataclasses import replace

import numpy as np
import pytest

from sorting_vision.config import load_config
from sorting_vision.geometry_rgbd_model import DepthGeometryModel, FUSED_FEATURE_NAMES
from sorting_vision.pipeline3d import VisionPipeline3D
from sorting_vision.rgbd import CameraIntrinsics, RGBDCalibration, RGBDFrame, Plane
from sorting_vision.side_geometry import load_side_training_samples
from sorting_vision.shape_registry import load_shape_registry


class RecordingModel:
    input_contract = "rgb_silhouette_depth_owned_v2"

    def classify(self, points, rgb, depth, mask, intrinsics=None, crop_origin_uv=(0, 0)):
        self.inputs = (points.copy(), rgb.copy(), depth.copy(), mask.copy(), crop_origin_uv)
        return "quadrangular_prism", .9


def test_new_contract_rejects_legacy_segmentation_config():
    with pytest.raises(ValueError, match="requires hsv"):
        VisionPipeline3D(shape_model=RecordingModel())


def test_rgb_contract_preserves_silhouette_without_fabricating_depth(monkeypatch):
    config = load_config()
    config = replace(config, rgbd=replace(config.rgbd, processing_scale=1,
                                         instance_segmentation="hsv", min_area_px=40))
    rgb = np.full((120, 160, 3), 240, np.uint8)
    rgb[35:85, 45:115] = (10, 150, 10)
    depth = np.full((120, 160), 600., np.float32)
    depth[35:85, 45:115] = 580
    depth[35:85, 77:82] = 0
    original = depth.copy()
    intrinsics = CameraIntrinsics(160, 120, 180, 180, 80, 60, 1)
    calibration = RGBDCalibration(intrinsics, np.eye(4), Plane(np.array([0., 0., -1.]), 600))
    monkeypatch.setattr("sorting_vision.pipeline3d.detect_tray_roi_mask",
                        lambda image: np.full(image.shape[:2], 255, np.uint8))
    model = RecordingModel()
    pipeline = VisionPipeline3D(config=config, calibration=calibration, shape_model=model)
    results = pipeline.process(RGBDFrame(rgb, depth, intrinsics, 1, "contract"))
    assert len(results) == 1
    points, crop, observed, silhouette, origin = model.inputs
    u, v = 79 - origin[0], 50 - origin[1]
    assert silhouette[v, u] > 0
    assert observed[v, u] == 0
    assert crop[v, u, 1] == 150
    assert np.all(points[:, 2] > 0)
    np.testing.assert_array_equal(original, depth)
    assert results[0].diagnostics["shape_input_contract"] == model.input_contract
    assert not results[0].selected


def test_model_contract_is_saved_loaded_and_unknown_is_rejected(tmp_path):
    features = np.random.default_rng(42).normal(size=(12, len(FUSED_FEATURE_NAMES))).astype(np.float32)
    model = DepthGeometryModel.fit(features, ["a"] * 6 + ["b"] * 6, feature_names=FUSED_FEATURE_NAMES)
    assert model.input_contract == "depth_owned_v1"
    model.edge_parameters["input_contract"] = "rgb_silhouette_depth_owned_v2"
    model.save(tmp_path / "model.npz")
    restored = DepthGeometryModel.load(tmp_path / "model.npz")
    assert restored.input_contract == model.input_contract
    np.testing.assert_allclose(restored.predict_features(features[0])[1], model.predict_features(features[0])[1])
    model.edge_parameters["input_contract"] = "unknown"
    model.save(tmp_path / "bad.npz")
    with pytest.raises(ValueError, match="input contract"):
        DepthGeometryModel.load(tmp_path / "bad.npz")


def test_side_image_cache_defers_wrong_background_segmentation(tmp_path):
    import cv2
    import json

    directory = tmp_path / "samples" / "object"
    directory.mkdir(parents=True)
    image = np.full((50, 70, 3), 200, np.uint8)
    cv2.imwrite(str(directory / "side-color.png"), image)
    (directory / "metadata.json").write_text(json.dumps({"label_id": "cube", "split": "train"}), encoding="utf-8")
    samples, errors = load_side_training_samples(tmp_path / "samples", load_shape_registry(), image,
        require_reviewed=False, defer_segmentation=True, image_cache_directory=tmp_path / "cache")
    assert not errors
    assert len(samples) == 1
    assert isinstance(samples[0].image_bgr, np.memmap)
    assert not np.any(samples[0].mask)
    np.testing.assert_array_equal(samples[0].image_bgr, image)
