import json
from argparse import Namespace

import numpy as np

from sorting_vision.geometry_rgbd_model import (
    BASE_FEATURE_NAMES,
    DepthGeometryModel,
    FEATURE_NAMES,
    FUSED_FEATURE_NAMES,
    detect_rgb_object_support,
    detect_tray_roi_mask,
    constrain_tray_roi,
)
import cv2
import pytest
from sorting_vision.rgbd import CameraIntrinsics, RGBDFrame
from sorting_vision.rgbd_dataset import audit_rgbd_dataset, save_rgbd_dataset_sample
from sorting_vision import cli


def _frame(frame_id="d415-1"):
    intrinsics = CameraIntrinsics(8, 6, 10, 10, 4, 3, 1.0)
    return RGBDFrame(
        np.full((6, 8, 3), 100, np.uint8),
        np.full((6, 8), 420, np.uint16),
        intrinsics, 1000, frame_id, 900, 950,
    )


def test_rgbd_sample_is_saved_as_self_contained_folder(tmp_path):
    target = save_rgbd_dataset_sample(
        _frame(), tmp_path, "batch-01", "三棱柱",
        {"camera_model": "Intel RealSense D415"}, sample_id="sample-01",
    )
    assert (target / "color.png").is_file()
    assert (target / "depth.npy").is_file()
    assert (target / "depth-preview.png").is_file()
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["label_id"] == "triangular_prism"
    assert metadata["depth_scale_to_mm"] == 1.0
    report = audit_rgbd_dataset(tmp_path)
    assert report["samples"] == 1
    assert report["errors"] == []


def test_depth_geometry_model_save_load_is_deterministic(tmp_path):
    width = len(FEATURE_NAMES)
    features = np.vstack(
        [np.zeros((3, width), np.float32), np.full((3, width), 3, np.float32)]
    )
    labels = ["triangular_prism"] * 3 + ["octahedron"] * 3
    model = DepthGeometryModel.fit(features, labels)
    expected = model.predict_features(features[0])
    path = tmp_path / "depth-model.npz"
    model.save(path)
    loaded = DepthGeometryModel.load(path)
    assert loaded.predict_features(features[0]) == expected
    assert loaded.exemplars is not None
    assert loaded.neighbors == 3


def test_depth_geometry_model_recovers_training_data():
    width = len(FEATURE_NAMES)
    features = np.arange(8 * width, dtype=np.float32).reshape(8, width)
    labels = ["a"] * 4 + ["b"] * 4
    model = DepthGeometryModel.fit(features, labels)

    recovered, recovered_labels = model.training_data()

    assert recovered_labels == labels
    assert np.allclose(recovered, features, atol=1e-4)


def test_weighted_depth_model_preserves_raw_data_and_v3_roundtrip(tmp_path):
    rng = np.random.default_rng(22)
    x = rng.normal(size=(24, len(FEATURE_NAMES))).astype(np.float32)
    x[12:] += 3
    labels = ['a'] * 12 + ['b'] * 12
    weights = np.ones(len(FEATURE_NAMES), np.float32)
    weights[20:] = .25
    model = DepthGeometryModel.fit(x, labels, feature_weights=weights)
    plain = DepthGeometryModel.fit(x, labels)
    assert not np.allclose(model.scale, plain.scale)
    np.testing.assert_allclose(model.training_data()[0], x, atol=1e-5)
    path = tmp_path / 'weighted.npz'
    model.save(path)
    loaded = DepthGeometryModel.load(path)
    with np.load(path, allow_pickle=False) as data:
        assert str(data['model_type']) == 'rgbd_geometry_v3_multipose_knn'
    for row in x:
        assert loaded.predict_features(row) == model.predict_features(row)
    assert loaded.predict_features(np.full(x.shape[1], 1000))[0] == 'unknown'
    assert loaded.predict_features(np.full(x.shape[1], np.nan)) == ('unknown', 0., 'invalid_features')
    assert loaded.predict_features(np.empty(0)) == ('unknown', 0., 'invalid_features')


def test_fused_edge_model_uses_v4_schema_and_roundtrips(tmp_path):
    rng = np.random.default_rng(23)
    features = rng.normal(size=(24, len(FUSED_FEATURE_NAMES))).astype(np.float32)
    features[12:] += 2.5
    labels = ["a"] * 12 + ["b"] * 12
    model = DepthGeometryModel.fit(features, labels)
    assert model.feature_names == FUSED_FEATURE_NAMES
    assert model.edge_parameters["version"] == 1
    path = tmp_path / "fused-v4.npz"
    model.save(path)
    loaded = DepthGeometryModel.load(path)
    with np.load(path, allow_pickle=False) as data:
        assert str(data["model_type"]) == "rgbd_geometry_v4_fused_edges"
    assert loaded.feature_names == FUSED_FEATURE_NAMES
    assert loaded.edge_parameters == model.edge_parameters
    for row in features:
        assert loaded.predict_features(row) == model.predict_features(row)


def test_weighted_depth_model_validates_inputs_and_uniform_equivalence():
    x = np.arange(8 * len(FEATURE_NAMES), dtype=np.float32).reshape(8, -1)
    labels = ['a'] * 4 + ['b'] * 4
    plain = DepthGeometryModel.fit(x, labels)
    equal = DepthGeometryModel.fit(x, labels, feature_weights=np.ones(x.shape[1]))
    np.testing.assert_array_equal(equal.scale, plain.scale)
    np.testing.assert_array_equal(equal.thresholds, plain.thresholds)
    for value in (0., -1., np.nan, np.inf):
        weights = np.ones(x.shape[1])
        weights[0] = value
        with pytest.raises(ValueError, match='weights'):
            DepthGeometryModel.fit(x, labels, feature_weights=weights)
    with pytest.raises(ValueError, match='weights'):
        DepthGeometryModel.fit(x, labels, feature_weights=np.ones(3))
    with pytest.raises(ValueError, match='samples or labels'):
        DepthGeometryModel.fit(x, labels[:-1])
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match='samples or labels'):
        DepthGeometryModel.fit(x, labels)


def test_depth_geometry_multipose_uses_nearby_exemplar():
    width = len(FEATURE_NAMES)
    a = np.zeros((4, width), np.float32)
    a[2:] = 8.0
    b = np.full((4, width), 4.0, np.float32)
    model = DepthGeometryModel.fit(np.vstack((a, b)), ["a"] * 4 + ["b"] * 4)
    label, confidence, reason = model.predict_features(np.full(width, 8.0, np.float32))
    assert label == "a"
    assert confidence > 0.75
    assert reason == "accepted"


def test_headless_rgbd_capture_writes_one_bundle(monkeypatch, tmp_path):
    class Source:
        def __init__(self):
            self.closed = False

        def read(self):
            return _frame()

        def capture_metadata(self):
            return {"camera_model": "fake D415"}

        def close(self):
            self.closed = True

    source = Source()
    monkeypatch.setattr(cli, "_make_camera_source", lambda args, config: source)
    args = Namespace(
        config=None, headless=True, count=1, discard_frames=0,
        dataset_root=str(tmp_path), batch_id="batch-01", label="正八面体",
    )
    assert cli._run_rgbd_capture(args) == 0
    assert audit_rgbd_dataset(tmp_path)["class_counts"] == {"octahedron": 1}
    assert source.closed is True


def test_depth_model_loader_keeps_v1_feature_compatibility(tmp_path):
    feature_count = len(BASE_FEATURE_NAMES)
    model = DepthGeometryModel(
        ["octahedron", "triangular_prism"],
        np.zeros(feature_count, np.float32),
        np.ones(feature_count, np.float32),
        np.vstack((np.zeros(feature_count), np.full(feature_count, 3))).astype(np.float32),
        np.ones(2, np.float32),
        feature_names=BASE_FEATURE_NAMES,
    )
    path = tmp_path / "legacy-rgbd-model.npz"
    model.save(path)
    loaded = DepthGeometryModel.load(path)
    assert loaded.feature_names == BASE_FEATURE_NAMES
    assert loaded.predict_features(np.zeros(len(FEATURE_NAMES), np.float32))[0] == "octahedron"


def test_tray_roi_selects_large_cool_white_rectangle():
    image = np.full((300, 400, 3), (85, 110, 135), np.uint8)
    cv2.rectangle(image, (170, 35), (375, 275), (178, 166, 146), -1)
    cv2.rectangle(image, (15, 20), (70, 80), (178, 166, 146), -1)
    roi = detect_tray_roi_mask(image)
    assert roi[150, 270] == 255
    assert roi[50, 40] == 0
    assert roi[10, 10] == 0


def test_tray_roi_rejects_larger_bright_region_touching_image_border():
    image = np.full((300, 500, 3), (80, 105, 130), np.uint8)
    cv2.rectangle(image, (0, 0), (190, 299), (205, 210, 215), -1)
    cv2.rectangle(image, (230, 35), (470, 275), (235, 235, 235), -1)
    roi = detect_tray_roi_mask(image)
    assert roi[150, 350] == 255
    assert roi[150, 100] == 0


def test_rgb_object_support_selects_coloured_block():
    image = np.full((180, 240, 3), (220, 225, 230), np.uint8)
    tray = np.zeros((180, 240), np.uint8)
    tray[20:160, 30:210] = 255
    image[70:120, 90:150] = (180, 50, 30)
    support = detect_rgb_object_support(image, tray)
    assert support[95, 120] == 255
    assert support[40, 60] == 0


def test_rgb_object_support_rejects_thin_tray_rim_colour_band():
    image = np.full((300, 400, 3), (230, 230, 230), np.uint8)
    tray = np.full((300, 400), 255, np.uint8)
    image[15:21, 40:360] = (170, 185, 195)
    image[100:180, 150:240] = (180, 50, 30)
    support = detect_rgb_object_support(image, tray)
    assert support[18, 200] == 0
    assert support[140, 195] == 255


def test_workspace_constraint_blocks_roi_expansion_and_rejects_mismatch():
    reference = np.zeros((100, 150), np.uint8)
    reference[20:80, 60:130] = 255
    proposal = np.zeros_like(reference)
    proposal[5:90, 5:145] = 255
    np.testing.assert_array_equal(constrain_tray_roi(proposal, reference), reference)
    proposal[:] = 0
    proposal[20:80, :45] = 255
    with pytest.raises(ValueError, match='reference_mismatch'):
        constrain_tray_roi(proposal, reference)


def test_workspace_constraint_preserves_normal_tray_translation():
    reference = np.zeros((100, 150), np.uint8)
    reference[20:80, 60:130] = 255
    translated = np.zeros_like(reference)
    translated[20:80, 70:140] = 255
    np.testing.assert_array_equal(constrain_tray_roi(translated, reference), translated)
    with pytest.raises(ValueError, match='dimensions'):
        constrain_tray_roi(translated[:20], reference)
