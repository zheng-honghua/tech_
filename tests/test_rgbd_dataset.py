import json
from argparse import Namespace

import numpy as np

from sorting_vision.geometry_rgbd_model import (
    BASE_FEATURE_NAMES,
    DepthGeometryModel,
    FEATURE_NAMES,
    detect_rgb_object_support,
    detect_tray_roi_mask,
)
import cv2
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
