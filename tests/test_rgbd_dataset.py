import json
from argparse import Namespace

import numpy as np

from sorting_vision.geometry_rgbd_model import DepthGeometryModel, FEATURE_NAMES
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
    assert DepthGeometryModel.load(path).predict_features(features[0]) == expected


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
