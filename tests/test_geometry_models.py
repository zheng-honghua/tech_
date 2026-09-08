import json
from pathlib import Path

import cv2
import numpy as np
from types import SimpleNamespace
import pytest

from sorting_vision.classification import HybridShapeClassifier
from sorting_vision.classification3d import HybridShapeClassifier3D
from sorting_vision.config import load_config
from sorting_vision.geometry_cnn import (
    OpenVINOGeometryModel,
    _predict_torch,
    benchmark_geometry_backend,
    cnn_input_tensor,
    export_geometry_backend_results,
    load_cnn_training_samples,
    load_geometry_shape_model,
)
from sorting_vision.geometry_models import EnsembleGeometryModel, GeometryPrediction
from sorting_vision.geometry_rgb import train_geometry_model
from sorting_vision.geometry_rgb import GeometrySample
from sorting_vision.rgbd import CameraIntrinsics, RGBDFrame


def _object_image() -> np.ndarray:
    image = np.full((240, 320, 3), 205, np.uint8)
    points = np.asarray([[160, 45], [95, 185], [225, 185]])
    cv2.fillPoly(image, [points], (205, 150, 35))
    cv2.line(image, (160, 45), (160, 155), (90, 70, 20), 4)
    return image


class _FakeCompiledModel:
    def __init__(self, rows):
        self.rows = np.asarray(rows, np.float32)
        self.calls = 0
        self.batch_sizes = []

    def __call__(self, inputs):
        self.calls += 1
        batch = len(inputs[0])
        self.batch_sizes.append(batch)
        return {"logits": np.repeat(self.rows[:1], batch, axis=0)}


def test_geometry_prediction_schema_and_cnn_tensor():
    prediction = GeometryPrediction("a", "A", 0.8, True, "opencv", "accepted")
    assert prediction.to_dict()["backend"] == "opencv"
    tensor = cnn_input_tensor(np.full((192, 192, 3), 127, np.uint8))
    assert tensor.shape == (3, 192, 192)
    assert tensor.dtype == np.float32


def test_openvino_backend_accepts_and_rejects_predictions():
    accepted_runtime = _FakeCompiledModel([[5.0, 0.1, -1.0]])
    model = OpenVINOGeometryModel(
        accepted_runtime,
        ["one", "two", "three"],
        {"one": "一", "two": "二", "three": "三"},
    )
    prediction = model.predict_geometry(_object_image())
    assert prediction.label_id == "one"
    assert prediction.accepted
    assert prediction.backend == "openvino"
    assert prediction.top_candidates[0].label_id == "one"

    uncertain_runtime = _FakeCompiledModel([[1.0, 0.95, 0.1]])
    uncertain = OpenVINOGeometryModel(
        uncertain_runtime,
        ["one", "two", "three"],
        {"one": "一", "two": "二", "three": "三"},
        confidence_threshold=0.0,
        margin_threshold=0.12,
    ).predict_geometry(_object_image())
    assert uncertain.label_id == "unknown"
    assert uncertain.reason == "margin_rejected"


def test_openvino_batch_uses_one_runtime_call_and_preserves_rejections():
    runtime = _FakeCompiledModel([[4.0, 0.0]])
    model = OpenVINOGeometryModel(runtime, ["one", "two"], {"one": "一", "two": "二"})
    blank = np.full((240, 320, 3), 205, np.uint8)
    predictions = model.predict_batch(
        [(_object_image(), None), (_object_image(), None), (blank, None)]
    )
    assert len(predictions) == 3
    assert runtime.calls == 1
    assert runtime.batch_sizes == [2]
    assert predictions[2].label_id == "unknown"
    assert predictions[2].reason == "object_not_found"


def test_shape_classifier_uses_common_model_without_2d_fusion():
    runtime = _FakeCompiledModel([[5.0, 0.0]])
    model = OpenVINOGeometryModel(
        runtime,
        ["triangular_pyramid", "octahedron"],
        {"triangular_pyramid": "三棱锥", "octahedron": "正八面体"},
    )
    classifier = HybridShapeClassifier(load_config().classification, model=model)
    mask = np.full(_object_image().shape[:2], 255, np.uint8)
    prediction = classifier.classify(_object_image(), mask)
    assert prediction.label_id == "triangular_pyramid"


def test_rgbd_classifier_recovers_large_flat_hexagonal_prism_from_octahedron():
    class OctahedronModel:
        last_diagnostics = {}

        def classify(self, *args, **kwargs):
            return "octahedron", 0.92

    x, y = np.meshgrid(np.linspace(0, 50, 12), np.linspace(0, 42, 10))
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 300.0)))
    mask = np.full((80, 100), 255, np.uint8)
    depth = np.full(mask.shape, 300.0, np.float32)
    classifier = HybridShapeClassifier3D(
        load_config().classification, model=OctahedronModel()
    )

    prediction = classifier.classify(
        points, np.zeros((*mask.shape, 3), np.uint8), depth, mask
    )

    assert prediction.label_id == "hexagonal_prism"
    assert prediction.features["metric_hexagonal_override"] == 1.0


def test_rgbd_classifier_rejects_sparse_cloud_before_model():
    class NeverCalled:
        def classify(self, *args, **kwargs):
            raise AssertionError('sparse cloud must not reach the model')

    classifier = HybridShapeClassifier3D(load_config().classification, model=NeverCalled())
    mask = np.ones((10, 10), np.uint8) * 255
    result = classifier.classify(
        np.zeros((12, 3)), np.zeros((10, 10, 3), np.uint8),
        np.ones((10, 10), np.float32) * 300, mask,
    )
    assert result.label_id == 'unknown'
    assert result.confidence == 0


def test_rgbd_training_strict_mode_rejects_extra_components(monkeypatch, tmp_path):
    import sorting_vision.geometry_rgbd_model as model_module

    entries = [
        {"label_id": "empty_tray", "batch_id": "batch", "sample_dir": "empty",
         "absolute_sample_dir": "empty"},
        *[
            {"label_id": label, "batch_id": "batch", "sample_dir": label,
             "absolute_sample_dir": label}
            for label in ("alpha", "beta", "gamma")
        ],
    ]
    monkeypatch.setattr(model_module, "load_rgbd_dataset_entries", lambda _: entries)
    frame = RGBDFrame(
        np.zeros((80, 80, 3), np.uint8), np.full((80, 80), 300, np.float32),
        CameraIntrinsics(80, 80, 70, 70, 40, 40), 0, "test-frame",
    )
    monkeypatch.setattr(model_module, "load_rgbd_frame", lambda _: frame)
    monkeypatch.setattr(
        model_module, "detect_tray_roi_mask", lambda image: np.full(image.shape[:2], 255, np.uint8)
    )
    monkeypatch.setattr(model_module, "_fit_background_plane", lambda *args: object())
    monkeypatch.setattr(
        model_module, "detect_rgb_object_support", lambda image, roi: np.full(image.shape[:2], 255, np.uint8)
    )
    object_mask = np.full(frame.depth_mm.shape, 255, np.uint8)
    candidates = iter([
        [SimpleNamespace(area=20, bbox=(0, 0, 2, 2), mask=object_mask),
         SimpleNamespace(area=10, bbox=(0, 0, 2, 2), mask=object_mask)],
        [SimpleNamespace(area=20, bbox=(0, 0, 2, 2), mask=object_mask)],
        [SimpleNamespace(area=20, bbox=(0, 0, 2, 2), mask=object_mask)],
    ])
    monkeypatch.setattr(model_module, "segment_depth_objects", lambda *args, **kwargs: (next(candidates), None))
    monkeypatch.setattr(model_module, "object_point_cloud", lambda *args, **kwargs: (np.ones((50, 3)), None))
    feature_values = iter([np.ones(len(model_module.FEATURE_NAMES)), np.full(len(model_module.FEATURE_NAMES), 2)])
    monkeypatch.setattr(model_module, "extract_rgbd_geometry_features", lambda *args, **kwargs: next(feature_values))

    report = model_module.train_rgbd_geometry_model(
        "ignored", tmp_path / "strict.npz", strict_single_object=True
    )

    assert report["accepted_samples"] == 2
    assert report["samples_with_extra_components"] == 1
    assert report["training_object_selection"] == "strict_single_rgb_depth_component"
    assert report["rejected"] == [{"sample_dir": "alpha", "reason": "expected_one_object_got_2"}]


def test_ensemble_is_reserved_but_not_silently_enabled():
    runtime = _FakeCompiledModel([[5.0, 0.0]])
    model = OpenVINOGeometryModel(runtime, ["one", "two"], {"one": "一", "two": "二"})
    with pytest.raises(RuntimeError, match="intentionally disabled"):
        EnsembleGeometryModel(model, model).predict_geometry(_object_image())


def _small_dataset(tmp_path):
    root = tmp_path / "geometry"
    for folder, shape in (("三棱锥", "triangle"), ("五棱柱", "rectangle")):
        target = root / folder
        target.mkdir(parents=True)
        for index, shift in enumerate((-8, 0, 8)):
            image = np.full((240, 320, 3), 205, np.uint8)
            if shape == "triangle":
                points = np.asarray([[160 + shift, 55], [95 + shift, 185], [225 + shift, 185]])
                cv2.fillPoly(image, [points], (205, 150, 35))
            else:
                cv2.rectangle(image, (95 + shift, 65), (225 + shift, 180), (205, 150, 35), -1)
            assert cv2.imwrite(str(target / f"{index}.png"), image)
    return root


def test_cnn_multi_batch_loader_deduplicates_exact_images(tmp_path):
    root = _small_dataset(tmp_path)
    samples, errors, duplicates, roots = load_cnn_training_samples(
        root, [root]
    )
    assert errors == []
    assert len(samples) == 6
    assert len(duplicates) == 6
    assert roots == [root, root]


def test_torch_evaluation_safely_rejects_unpreprocessable_sample():
    pytest.importorskip("torch")
    blank = np.full((240, 320, 3), 205, np.uint8)
    sample = GeometrySample(
        Path("blank.png"), "one", "一", blank, "hash"
    )

    class NeverCalled:
        def __call__(self, inputs):
            raise AssertionError("invalid samples must not reach the CNN")

    assert _predict_torch(NeverCalled(), [sample], ["one", "two"]) == ["unknown"]


def test_common_loader_and_opencv_benchmark(tmp_path):
    root = _small_dataset(tmp_path)
    trained, _ = train_geometry_model(root)
    model_path = tmp_path / "geometry.npz"
    trained.save(model_path)
    loaded = load_geometry_shape_model("opencv", model_path)
    assert loaded.backend == "opencv"
    report = benchmark_geometry_backend(
        root, "opencv", model_path, batch_size=1, warmup=1, iterations=2
    )
    assert report["iterations"] == 2
    assert report["p95_ms"] >= 0.0


def test_backend_export_writes_images_manifest_and_summary(tmp_path, monkeypatch):
    root = _small_dataset(tmp_path)
    runtime = _FakeCompiledModel([[5.0, 0.0]])
    model = OpenVINOGeometryModel(
        runtime,
        ["triangular_pyramid", "pentagonal_prism"],
        {"triangular_pyramid": "三棱锥", "pentagonal_prism": "五棱柱"},
    )
    monkeypatch.setattr(
        "sorting_vision.geometry_cnn.load_geometry_shape_model",
        lambda backend, model_path, device: model,
    )
    output = tmp_path / "exported"
    summary = export_geometry_backend_results(
        root, "openvino", tmp_path / "unused", output
    )
    assert summary["exported_images"] == 6
    assert summary["same_batch_only"] is True
    assert len((output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 6
    assert len(list((output / "按真实类别").rglob("annotated.jpg"))) == 6
    assert (output / "confusion_matrix.csv").is_file()


def test_openvino_metadata_validation_happens_before_optional_import(tmp_path):
    model_dir = tmp_path / "cnn"
    model_dir.mkdir()
    (model_dir / "metadata.json").write_text(
        json.dumps({"model_version": -1}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="version"):
        OpenVINOGeometryModel.load(model_dir)


def test_openvino_metadata_rejects_missing_model_before_optional_import(tmp_path):
    model_dir = tmp_path / "cnn"
    model_dir.mkdir()
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_version": 1,
                "model_file": "missing.xml",
                "labels": ["one", "two"],
                "class_names": {"one": "一", "two": "二"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="model does not exist"):
        OpenVINOGeometryModel.load(model_dir)
