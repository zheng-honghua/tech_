import cv2
import numpy as np

from sorting_vision.geometry_rgb import (
    GeometryRGBModel,
    audit_geometry_dataset,
    evaluate_geometry_model,
    export_geometry_results,
    extract_geometry_features,
    preprocess_geometry_object,
    train_geometry_model,
)


def _scene(kind: str, shift: int = 0) -> np.ndarray:
    image = np.full((240, 320, 3), 205, np.uint8)
    color = (205, 150, 35)
    if kind == "triangle":
        points = np.asarray([[160 + shift, 55], [95 + shift, 185], [225 + shift, 185]])
        cv2.fillPoly(image, [points], color)
        cv2.line(image, tuple(points[0]), (160 + shift, 150), (110, 80, 25), 4)
    else:
        cv2.rectangle(image, (95 + shift, 65), (225 + shift, 180), color, -1)
        cv2.line(image, (95 + shift, 120), (225 + shift, 120), (110, 80, 25), 4)
    return image


def _dataset(tmp_path):
    root = tmp_path / "geometry"
    for folder, kind in (("三棱锥", "triangle"), ("五棱柱", "rectangle")):
        target = root / folder
        target.mkdir(parents=True)
        for index, shift in enumerate((-8, 0, 8)):
            assert cv2.imwrite(str(target / f"{index}.png"), _scene(kind, shift))
    return root


def test_preprocessing_prefers_central_object_over_border_clutter():
    image = _scene("triangle")
    cv2.rectangle(image, (0, 20), (35, 190), (205, 150, 35), -1)
    prepared = preprocess_geometry_object(image)
    assert prepared is not None
    x, _, width, _ = prepared.bbox_px
    assert x > 50
    assert width > 100


def test_audit_train_save_load_and_unknown_rejection(tmp_path):
    root = _dataset(tmp_path)
    audit = audit_geometry_dataset(root)
    assert audit["total_images"] == 6
    assert audit["preprocessing_success"] == 6
    assert audit["errors"] == []

    model, report = train_geometry_model(root)
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = GeometryRGBModel.load(path)
    prepared = preprocess_geometry_object(_scene("triangle"))
    feature = extract_geometry_features(prepared)
    loaded_label, loaded_confidence, loaded_diagnostics = loaded.predict_feature(feature)
    original_label, original_confidence, original_diagnostics = model.predict_feature(feature)
    assert loaded_label == original_label
    assert loaded_confidence == original_confidence
    assert loaded_diagnostics["reason"] == original_diagnostics["reason"]
    assert report["training_samples"] == 6
    label, confidence, diagnostics = loaded.predict(
        np.full((128, 128, 3), 205, np.uint8)
    )
    assert label == "unknown"
    assert confidence == 0.0
    assert diagnostics["reason"] == "object_not_found"


def test_leave_one_out_report_is_explicitly_same_batch_only(tmp_path):
    root = _dataset(tmp_path)
    model, _ = train_geometry_model(root)
    path = tmp_path / "model.npz"
    model.save(path)
    report = evaluate_geometry_model(root, path)
    assert report["same_batch_only"] is True
    assert report["evaluation"] == "leave_one_out"
    assert report["samples"] == 6
    assert len(report["confusion_matrix"]) == 2


def test_model_rejects_multiple_central_objects(tmp_path):
    root = _dataset(tmp_path)
    model, _ = train_geometry_model(root)
    image = np.full((240, 320, 3), 205, np.uint8)
    cv2.circle(image, (100, 120), 38, (205, 150, 35), -1)
    cv2.circle(image, (220, 120), 38, (205, 150, 35), -1)
    label, confidence, diagnostics = model.predict(image)
    assert label == "unknown"
    assert confidence == 0.0
    assert diagnostics["reason"] == "multiple_objects"


def test_export_collects_images_masks_predictions_and_summary(tmp_path):
    root = _dataset(tmp_path)
    model, _ = train_geometry_model(root)
    model_path = tmp_path / "model.npz"
    model.save(model_path)
    output = tmp_path / "results"
    summary = export_geometry_results(root, model_path, output)
    assert summary["exported_images"] == 6
    assert len((output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 6
    assert (output / "summary.json").is_file()
    assert (output / "confusion_matrix.csv").is_file()
    assert len(list((output / "按真实类别").rglob("annotated.jpg"))) == 6
