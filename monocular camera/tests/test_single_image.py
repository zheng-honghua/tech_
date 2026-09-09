import json

import cv2
import numpy as np

from sorting_vision.cli import main
from sorting_vision.geometry_rgb import train_geometry_model
from sorting_vision.single_image import (
    GeometryImagePredictor,
    save_single_image_result,
)


def _image(kind: str, shift: int = 0) -> np.ndarray:
    image = np.full((240, 320, 3), 205, np.uint8)
    if kind == "triangle":
        points = np.asarray([[160 + shift, 50], [95 + shift, 185], [225 + shift, 185]])
        cv2.fillPoly(image, [points], (205, 150, 35))
        cv2.line(image, tuple(points[0]), (160 + shift, 150), (90, 70, 20), 4)
    else:
        cv2.rectangle(image, (95 + shift, 65), (225 + shift, 180), (205, 150, 35), -1)
        cv2.line(image, (95 + shift, 120), (225 + shift, 120), (90, 70, 20), 4)
    return image


def _model_and_image(tmp_path):
    root = tmp_path / "data"
    for folder, kind in (("三棱锥", "triangle"), ("五棱柱", "rectangle")):
        target = root / folder
        target.mkdir(parents=True)
        for index, shift in enumerate((-8, 0, 8)):
            assert cv2.imwrite(str(target / f"{index}.png"), _image(kind, shift))
    model, _ = train_geometry_model(root)
    model_path = tmp_path / "model.npz"
    model.save(model_path)
    image_path = root / "三棱锥" / "1.png"
    return model_path, image_path


def test_single_image_api_and_artifact_export(tmp_path):
    model_path, image_path = _model_and_image(tmp_path)
    predictor = GeometryImagePredictor.load(model_path)
    result = predictor.predict_file(image_path)
    assert result.image_width == 320
    assert result.image_height == 240
    assert result.bbox_px is not None
    assert result.prediction.backend == "opencv"
    assert result.total_ms >= result.prediction.inference_ms
    assert result.analysis_scale == 1.0
    assert result.to_dict()["safe_for_robot"] is False

    output = tmp_path / "prediction"
    artifacts = save_single_image_result(result, output)
    assert set(artifacts) == {
        "annotated_image",
        "result_json",
        "normalized_crop",
        "object_mask",
    }
    payload = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert payload["task"] == "single_image_geometry_prediction"
    assert (output / "annotated.jpg").is_file()


def test_single_image_blank_is_safely_rejected(tmp_path):
    model_path, _ = _model_and_image(tmp_path)
    result = GeometryImagePredictor.load(model_path).predict(
        np.full((240, 320, 3), 205, np.uint8)
    )
    assert result.prediction.label_id == "unknown"
    assert result.prediction.reason == "object_not_found"
    assert not result.prediction.accepted


def test_predict_image_cli_outputs_json_and_files(tmp_path, capsys):
    model_path, image_path = _model_and_image(tmp_path)
    output = tmp_path / "cli-output"
    exit_code = main(
        [
            "predict-image",
            str(image_path),
            "--model",
            str(model_path),
            "--output-dir",
            str(output),
            "--json-only",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["prediction"]["backend"] == "opencv"
    assert payload["safe_for_robot"] is False
    assert (output / "result.json").is_file()
