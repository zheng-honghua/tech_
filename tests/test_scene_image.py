import json

import cv2
import numpy as np

from sorting_vision.cli import main
from sorting_vision.geometry_rgb import train_geometry_model
from sorting_vision.scene_image import GeometryScenePredictor, save_scene_image_result


def _object_image(kind: str, shift: int = 0) -> np.ndarray:
    image = np.full((240, 320, 3), 205, np.uint8)
    colour = (190, 150, 25)
    dark = (80, 65, 15)
    if kind == "triangle":
        points = np.asarray([[160 + shift, 48], [95 + shift, 188], [225 + shift, 188]])
        cv2.fillPoly(image, [points], colour)
        cv2.line(image, tuple(points[0]), (160 + shift, 148), dark, 4)
        cv2.line(image, (160 + shift, 148), tuple(points[1]), dark, 4)
    else:
        cv2.rectangle(image, (92 + shift, 62), (228 + shift, 182), colour, -1)
        cv2.line(image, (92 + shift, 122), (228 + shift, 122), dark, 4)
    return image


def _edge_model(tmp_path):
    root = tmp_path / "training"
    for folder, kind in (("三棱锥", "triangle"), ("五棱柱", "rectangle")):
        target = root / folder
        target.mkdir(parents=True)
        for index, shift in enumerate((-8, 0, 8)):
            assert cv2.imwrite(str(target / f"{index}.png"), _object_image(kind, shift))
    model, _ = train_geometry_model(root, feature_set="edge-topology")
    path = tmp_path / "scene-model.npz"
    model.save(path)
    return path


def _scene() -> np.ndarray:
    image = np.full((480, 720, 3), 205, np.uint8)
    colour = (190, 150, 25)
    dark = (80, 65, 15)
    first = np.asarray([[115, 65], [55, 190], [175, 190]])
    cv2.fillPoly(image, [first], colour)
    cv2.line(image, tuple(first[0]), (115, 155), dark, 4)
    cv2.rectangle(image, (285, 85), (430, 205), colour, -1)
    cv2.line(image, (285, 145), (430, 145), dark, 4)
    second = np.asarray([[600, 255], [535, 400], [665, 400]])
    cv2.fillPoly(image, [second], colour)
    cv2.line(image, tuple(second[0]), (600, 360), dark, 4)
    return image


def test_scene_predictor_detects_and_exports_three_objects(tmp_path):
    predictor = GeometryScenePredictor.load(_edge_model(tmp_path))
    result = predictor.predict(_scene())
    assert len(result.objects) == 3
    assert [item.object_id for item in result.objects] == [
        "object-001",
        "object-002",
        "object-003",
    ]
    assert all(item.topology is not None for item in result.objects)
    assert result.to_dict()["safe_for_robot"] is False

    output = tmp_path / "scene-output"
    save_scene_image_result(result, output)
    payload = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert payload["detected_count"] == 3
    assert (output / "annotated.jpg").is_file()
    assert (output / "object-001" / "topology.png").is_file()
    assert (output / "object-003" / "topology.json").is_file()


def test_predict_scene_cli_is_machine_readable(tmp_path, capsys):
    model_path = _edge_model(tmp_path)
    image_path = tmp_path / "scene.png"
    assert cv2.imwrite(str(image_path), _scene())
    output = tmp_path / "cli-output"
    exit_code = main(
        [
            "predict-scene",
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
    assert payload["detected_count"] == 3
    assert len(payload["objects"]) == 3
