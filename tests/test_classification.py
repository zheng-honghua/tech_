import cv2
import numpy as np
import pytest

from sorting_vision.classification import GeometricShapeClassifier, LabColorClassifier
from sorting_vision.config import load_config
from sorting_vision.synthetic import DEFAULT_BGR, SyntheticObject, make_scene


@pytest.mark.parametrize("color_id", list(DEFAULT_BGR))
def test_lab_color_classifier(color_id):
    config = load_config()
    background, image = make_scene(
        [SyntheticObject(color_id, "circle", (200, 200), (100, 100))],
        width=400,
        height=400,
    )
    mask = (cv2.cvtColor(cv2.absdiff(image, background), cv2.COLOR_BGR2GRAY) > 5).astype(np.uint8) * 255
    prediction = LabColorClassifier(config.classification).classify(image, mask)
    assert prediction.label_id == color_id
    assert prediction.confidence >= config.classification.min_color_confidence


@pytest.mark.parametrize(
    "shape_id,size",
    [
        ("triangle", (100, 100)),
        ("square", (100, 100)),
        ("rectangle", (135, 75)),
        ("pentagon", (100, 100)),
        ("hexagon", (100, 100)),
        ("circle", (100, 100)),
    ],
)
def test_geometric_shape_classifier(shape_id, size):
    config = load_config()
    background, image = make_scene(
        [SyntheticObject("red", shape_id, (200, 200), size, 13)],
        width=400,
        height=400,
    )
    mask = (cv2.cvtColor(cv2.absdiff(image, background), cv2.COLOR_BGR2GRAY) > 5).astype(np.uint8) * 255
    prediction = GeometricShapeClassifier(config.classification).classify(mask)
    assert prediction.label_id == shape_id
    assert prediction.confidence >= config.classification.min_shape_confidence


def test_unknown_colour_is_rejected():
    config = load_config()
    image = np.full((200, 200, 3), 238, np.uint8)
    cv2.circle(image, (100, 100), 45, (130, 130, 130), -1)
    mask = np.zeros((200, 200), np.uint8)
    cv2.circle(mask, (100, 100), 45, 255, -1)
    prediction = LabColorClassifier(config.classification).classify(image, mask)
    assert prediction.label_id == "unknown"

