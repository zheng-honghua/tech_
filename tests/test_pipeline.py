import cv2

from sorting_vision.calibration import PerspectiveCalibration
from sorting_vision.config import load_config
from sorting_vision.pipeline import VisionPipeline
from sorting_vision.synthetic import SyntheticObject, competition_demo_scene, make_scene
from sorting_vision.types import DetectionStatus


def _pipeline(background):
    config = load_config()
    height, width = background.shape[:2]
    calibration = PerspectiveCalibration.identity(
        width, height, config.tray.width_mm, config.tray.height_mm
    )
    return VisionPipeline(config=config, calibration=calibration, background=background)


def test_complete_scene_and_temporal_selection():
    background, scene, expected = competition_demo_scene()
    pipeline = _pipeline(background)

    first = pipeline.process(scene, frame_id="scene")
    second = pipeline.process(scene, frame_id="scene")

    assert len(second) == len(expected)
    assert not any(result.selected for result in first)
    assert sum(result.selected for result in second) == 1
    assert all(result.status == DetectionStatus.PICKABLE for result in second)
    expected_classes = sorted(f"{item.color_id}:{item.shape_id}" for item in expected)
    assert sorted(result.class_key for result in second) == expected_classes


def test_pose_uses_millimetres_and_symmetric_angle_is_null():
    background, scene = make_scene(
        [SyntheticObject("blue", "circle", (400, 600), (100, 100))]
    )
    pipeline = _pipeline(background)
    pipeline.process(scene)
    result = pipeline.process(scene)[0]

    assert abs(result.center_mm.x - 80.0) < 1.0
    assert abs(result.center_mm.y - 40.0) < 1.0
    assert result.angle_deg is None


def test_unknown_object_is_not_selected():
    background, scene = make_scene([])
    cv2.circle(scene, (400, 400), 50, (130, 130, 130), -1)
    pipeline = _pipeline(background)
    pipeline.process(scene)
    result = pipeline.process(scene)[0]

    assert result.color_id == "unknown"
    assert result.status == DetectionStatus.UNCERTAIN
    assert result.selected is False


def test_reset_requires_new_stable_observation():
    background, scene = make_scene(
        [SyntheticObject("red", "rectangle", (300, 300), (130, 75), 20)]
    )
    pipeline = _pipeline(background)
    pipeline.process(scene)
    assert pipeline.process(scene)[0].selected
    pipeline.reset_tracking()
    assert not pipeline.process(scene)[0].selected

