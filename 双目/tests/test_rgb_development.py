from sorting_vision.calibration import PerspectiveCalibration
from sorting_vision.camera import RGBFrame
from sorting_vision.config import load_config
from sorting_vision.pipeline import VisionPipeline
from sorting_vision.rgb_development import RGBDevelopmentPipeline
from sorting_vision.synthetic import SyntheticObject, make_scene


def test_rgb_mode_never_authorizes_pick():
    background, scene = make_scene(
        [SyntheticObject("red", "rectangle", (300, 300), (130, 75), 20)]
    )
    config = load_config()
    height, width = scene.shape[:2]
    calibration = PerspectiveCalibration.identity(
        width, height, config.tray.width_mm, config.tray.height_mm
    )
    pipeline = RGBDevelopmentPipeline(
        VisionPipeline(config=config, calibration=calibration, background=background)
    )
    frame = RGBFrame(scene, 123, "rgb-test")

    pipeline.process(frame)
    result = pipeline.process(frame)[0]
    assert result.status.value == "DEPTH_REQUIRED"
    assert result.pose_3d is None
    assert result.grasp is None
    assert result.selected is False
    assert pipeline.health()["reason"] == "depth_required"
