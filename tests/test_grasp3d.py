import numpy as np
from dataclasses import replace

from sorting_vision.config import load_config
from sorting_vision.geometry3d import segment_depth_objects
from sorting_vision.grasp3d import find_suction_grasp
from sorting_vision.rgbd import Plane, RGBDCalibration
from sorting_vision.synthetic3d import SyntheticSolid, make_rgbd_scene


def _candidate(solid, min_area_px=None):
    config = load_config()
    if min_area_px is not None:
        config = replace(config, rgbd=replace(config.rgbd, min_area_px=min_area_px))
    _, scene = make_rgbd_scene([solid])
    plane = Plane([0, 0, -1], 700)
    calibration = RGBDCalibration(scene.intrinsics, np.eye(4), plane)
    objects, _ = segment_depth_objects(
        scene.color_bgr, scene.depth_mm, scene.intrinsics, plane, config.rgbd
    )
    assert len(objects) == 1
    return find_suction_grasp(objects[0], scene.depth_mm, calibration, config.grasp)


def test_flat_surface_produces_robot_pose():
    candidate = _candidate(
        SyntheticSolid("red", "cube", (320, 240), (50, 50), 28, 10, 5, -3)
    )
    assert candidate is not None
    assert candidate.info.score >= 0.72
    assert candidate.pose.position_mm.z < 680
    normal = candidate.pose.surface_normal
    assert normal.z < -0.98
    approach = candidate.pose.approach_vector
    assert approach.z > 0.98


def test_surface_smaller_than_suction_cup_is_rejected():
    assert _candidate(
        SyntheticSolid("red", "cube", (320, 240), (10, 10), 20),
        min_area_px=50,
    ) is None


def test_excessive_surface_tilt_is_rejected():
    assert _candidate(
        SyntheticSolid("red", "cube", (320, 240), (60, 60), 30, 0, 48, 0)
    ) is None
