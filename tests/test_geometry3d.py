import numpy as np

from sorting_vision.config import load_config
from sorting_vision.geometry3d import (
    estimate_plane_shift_mm,
    segment_depth_objects,
)
from sorting_vision.rgbd import Plane
from sorting_vision.synthetic3d import SyntheticSolid, make_rgbd_scene


def test_depth_plane_segmentation_ignores_empty_tray():
    config = load_config()
    background, _ = make_rgbd_scene([])
    plane = Plane([0, 0, -1], 700)
    objects, heights = segment_depth_objects(
        background.color_bgr,
        background.depth_mm,
        background.intrinsics,
        plane,
        config.rgbd,
    )
    assert objects == []
    assert np.max(np.abs(heights)) < 1e-5


def test_depth_segmentation_extracts_separate_solids():
    config = load_config()
    _, scene = make_rgbd_scene(
        [
            SyntheticSolid("red", "cube", (180, 220), (48, 48)),
            SyntheticSolid("blue", "cylinder", (440, 220), (48, 48)),
        ]
    )
    plane = Plane([0, 0, -1], 700)
    objects, _ = segment_depth_objects(
        scene.color_bgr, scene.depth_mm, scene.intrinsics, plane, config.rgbd
    )
    assert len(objects) == 2
    assert all(item.height_max_mm > 20 for item in objects)
    assert all(item.valid_depth_ratio > 0.99 for item in objects)


def test_depth_segmentation_respects_tray_roi():
    config = load_config()
    _, scene = make_rgbd_scene(
        [
            SyntheticSolid("red", "cube", (180, 220), (48, 48)),
            SyntheticSolid("blue", "cylinder", (440, 220), (48, 48)),
        ]
    )
    roi = np.zeros(scene.depth.shape, np.uint8)
    roi[:, :320] = 255
    objects, _ = segment_depth_objects(
        scene.color_bgr,
        scene.depth_mm,
        scene.intrinsics,
        Plane([0, 0, -1], 700),
        config.rgbd,
        roi_mask=roi,
    )
    assert len(objects) == 1
    assert objects[0].bbox[0] < 320


def test_depth_segmentation_respects_rgb_support():
    config = load_config()
    _, scene = make_rgbd_scene(
        [
            SyntheticSolid("red", "cube", (180, 220), (48, 48)),
            SyntheticSolid("blue", "cylinder", (440, 220), (48, 48)),
        ]
    )
    support = np.zeros(scene.depth.shape, np.uint8)
    support[190:250, 150:210] = 255
    objects, _ = segment_depth_objects(
        scene.color_bgr, scene.depth_mm, scene.intrinsics,
        Plane([0, 0, -1], 700), config.rgbd, support_mask=support,
    )
    assert len(objects) == 1
    assert objects[0].bbox[0] < 320


def test_single_target_mode_does_not_watershed_one_object():
    config = load_config()
    _, scene = make_rgbd_scene(
        [SyntheticSolid("red", "cube", (320, 220), (90, 60))]
    )
    objects, _ = segment_depth_objects(
        scene.color_bgr, scene.depth_mm, scene.intrinsics,
        Plane([0, 0, -1], 700), config.rgbd,
        split_touching_objects=False,
    )
    assert len(objects) == 1


def test_plane_shift_detects_moved_tray():
    config = load_config()
    _, frame = make_rgbd_scene([], tray_depth_mm=708)
    shift = estimate_plane_shift_mm(
        frame.depth_mm,
        frame.intrinsics,
        Plane([0, 0, -1], 700),
        config.rgbd,
    )
    assert abs(shift + 8.0) < 0.1
