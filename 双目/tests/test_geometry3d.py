import cv2
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


def test_depth_segmentation_merges_small_detached_face_fragment():
    config = load_config()
    height = 480
    width = 640
    color = np.full((height, width, 3), 210, np.uint8)
    depth = np.full((height, width), 700.0, np.float32)
    # Same cyan solid separated by a narrow invalid-height band.  The smaller
    # upper face must rejoin the main face instead of becoming a second object.
    cv2.rectangle(color, (270, 180), (330, 215), (180, 150, 20), -1)
    cv2.rectangle(color, (250, 222), (350, 290), (180, 150, 20), -1)
    color[215:223, 270:331] = (180, 150, 20)  # RGB surface spans depth dropout
    depth[180:216, 270:331] = 691.0
    depth[222:291, 250:351] = 678.0
    objects, _ = segment_depth_objects(
        color, depth,
        make_rgbd_scene([])[0].intrinsics,
        Plane([0, 0, -1], 700),
        config.rgbd,
        split_touching_objects=False,
    )
    assert len(objects) == 1
    assert objects[0].bbox[3] >= 105


def test_fragment_merge_does_not_cross_white_tray_or_different_color_gap():
    from sorting_vision.geometry3d import _merge_fragmented_components

    first = np.zeros((100, 100), np.uint8)
    second = first.copy()
    first[10:30, 35:60] = 255
    second[36:80, 20:80] = 255
    heights = np.full(first.shape, 12.0, np.float32)
    for gap_color in [(200, 200, 200), (200, 190, 170), (20, 20, 200)]:
        color = np.full((100, 100, 3), gap_color, np.uint8)
        color[(first > 0) | (second > 0)] = (180, 150, 20)
        merged = _merge_fragmented_components([first, second], color, heights, 10)
        assert len(merged) == 2


def test_fragment_merge_preserves_missing_depth_values():
    from sorting_vision.geometry3d import _merge_fragmented_components

    first = np.zeros((100, 100), np.uint8)
    second = first.copy()
    first[10:30, 35:60] = 255
    second[36:80, 20:80] = 255
    heights = np.full(first.shape, 12.0, np.float32)
    heights[30:36] = np.nan
    original = heights.copy()
    color = np.full((100, 100, 3), (180, 150, 20), np.uint8)
    assert len(_merge_fragmented_components([first, second], color, heights, 10)) == 1
    np.testing.assert_equal(heights, original)


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
