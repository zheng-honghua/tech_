from dataclasses import replace

import numpy as np

from sorting_vision.config import RGBDConfig
from sorting_vision.geometry3d import segment_depth_objects
from sorting_vision.rgbd import CameraIntrinsics, Plane
from sorting_vision.config import DualViewConfig
from sorting_vision.dual_view import ProjectedROI, _side_mask


def test_hsv_groups_depth_fragments_without_filling_missing_depth():
    color = np.full((80, 100, 3), 240, np.uint8)
    color[20:60, 20:65] = (0, 150, 0)
    depth = np.full((80, 100), 600., np.float32)
    depth[20:60, 20:65] = 580
    depth[20:60, 40:45] = 0
    intrinsics = CameraIntrinsics(100, 80, 100, 100, 50, 40, 1)
    plane = Plane(np.array([0., 0., -1.]), 600)
    cfg = RGBDConfig(instance_segmentation="hsv", min_area_px=30)
    objects, _ = segment_depth_objects(color, depth, intrinsics, plane, cfg)
    assert len(objects) == 1
    assert not np.any(objects[0].mask[20:60, 40:45])
    assert objects[0].valid_depth_ratio < 1
    color[20:60, 40:45] = 240
    separated, _ = segment_depth_objects(color, depth, intrinsics, plane, cfg)
    assert len(separated) == 2
    empty, _ = segment_depth_objects(color, np.zeros_like(depth), intrinsics, plane, cfg)
    assert not empty


def test_side_hsv_keeps_similar_faces_together_and_excludes_shadow():
    image = np.full((100, 100, 3), 220, np.uint8)
    background = image.copy()
    image[30:65, 30:48] = (15, 150, 15)
    image[30:65, 48:65] = (10, 70, 15)
    image[68:78, 30:65] = 90  # neutral shadow, not coloured foreground
    roi = ProjectedROI((0, 0, 100, 100), np.array([[10, 10], [90, 10], [90, 90], [10, 90]]), 1.)
    mask, pixels, _, _ = _side_mask(image, background, roi,
        DualViewConfig(side_foreground_method="hsv"))
    assert pixels > 900
    assert mask[45, 40] and mask[45, 55]
    assert not mask[73, 45]
