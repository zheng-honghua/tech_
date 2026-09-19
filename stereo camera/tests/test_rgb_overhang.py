from dataclasses import replace

import cv2
import numpy as np
import pytest

from sorting_vision.config import RGBDConfig, load_config
from sorting_vision.geometry3d import depth_footprint_in_tray, segment_depth_objects
from sorting_vision.geometry_rgbd_model import detect_tray_roi_mask
from sorting_vision.pipeline3d import VisionPipeline3D
from sorting_vision.rgbd import CameraIntrinsics, Plane, RGBDCalibration, RGBDFrame


def scene():
    image = np.full((120, 180, 3), 240, np.uint8)
    image[35:85, 110:135] = (10, 150, 10)
    depth = np.full(image.shape[:2], 600., np.float32)
    depth[35:85, 110:135] = 550
    roi = np.zeros(image.shape[:2], np.uint8)
    roi[15:105, 25:131] = 255
    intrinsics = CameraIntrinsics(180, 120, 100, 100, 80, 60, 1)
    plane = Plane(np.array([0., 0., -1.]), 600)
    cfg = RGBDConfig(instance_segmentation="hsv", allow_rgb_overhang=True, min_area_px=30)
    return image, depth, roi, intrinsics, plane, cfg


def test_complete_overhang_has_real_plane_ownership_without_roi_dilation():
    image, depth, roi, intrinsics, plane, cfg = scene()
    footprint = depth_footprint_in_tray(depth, intrinsics, plane, roi, cfg)
    assert roi[50, 134] == 0 and footprint[50, 134]
    objects, _ = segment_depth_objects(image, depth, intrinsics, plane, cfg, roi_mask=roi)
    assert len(objects) == 1
    assert objects[0].bbox == (110, 35, 25, 50)
    assert objects[0].workspace_support_ratio == 1.
    assert objects[0].rgb_mask[50, 134] and objects[0].mask[50, 134]
    assert not objects[0].touches_border
    legacy, _ = segment_depth_objects(image, depth, intrinsics, plane,
                                     replace(cfg, allow_rgb_overhang=False), roi_mask=roi)
    assert legacy[0].bbox[0] + legacy[0].bbox[2] == 131
    assert legacy[0].rgb_mask is None


def test_outside_object_and_invalid_depth_do_not_become_owned():
    image, depth, roi, intrinsics, plane, cfg = scene()
    image[35:85, 110:135] = 240
    depth[35:85, 110:135] = 600
    image[35:85, 150:175] = (10, 150, 10)
    depth[35:85, 150:175] = 550
    objects, _ = segment_depth_objects(image, depth, intrinsics, plane, cfg, roi_mask=roi)
    assert not objects
    depth[:] = 0
    assert not np.any(depth_footprint_in_tray(depth, intrinsics, plane, roi, cfg))
    with pytest.raises(ValueError, match="dimensions"):
        depth_footprint_in_tray(depth, intrinsics, plane, roi[:20], cfg)


def test_holes_stay_rgb_only_and_original_depth_is_unchanged():
    image, depth, roi, intrinsics, plane, cfg = scene()
    depth[35:85, 123:128] = 0
    original = depth.copy()
    objects, _ = segment_depth_objects(image, depth, intrinsics, plane, cfg, roi_mask=roi)
    assert len(objects) == 1
    assert np.all(objects[0].rgb_mask[35:85, 123:128])
    assert not np.any(objects[0].mask[35:85, 123:128])
    np.testing.assert_array_equal(original, depth)


def test_pipeline_passes_full_overhang_to_shape_model(monkeypatch):
    image, depth, roi, intrinsics, plane, cfg = scene()
    config = load_config()
    config = replace(config, rgbd=replace(cfg, processing_scale=1))
    monkeypatch.setattr("sorting_vision.pipeline3d.detect_tray_roi_mask", lambda _: roi)

    class Recorder:
        input_contract = "rgb_silhouette_depth_owned_v2"

        def classify(self, points, rgb, observed, mask, intrinsics=None, crop_origin_uv=(0, 0)):
            self.mask = mask.copy()
            self.origin = crop_origin_uv
            return "quadrangular_prism", .9

    model = Recorder()
    calibration = RGBDCalibration(intrinsics, np.eye(4), plane)
    pipeline = VisionPipeline3D(config=config, calibration=calibration, shape_model=model)
    results = pipeline.process(RGBDFrame(image, depth, intrinsics, 1, "overhang"))
    assert len(results) == 1
    assert results[0].bbox_px == (110, 35, 25, 50)
    assert model.mask[50 - model.origin[1], 134 - model.origin[0]]
    assert results[0].diagnostics["rgb_mask_source"] == "depth_verified_full_hsv"
    assert results[0].diagnostics["workspace_support_ratio"] == 1.
    assert cv2.countNonZero(model.mask) == 1250
    assert not results[0].selected


def test_white_balance_change_does_not_select_only_cool_half_tray():
    image = np.full((300, 400, 3), 80, np.uint8)
    image[45:255, 85:315] = (200, 200, 200)
    image[45:255, 200:315] = (220, 215, 195)
    legacy = detect_tray_roi_mask(image)
    robust = detect_tray_roi_mask(image, compare_neutral=True)
    assert not legacy[150, 130]
    assert robust[150, 130] and robust[150, 250]
    assert not robust[150, 30]


def test_separate_nearby_object_is_not_absorbed_into_full_silhouette():
    image, depth, roi, intrinsics, plane, cfg = scene()
    image[40:80, 80:100] = (0, 0, 180)
    depth[40:80, 80:100] = 550
    objects, _ = segment_depth_objects(image, depth, intrinsics, plane, cfg, roi_mask=roi)
    assert len(objects) == 2
    assert not np.any((objects[0].rgb_mask > 0) & (objects[1].rgb_mask > 0))
    assert all(item.clearance_px == 10 for item in objects)


def test_touching_different_hues_split_but_similar_dark_faces_do_not():
    image, depth, roi, intrinsics, plane, cfg = scene()
    image[35:85, 85:110] = (0, 0, 180)
    depth[35:85, 85:110] = 550
    cfg = replace(cfg, hsv_split_hue_gap=18)
    objects, _ = segment_depth_objects(image, depth, intrinsics, plane, cfg, roi_mask=roi)
    assert len(objects) == 2
    assert all(item.clearance_px == 0 for item in objects)
    image[35:85, 85:110] = (5, 70, 5)
    objects, _ = segment_depth_objects(image, depth, intrinsics, plane, cfg, roi_mask=roi)
    assert len(objects) == 1
    assert cv2.countNonZero(objects[0].rgb_mask) == 2500


def test_red_hue_wraparound_is_one_object():
    image, depth, roi, intrinsics, plane, cfg = scene()
    hsv = np.full((50, 25, 3), (178, 220, 160), np.uint8)
    hsv[:, 12:, 0] = 2
    image[35:85, 110:135] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    objects, _ = segment_depth_objects(image, depth, intrinsics, plane,
        replace(cfg, hsv_split_hue_gap=18), roi_mask=roi)
    assert len(objects) == 1
