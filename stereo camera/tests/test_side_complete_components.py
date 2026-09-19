from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from sorting_vision.config import DualViewConfig
from sorting_vision.dual_view import (
    DualViewFusion, ProjectedROI, associate_side_color_components,
    measure_side_color_segmentation, _side_mask,
)
from sorting_vision.rgbd import Plane
from sorting_vision.config import RGBDConfig
from sorting_vision.geometry3d import _hsv_component_masks


def scene():
    image = np.full((100, 140, 3), 220, np.uint8)
    image[40:68, 30:88] = (10, 150, 10)
    roi = ProjectedROI((40, 35, 40, 40), np.array([[40, 35], [79, 35], [79, 74], [40, 74]]), 1.)
    cfg = DualViewConfig(side_foreground_method="hsv", side_complete_color_components=True)
    return image, roi, cfg


def test_side_component_is_not_clipped_to_projection_polygon_or_box():
    image, roi, cfg = scene()
    segmentation = associate_side_color_components(image, {"one": roi}, cfg)["one"]
    assert segmentation.reason == "accepted"
    x, y, width, height = segmentation.roi.bbox
    assert x < 30 and x + width > 88
    assert cv2.countNonZero(segmentation.mask) == 58 * 28
    assert segmentation.mask[50 - y, 31 - x]
    np.testing.assert_array_equal(segmentation.roi.polygon, roi.polygon)
    legacy, _, _, _ = _side_mask(image, np.full_like(image, 220), roi, cfg)
    assert cv2.countNonZero(legacy) < cv2.countNonZero(segmentation.mask)
    pixels, blur, quality = measure_side_color_segmentation(image, segmentation, cfg)
    assert pixels == 58 * 28 and blur > 35 and quality >= .6


def test_shared_colour_component_has_no_unique_owner():
    image, _, cfg = scene()
    rois = {name: ProjectedROI((x, 35, 20, 40), np.array([[x, 35], [x + 19, 35], [x + 19, 74], [x, 74]]), 1.)
            for name, x in (("a", 35), ("b", 65))}
    items = associate_side_color_components(image, rois, cfg)
    assert all(item.reason == "ambiguous_side_color_correspondence" for item in items.values())
    assert all(cv2.countNonZero(item.mask) == 0 for item in items.values())


def test_empty_foreign_colour_and_neutral_shadow_are_not_objects():
    image, roi, cfg = scene()
    image[:] = 220
    image[40:68, 100:130] = (10, 150, 10)
    image[40:68, 45:70] = 80
    item = associate_side_color_components(image, {"one": roi}, cfg)["one"]
    assert item.reason == "no_matching_side_color_component"
    assert measure_side_color_segmentation(image, item, cfg)[2] == 0
    with pytest.raises(ValueError, match="require hsv"):
        associate_side_color_components(image, {"one": roi}, replace(cfg, side_foreground_method="background"))


def test_image_boundary_truncation_is_not_accepted_side_evidence():
    image, _, cfg = scene()
    image[:] = 220
    image[35:65, 0:50] = (10, 150, 10)
    roi = ProjectedROI((0, 30, 60, 40), np.array([[0, 30], [59, 30], [59, 69], [0, 69]]), 1.)
    item = associate_side_color_components(image, {"one": roi}, cfg)["one"]
    assert item.reason == "image_boundary_truncated"
    assert cv2.countNonZero(item.mask) > 0
    assert measure_side_color_segmentation(image, item, cfg)[2] == 0


def test_complete_side_mask_reaches_classifier_and_diagnostics():
    image, roi, cfg = scene()
    item = associate_side_color_components(image, {"one": roi}, cfg)["one"]

    class Recorder:
        last_class_scores = {"quadrangular_prism": .95, "cone": .05}

        def predict(self, rgb, mask):
            self.mask = mask.copy()
            return "quadrangular_prism", .95, {"reason": "accepted"}

    model = Recorder()
    fusion = DualViewFusion(None, np.full_like(image, 220), model, cfg, Plane(np.array([0., 0., -1.]), 500))
    evidence = fusion._classify_side(image, roi, np.empty((0, 3)), item)
    assert model.mask[50, 31]
    assert cv2.countNonZero(model.mask) == 58 * 28
    assert evidence.contours_px and evidence.segmentation_diagnostics["method"] == "full_image_color_then_projection"


def test_default_side_annotation_draws_observed_outline_not_reference_boxes(monkeypatch):
    image, roi, _ = scene()
    result = SimpleNamespace(diagnostics={"dual_view": {
        "side_roi": roi.bbox, "fusion_state": "TOP_ONLY", "side_class_label": "cone",
        "side_contours_px": [[[30, 40], [87, 40], [87, 67], [30, 67]]],
        "cross_view_topology": {"projected_obb_edges_px": [[45, 36, 75, 36]],
                                "segments_px": [[45, 45, 75, 45]]},
    }})
    calls = []
    original_rectangle, original_line = cv2.rectangle, cv2.line
    monkeypatch.setattr(cv2, "rectangle", lambda *args, **kwargs: (calls.append("rectangle"), original_rectangle(*args, **kwargs))[1])
    monkeypatch.setattr(cv2, "line", lambda *args, **kwargs: (calls.append("line"), original_line(*args, **kwargs))[1])
    default = DualViewFusion.annotate_side(image, [result])
    assert not calls
    assert default[40, 30, 1] >= 200
    DualViewFusion.annotate_side(image, [result], show_geometry=True, show_edges=True)
    assert "rectangle" in calls and "line" in calls


def test_dark_similar_hue_face_is_retained_without_neutral_shadow():
    image, roi, cfg = scene()
    image[40:68, 60:88] = (0, 14, 0)
    image[70:76, 35:85] = 14
    legacy = associate_side_color_components(image, {"one": roi}, cfg)["one"]
    complete = associate_side_color_components(image, {"one": roi}, replace(cfg, side_hsv_min_value=12))["one"]
    assert complete.reason == "accepted"
    assert cv2.countNonZero(complete.mask) == 58 * 28
    assert cv2.countNonZero(legacy.mask) < cv2.countNonZero(complete.mask)
    x, y, _, _ = complete.roi.bbox
    assert complete.mask[50 - y, 80 - x]
    assert not complete.mask[72 - y, 45 - x]


def test_noise_prefilter_keeps_later_components_after_a_hue_split():
    image = np.full((120, 180, 3), 220, np.uint8)
    image[20:60, 20:45] = (0, 0, 180)
    image[20:60, 45:70] = (0, 160, 0)
    image[70:100, 100:130] = (180, 0, 0)
    image[105, ::5] = (180, 0, 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    foreground = (hsv[:, :, 1] >= 70).astype(np.uint8) * 255
    masks = list(_hsv_component_masks(foreground, hsv,
        RGBDConfig(min_area_px=80, hsv_split_hue_gap=18), minimum_pixels=80))
    assert len(masks) == 3
    assert sum(cv2.countNonZero(mask) for mask in masks) == 2900
