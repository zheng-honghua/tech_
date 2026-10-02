import json

import cv2
import numpy as np
import pytest

from sorting_vision.color_v7 import CameraColorProfile, ColorFrameContext, canonical_color, color_vector, neutral_correction, recover_components
from sorting_vision.ridge_v7 import extract_ridges, _supported_merge
from sorting_vision.visual_contract import LEGACY, V7, active_contract, visual_contract
from sorting_vision.fusion_policy import accepted_fusion_scores
from sorting_vision.geometry_edges import extract_edge_topology
from sorting_vision.side_geometry import SideGeometryModel
from sorting_vision.sparse_stereo import observations, triangulate_corner, SparseStereoLimits
from test_sparse_stereo import stereo, pixels


def context(hsv):
    return ColorFrameContext.create(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), {"gains_bgr": [1., 1., 1.]})


def scene():
    hsv = np.zeros((100, 140, 3), np.uint8)
    hsv[:, :, 2] = 220
    hsv[20:80, 20:55] = [179, 220, 180]
    hsv[20:80, 55:85] = [1, 160, 19]
    return hsv


def test_circular_red_recovers_connected_dark_face_without_white_gap():
    hsv = scene()
    hsv[20:80, 95:120] = [1, 220, 180]
    masks = recover_components(context(hsv), 70, 25, 100)
    assert len(masks) == 2
    assert any(np.all(mask[30:70, 60:80] > 0) for mask in masks)
    assert all(np.all(mask[:, 86:94] == 0) for mask in masks)


def test_neutral_highlight_requires_depth_and_shadow_does_not_bridge():
    hsv = scene()
    hsv[40:60, 40:50] = [0, 0, 235]
    ctx = context(hsv)
    unsupported = recover_components(ctx, 70, 25, 100)[0]
    support = np.zeros(hsv.shape[:2], np.uint8)
    support[20:80, 20:85] = 255
    supported = recover_components(ctx, 70, 25, 100, geometric_support=support)[0]
    assert unsupported[50, 45] == 0 and supported[50, 45] > 0
    assert supported[50, 90] == 0


def test_hue_boundary_alone_does_not_split_physical_instances():
    hsv = scene()
    hsv[20:80, 85:115] = [60, 220, 180]
    masks = recover_components(context(hsv), 70, 25, 100)
    assert len(masks) == 1
    assert masks[0][50, 30] and masks[0][50, 100]
    with pytest.raises(ValueError, match="multiple_hues"):
        color_vector(context(hsv), masks[0])


def test_fixed_neutral_reference_fallback_and_invalid_gains():
    assert neutral_correction(np.full((50, 50, 3), 255, np.uint8))["gains_bgr"] == [1., 1., 1.]
    background = np.full((50, 50, 3), [150, 155, 160], np.uint8)
    assert neutral_correction(background)["reason"] == "batch_empty_tray"
    with pytest.raises(ValueError):
        ColorFrameContext.create(background, {"gains_bgr": [1, np.nan, 1]})


def test_raw_typo_preserved_canonical_profile_and_hash(tmp_path):
    ctx = context(scene())
    mask = np.zeros(ctx.hsv.shape[:2], np.uint8)
    mask[20:80, 20:85] = 255
    vector = color_vector(ctx, mask)
    samples = [("bule", vector)] * 6 + [("green", -vector)] * 6
    profile = CameraColorProfile.fit("primary", samples, {"reviewed_ids": ["fixture"]})
    assert canonical_color("bule") == "blue"
    assert "bule" not in profile.prototypes
    assert profile.predict_vector(vector).label_id == "blue"
    profile.save(tmp_path / "profile.json")
    assert CameraColorProfile.load(tmp_path / "profile.json", "primary").content_hash == profile.content_hash
    with pytest.raises(ValueError):
        CameraColorProfile.load(tmp_path / "profile.json", "side")
    assert profile.predict_vector(vector + 5).label_id == "unknown"


def test_smooth_curve_has_no_polygon_chord_ridges_or_corners():
    mask = np.zeros((160, 160), np.uint8)
    cv2.circle(mask, (80, 80), 55, 255, -1)
    image = np.full((160, 160, 3), 240, np.uint8)
    image[mask > 0] = [20, 40, 180]
    result = extract_ridges(image, mask)
    assert len(result.segments) == 0
    with visual_contract(V7):
        assert len(observations(image, mask)[0]) == 0


def test_texture_ridge_stays_2d_and_shared_topology_is_finite():
    image = np.full((120, 120, 3), 245, np.uint8)
    mask = np.zeros((120, 120), np.uint8)
    mask[15:105, 15:105] = 255
    image[mask > 0] = [20, 70, 160]
    image[15:105, 60:105] = [10, 35, 80]
    result = extract_ridges(image, mask)
    assert len(result.internal) > 0
    assert all(record["geometry_support"] is None for record in result.evidence)
    assert all(record["state"] != "CONFIRMED_3D" for record in result.evidence)
    with visual_contract(V7):
        assert np.isfinite(extract_edge_topology(image, mask).quality)


def test_supported_merge_keeps_parallel_lines_and_unsupported_gaps():
    edges = np.zeros((50, 100), np.uint8)
    edges[20, 5:45] = 255
    edges[20, 50:90] = 255
    edges[25, 5:90] = 255
    lines = [np.array([5., 20., 44., 20.]), np.array([50., 20., 89., 20.]), np.array([5., 25., 89., 25.])]
    assert len(_supported_merge(lines, edges, 3)) == 3


def test_degraded_probability_is_not_renormalized_over_top_two():
    output = accepted_fusion_scores({"a": .55, "b": .2, "c": .25}, {"a": 1.}, .1)
    assert output["probability"] == .55 and not output["accepted"]
    assert output["scores"]["c"] == .25


def test_contract_scope_resets_after_error():
    with pytest.raises(RuntimeError):
        with visual_contract(V7):
            assert active_contract() == V7
            raise RuntimeError()
    assert active_contract() == LEGACY


def test_v7_refinement_reports_sensitivity_without_claiming_accuracy():
    calibration = stereo()
    point = np.array([0., 10., 200.])
    a, b = pixels(point, calibration)
    with visual_contract(V7):
        result = triangulate_corner(a, b, calibration, SparseStereoLimits(), point)
    assert result["accepted"]
    assert result["position_sensitivity"]["maximum_shift_mm"] > 0
    assert result["position_sensitivity"]["is_measured_accuracy"] is False
    json.dumps(result, allow_nan=False)


def test_saved_side_contract_cannot_silently_consume_other_version(tmp_path):
    from sorting_vision.shape_registry import ShapeClass, ShapeRegistry
    registry = ShapeRegistry(1,(ShapeClass("a","A","x"),ShapeClass("b","B","x")))
    raw = np.array([[0.,0.],[.1,0.],[2.,2.],[2.1,2.]],np.float32)
    with visual_contract(V7):
        model = SideGeometryModel(raw,np.array(["a","a","b","b"]),np.zeros(2),np.ones(2),registry,
                                  group_ids=np.zeros(2,np.int32))
        model.save(tmp_path/"side.npz")
    loaded = SideGeometryModel.load(tmp_path/"side.npz",registry)
    with pytest.raises(ValueError,match="contract mismatch"):
        loaded.predict_features(raw[0])
    with visual_contract(V7):
        assert loaded.predict_features(raw[0])[2]["nearest_label"] == "a"


def test_v7_depth_holes_are_not_repaired_into_grasp_support():
    from dataclasses import replace
    from sorting_vision.config import load_config
    from sorting_vision.pipeline3d import VisionPipeline3D
    from sorting_vision.synthetic3d import competition_rgbd_demo
    background, scene, _ = competition_rgbd_demo()
    cfg=load_config()
    cfg=replace(cfg,rgbd=replace(cfg.rgbd,visual_version=7))
    broken=scene.depth.copy()
    foreground=broken<np.median(background.depth)
    broken[foreground]=0
    frame=type(scene)(scene.color_bgr,broken,scene.intrinsics,scene.timestamp_ns,"holes")
    pipeline=VisionPipeline3D(config=cfg,background_frame=background)
    results=pipeline.process(frame)
    assert not any(item.selected or item.status.value=="PICKABLE" for item in results)
    np.testing.assert_array_equal(frame.depth,broken)


def test_v7_restores_ridge_pixels_after_processing_resize():
    from sorting_vision.pipeline3d import VisionPipeline3D
    from test_dual_view import result
    item=result()
    item.diagnostics["candidate_ridges_2d"]=[{"endpoints_px":[[10.,20.],[30.,40.]]}]
    item.diagnostics["rgb_crop_origin_uv"]=[5.,8.]
    VisionPipeline3D._restore_image_coordinates([item],.5)
    assert item.diagnostics["candidate_ridges_2d"][0]["endpoints_px"]==[[20.,40.],[60.,80.]]


def test_v7_same_color_contact_stays_one_ambiguous_component():
    hsv=scene()
    hsv[40:90,85:120]=[1,210,180]
    masks=recover_components(context(hsv),70,25,100)
    assert len(masks)==1


def test_v7_metric_merge_keeps_parallel_edges_and_gaps():
    from sorting_vision.face_topology3d import _line_from_pixels, _merge_fused_edges
    lines=[_line_from_pixels(np.array(points),1.,1.,.9,"both") for points in (
        [[5.,20.],[40.,20.]],[[45.,20.],[85.,20.]],[[5.,24.],[85.,24.]])]
    with visual_contract(V7):
        assert len(_merge_fused_edges(lines,100.))==3


@pytest.mark.parametrize("initial,expected",[("PICKABLE","UNCERTAIN"),("DEPTH_INVALID","DEPTH_INVALID")])
def test_v7_high_confidence_color_conflict_veto_preserves_depth_state(monkeypatch,initial,expected):
    from sorting_vision.config import DualViewConfig
    from sorting_vision.dual_view import DualViewFusion, SideEvidence
    from sorting_vision.rgbd import Plane
    from sorting_vision.types import DetectionStatus
    from test_dual_view import SideModel, calibration, pair, points, result
    model=SideModel({"cube":.99,"hexagonal_prism":.01})
    model.feature_contract=V7
    cfg=DualViewConfig(enabled=True,visual_version=7)
    fusion=DualViewFusion(calibration(),np.zeros((100,100,3),np.uint8),model,cfg,
        Plane(np.array([0,0,-1.]),500.))
    evidence=SideEvidence(model.last_class_scores,"cube",.99,"accepted",1.,(40,40,20,20),400,100.,
        np.array([[40,40],[60,40],[60,60],[40,60]]),topology_diagnostics={"color":{"label":"green","confidence":.99}})
    monkeypatch.setattr(fusion,"_classify_side",lambda *args:evidence)
    item=result(status=DetectionStatus(initial))
    item.diagnostics["primary_color"]={"label":"red","confidence":.99}
    fusion.apply([item],{item.object_id:points()},pair())
    assert item.status.value==expected and not item.selected
    assert item.color_id=="red" and item.diagnostics["dual_view"]["color_conflict"]


def test_v7_color_training_and_runtime_share_native_pixel_domain():
    from dataclasses import replace
    from sorting_vision.classification import LabelPrediction
    from sorting_vision.config import load_config
    from sorting_vision.pipeline3d import VisionPipeline3D
    from sorting_vision.synthetic3d import competition_rgbd_demo
    background,scene,_=competition_rgbd_demo()
    cfg=load_config()
    cfg=replace(cfg,rgbd=replace(cfg.rgbd,visual_version=7,processing_scale=.5))
    pipeline=VisionPipeline3D(config=cfg,background_frame=background)
    calls=[]
    class NativeProfile:
        def classify(self,ctx,mask):
            calls.append((ctx.original.shape,mask.shape))
            return LabelPrediction("red","红色",.99,{"camera":"primary"})
    pipeline.color_profile=NativeProfile()
    pipeline.process(scene)
    assert calls
    assert all(image_shape==scene.color_bgr.shape and mask_shape==scene.color_bgr.shape[:2] for image_shape,mask_shape in calls)
