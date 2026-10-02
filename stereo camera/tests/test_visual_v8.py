from dataclasses import replace
import numpy as np
import pytest
from sorting_vision.classification import LabelPrediction
from sorting_vision.config import load_config, DualViewConfig
from sorting_vision.pipeline3d import VisionPipeline3D
from sorting_vision.rgbd import Plane
from sorting_vision.dual_view import DualViewFusion, SideEvidence
from sorting_vision.sparse_stereo import SparseStereoLimits, triangulate_corner, observations
from sorting_vision.ridge_v7 import combine_recall_candidates
from sorting_vision.visual_contract import LEGACY, V7, active_contract
from sorting_vision.synthetic3d import competition_rgbd_demo
from test_sparse_stereo import stereo, pixels


def test_hybrid_colour_does_not_change_legacy_model_input():
    background,scene,_=competition_rgbd_demo()
    cfg=load_config()
    cfg=replace(cfg,rgbd=replace(cfg.rgbd,instance_segmentation='hsv'))
    calls=[]
    class Recorder:
        edge_parameters={'feature_contract':LEGACY}
        input_contract='rgb_silhouette_depth_owned_v2'
        def classify(self,points,rgb,depth,mask,intrinsics,origin):
            assert active_contract()==LEGACY
            calls.append((rgb.copy(),depth.copy(),mask.copy(),origin))
            return 'cube',.99
    baseline=VisionPipeline3D(cfg,background_frame=background,shape_model=Recorder())
    baseline.process(scene)
    original=calls[:];calls.clear()
    hybrid=VisionPipeline3D(replace(cfg,rgbd=replace(cfg.rgbd,visual_version=8)),background_frame=background,shape_model=Recorder())
    domains=[]
    class Profile:
        def classify(self,context,mask):
            domains.append((context.original.shape,mask.shape))
            return LabelPrediction('red','red',.99,{})
    hybrid.color_profile=Profile()
    hybrid.process(scene)
    assert len(original)==len(calls)>0
    for before,after in zip(original,calls):
        for a,b in zip(before[:3],after[:3]): np.testing.assert_array_equal(a,b)
        assert before[3]==after[3]
    assert all(a==scene.color_bgr.shape and b==scene.color_bgr.shape[:2] for a,b in domains)


def test_hybrid_rejects_v7_shape_semantics():
    background,_,_=competition_rgbd_demo()
    cfg=load_config();cfg=replace(cfg,rgbd=replace(cfg.rgbd,visual_version=8))
    class WrongModel:
        edge_parameters={'feature_contract':V7}
    with pytest.raises(ValueError,match='contract mismatch'):
        VisionPipeline3D(cfg,background_frame=background,shape_model=WrongModel())


def test_recall_candidates_do_not_gain_verified_image_status():
    verified=[{'endpoints_px':[[10,10],[60,10]],'state':'CANDIDATE_2D'}]
    rows=combine_recall_candidates(verified,[[10,10,60,10],[10,15,60,15]])
    assert len(rows)==2 and rows[0]['image_verified']
    assert not rows[1]['image_verified'] and rows[1]['geometry_support'] is None
    assert 'image_verified' not in verified[0]


def test_optional_refinement_retains_valid_point_when_depth_would_fail(monkeypatch):
    cal=stereo();point=np.array([0.,10.,200.]);a,b=pixels(point,cal)
    monkeypatch.setattr('sorting_vision.sparse_stereo._refine_position',lambda *args:point+[0,0,20.])
    result=triangulate_corner(a,b,cal,SparseStereoLimits(),point,refine_position=True)
    assert result['accepted'] and result['position_sensitivity']['is_measured_accuracy'] is False
    assert not result['refinement_applied'] and result['refinement_fallback_reason']
    np.testing.assert_allclose(result['position_primary_mm'],point,atol=1e-5)


def test_hybrid_keeps_contour_candidates_without_two_incident_lines(monkeypatch):
    monkeypatch.setattr('sorting_vision.sparse_stereo._side_lsd_segments',lambda *args:[])
    mask=np.zeros((100,100),np.uint8);mask[20:80,20:80]=255
    corners,_=observations(np.zeros((100,100,3),np.uint8),mask)
    assert len(corners)==4


@pytest.mark.parametrize('initial,expected',[('PICKABLE','UNCERTAIN'),('DEPTH_INVALID','DEPTH_INVALID')])
def test_hybrid_colour_conflict_only_vetoes(initial,expected,monkeypatch):
    from sorting_vision.types import DetectionStatus
    from test_dual_view import SideModel, calibration, pair, points, result
    model=SideModel({'cube':.99,'hexagonal_prism':.01})
    cfg=DualViewConfig(enabled=True,visual_version=8)
    fusion=DualViewFusion(calibration(),np.zeros((100,100,3),np.uint8),model,cfg,Plane(np.array([0,0,-1.]),500.))
    evidence=SideEvidence(model.last_class_scores,'cube',.99,'accepted',1.,(40,40,20,20),400,100.,
        np.array([[40,40],[60,40],[60,60],[40,60]]),topology_diagnostics={'color':{'label':'green','confidence':.99}})
    monkeypatch.setattr(fusion,'_classify_side',lambda *args:evidence)
    item=result(status=DetectionStatus(initial));item.diagnostics['primary_color']={'label':'red','confidence':.99}
    fusion.apply([item],{item.object_id:points()},pair())
    assert item.status.value==expected and not item.selected and item.color_id=='red'


@pytest.mark.parametrize('side_verified,expected', [(False,0),(True,1)])
def test_geometric_ridge_requires_verified_image_support_in_both_views(side_verified,expected,monkeypatch):
    from sorting_vision.sparse_stereo import project_primary
    from test_dual_view import SideModel, calibration, pair, points, result
    model=SideModel({'cube':.99,'hexagonal_prism':.01})
    cal=calibration()
    fusion=DualViewFusion(cal,np.zeros((100,100,3),np.uint8),model,DualViewConfig(enabled=True,visual_version=8),Plane(np.array([0,0,-1.]),500.))
    endpoints=np.array([[-30.,0.,500.],[30.,0.,500.]])
    top_uv=project_primary(endpoints,cal).tolist()
    side_uv=cal.project_primary_points(endpoints).tolist()
    edge={'endpoints_primary_mm':endpoints.tolist(),'sources':['TOP_PLANE_INTERSECTION','SIDE_DEPTH_SUPPORTED']}
    evidence=SideEvidence(model.last_class_scores,'cube',.99,'accepted',1.,(40,40,20,20),400,100.,
        np.array([[40,40],[60,40],[60,60],[40,60]]),topology_diagnostics={
            'joint_topology_graph':{'edges':[edge]},
            'candidate_ridges_2d':[{'endpoints_px':side_uv,'image_verified':side_verified}]})
    monkeypatch.setattr(fusion,'_classify_side',lambda *args:evidence)
    item=result();item.diagnostics['candidate_ridges_2d']=[{'endpoints_px':top_uv,'image_verified':True}]
    fusion.apply([item],{item.object_id:points()},pair())
    assert len(item.diagnostics['geometry_candidates_3d'])==1
    assert len(item.diagnostics['confirmed_ridges_3d'])==expected
