import cv2
import numpy as np

from sorting_vision.pipeline3d import VisionPipeline3D
from sorting_vision.rgbd import CameraIntrinsics, RGBDFrame
from sorting_vision.synthetic3d import competition_rgbd_demo, make_rgbd_scene
from sorting_vision.types import DetectionStatus


def test_rgbd_pipeline_classifies_solids_and_selects_after_two_frames():
    background, scene, truth = competition_rgbd_demo()
    pipeline = VisionPipeline3D(background_frame=background)
    first = pipeline.process(scene)
    second = pipeline.process(scene)

    assert len(second) == len(truth)
    assert not any(item.selected for item in first)
    assert sum(item.selected for item in second) == 1
    assert sorted(item.class_key for item in second) == sorted(
        f"{solid.color_id}:{solid.shape_id}" for solid in truth
    )
    selected = next(item for item in second if item.selected)
    payload = selected.to_dict("crop.png", "depth.npy")
    assert payload["schema_version"] == 2
    assert payload["pose_3d"]["position_mm"]
    assert payload["grasp"]["score"] >= 0.72


def test_acknowledge_pick_resets_stability():
    background, scene, _ = competition_rgbd_demo()
    pipeline = VisionPipeline3D(background_frame=background)
    pipeline.process(scene)
    assert any(item.selected for item in pipeline.process(scene))
    pipeline.acknowledge_pick()
    assert not any(item.selected for item in pipeline.process(scene))


def test_global_depth_failure_blocks_selection():
    background, scene, _ = competition_rgbd_demo()
    broken_depth = scene.depth.copy()
    broken_depth[:, :] = 0
    broken = type(scene)(
        scene.color_bgr,
        broken_depth,
        scene.intrinsics,
        scene.timestamp_ns,
        "broken",
    )
    pipeline = VisionPipeline3D(background_frame=background)
    assert pipeline.process(broken) == []
    assert pipeline.health()["ok"] is False
    assert pipeline.health()["reason"] == "insufficient_tray_depth"


def test_pipeline_only_detects_objects_inside_tray_and_keeps_edge_object():
    width, height = 400, 300
    intrinsics = CameraIntrinsics(width, height, 300, 300, 200, 150, 1.0)
    background_color = np.full((height, width, 3), (135, 141, 140), np.uint8)
    cv2.rectangle(background_color, (100, 30), (360, 280), (174, 157, 134), -1)
    background_depth = np.zeros((height, width), np.float32)
    background_depth[30:281, 100:361] = 650
    background = RGBDFrame(
        background_color, background_depth, intrinsics, 1, "tray-background"
    )

    scene_color = background_color.copy()
    scene_depth = background_depth.copy()
    # A valid object touches the tray's inner edge.
    cv2.rectangle(scene_color, (110, 105), (165, 165), (180, 40, 20), -1)
    scene_depth[105:166, 110:166] = 620
    # A second raised, coloured object is outside the tray and must be ignored.
    cv2.rectangle(scene_color, (15, 105), (70, 165), (180, 40, 20), -1)
    scene_depth[105:166, 15:71] = 620
    scene = RGBDFrame(scene_color, scene_depth, intrinsics, 2, "tray-scene")

    pipeline = VisionPipeline3D(background_frame=background)
    results = pipeline.process(scene)
    assert len(results) == 1, pipeline.health()
    assert results[0].bbox_px[0] >= 100
    health = pipeline.health()
    assert health["tray_roi_valid"] is True
    assert health["tray_valid_depth_ratio"] > 0.9
    assert health["global_valid_depth_ratio"] < 0.7


def test_shifted_tray_marks_results_depth_invalid():
    background, _, truth = competition_rgbd_demo()
    _, shifted = make_rgbd_scene(truth, tray_depth_mm=708, frame_id="shifted")
    pipeline = VisionPipeline3D(background_frame=background)
    results = pipeline.process(shifted)
    assert results
    assert all(item.status == DetectionStatus.DEPTH_INVALID for item in results)
    assert not any(item.selected for item in results)


def test_out_of_sync_rgb_and_depth_blocks_selection():
    background, scene, _ = competition_rgbd_demo()
    unsynchronised = type(scene)(
        scene.color_bgr,
        scene.depth,
        scene.intrinsics,
        scene.timestamp_ns,
        "unsynchronised",
        scene.timestamp_ns,
        scene.timestamp_ns + 50_000_000,
    )
    pipeline = VisionPipeline3D(background_frame=background)
    results = pipeline.process(unsynchronised)
    assert pipeline.health()["reason"] == "rgb_depth_out_of_sync"
    assert all(item.status == DetectionStatus.DEPTH_INVALID for item in results)
    assert not any(item.selected for item in results)
