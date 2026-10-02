"""Offline review probes. No camera or robot is opened."""
from __future__ import annotations

import json
import sys
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stereo camera" / "src"))
sys.path.insert(0, str(ROOT / "stereo camera" / "scripts"))

from sorting_vision.pipeline3d import VisionPipeline3D
from sorting_vision.rgbd import RGBDCalibration
from sorting_vision.server import VisionService3D
from sorting_vision.synthetic3d import competition_rgbd_demo
from sorting_vision.config import load_config
from sorting_vision.geometry3d import DepthSegmentedObject
from sorting_vision.grasp3d import find_suction_grasp
from sorting_vision.rgbd import CameraIntrinsics, Plane
from train_dual_fusion_holdout import _apply_candidate


class Source:
    def __init__(self, frame):
        self.frame = frame

    def read(self):
        return self.frame


def main():
    background, scene, _ = competition_rgbd_demo()
    pipeline = VisionPipeline3D(background_frame=background)
    service = VisionService3D(pipeline, Source(scene), mode="RGBD")
    service.update()
    previous = service.update()
    output = {
        "identity_robot_calibration": {
            "uses_identity": bool(np.array_equal(pipeline.calibration.camera_to_robot, np.eye(4))),
            "selected_count": sum(x.selected for x in previous),
            "health_can_pick": service.health()["can_pick"],
        },
        "same_frame_counted_twice": {
            "frame_id": scene.frame_id,
            "stable_count": pipeline._stable_count,
            "selected_count": sum(x.selected for x in previous),
        },
    }

    original_process = pipeline.process

    def fail(_frame):
        raise ValueError("review_simulated_processing_failure")

    pipeline.process = fail
    try:
        service.update()
    except ValueError:
        pass
    output["processing_failure_retains_state"] = {
        "health_can_pick": service.health()["can_pick"],
        "cached_selected_count": sum(x.selected for x in service._latest_results),
        "stable_count": pipeline._stable_count,
    }
    pipeline.process = original_process

    transform = np.eye(4)
    transform[0, 3] = np.nan
    calibration = RGBDCalibration(
        pipeline.calibration.intrinsics,
        transform,
        pipeline.calibration.tray_plane_camera,
        pipeline.calibration.tray_roi_polygon,
    )
    invalid_pipeline = VisionPipeline3D(calibration=calibration)
    invalid_results = invalid_pipeline.process(scene)
    output["nonfinite_translation"] = {
        "calibration_accepted": True,
        "pickable_count": sum(x.status.value == "PICKABLE" for x in invalid_results),
        "nonfinite_pose_count": sum(
            x.pose_3d is not None and not np.isfinite(x.pose_3d.position_mm.x)
            for x in invalid_results
        ),
    }

    # Force the legal schedule: update releases its lock, another TCP client
    # starts motion, then detect builds its response from the returned results.
    reached = threading.Event()
    proceed = threading.Event()
    original_update = service.update

    def paused_update():
        result = original_update()
        reached.set()
        if not proceed.wait(5):
            raise TimeoutError("review scheduling timeout")
        return result

    service.update = paused_update
    response = {}
    worker = threading.Thread(target=lambda: response.update(service.handle({"type": "detect"})))
    worker.start()
    if not reached.wait(5):
        raise TimeoutError("review detect did not reach response boundary")
    service.motion_start()
    proceed.set()
    worker.join(5)
    output["detect_motion_race"] = {
        "response_status": response["status"],
        "can_pick": response["health"]["can_pick"],
        "selected_count": sum(x["selected"] for x in response["results"]),
        "selected_statuses": [x["status"] for x in response["results"] if x["selected"]],
    }

    candidate = _apply_candidate(
        [{"top_class_scores": {"cube": .40, "cuboid": .10, "cone": .09,
            "cylinder": .09, "sphere": .08, "triangular_prism": .08,
            "pentagonal_prism": .08, "hexagonal_prism": .08},
          "side_class_scores": {"cube": .9, "cuboid": .1}, "side_quality": .1,
          "top_reason": "accepted", "top_prediction": "cube", "topology_quality": .5}],
        method="log_product", side_weight=.3, same_family_scale=.35,
        topology_guard=.65, top_temperature=1., side_temperature=1.,
        probability_threshold=.72, margin_threshold=.12,
    )[0]
    output["training_runtime_degraded_gate_mismatch"] = {
        "full_top_probability": .4,
        "training_top2_probability": candidate["fused_probability"],
        "training_accepted": candidate["fused_accepted"],
        "default_runtime_top_only_threshold": .85,
        "runtime_top_only_would_accept": False,
    }

    # A rough center and a flat region elsewhere on the same object.
    import cv2
    mask = np.zeros((300, 300), np.uint8)
    mask[50:250, 50:250] = 255
    depth = np.full(mask.shape, 700., np.float32)
    depth[mask > 0] = 650.
    depth[140:160, 140:160] += np.random.default_rng(0).uniform(-6., 6., (20, 20))
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    item = DepthSegmentedObject(mask, contour, (50, 50, 200, 200), 40000., 44., 56., 1., 1.)
    calib = RGBDCalibration(CameraIntrinsics(300, 300, 700., 700., 150., 150.), np.eye(4), Plane([0, 0, -1], 700.))
    cfg = load_config().grasp
    few = find_suction_grasp(item, depth, calib, cfg)
    many = find_suction_grasp(item, depth, calib, replace(cfg, max_candidates=100))
    output["center_only_grasp_search"] = {
        "default_three_candidates_found": few is not None,
        "hundred_candidates_found": many is not None,
        "expanded_score": None if many is None else many.info.score,
        "expanded_pixel": None if many is None else many.pixel_uv,
    }

    path = Path(__file__).with_name("evidence.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
