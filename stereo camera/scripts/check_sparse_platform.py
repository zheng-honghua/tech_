"""Offline geometric preflight. A pass is NOT competition/robot acceptance."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import cv2

from sorting_vision.config import load_config
from sorting_vision.dual_view import DualViewCalibration, DualCalibrationQualityLimits
from sorting_vision.sparse_stereo import SparseStereoLimits, camera_poses


def check(config_path):
    cfg = load_config(config_path)
    calibration = DualViewCalibration.load(cfg.dual_view.calibration_path)
    strict = DualCalibrationQualityLimits()
    metrics = calibration.metrics
    gates = {"calibration_valid": calibration.valid, "platform_matches": calibration.platform_id == cfg.dual_view.platform_id,
        "tray_transform_present": calibration.tray_from_primary is not None,
        "tray_plane_present": calibration.tray_plane_primary is not None,
        "strict_primary_rms": metrics.primary_rms_px <= strict.max_primary_rms_px,
        "strict_side_rms": metrics.side_rms_px <= strict.max_side_rms_px,
        "strict_projection_p95": metrics.joint_projection_p95_px <= strict.max_joint_projection_p95_px,
        "strict_scale": metrics.scale_error_ratio <= strict.max_scale_error_ratio,
        "nondegenerate_baseline": camera_poses(calibration)["baseline_mm"] >= 1.}
    background = cv2.imread(cfg.dual_view.side_background_path)
    expected = calibration.side_intrinsics
    gates["side_background_size_matches"] = background is not None and background.shape[:2] == (expected.height, expected.width)
    limits = SparseStereoLimits(cfg.dual_view.sparse_epipolar_px, cfg.dual_view.sparse_reprojection_px,
        cfg.dual_view.sparse_minimum_ray_angle_deg, cfg.dual_view.sparse_depth_consistency_mm,
        cfg.dual_view.sparse_prior_side_distance_px, cfg.dual_view.sparse_ambiguity_gap, cfg.dual_view.sparse_maximum_features)
    return {"geometric_preflight_passed": all(gates.values()), "gates": gates,
        "effective_config": asdict(cfg), "sparse_limits": vars(limits),
        "camera_poses": camera_poses(calibration), "calibration_metrics": metrics.to_dict(),
        "background_sha256": hashlib.sha256(Path(cfg.dual_view.side_background_path).read_bytes()).hexdigest() if background is not None else None,
        "robot_execution_approved": False,
        "limitations": ["OFFLINE_ONLY_NO_CURRENT_CAMERA_IDENTITY_CHECK", "NO_MANUAL_ROI_REVIEW",
                        "NO_MODEL_OR_DATA_ACCEPTANCE", "NO_ROBOT_COORDINATE_CALIBRATION"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = check(args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report["gates"], indent=2))
    raise SystemExit(0 if report["geometric_preflight_passed"] else 2)


if __name__ == "__main__":
    main()
