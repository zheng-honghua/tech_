"""Read factory rays and solve a separate, provenance-labelled extrinsic file."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sorting_vision.apriltag_calibration import calibrate_apriltag_pairs
from sorting_vision.dual_view import DualViewCalibration
from sorting_vision.intrinsic_calibration import CameraCalibration
from sorting_vision.rgbd import CameraIntrinsics
from dual_apriltag_calibrate import _load_saved_session


@dataclass(frozen=True)
class FactoryRays:
    intrinsics: CameraIntrinsics
    distortion: np.ndarray
    serial: str
    model: str
    camera_id: str = "primary-sdk-factory"

    @property
    def valid(self) -> bool:
        # Nonzero inverse Brown cannot be passed straight to cv2.projectPoints.
        parameters = (self.intrinsics.fx, self.intrinsics.fy, self.intrinsics.cx, self.intrinsics.cy)
        return bool(np.all(np.isfinite(parameters)) and self.intrinsics.fx > 0
                    and self.intrinsics.fy > 0 and self.intrinsics.width > 0
                    and self.intrinsics.height > 0 and self.distortion.shape == (5,)
                    and np.all(self.distortion == 0) and self.serial)

    def to_dict(self) -> dict:
        value = {"source": "REALSENSE_SDK_FACTORY", "serial": self.serial,
                 "intrinsics": self.intrinsics.to_dict(), "model": self.model,
                 "distortion": self.distortion.tolist(), "checkerboard_metrics": None}
        value["calibration_hash"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    primary, side, reference, fixed = _load_saved_session(Path(args.session), (45, 17), 50, 20)
    import pyrealsense2 as rs
    devices = list(rs.context().query_devices())
    if len(devices) != 1:
        raise ValueError("exactly one RealSense must be connected")
    device = devices[0]
    profiles = [profile.as_video_stream_profile() for sensor in device.query_sensors()
                for profile in sensor.get_stream_profiles() if profile.stream_type() == rs.stream.color
                and profile.as_video_stream_profile().width() == reference.intrinsics.width
                and profile.as_video_stream_profile().height() == reference.intrinsics.height and profile.fps() == 15]
    if not profiles:
        raise ValueError("matching SDK colour stream not available")
    native = profiles[0].get_intrinsics()
    rays = FactoryRays(CameraIntrinsics(native.width, native.height, native.fx, native.fy, native.ppx, native.ppy,
                                       reference.intrinsics.depth_scale_to_mm), np.asarray(native.coeffs),
                       device.get_info(rs.camera_info.serial_number), str(native.model))
    if not rays.valid:
        raise ValueError("unsupported nonzero SDK distortion; do not assume OpenCV equivalence")
    for name in ("fx", "fy", "cx", "cy"):
        if not np.isclose(getattr(rays.intrinsics, name), getattr(reference.intrinsics, name), atol=.001, rtol=0):
            raise ValueError("connected camera rays differ from recorded reference")
    (output / "factory-rays.json").write_text(json.dumps(rays.to_dict(), indent=2), encoding="utf-8")
    old = DualViewCalibration.load("config/dual/temporary/calibration.json")
    calibration = calibrate_apriltag_pairs(primary, side, reference, fixed, rays,
        CameraCalibration.load("config/dual/temporary/side-intrinsics.json"), platform_id="temporary",
        tag_size_mm=24, fixed_tag_ids=(45, 17), free_tag_id=50, tray_width_mm=154, tray_height_mm=154,
        fixed_tag_inset_mm=15, maximum_detection_width=1280, quality_limits=old.quality_limits,
        quality_profile=old.quality_profile)
    calibration.save(output / "calibration-native.json")
    summary = {"poses": len(primary), "valid": calibration.valid, "metrics": calibration.metrics.to_dict(),
               "factory": rays.to_dict(), "new_calibration_hash": calibration.calibration_hash,
               "original_preserved": True, "quality_profile": calibration.quality_profile,
               "quality_limits": calibration.quality_limits,
               "strict_competition_valid": calibration.metrics.to_dict()["valid"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
