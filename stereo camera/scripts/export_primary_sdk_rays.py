"""Read a chosen RealSense factory colour profile without starting streaming."""
import argparse
import json
from pathlib import Path

import numpy as np

from sorting_vision.intrinsic_calibration import SDKProjectionCalibration
from sorting_vision.rgbd import CameraIntrinsics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("output already exists; archive it or choose a new file")
    import pyrealsense2 as rs

    devices = [device for device in rs.context().query_devices() if args.serial is None or
               device.get_info(rs.camera_info.serial_number) == args.serial]
    if len(devices) != 1:
        raise ValueError("select exactly one RealSense with --serial")
    device = devices[0]
    profiles = [profile.as_video_stream_profile() for sensor in device.query_sensors()
                for profile in sensor.get_stream_profiles() if profile.stream_type() == rs.stream.color
                and profile.as_video_stream_profile().width() == args.width
                and profile.as_video_stream_profile().height() == args.height and profile.fps() == args.fps]
    if not profiles:
        raise ValueError("requested SDK colour profile is unavailable")
    native = profiles[0].get_intrinsics()
    scale = device.first_depth_sensor().get_depth_scale() * 1000
    calibration = SDKProjectionCalibration(CameraIntrinsics(native.width, native.height, native.fx, native.fy,
        native.ppx, native.ppy, scale), np.asarray(native.coeffs),
        device.get_info(rs.camera_info.serial_number), str(native.model))
    if not calibration.valid:
        raise ValueError("nonzero SDK distortion is not supported by this OpenCV reconstruction path")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(calibration.to_dict(), stream, indent=2)
    print(json.dumps(calibration.to_dict(), indent=2))


if __name__ == "__main__":
    main()
