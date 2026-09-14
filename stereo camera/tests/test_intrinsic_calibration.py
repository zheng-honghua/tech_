import json

import cv2
import numpy as np

from scripts import rgb_intrinsics_calibrate
from sorting_vision.intrinsic_calibration import (
    CameraCalibration,
    calibrate_camera_intrinsics,
    checkerboard_points_from_image,
    generate_checkerboard_asset,
)
from sorting_vision.rgbd import CameraIntrinsics


def test_rgb_intrinsic_parser_selects_one_rgb_camera_without_depth_options():
    args = rgb_intrinsics_calibrate.build_parser().parse_args(
        [
            "--source", "realsense",
            "--camera-id", "primary",
            "--width", "1920",
            "--height", "1080",
            "--fps", "15",
            "--output", "primary.json",
        ]
    )
    assert args.source == "realsense"
    assert (args.width, args.height, args.fps) == (1920, 1080, 15)
    assert (args.corners_x, args.corners_y, args.square_size_mm) == (10, 7, 20.0)
    assert args.video_sample_interval_ms == 500
    assert args.video_max_adjacent_difference == 4.0
    assert not hasattr(args, "depth_width")
    assert not hasattr(args, "dictionary")


def test_rgb_intrinsic_paths_follow_platform_layout():
    args = rgb_intrinsics_calibrate._resolve_storage_paths(
        rgb_intrinsics_calibrate.build_parser().parse_args(
            ["--source", "video", "--camera-id", "primary"]
        )
    )
    assert args.video_file == "data\\calibration\\temporary\\intrinsics\\videos\\primary.mp4"
    assert args.session_dir == "data\\calibration\\temporary\\intrinsics\\sessions"
    assert args.output == "config\\dual\\temporary\\primary-intrinsics.json"


def test_implausible_principal_point_is_rejected():
    calibration = CameraCalibration(
        camera_id="bad",
        intrinsics=CameraIntrinsics(1280, 720, 700.0, 700.0, 640.0, 650.0),
        distortion=np.zeros(5),
        rms_px=0.2,
        p95_px=0.4,
        coverage_ratio=0.8,
        maximum_tilt_deg=45.0,
        tilt_span_deg=20.0,
        frame_count=25,
        target={"type": "test"},
        version="test",
    )
    assert not calibration.valid
    assert "principal_point_y_implausible" in calibration.rejection_reasons


def test_checkerboard_asset_is_detectable_and_has_physical_svg(tmp_path):
    image_path, svg_path, metadata_path = generate_checkerboard_asset(
        tmp_path,
        corners_x=10,
        corners_y=7,
        square_size_mm=20.0,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["board_width_mm"] == 220.0
    assert metadata["board_height_mm"] == 160.0
    assert metadata["page_width_mm"] == 240.0
    assert metadata["page_height_mm"] == 180.0
    assert metadata["squares_x"] == 11
    assert metadata["squares_y"] == 8
    assert 'width="240.0mm"' in svg_path.read_text(encoding="utf-8")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    objects, pixels = checkerboard_points_from_image(
        image,
        corners_x=10,
        corners_y=7,
        square_size_mm=20.0,
        maximum_detection_width=None,
    )
    assert objects.shape == (70, 1, 3)
    assert pixels.shape == (70, 1, 2)


def test_synthetic_rgb_intrinsic_calibration_and_round_trip(tmp_path):
    width, height = 1280, 720
    expected = np.asarray(
        [[820.0, 0.0, 640.0], [0.0, 815.0, 360.0], [0.0, 0.0, 1.0]],
        np.float64,
    )
    distortion = np.asarray([-0.04, 0.01, 0.001, -0.001, 0.0], np.float64)
    xy = np.asarray(
        [(x, y, 0.0) for y in range(-75, 76, 25) for x in range(-100, 101, 25)],
        np.float32,
    ).reshape(-1, 1, 3)
    objects = []
    images = []
    for index in range(25):
        row, column = divmod(index, 5)
        rotation = np.asarray(
            [
                -0.35 + 0.16 * row,
                -0.30 + 0.15 * column,
                -0.20 + 0.09 * index,
            ],
            np.float64,
        )
        translation = np.asarray(
            [-250.0 + 125.0 * column, -120.0 + 60.0 * row, 650.0 + 8.0 * index],
            np.float64,
        )
        projected, _ = cv2.projectPoints(xy, rotation, translation, expected, distortion)
        objects.append(xy.copy())
        images.append(projected.astype(np.float32))
    images[7] = images[7] + np.random.default_rng(7).normal(
        0.0, 5.0, images[7].shape
    ).astype(np.float32)
    calibration = calibrate_camera_intrinsics(
        objects,
        images,
        (width, height),
        camera_id="synthetic",
        target={"type": "synthetic_checkerboard"},
    )
    assert calibration.valid
    assert calibration.rms_px < 0.01
    assert calibration.target["view_selection"]["input_frame_count"] == 25
    assert calibration.target["view_selection"]["retained_frame_count"] == 24
    assert calibration.target["view_selection"]["rejected_frame_indices"] == [7]
    np.testing.assert_allclose(calibration.intrinsics.fx, expected[0, 0], rtol=0.01)
    np.testing.assert_allclose(calibration.intrinsics.fy, expected[1, 1], rtol=0.01)
    path = tmp_path / "intrinsics.json"
    calibration.save(path)
    loaded = CameraCalibration.load(path)
    assert loaded.valid
    assert loaded.to_dict() == calibration.to_dict()
