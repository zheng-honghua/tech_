import numpy as np

from sorting_vision.rgbd import (
    CameraIntrinsics,
    Plane,
    RGBDCalibration,
    backproject_pixels,
    fit_plane_ransac,
    project_points,
    resize_rgbd_frame,
)


def test_projection_round_trip():
    intrinsics = CameraIntrinsics(640, 480, 700, 710, 319.5, 239.5)
    pixels = np.array([[100.0, 80.0], [320.0, 240.0], [500.0, 400.0]])
    depths = np.array([500.0, 700.0, 900.0])
    points = backproject_pixels(pixels, depths, intrinsics)
    assert np.allclose(project_points(points, intrinsics), pixels)


def test_resize_rgbd_frame_scales_intrinsics_and_keeps_alignment():
    intrinsics = CameraIntrinsics(8, 6, 10, 12, 3.5, 2.5)
    from sorting_vision.rgbd import RGBDFrame

    frame = RGBDFrame(
        np.zeros((6, 8, 3), np.uint8), np.full((6, 8), 400, np.uint16),
        intrinsics, 123, "frame-1",
    )
    resized = resize_rgbd_frame(frame, 0.5)

    assert resized.color_bgr.shape == (3, 4, 3)
    assert resized.depth.shape == (3, 4)
    assert resized.intrinsics.fx == 5
    assert resized.intrinsics.fy == 6
    assert resized.intrinsics.cx == 1.5
    assert resized.intrinsics.cy == 1.0
    assert np.all(resized.depth == 400)


def test_ransac_plane_rejects_outliers():
    generator = np.random.default_rng(3)
    x, y = generator.uniform(-100, 100, (2, 1000))
    z = 700.0 + generator.normal(0, 0.1, 1000)
    points = np.column_stack((x, y, z))
    points[:30, 2] -= 50
    plane = fit_plane_ransac(points, threshold_mm=0.8)
    assert np.allclose(plane.normal, [0, 0, -1], atol=0.01)
    assert abs(plane.offset - 700.0) < 0.3
    assert plane.rmse_mm < 0.2


def test_extrinsic_transform_and_serialization(tmp_path):
    intrinsics = CameraIntrinsics(640, 480, 700, 700, 319.5, 239.5)
    transform = np.eye(4)
    transform[:3, 3] = [10, 20, 30]
    calibration = RGBDCalibration(intrinsics, transform, Plane([0, 0, -1], 700))
    point = calibration.transform_points(np.array([[1.0, 2.0, 3.0]]))[0]
    assert np.allclose(point, [11, 22, 33])

    path = tmp_path / "rgbd-calibration.json"
    calibration.save(path)
    loaded = RGBDCalibration.load(path)
    assert np.allclose(loaded.camera_to_robot, transform)
    assert loaded.intrinsics.fx == 700
