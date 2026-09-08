import numpy as np

from sorting_vision.calibration import PerspectiveCalibration


def test_identity_coordinates_and_y_axis():
    calibration = PerspectiveCalibration.identity(801, 801, 160.0, 160.0)
    bottom_left = calibration.pixel_to_mm(0, 800)
    top_right = calibration.pixel_to_mm(800, 0)
    center = calibration.pixel_to_mm(400, 400)

    assert bottom_left.to_dict() == {"x": 0.0, "y": 0.0}
    assert top_right.to_dict() == {"x": 160.0, "y": 160.0}
    assert center.to_dict() == {"x": 80.0, "y": 80.0}


def test_calibration_round_trip(tmp_path):
    calibration = PerspectiveCalibration(
        source_points=np.array([[10, 20], [700, 18], [710, 500], [8, 505]], np.float32),
        output_width_px=800,
        output_height_px=600,
        tray_width_mm=200.0,
        tray_height_mm=160.0,
    )
    path = tmp_path / "calibration.json"
    calibration.save(path)
    loaded = PerspectiveCalibration.load(path)

    assert np.allclose(loaded.source_points, calibration.source_points)
    assert loaded.output_width_px == 800
    assert loaded.tray_height_mm == 160.0

