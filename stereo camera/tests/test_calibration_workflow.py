import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts import dual_apriltag_calibrate as tags
from scripts import rgb_intrinsics_calibrate as board
from sorting_vision.apriltag_calibration import (
    AprilTagObservation, _epipolar_p95, estimate_tray_frame_from_diagonal_tags,
    stereo_projection_diagnostics, tag_object_points,
)
from sorting_vision.calibration_capture import (
    CalibrationCaptureTracker, prepare_calibration_session, publish_calibration,
)
from sorting_vision.camera import RGBFrame
from sorting_vision.intrinsic_calibration import CameraCalibration, checkerboard_points_from_image
from sorting_vision.rgbd import CameraIntrinsics, RGBDFrame


class Candidate:
    def __init__(self, valid):
        self.valid = valid

    def save(self, path):
        Path(path).write_text(json.dumps({"valid": self.valid}), encoding="utf-8")


def test_failed_calibration_never_replaces_existing_output(tmp_path):
    output = tmp_path / "active.json"
    output.write_bytes(b"previous calibrated content")
    result = publish_calibration(Candidate(False), output, tmp_path / "session")
    assert output.read_bytes() == b"previous calibrated content"
    assert not result["published"]
    assert json.loads((tmp_path / "session/calibration-summary.json").read_text())["valid"] is False
    result = publish_calibration(Candidate(True), output, tmp_path / "session")
    assert result["published"]
    assert Path(result["backup_file"]).read_bytes() == b"previous calibrated content"
    assert json.loads(output.read_text())["valid"] is True


def test_live_sessions_keep_old_observations(tmp_path):
    root = tmp_path / "session"
    first = prepare_calibration_session(root)
    (first / "observation.png").write_bytes(b"saved")
    second = prepare_calibration_session(root)
    assert second != first and second.parent == first
    assert (first / "observation.png").read_bytes() == b"saved"


def test_capture_rejects_blur_border_motion_and_missing_target():
    image = np.repeat((np.indices((100, 120)).sum(axis=0) % 2 * 255).astype(np.uint8)[..., None], 3, axis=2)
    points = np.array([[20, 20], [70, 20], [70, 70], [20, 70]], dtype=float)
    tracker = CalibrationCaptureTracker(stable_frames=3)
    assert not tracker.update(image, points)["ready"]
    assert not tracker.update(image, points)["ready"]
    assert tracker.update(image, points)["ready"]
    assert not tracker.update(image, points + 5)["ready"]
    assert "target_blurred" in tracker.update(np.zeros_like(image), points)["reasons"]
    assert "target_near_image_border" in tracker.update(image, points - 19)["reasons"]
    assert tracker.update(image, np.empty((0, 2)))["stable_frames"] == 0


def test_headless_live_auto_capture_solves_without_keyboard(tmp_path, monkeypatch):
    class Source:
        index = 0
        closed = False

        def read(self):
            self.index += 1
            if self.index > 20:
                pytest.fail("headless camera kept reading after enough captures")
            return RGBFrame(np.zeros((400, 800, 3), np.uint8), self.index, str(self.index))

        def close(self):
            self.closed = True

    source = Source()
    ticks = iter(range(2_000_000, 100_000_000, 2_000_000))
    monkeypatch.setattr(board.time, "monotonic_ns", lambda: next(ticks))
    monkeypatch.setattr(board, "_camera_source", lambda args: source)
    monkeypatch.setattr(board.CalibrationCaptureTracker, "update", lambda *a: {"ready": True, "reasons": []})
    monkeypatch.setattr(board, "checkerboard_points_from_image", lambda *a, **k: (
        np.zeros((70, 1, 3), np.float32), np.full((70, 1, 2), source.index * 40, np.float32)))
    monkeypatch.setattr(board, "calibration_view_signature", lambda pixels: np.array([source.index * 40, 0, 5, 0]))
    monkeypatch.setattr(board, "_solve_and_save", lambda *a: SimpleNamespace(valid=True))
    monkeypatch.setattr(board.cv2, "destroyAllWindows", lambda: None)
    assert board.main(["--headless", "--auto-capture", "--auto-interval-ms", "1", "--required-frames", "20",
                       "--session-dir", str(tmp_path), "--output", str(tmp_path / "intrinsics.json")]) == 0
    assert source.closed and source.index == 20


def test_replay_does_not_open_camera_and_checks_target_dimensions(tmp_path, monkeypatch):
    session = tmp_path / "camera"
    (session / "frames").mkdir(parents=True)
    cv2.imwrite(str(session / "frames/frame-000.png"), np.zeros((80, 120, 3), np.uint8))
    (session / "frame-000.json").write_text(json.dumps({"target": {"type": "wrong-size"}}))
    def forbidden(args):
        pytest.fail("replay opened a camera")
    monkeypatch.setattr(board, "_camera_source", forbidden)
    assert board.main(["--replay-session", "--session-dir", str(session)]) == 1


def test_checkerboard_downscale_uses_both_actual_resize_ratios(monkeypatch):
    image = np.zeros((721, 1281, 3), np.uint8)
    corners = np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]], np.float32)
    monkeypatch.setattr(cv2, "findChessboardCornersSB", lambda *a, **k: (True, corners.copy()))
    monkeypatch.setattr(cv2, "cornerSubPix", lambda *a: a[1])
    _, pixels = checkerboard_points_from_image(image, corners_x=3, corners_y=3,
        square_size_mm=20, maximum_detection_width=640)
    np.testing.assert_allclose(pixels.reshape(-1, 2),
        (corners.reshape(-1, 2) + 0.5) * [1281 / 640, 721 / 360] - 0.5, atol=1e-4)


def test_stereo_quality_uses_undistorted_epipolar_and_original_reprojection():
    matrix = np.array([[800., 0, 640], [0, 810, 360], [0, 0, 1]])
    d1 = np.array([-0.22, 0.03, .005, -.004, 0])
    d2 = np.array([.12, -.02, -.003, .002, 0])
    rotation = cv2.Rodrigues(np.array([.03, -.12, .01]))[0]
    translation = np.array([-180., 10, 30])
    tx, ty, tz = translation
    cross = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
    fundamental = np.linalg.inv(matrix).T @ cross @ rotation @ np.linalg.inv(matrix)
    primary, side = [], []
    for index in range(5):
        xyz = tag_object_points(40).astype(np.float64) + [-250 + index * 100, 120, 550 + index * 25]
        p1 = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), matrix, d1)[0]
        p2 = cv2.projectPoints(xyz @ rotation.T + translation, np.zeros(3), np.zeros(3), matrix, d2)[0]
        primary.append(p1)
        side.append(p2)
    assert _epipolar_p95(primary, side, fundamental) > 1
    result = stereo_projection_diagnostics(primary, side, matrix, d1, matrix, d2, rotation, translation, fundamental)
    assert result["undistorted_epipolar_p95_px"] < .01
    assert result["reprojection_p95_px"] < .01
    mismatched = [value + [0, 10] for value in side]
    result = stereo_projection_diagnostics(primary, mismatched, matrix, d1, matrix, d2, rotation, translation, fundamental)
    assert result["reprojection_p95_px"] > 2


def test_fixed_tags_rotated_180_degrees_are_rejected():
    matrix = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])
    corners = {}
    for tag_id, angle, center in ((0, 0., [10., 10, 600]), (1, np.pi, [150., 150, 600])):
        corners[tag_id] = cv2.projectPoints(tag_object_points(20), np.array([0., 0, angle]),
                                           np.array(center), matrix, np.zeros(5))[0].reshape(4, 2)
    with pytest.raises(ValueError, match="printed orientations"):
        estimate_tray_frame_from_diagonal_tags(AprilTagObservation(corners),
            CameraIntrinsics(640, 480, 600, 600, 320, 240), np.zeros(5), fixed_tag_ids=(0, 1),
            tag_size_mm=20, tray_width_mm=160, tray_height_mm=160, fixed_tag_inset_mm=10)


def test_complete_synthetic_three_tag_solve_keeps_intrinsics_fixed(monkeypatch):
    from sorting_vision import apriltag_calibration as calibration
    intrinsics = CameraIntrinsics(640, 480, 600, 600, 320, 240)
    matrix = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])
    rotation = cv2.Rodrigues(np.array([.12, -.05, .01]))[0]
    translation = np.array([-120., 0, 20.])
    def project(xyz):
        return cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), matrix, np.zeros(5))[0].reshape(4, 2).astype(np.float32)
    template = tag_object_points(20)
    observations = {250: AprilTagObservation({
        0: project(template + [10, 10, 650]), 1: project(template + [150, 150, 650])})}
    primary, side = [], []
    for index in range(20):
        rvec = np.array([-.3 + index * .03, .2 * np.sin(index), index * .08])
        xyz = template @ cv2.Rodrigues(rvec)[0].T + [-80 + index * 10, -40 + index * 4, 550 + index * 8]
        observations[index + 1] = AprilTagObservation({2: project(xyz)})
        observations[index + 41] = AprilTagObservation({2: project(xyz @ rotation.T + translation)})
        primary.append(np.full((480, 640, 3), index + 1, np.uint8))
        side.append(np.full((480, 640, 3), index + 41, np.uint8))
    reference_image = np.full((480, 640, 3), 250, np.uint8)
    reference = RGBDFrame(reference_image, np.full((480, 640), 650, np.uint16), intrinsics, 1, "reference")
    cameras = [CameraCalibration(name, intrinsics, np.zeros(5), .1, .2, .6, 35, 20, 25, {}, "test")
               for name in ("primary", "side")]
    monkeypatch.setattr(calibration, "detect_apriltags", lambda image, *args: observations[int(image[0, 0, 0])])
    result = calibration.calibrate_apriltag_pairs(primary, side, reference, reference_image, *cameras,
        platform_id="temporary", tag_size_mm=20, fixed_tag_ids=(0, 1), free_tag_id=2,
        tray_width_mm=160, tray_height_mm=160, fixed_tag_inset_mm=10)
    assert result.valid
    np.testing.assert_allclose(result.side_from_primary[:3, :3], rotation, atol=1e-5)
    np.testing.assert_allclose(result.side_from_primary[:3, 3], translation, atol=.01)
    assert result.metrics.joint_projection_p95_px < .01
    assert result.primary_intrinsics == intrinsics and result.side_intrinsics == intrinsics
    assert len(result.board["projection_diagnostics"]["per_pose"]) == 20


@pytest.mark.parametrize("metadata, expected_spec, message", [
    ({"session_spec": {"tag_size_mm": 24}}, {"tag_size_mm": 30}, "dimensions"),
    ({"pair_delta_ms": 60}, None, "unsynchronised"),
    ({"primary_rgb_depth_sync_ms": 30}, None, "RGB-depth"),
])
def test_replay_rejects_changed_geometry_and_bad_saved_timestamps(tmp_path, metadata, expected_spec, message):
    cv2.imwrite(str(tmp_path / "fixed-reference-primary.png"), np.zeros((40, 40, 3), np.uint8))
    np.save(tmp_path / "fixed-reference-depth.npy", np.ones((40, 40), np.uint16))
    metadata["fixed_tag_ids"] = [0, 1]
    (tmp_path / "fixed-reference-metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        tags._load_saved_session(tmp_path, (0, 1), 2, 20, expected_spec, 50)
