import json

import cv2
import numpy as np
import pytest
from scripts import dual_apriltag_calibrate, dual_rgbd_side_capture

from sorting_vision.apriltag_calibration import (
    AprilTagObservation,
    detect_apriltags,
    estimate_tray_frame_from_diagonal_tags,
    fixed_intrinsics_reprojection_rms,
    free_tag_pose_signature,
    generate_three_tag_assets,
    pose_is_diverse,
    tag_object_points,
    validate_apriltag_ids,
)
from sorting_vision.camera import RGBFrame, SynchronizedFramePair
from sorting_vision.capture_assistant import CaptureAssistantState
from sorting_vision.dual_capture_app import (
    DualCaptureQualityTracker,
    load_dual_batch_counts,
    render_dual_capture_assistant,
    save_dual_capture_sample,
)
from sorting_vision.rgbd import CameraIntrinsics, RGBDFrame
from sorting_vision.config import load_config


@pytest.mark.parametrize(
    ("parser", "required"),
    (
        (
            dual_apriltag_calibrate.build_parser(),
            ["--platform-id", "temporary", "--tag-size-mm", "30"],
        ),
        (
            dual_rgbd_side_capture.build_parser(),
            ["--batch-id", "resolution-test", "--platform-id", "temporary"],
        ),
    ),
)
def test_dual_programs_accept_all_three_resolution_overrides(parser, required):
    args = parser.parse_args(
        required
        + [
            "--color-width", "640", "--color-height", "480",
            "--depth-width", "848", "--depth-height", "480",
            "--fps", "15", "--side-width", "1280",
            "--side-height", "720", "--side-fps", "30",
        ]
    )
    assert (args.color_width, args.color_height) == (640, 480)
    assert (args.depth_width, args.depth_height) == (848, 480)
    assert args.fps == 15
    assert (args.side_width, args.side_height, args.side_fps) == (1280, 720, 30)


def test_apriltag_program_resolves_yaml_ids_and_command_overrides():
    parser = dual_apriltag_calibrate.build_parser()
    config = load_config("config/dual/temporary.yaml")
    defaults = parser.parse_args(
        ["--platform-id", "temporary", "--tag-size-mm", "30"]
    )
    assert dual_apriltag_calibrate._resolved_tag_ids(defaults, config) == (
        (config.dual_view.fixed_tag_a_id, config.dual_view.fixed_tag_b_id),
        config.dual_view.free_tag_id,
    )
    overridden = parser.parse_args(
        [
            "--platform-id", "temporary", "--tag-size-mm", "30",
            "--fixed-tag-a", "10", "--fixed-tag-b", "21", "--free-tag", "35",
        ]
    )
    assert dual_apriltag_calibrate._resolved_tag_ids(overridden, config) == (
        (10, 21), 35,
    )


def test_apriltag_program_accepts_saved_session_replay():
    args = dual_apriltag_calibrate.build_parser().parse_args(
        [
            "--platform-id", "temporary", "--tag-size-mm", "30",
            "--replay-session",
        ]
    )
    assert args.replay_session is True


def test_custom_apriltag_ids_are_generated_and_invalid_ids_are_rejected(tmp_path):
    paths = generate_three_tag_assets(
        tmp_path,
        fixed_tag_ids=(10, 21),
        free_tag_id=35,
        tag_size_mm=30.0,
    )
    assert {path.name for path in paths} >= {
        "apriltag-10-dict_apriltag_36h11.png",
        "apriltag-21-dict_apriltag_36h11.png",
        "apriltag-35-dict_apriltag_36h11.png",
    }
    metadata = json.loads(
        (tmp_path / "three-apriltag-printing.json").read_text(encoding="utf-8")
    )
    assert metadata["fixed_tag_ids"] == [10, 21]
    assert metadata["free_tag_id"] == 35
    with pytest.raises(ValueError, match="distinct"):
        validate_apriltag_ids((10, 10), 35)
    with pytest.raises(ValueError, match="outside.*range"):
        validate_apriltag_ids((10, 21), 999_999)


def _pair(delta_ms: float = 10.0, side: bool = True) -> SynchronizedFramePair:
    intrinsics = CameraIntrinsics(160, 120, 140, 140, 80, 60, 1.0)
    checker = (np.indices((120, 160)).sum(axis=0) % 2 * 255).astype(np.uint8)
    image = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    primary = RGBDFrame(
        image.copy(), np.full((120, 160), 600, np.uint16), intrinsics,
        10, "primary-10", 10, 11,
    )
    side_frame = RGBFrame(image.copy(), 20, "side-20", 20) if side else None
    return SynchronizedFramePair(
        primary,
        side_frame,
        1_000_000_000,
        None if not side else 1_000_000_000 + int(delta_ms * 1_000_000),
        50.0,
    )


def test_apriltag_detector_reads_generated_36h11_marker():
    aruco = pytest.importorskip("cv2.aruco")
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    marker = aruco.generateImageMarker(dictionary, 2, 180)
    canvas = np.full((260, 260), 255, np.uint8)
    canvas[40:220, 40:220] = marker
    observation = detect_apriltags(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
    assert observation.has(2)
    assert observation.corners_by_id[2].shape == (4, 2)


def test_apriltag_detector_downscales_and_returns_full_resolution_corners():
    aruco = pytest.importorskip("cv2.aruco")
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    marker = aruco.generateImageMarker(dictionary, 2, 400)
    canvas = np.full((800, 1600), 255, np.uint8)
    canvas[200:600, 600:1000] = marker
    observation = detect_apriltags(
        cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), maximum_detection_width=800
    )
    assert observation.has(2)
    center = observation.corners_by_id[2].mean(axis=0)
    np.testing.assert_allclose(center, [799.5, 399.5], atol=2.0)


def test_generate_three_tag_assets_are_detectable(tmp_path):
    paths = generate_three_tag_assets(tmp_path, tag_size_mm=30.0)
    assert len(paths) == 5
    for tag_id in (0, 1, 2):
        image = cv2.imread(str(tmp_path / f"apriltag-{tag_id}-dict_apriltag_36h11.png"))
        assert detect_apriltags(image).has(tag_id)
    metadata = json.loads(
        (tmp_path / "three-apriltag-printing.json").read_text(encoding="utf-8")
    )
    assert metadata["black_square_width_mm"] == 30.0
    assert metadata["layout_is_to_scale"] is False


def test_diagonal_tags_recover_tray_frame_and_scale():
    intrinsics = CameraIntrinsics(640, 480, 600, 600, 320, 240, 1.0)
    camera_matrix = np.asarray(
        [[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]]
    )
    tag_size = 20.0
    centers = ((10.0, 10.0, 600.0), (150.0, 150.0, 600.0))
    corners = {}
    for tag_id, center in zip((0, 1), centers):
        points = tag_object_points(tag_size) + np.asarray(center, np.float32)
        projected, _ = cv2.projectPoints(
            points, np.zeros(3), np.zeros(3), camera_matrix, np.zeros(5)
        )
        corners[tag_id] = projected.reshape(4, 2)
    tray_from_primary, scale_error = estimate_tray_frame_from_diagonal_tags(
        AprilTagObservation(corners),
        intrinsics,
        np.zeros(5),
        fixed_tag_ids=(0, 1),
        tag_size_mm=tag_size,
        tray_width_mm=160.0,
        tray_height_mm=160.0,
        fixed_tag_inset_mm=10.0,
    )
    for center, expected in zip(centers, ((10.0, 10.0), (150.0, 150.0))):
        mapped = tray_from_primary @ np.asarray([*center, 1.0])
        np.testing.assert_allclose(mapped[:2], expected, atol=0.6)
        assert abs(mapped[2]) < 0.6
    assert scale_error < 0.01


def test_fixed_intrinsics_reprojection_rms_does_not_refit_camera():
    camera_matrix = np.asarray(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
    )
    object_points = []
    image_points = []
    template = tag_object_points(30.0).reshape(-1, 1, 3)
    for rotation, translation in (
        ((0.1, 0.2, 0.0), (-80.0, -40.0, 500.0)),
        ((-0.2, 0.05, 0.3), (70.0, 55.0, 650.0)),
    ):
        projected, _ = cv2.projectPoints(
            template,
            np.asarray(rotation, np.float64),
            np.asarray(translation, np.float64),
            camera_matrix,
            np.zeros(5),
        )
        object_points.append(template.copy())
        image_points.append(projected)
    original = camera_matrix.copy()
    rms = fixed_intrinsics_reprojection_rms(
        object_points, image_points, camera_matrix, np.zeros(5)
    )
    assert rms < 1e-4
    np.testing.assert_array_equal(camera_matrix, original)


def test_saved_apriltag_session_loads_without_opening_cameras(tmp_path):
    session = tmp_path / "temporary"
    primary_dir = session / "free-poses" / "primary"
    side_dir = session / "free-poses" / "side"
    primary_dir.mkdir(parents=True)
    side_dir.mkdir(parents=True)
    intrinsics = CameraIntrinsics(160, 120, 140, 140, 80, 60, 1.0)
    image = np.full((120, 160, 3), 127, np.uint8)
    cv2.imwrite(str(session / "fixed-reference-primary.png"), image)
    np.save(session / "fixed-reference-depth.npy", np.full((120, 160), 600, np.uint16))
    (session / "fixed-reference-metadata.json").write_text(
        json.dumps(
            {
                "primary_frame_id": "primary-reference",
                "primary_timestamp_ns": 123,
                "primary_intrinsics": intrinsics.to_dict(),
                "fixed_tag_ids": [45, 17],
            }
        ),
        encoding="utf-8",
    )
    for index in range(20):
        name = f"pose-{index:03d}"
        cv2.imwrite(str(primary_dir / f"{name}.png"), image)
        cv2.imwrite(str(side_dir / f"{name}.png"), image)
        (session / "free-poses" / f"{name}.json").write_text(
            json.dumps({"free_tag_id": 50}), encoding="utf-8"
        )
    primary, side, reference, reference_image = (
        dual_apriltag_calibrate._load_saved_session(
            session, (45, 17), 50, 20
        )
    )
    assert len(primary) == len(side) == 20
    assert reference.frame_id == "primary-reference"
    assert reference_image.shape == (120, 160, 3)


def test_diagonal_tag_plane_error_reports_measured_angle():
    intrinsics = CameraIntrinsics(640, 480, 600, 600, 320, 240, 1.0)
    camera_matrix = np.asarray(
        [[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]]
    )
    corners = {}
    for tag_id, rotation, translation in (
        (0, (0.0, 0.0, 0.0), (-70.0, -70.0, 600.0)),
        (1, (0.45, 0.0, 0.0), (70.0, 70.0, 600.0)),
    ):
        projected, _ = cv2.projectPoints(
            tag_object_points(20.0),
            np.asarray(rotation, np.float64),
            np.asarray(translation, np.float64),
            camera_matrix,
            np.zeros(5),
        )
        corners[tag_id] = projected.reshape(4, 2)
    with pytest.raises(ValueError, match=r"disagree by \d+\.\d degrees"):
        estimate_tray_frame_from_diagonal_tags(
            AprilTagObservation(corners),
            intrinsics,
            np.zeros(5),
            fixed_tag_ids=(0, 1),
            tag_size_mm=20.0,
            tray_width_mm=160.0,
            tray_height_mm=160.0,
            fixed_tag_inset_mm=10.0,
        )


def test_two_view_pose_diversity_uses_both_cameras():
    corners = np.asarray([[0, 0], [40, 0], [40, 40], [0, 40]], np.float32)
    signature = np.concatenate(
        (free_tag_pose_signature(corners), free_tag_pose_signature(corners + 100))
    )
    assert pose_is_diverse([], signature)
    assert not pose_is_diverse([signature], signature.copy())
    moved_side = signature.copy()
    moved_side[4] += 50
    assert pose_is_diverse([signature], moved_side)
    with pytest.raises(ValueError, match="four-value"):
        pose_is_diverse([], np.zeros(3))


def test_dual_capture_quality_render_save_and_resume(tmp_path):
    pair = _pair()
    quality = DualCaptureQualityTracker(
        stable_frames_required=2,
        motion_threshold=0.1,
        minimum_side_blur_variance=10.0,
    )
    for _ in range(3):
        quality.update(pair)
    assert quality.ready(pair)
    state = CaptureAssistantState(target_per_label=2)
    canvas = render_dual_capture_assistant(pair, state, quality, "test")
    assert canvas.shape == (490, 1440, 3)
    target = save_dual_capture_sample(
        pair,
        tmp_path,
        "batch-1",
        "temporary",
        "empty_tray",
        "calibration-hash",
        quality,
    )
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["capture_quality"]["ready"] is True
    assert load_dual_batch_counts(tmp_path, "batch-1", "temporary") == {
        "empty_tray": 1
    }
    assert load_dual_batch_counts(tmp_path, "batch-1", "competition") == {}


def test_dual_capture_rejects_missing_or_unsynchronised_side():
    quality = DualCaptureQualityTracker(stable_frames_required=1)
    missing = _pair(side=False)
    quality.update(missing)
    assert "side_camera_missing" in quality.rejection_reasons(missing)
    unsynchronised = _pair(delta_ms=80.0)
    quality.update(unsynchronised)
    assert "camera_pair_out_of_sync" in quality.rejection_reasons(unsynchronised)
