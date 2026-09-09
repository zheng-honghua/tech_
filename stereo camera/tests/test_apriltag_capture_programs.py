import json

import cv2
import numpy as np
import pytest
from scripts import dual_apriltag_calibrate, dual_rgbd_side_capture

from sorting_vision.apriltag_calibration import (
    AprilTagObservation,
    detect_apriltags,
    estimate_tray_frame_from_diagonal_tags,
    free_tag_pose_signature,
    generate_three_tag_assets,
    pose_is_diverse,
    tag_object_points,
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
