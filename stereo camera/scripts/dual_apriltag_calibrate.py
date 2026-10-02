"""Live three-AprilTag stereo calibration for top RGB-D plus side RGB."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from sorting_vision.apriltag_calibration import (
    AprilTagObservation,
    calibrate_apriltag_pairs,
    detect_apriltags,
    draw_apriltags,
    free_tag_pose_signature,
    generate_three_tag_assets,
    pose_is_diverse,
    validate_apriltag_ids,
)
from sorting_vision.camera import (
    DualCameraSource,
    OpenCVCameraSource,
    ProcessOpenCVCameraSource,
    RealSenseSource,
    ThreadedRealSenseSource,
)
from sorting_vision.config import load_config
from sorting_vision.calibration_capture import (
    CalibrationCaptureTracker, prepare_calibration_session,
    publish_calibration, write_calibration_image,
)
from sorting_vision.intrinsic_calibration import CameraCalibration, load_projection_calibration
from sorting_vision.dual_view import DualCalibrationQualityLimits
from sorting_vision.rgbd import CameraIntrinsics, RGBDFrame
from sorting_vision.rgbd_dataset import depth_preview


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve stereo/tray geometry from three AprilTags using independently "
            "calibrated RGB intrinsics"
        )
    )
    parser.add_argument("--config")
    parser.add_argument("--platform-id", choices=("temporary", "competition"), required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--primary-intrinsics",
        help="independent top RGB intrinsic calibration JSON",
    )
    parser.add_argument(
        "--side-intrinsics",
        help="independent side RGB intrinsic calibration JSON",
    )
    parser.add_argument(
        "--generate-tags-dir",
        help="generate IDs and a placement preview, then exit without opening cameras",
    )
    parser.add_argument("--session-dir")
    parser.add_argument(
        "--replay-session",
        action="store_true",
        help="recalculate from the saved fixed reference and free poses without opening cameras",
    )
    parser.add_argument("--side-camera-index", type=int)
    parser.add_argument("--color-width", type=_positive_int, help="top RealSense RGB width")
    parser.add_argument("--color-height", type=_positive_int, help="top RealSense RGB height")
    parser.add_argument("--depth-width", type=_positive_int, help="top RealSense depth width")
    parser.add_argument("--depth-height", type=_positive_int, help="top RealSense depth height")
    parser.add_argument("--fps", type=_positive_int, help="top RealSense RGB/depth FPS")
    parser.add_argument("--side-width", type=_positive_int, help="side RGB width")
    parser.add_argument("--side-height", type=_positive_int, help="side RGB height")
    parser.add_argument("--side-fps", type=_positive_int, help="side RGB FPS")
    parser.add_argument("--tag-size-mm", type=float, required=True)
    parser.add_argument("--fixed-tag-a", type=int, help="override fixed AprilTag A ID")
    parser.add_argument("--fixed-tag-b", type=int, help="override fixed AprilTag B ID")
    parser.add_argument("--free-tag", type=int, help="override moving AprilTag ID")
    parser.add_argument("--tag-ids", type=int, nargs=3, metavar=("FIXED_A", "FIXED_B", "FREE"),
                        help="three custom IDs in fixed-A, fixed-B, free order; also usable for calibration")
    parser.add_argument("--fixed-tag-inset-mm", type=float)
    parser.add_argument("--tray-width-mm", type=float)
    parser.add_argument("--tray-height-mm", type=float)
    parser.add_argument("--dictionary", default="DICT_APRILTAG_36h11")
    parser.add_argument("--required-poses", type=int, default=20)
    parser.add_argument("--discard-frames", type=int, default=30)
    parser.add_argument("--stable-frames", type=_positive_int, default=3)
    parser.add_argument("--max-corner-motion-px", type=float, default=1.5)
    parser.add_argument("--min-sharpness", type=float, default=50.0)
    parser.add_argument("--min-reference-depth-ratio", type=float, default=0.85)
    parser.add_argument(
        "--detection-width",
        type=int,
        default=960,
        help="downscale only AprilTag detection for live speed; corners are refined at full resolution",
    )
    return parser


def _resolve_storage_paths(args):
    if not args.config:
        args.config = str(Path("config") / "dual" / f"{args.platform_id}.yaml")
    if not args.session_dir:
        args.session_dir = str(
            Path("data")
            / "calibration"
            / args.platform_id
            / "extrinsics"
            / "sessions"
            / "current"
        )
    if not args.output and not args.generate_tags_dir:
        args.output = str(
            Path("config") / "dual" / args.platform_id / "calibration.json"
        )
    return args


def _resolved_tag_ids(args, config) -> tuple[tuple[int, int], int]:
    explicit = _explicit_tag_ids(args)
    dual = config.dual_view
    fixed_ids = (
        dual.fixed_tag_a_id if explicit[0] is None else explicit[0],
        dual.fixed_tag_b_id if explicit[1] is None else explicit[1],
    )
    free_tag_id = dual.free_tag_id if explicit[2] is None else explicit[2]
    validate_apriltag_ids(fixed_ids, free_tag_id, args.dictionary)
    return fixed_ids, free_tag_id


def _explicit_tag_ids(args) -> tuple[int | None, int | None, int | None]:
    named = (args.fixed_tag_a, args.fixed_tag_b, args.free_tag)
    combined = getattr(args, "tag_ids", None)
    if combined is not None:
        if any(value is not None for value in named):
            raise ValueError("use --tag-ids OR the three individual ID options, not both")
        return tuple(combined)
    return named


def _generation_tag_ids(args, input_fn=None) -> tuple[tuple[int, int], int]:
    """Require an intentional three-ID choice before generating print assets."""
    explicit = _explicit_tag_ids(args)
    if any(value is not None for value in explicit):
        if any(value is None for value in explicit):
            raise ValueError("generation needs all three IDs; use --tag-ids FIXED_A FIXED_B FREE")
        validate_apriltag_ids(explicit[:2], explicit[2], args.dictionary)
        return explicit[:2], explicit[2]
    if input_fn is None:
        if not sys.stdin.isatty():
            raise ValueError("non-interactive generation requires --tag-ids FIXED_A FIXED_B FREE")
        input_fn = input
    print("请输入三个不同的AprilTag ID，顺序为：固定A、固定B、自由码。")
    print("例如：10 21 35；输入q取消。这里不会沿用配置中的旧ID。")
    while True:
        text = input_fn("三个ID：").strip()
        if text.lower() in {"q", "quit", "exit"}:
            raise ValueError("tag generation cancelled")
        try:
            parts = re.split(r"[\s,，、;；]+", text)
            if len(parts) != 3:
                raise ValueError("enter exactly three integer IDs")
            values = tuple(int(value) for value in parts)
            validate_apriltag_ids(values[:2], values[2], args.dictionary)
        except ValueError as error:
            print(f"ID无效，请重新输入：{error}")
            continue
        return values[:2], values[2]


def _load_camera_calibrations(args, config):
    primary_path = args.primary_intrinsics or config.dual_view.primary_intrinsics_path
    side_path = args.side_intrinsics or config.dual_view.side_intrinsics_path
    primary = load_projection_calibration(primary_path)
    side = CameraCalibration.load(side_path)
    if not primary.valid:
        raise ValueError(
            f"primary RGB intrinsics are invalid: {primary.rejection_reasons}"
        )
    if not side.valid:
        raise ValueError(f"side RGB intrinsics are invalid: {side.rejection_reasons}")
    requested_primary = (
        args.color_width or config.camera.realsense_color_width,
        args.color_height or config.camera.realsense_color_height,
    )
    requested_side = (
        args.side_width or config.dual_view.side_width,
        args.side_height or config.dual_view.side_height,
    )
    calibrated_primary = (primary.intrinsics.width, primary.intrinsics.height)
    calibrated_side = (side.intrinsics.width, side.intrinsics.height)
    if requested_primary != calibrated_primary:
        raise ValueError(
            f"requested primary RGB size {requested_primary} differs from "
            f"intrinsics {calibrated_primary}"
        )
    if requested_side != calibrated_side:
        raise ValueError(
            f"requested side RGB size {requested_side} differs from "
            f"intrinsics {calibrated_side}"
        )
    return primary, side


def _calibration_quality(platform_id: str, dual):
    if platform_id == "competition":
        return DualCalibrationQualityLimits(), "strict"
    return (
        DualCalibrationQualityLimits(
            dual.calibration_max_primary_rms_px,
            dual.calibration_max_side_rms_px,
            dual.calibration_max_joint_projection_p95_px,
            dual.calibration_max_scale_error_ratio,
        ),
        dual.calibration_quality_profile,
    )


def _camera_source(args, config):
    camera = config.camera
    dual = config.dual_view
    primary_type = (
        ThreadedRealSenseSource if dual.side_process_isolation else RealSenseSource
    )
    primary = primary_type(
        depth_width=getattr(args, "depth_width", None) or camera.realsense_depth_width,
        depth_height=getattr(args, "depth_height", None) or camera.realsense_depth_height,
        color_width=getattr(args, "color_width", None) or camera.realsense_color_width,
        color_height=getattr(args, "color_height", None) or camera.realsense_color_height,
        fps=getattr(args, "fps", None) or camera.realsense_fps,
        camera_model=camera.realsense_model,
        frame_prefix=camera.realsense_frame_prefix,
    )
    try:
        if not dual.side_process_isolation:
            primary.read()
        side_kwargs = dict(
            camera_index=dual.side_camera_index if args.side_camera_index is None else args.side_camera_index,
            width=getattr(args, "side_width", None) or dual.side_width,
            height=getattr(args, "side_height", None) or dual.side_height,
            fps=getattr(args, "side_fps", None) or dual.side_fps,
            backend=dual.side_backend,
            fourcc=dual.side_fourcc,
            warmup_frames=camera.warmup_frames,
            reconnect_attempts=camera.reconnect_attempts,
            auto_exposure=dual.side_auto_exposure,
            exposure=dual.side_exposure,
            auto_white_balance=dual.side_auto_white_balance,
            white_balance=dual.side_white_balance,
            autofocus=dual.side_autofocus,
            focus=dual.side_focus,
        )
        side = (
            ProcessOpenCVCameraSource(**side_kwargs)
            if dual.side_process_isolation
            else OpenCVCameraSource(**side_kwargs)
        )
    except Exception:
        primary.close()
        raise
    print(
        json.dumps(
            {
                "side_camera_index": side.camera_index,
                "side_requested_size": [side.width, side.height],
                "side_control_status": side.control_status,
            },
            ensure_ascii=False,
        )
    )
    return DualCameraSource(
        primary,
        side,
        max_pair_delta_ms=dual.max_pair_delta_ms,
        pair_timeout_ms=dual.pair_timeout_ms,
        acquisition_mode=dual.acquisition_mode,
    )


def _render(pair, primary_observation, side_observation, captured, required, message):
    width, height = 480, 270
    primary = cv2.resize(draw_apriltags(pair.primary.color_bgr, primary_observation), (width, height))
    side_image = np.zeros_like(primary) if pair.side is None else draw_apriltags(pair.side.color_bgr, side_observation)
    side = cv2.resize(side_image, (width, height))
    depth = cv2.resize(depth_preview(pair.primary.depth_mm), (width, height), interpolation=cv2.INTER_NEAREST)
    views = np.hstack((primary, depth, side))
    panel = np.full((140, views.shape[1], 3), 25, np.uint8)
    canvas = np.vstack((views, panel))
    delta = pair.pair_delta_ms
    lines = [
        f"free poses {captured}/{required} | pair {'missing' if delta is None else f'{delta:.1f} ms'}",
        "R=fixed diagonal reference | SPACE=capture free tag pose | C=calibrate | Q=quit",
        message[:130],
    ]
    for index, text in enumerate(lines):
        cv2.putText(canvas, text, (14, height + 32 + 34 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.57,
                    (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def _session_spec(args, config, calibrations) -> dict:
    fixed_ids, free_id = _resolved_tag_ids(args, config)
    return {"dictionary": args.dictionary, "fixed_tag_ids": list(fixed_ids), "free_tag_id": free_id,
            "tag_size_mm": args.tag_size_mm,
            "tray_width_mm": config.tray.width_mm if args.tray_width_mm is None else args.tray_width_mm,
            "tray_height_mm": config.tray.height_mm if args.tray_height_mm is None else args.tray_height_mm,
            "fixed_tag_inset_mm": args.tag_size_mm / 2 if args.fixed_tag_inset_mm is None else args.fixed_tag_inset_mm,
            "primary_intrinsics_hash": calibrations[0].to_dict()["calibration_hash"],
            "side_intrinsics_hash": calibrations[1].to_dict()["calibration_hash"]}


def _pair_rejection_reasons(pair) -> list[str]:
    reasons = []
    if pair.side is None or not pair.synchronized:
        reasons.append("cameras_missing_or_unsynchronised")
    if pair.primary.sync_delta_ms > 20.0:
        reasons.append("primary_RGB_depth_sync_above_20_ms")
    return reasons


def _reference_depth_ratio(frame, corners) -> float:
    points = np.asarray(corners).reshape(-1, 2)
    if len(points) < 8:
        return 0.0
    low = np.maximum(np.floor(points.min(axis=0)).astype(int), 0)
    high = np.minimum(np.ceil(points.max(axis=0)).astype(int) + 1,
                      [frame.depth.shape[1], frame.depth.shape[0]])
    roi = frame.depth_mm[low[1]:high[1], low[0]:high[0]]
    return float(np.mean(np.isfinite(roi) & (roi > 0))) if roi.size else 0.0


def _load_saved_session(
    session: Path,
    fixed_ids: tuple[int, int],
    free_tag_id: int,
    required_poses: int,
    expected_spec: dict | None = None,
    max_pair_delta_ms: float | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], RGBDFrame, np.ndarray]:
    metadata_path = session / "fixed-reference-metadata.json"
    primary_path = session / "fixed-reference-primary.png"
    depth_path = session / "fixed-reference-depth.npy"
    for path in (metadata_path, primary_path, depth_path):
        if not path.is_file():
            raise ValueError(f"saved calibration session is missing {path.name}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if expected_spec is not None:
        if "session_spec" in metadata and metadata["session_spec"] != expected_spec:
            raise ValueError("saved tag/tray dimensions or intrinsic hashes differ from requested calibration")
        if "session_spec" not in metadata:
            print("Legacy session: physical dimensions and intrinsic hashes were not recorded; verify them manually")
    if tuple(map(int, metadata.get("fixed_tag_ids", ()))) != fixed_ids:
        raise ValueError("saved fixed-tag IDs differ from the requested IDs")
    delta = metadata.get("pair_delta_ms")
    if delta is not None and max_pair_delta_ms is not None and (not np.isfinite(delta) or delta < 0 or delta > max_pair_delta_ms):
        raise ValueError("saved fixed reference is unsynchronised")
    rgbd_delta = metadata.get("primary_rgb_depth_sync_ms")
    if rgbd_delta is not None and (not np.isfinite(rgbd_delta) or rgbd_delta < 0 or rgbd_delta > 20):
        raise ValueError("saved fixed reference has invalid primary RGB-depth synchronization")
    reference_image = cv2.imread(str(primary_path), cv2.IMREAD_COLOR)
    if reference_image is None:
        raise ValueError("saved fixed-reference-primary.png cannot be decoded")
    reference_frame = RGBDFrame(
        reference_image,
        np.load(depth_path, allow_pickle=False),
        CameraIntrinsics(**metadata["primary_intrinsics"]),
        int(metadata.get("primary_timestamp_ns", 0)),
        str(metadata.get("primary_frame_id", "saved-reference")),
    )
    primary_paths = sorted((session / "free-poses" / "primary").glob("pose-*.png"))
    side_paths = sorted((session / "free-poses" / "side").glob("pose-*.png"))
    if len(primary_paths) != len(side_paths):
        raise ValueError("saved primary/side pose counts do not match")
    side_by_stem = {path.stem: path for path in side_paths}
    reference_host_timestamp_ns = int(metadata.get("primary_host_timestamp_ns", 0))
    primary_images: list[np.ndarray] = []
    side_images: list[np.ndarray] = []
    for primary_file in primary_paths:
        side_file = side_by_stem.get(primary_file.stem)
        if side_file is None:
            raise ValueError("saved primary/side pose names do not match")
        pose_metadata_path = session / "free-poses" / f"{primary_file.stem}.json"
        if not pose_metadata_path.is_file():
            raise ValueError(f"saved session is missing {pose_metadata_path.name}")
        pose_metadata = json.loads(pose_metadata_path.read_text(encoding="utf-8"))
        if int(pose_metadata.get("free_tag_id", -1)) != free_tag_id:
            raise ValueError(f"saved {primary_file.stem} uses a different free-tag ID")
        delta = pose_metadata.get("pair_delta_ms")
        if delta is not None and max_pair_delta_ms is not None and (not np.isfinite(delta) or delta < 0 or delta > max_pair_delta_ms):
            raise ValueError(f"saved {primary_file.stem} is unsynchronised")
        rgbd_delta = pose_metadata.get("primary_rgb_depth_sync_ms")
        if rgbd_delta is not None and (not np.isfinite(rgbd_delta) or rgbd_delta < 0 or rgbd_delta > 20):
            raise ValueError(f"saved {primary_file.stem} has invalid primary RGB-depth synchronization")
        pose_host_timestamp_ns = int(pose_metadata.get("primary_host_timestamp_ns", 0))
        if (
            reference_host_timestamp_ns > 0
            and pose_host_timestamp_ns > 0
            and pose_host_timestamp_ns < reference_host_timestamp_ns
        ):
            continue
        primary_image = cv2.imread(str(primary_file), cv2.IMREAD_COLOR)
        side_image = cv2.imread(str(side_file), cv2.IMREAD_COLOR)
        if primary_image is None or side_image is None:
            raise ValueError(f"saved {primary_file.stem} image cannot be decoded")
        primary_images.append(primary_image)
        side_images.append(side_image)
    if len(primary_images) < required_poses:
        raise ValueError(
            f"saved current run has {len(primary_images)}/{required_poses} paired poses"
        )
    return primary_images, side_images, reference_frame, reference_image


def _solve_and_save(
    args,
    config,
    fixed_ids: tuple[int, int],
    free_tag_id: int,
    primary_images: list[np.ndarray],
    side_images: list[np.ndarray],
    reference_frame: RGBDFrame,
    reference_image: np.ndarray,
    session: Path,
    camera_calibrations,
):
    primary_camera_calibration, side_camera_calibration = camera_calibrations
    dual = config.dual_view
    quality_limits, quality_profile = _calibration_quality(args.platform_id, dual)
    calibration = calibrate_apriltag_pairs(
        primary_images,
        side_images,
        reference_frame,
        reference_image,
        primary_camera_calibration,
        side_camera_calibration,
        platform_id=args.platform_id,
        tag_size_mm=args.tag_size_mm,
        fixed_tag_ids=fixed_ids,
        free_tag_id=free_tag_id,
        tray_width_mm=config.tray.width_mm if args.tray_width_mm is None else args.tray_width_mm,
        tray_height_mm=config.tray.height_mm if args.tray_height_mm is None else args.tray_height_mm,
        fixed_tag_inset_mm=args.fixed_tag_inset_mm,
        dictionary_name=args.dictionary,
        maximum_detection_width=args.detection_width,
        quality_limits=quality_limits,
        quality_profile=quality_profile,
    )
    status = publish_calibration(calibration, args.output, session)
    print(json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2))
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return calibration


def main(argv: list[str] | None = None) -> int:
    args = _resolve_storage_paths(build_parser().parse_args(argv))
    if not np.isfinite(args.tag_size_mm) or args.tag_size_mm <= 0:
        raise ValueError("tag-size-mm must be finite and positive")
    if args.generate_tags_dir:
        try:
            fixed_ids, free_tag_id = _generation_tag_ids(args)
            paths = generate_three_tag_assets(
                args.generate_tags_dir, fixed_tag_ids=fixed_ids, free_tag_id=free_tag_id,
                tag_size_mm=args.tag_size_mm, dictionary_name=args.dictionary,
            )
        except (ValueError, RuntimeError, OSError, EOFError, KeyboardInterrupt, cv2.error) as error:
            print(f"Generation cancelled or rejected: {error}")
            return 1
        print(json.dumps({"fixed_tag_ids": list(fixed_ids), "free_tag_id": free_tag_id,
            "dictionary": args.dictionary, "black_square_width_mm": args.tag_size_mm,
            "generated": [str(path.resolve()) for path in paths]}, ensure_ascii=False, indent=2))
        print(f"标定时沿用：--tag-ids {fixed_ids[0]} {fixed_ids[1]} {free_tag_id}")
        return 0
    if args.required_poses < 20:
        raise ValueError("required-poses must be at least 20")
    if args.detection_width < 320:
        raise ValueError("detection-width must be at least 320 pixels")
    config = load_config(args.config)
    fixed_ids, free_tag_id = _resolved_tag_ids(args, config)
    for value in (args.tray_width_mm, args.tray_height_mm):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError("tray dimensions must be finite and positive")
    if args.fixed_tag_inset_mm is not None and (not np.isfinite(args.fixed_tag_inset_mm) or args.fixed_tag_inset_mm < args.tag_size_mm / 2):
        raise ValueError("fixed-tag-inset-mm must be at least half the tag size")
    if not np.isfinite(args.min_reference_depth_ratio) or not 0 < args.min_reference_depth_ratio <= 1:
        raise ValueError("min-reference-depth-ratio must be in (0, 1]")
    try:
        camera_calibrations = _load_camera_calibrations(args, config)
    except (ValueError, OSError, KeyError) as error:
        print(f"Calibration rejected: {error}")
        return 1
    session = Path(args.session_dir)
    spec = _session_spec(args, config, camera_calibrations)
    if args.replay_session:
        try:
            saved = _load_saved_session(
                session, fixed_ids, free_tag_id, args.required_poses, spec,
                config.dual_view.max_pair_delta_ms,
            )
            calibration = _solve_and_save(
                args,
                config,
                fixed_ids,
                free_tag_id,
                *saved,
                session,
                camera_calibrations,
            )
        except (ValueError, OSError, KeyError, cv2.error) as error:
            print(f"Calibration rejected: {error}")
            return 1
        return 0 if calibration.valid else 2
    trackers = {name: CalibrationCaptureTracker(args.stable_frames, args.max_corner_motion_px, args.min_sharpness)
                for name in ("primary_fixed", "side_fixed", "primary_free", "side_free")}
    session = prepare_calibration_session(session)
    print(f"Calibration session: {session.resolve()}")
    try:
        source = _camera_source(args, config)
    except (ValueError, RuntimeError, OSError, cv2.error) as error:
        print(f"Camera open failed: {error}")
        return 1
    primary_dir = session / "free-poses" / "primary"
    side_dir = session / "free-poses" / "side"
    depth_dir = session / "free-poses" / "depth"
    for directory in (primary_dir, side_dir, depth_dir):
        directory.mkdir(parents=True, exist_ok=True)
    primary_images: list[np.ndarray] = []
    side_images: list[np.ndarray] = []
    signatures: list[np.ndarray] = []
    reference_frame = None
    reference_image = None
    message = (
        f"Fix tags {fixed_ids[0]}/{fixed_ids[1]} diagonally; "
        f"move and tilt tag {free_tag_id}"
    )
    exit_code = 1
    try:
        for _ in range(max(0, args.discard_frames)):
            source.read()
        while True:
            pair = source.read()
            primary_observation = detect_apriltags(
                pair.primary.color_bgr,
                args.dictionary,
                maximum_detection_width=args.detection_width,
            )
            side_observation = (
                detect_apriltags(
                    pair.side.color_bgr,
                    args.dictionary,
                    maximum_detection_width=args.detection_width,
                )
                if pair.side is not None
                else AprilTagObservation({})
            )
            for image, calibration in ((pair.primary.color_bgr, camera_calibrations[0]),
                                       (None if pair.side is None else pair.side.color_bgr, camera_calibrations[1])):
                if image is not None and (image.shape[1], image.shape[0]) != (calibration.intrinsics.width, calibration.intrinsics.height):
                    raise ValueError("actual camera resolution differs from saved intrinsics")
            quality = {}
            for prefix, image, observation in (("primary", pair.primary.color_bgr, primary_observation),
                                               ("side", pair.primary.color_bgr if pair.side is None else pair.side.color_bgr, side_observation)):
                for kind, ids in (("fixed", fixed_ids), ("free", (free_tag_id,))):
                    points = np.concatenate([observation.corners_by_id[tag_id] for tag_id in ids]) if observation.has(*ids) else np.empty((0, 2))
                    quality[prefix + "_" + kind] = trackers[prefix + "_" + kind].update(image, points)
            pair_reasons = _pair_rejection_reasons(pair)
            free_ready = not pair_reasons and all(quality[name]["ready"] for name in ("primary_free", "side_free"))
            cv2.imshow(
                "Three-AprilTag stereo calibration",
                _render(
                    pair,
                    primary_observation,
                    side_observation,
                    len(primary_images),
                    args.required_poses,
                    ("FREE READY | " if free_ready else "FREE WAIT | ") + message,
                ),
            )
            key = cv2.waitKey(1) & 0xFF
            if ord("A") <= key <= ord("Z"):
                key += ord("a") - ord("A")
            if key == ord("q"):
                break
            if key == ord("r"):
                if pair_reasons:
                    message = "Reference rejected: " + ", ".join(pair_reasons)
                elif not primary_observation.has(*fixed_ids) or not side_observation.has(*fixed_ids):
                    message = "Reference rejected: both fixed diagonal tags must be visible in both views"
                elif not quality["primary_fixed"]["ready"] or not quality["side_fixed"]["ready"]:
                    message = "Reference rejected: hold fixed tags still and sharp, away from image borders"
                else:
                    ratio = _reference_depth_ratio(pair.primary, np.concatenate([primary_observation.corners_by_id[tag_id] for tag_id in fixed_ids]))
                    if ratio < args.min_reference_depth_ratio:
                        message = f"Reference rejected: valid depth {ratio:.1%} below {args.min_reference_depth_ratio:.1%}"
                        continue
                    if primary_images:
                        session = prepare_calibration_session(session)
                        primary_dir = session / "free-poses" / "primary"
                        side_dir = session / "free-poses" / "side"
                        depth_dir = session / "free-poses" / "depth"
                        for directory in (primary_dir, side_dir, depth_dir):
                            directory.mkdir(parents=True, exist_ok=True)
                        primary_images.clear()
                        side_images.clear()
                        signatures.clear()
                        print(f"New fixed reference; pose count reset. Session: {session.resolve()}")
                    reference_frame = pair.primary
                    reference_image = pair.primary.color_bgr.copy()
                    write_calibration_image(session / "fixed-reference-primary.png", pair.primary.color_bgr)
                    write_calibration_image(session / "fixed-reference-side.png", pair.side.color_bgr)
                    np.save(session / "fixed-reference-depth.npy", pair.primary.depth)
                    (session / "fixed-reference-metadata.json").write_text(
                        json.dumps(
                            {
                                "primary_frame_id": pair.primary.frame_id,
                                "side_frame_id": pair.side.frame_id,
                                "primary_timestamp_ns": pair.primary.timestamp_ns,
                                "side_timestamp_ns": pair.side.timestamp_ns,
                                "primary_host_timestamp_ns": pair.primary_host_timestamp_ns,
                                "side_host_timestamp_ns": pair.side_host_timestamp_ns,
                                "pair_delta_ms": pair.pair_delta_ms,
                                "primary_rgb_depth_sync_ms": pair.primary.sync_delta_ms,
                                "primary_intrinsics": pair.primary.intrinsics.to_dict(),
                                "fixed_tag_ids": list(fixed_ids),
                                "session_spec": spec,
                                "reference_depth_ratio": ratio,
                                "capture_quality": {name: quality[name] for name in ("primary_fixed", "side_fixed")},
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    message = "Fixed diagonal reference saved"
                continue
            if key == 32:
                if reference_frame is None:
                    message = "Pose rejected: press R to save the fixed reference first"
                    continue
                if pair_reasons:
                    message = "Pose rejected: " + ", ".join(pair_reasons)
                    continue
                if not primary_observation.has(free_tag_id) or not side_observation.has(free_tag_id):
                    message = f"Pose rejected: free tag {free_tag_id} must be visible in both views"
                    continue
                if not free_ready:
                    message = "Pose rejected: " + ", ".join(quality["primary_free"]["reasons"] + quality["side_free"]["reasons"])
                    continue
                signature = np.concatenate(
                    (
                        free_tag_pose_signature(primary_observation.corners_by_id[free_tag_id]),
                        free_tag_pose_signature(side_observation.corners_by_id[free_tag_id]),
                    )
                )
                if not pose_is_diverse(signatures, signature):
                    message = "Pose rejected: move/rotate/tilt the free tag more"
                    continue
                index = len(primary_images)
                write_calibration_image(primary_dir / f"pose-{index:03d}.png", pair.primary.color_bgr)
                write_calibration_image(side_dir / f"pose-{index:03d}.png", pair.side.color_bgr)
                np.save(depth_dir / f"pose-{index:03d}.npy", pair.primary.depth)
                (session / "free-poses" / f"pose-{index:03d}.json").write_text(
                    json.dumps(
                        {
                            "primary_frame_id": pair.primary.frame_id,
                            "side_frame_id": pair.side.frame_id,
                            "primary_timestamp_ns": pair.primary.timestamp_ns,
                            "side_timestamp_ns": pair.side.timestamp_ns,
                            "primary_host_timestamp_ns": pair.primary_host_timestamp_ns,
                            "side_host_timestamp_ns": pair.side_host_timestamp_ns,
                            "pair_delta_ms": pair.pair_delta_ms,
                            "primary_rgb_depth_sync_ms": pair.primary.sync_delta_ms,
                            "free_tag_id": free_tag_id,
                            "pose_signature": signature.tolist(),
                            "capture_quality": {name: quality[name] for name in ("primary_free", "side_free")},
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                primary_images.append(pair.primary.color_bgr.copy())
                side_images.append(pair.side.color_bgr.copy())
                signatures.append(signature)
                message = f"Saved diverse free-tag pose {len(primary_images)}"
                continue
            if key != ord("c"):
                continue
            if reference_frame is None or reference_image is None:
                message = "Calibration rejected: press R with both fixed tags visible first"
                continue
            if len(primary_images) < args.required_poses:
                message = f"Calibration rejected: need {args.required_poses} free-tag poses"
                continue
            try:
                calibration = _solve_and_save(
                    args,
                    config,
                    fixed_ids,
                    free_tag_id,
                    primary_images,
                    side_images,
                    reference_frame,
                    reference_image,
                    session,
                    camera_calibrations,
                )
            except (ValueError, OSError, cv2.error) as error:
                message = f"Calibration rejected: {error}"
                print(message)
                continue
            exit_code = 0 if calibration.valid else 2
            message = "VALID calibration published" if calibration.valid else "FAILED gates; output preserved; capture more poses"
            cv2.imshow(
                "Three-AprilTag stereo calibration",
                _render(pair, primary_observation, side_observation, len(primary_images), args.required_poses, message),
            )
            cv2.waitKey(1200)
            if not calibration.valid:
                continue
            break
    except (ValueError, RuntimeError, OSError, cv2.error) as error:
        print(f"Calibration failed: {error}")
        exit_code = 1
    finally:
        source.close()
        cv2.destroyAllWindows()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
