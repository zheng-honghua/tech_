"""Live three-AprilTag stereo calibration for top RGB-D plus side RGB."""

from __future__ import annotations

import argparse
import json
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
from sorting_vision.rgbd import CameraIntrinsics, RGBDFrame
from sorting_vision.rgbd_dataset import depth_preview


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate stereo cameras using two fixed diagonal AprilTags and one moving tag"
    )
    parser.add_argument("--config", default="config/dual/temporary.yaml")
    parser.add_argument("--platform-id", choices=("temporary", "competition"), required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--generate-tags-dir",
        help="generate IDs and a placement preview, then exit without opening cameras",
    )
    parser.add_argument("--session-dir", default="data/apriltag-calibration")
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
    parser.add_argument("--fixed-tag-inset-mm", type=float)
    parser.add_argument("--tray-width-mm", type=float)
    parser.add_argument("--tray-height-mm", type=float)
    parser.add_argument("--dictionary", default="DICT_APRILTAG_36h11")
    parser.add_argument("--required-poses", type=int, default=20)
    parser.add_argument("--discard-frames", type=int, default=30)
    parser.add_argument(
        "--detection-width",
        type=int,
        default=960,
        help="downscale only AprilTag detection for live speed; corners are refined at full resolution",
    )
    return parser


def _resolved_tag_ids(args, config) -> tuple[tuple[int, int], int]:
    dual = config.dual_view
    fixed_ids = (
        dual.fixed_tag_a_id if args.fixed_tag_a is None else args.fixed_tag_a,
        dual.fixed_tag_b_id if args.fixed_tag_b is None else args.fixed_tag_b,
    )
    free_tag_id = dual.free_tag_id if args.free_tag is None else args.free_tag
    validate_apriltag_ids(fixed_ids, free_tag_id, args.dictionary)
    return fixed_ids, free_tag_id


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


def _load_saved_session(
    session: Path,
    fixed_ids: tuple[int, int],
    free_tag_id: int,
    required_poses: int,
) -> tuple[list[np.ndarray], list[np.ndarray], RGBDFrame, np.ndarray]:
    metadata_path = session / "fixed-reference-metadata.json"
    primary_path = session / "fixed-reference-primary.png"
    depth_path = session / "fixed-reference-depth.npy"
    for path in (metadata_path, primary_path, depth_path):
        if not path.is_file():
            raise ValueError(f"saved calibration session is missing {path.name}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if tuple(map(int, metadata.get("fixed_tag_ids", ()))) != fixed_ids:
        raise ValueError("saved fixed-tag IDs differ from the requested IDs")
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
    if len(primary_paths) != len(side_paths) or len(primary_paths) < required_poses:
        raise ValueError(
            f"saved session has {min(len(primary_paths), len(side_paths))}/"
            f"{required_poses} paired poses"
        )
    primary_images: list[np.ndarray] = []
    side_images: list[np.ndarray] = []
    for primary_file, side_file in zip(primary_paths, side_paths):
        pose_metadata_path = session / "free-poses" / f"{primary_file.stem}.json"
        if not pose_metadata_path.is_file():
            raise ValueError(f"saved session is missing {pose_metadata_path.name}")
        pose_metadata = json.loads(pose_metadata_path.read_text(encoding="utf-8"))
        if int(pose_metadata.get("free_tag_id", -1)) != free_tag_id:
            raise ValueError(f"saved {primary_file.stem} uses a different free-tag ID")
        primary_image = cv2.imread(str(primary_file), cv2.IMREAD_COLOR)
        side_image = cv2.imread(str(side_file), cv2.IMREAD_COLOR)
        if primary_image is None or side_image is None:
            raise ValueError(f"saved {primary_file.stem} image cannot be decoded")
        primary_images.append(primary_image)
        side_images.append(side_image)
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
):
    calibration = calibrate_apriltag_pairs(
        primary_images,
        side_images,
        reference_frame,
        reference_image,
        platform_id=args.platform_id,
        tag_size_mm=args.tag_size_mm,
        fixed_tag_ids=fixed_ids,
        free_tag_id=free_tag_id,
        tray_width_mm=args.tray_width_mm or config.tray.width_mm,
        tray_height_mm=args.tray_height_mm or config.tray.height_mm,
        fixed_tag_inset_mm=args.fixed_tag_inset_mm,
        dictionary_name=args.dictionary,
    )
    calibration.save(args.output)
    (session / "calibration-summary.json").write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2))
    return calibration


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.required_poses < 20:
        raise ValueError("required-poses must be at least 20")
    if args.detection_width < 320:
        raise ValueError("detection-width must be at least 320 pixels")
    config = load_config(args.config)
    fixed_ids, free_tag_id = _resolved_tag_ids(args, config)
    if args.generate_tags_dir:
        paths = generate_three_tag_assets(
            args.generate_tags_dir,
            fixed_tag_ids=fixed_ids,
            free_tag_id=free_tag_id,
            tag_size_mm=args.tag_size_mm,
            dictionary_name=args.dictionary,
        )
        print(json.dumps({"generated": [str(path.resolve()) for path in paths]}, ensure_ascii=False, indent=2))
        return 0
    if not args.output:
        raise ValueError("--output is required unless --generate-tags-dir is used")
    session = Path(args.session_dir) / args.platform_id
    if args.replay_session:
        try:
            saved = _load_saved_session(
                session, fixed_ids, free_tag_id, args.required_poses
            )
            calibration = _solve_and_save(
                args, config, fixed_ids, free_tag_id, *saved, session
            )
        except (ValueError, OSError, KeyError, cv2.error) as error:
            print(f"Calibration rejected: {error}")
            return 1
        return 0 if calibration.valid else 2
    source = _camera_source(args, config)
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
            cv2.imshow(
                "Three-AprilTag stereo calibration",
                _render(
                    pair,
                    primary_observation,
                    side_observation,
                    len(primary_images),
                    args.required_poses,
                    message,
                ),
            )
            key = cv2.waitKey(1) & 0xFF
            if ord("A") <= key <= ord("Z"):
                key += ord("a") - ord("A")
            if key == ord("q"):
                break
            if key == ord("r"):
                if pair.side is None or not pair.synchronized:
                    message = "Reference rejected: cameras missing or unsynchronised"
                elif not primary_observation.has(*fixed_ids) or not side_observation.has(*fixed_ids):
                    message = "Reference rejected: both fixed diagonal tags must be visible in both views"
                else:
                    reference_frame = pair.primary
                    reference_image = pair.primary.color_bgr.copy()
                    cv2.imwrite(str(session / "fixed-reference-primary.png"), pair.primary.color_bgr)
                    cv2.imwrite(str(session / "fixed-reference-side.png"), pair.side.color_bgr)
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
                                "primary_intrinsics": pair.primary.intrinsics.to_dict(),
                                "fixed_tag_ids": list(fixed_ids),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    message = "Fixed diagonal reference saved"
                continue
            if key == 32:
                if pair.side is None or not pair.synchronized:
                    message = "Pose rejected: cameras missing or unsynchronised"
                    continue
                if not primary_observation.has(free_tag_id) or not side_observation.has(free_tag_id):
                    message = f"Pose rejected: free tag {free_tag_id} must be visible in both views"
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
                primary_images.append(pair.primary.color_bgr.copy())
                side_images.append(pair.side.color_bgr.copy())
                signatures.append(signature)
                cv2.imwrite(str(primary_dir / f"pose-{index:03d}.png"), pair.primary.color_bgr)
                cv2.imwrite(str(side_dir / f"pose-{index:03d}.png"), pair.side.color_bgr)
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
                            "free_tag_id": free_tag_id,
                            "pose_signature": signature.tolist(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
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
                )
            except (ValueError, cv2.error) as error:
                message = f"Calibration rejected: {error}"
                print(message)
                continue
            exit_code = 0 if calibration.valid else 2
            message = "VALID calibration saved" if calibration.valid else "INVALID metrics saved for diagnosis"
            cv2.imshow(
                "Three-AprilTag stereo calibration",
                _render(pair, primary_observation, side_observation, len(primary_images), args.required_poses, message),
            )
            cv2.waitKey(1200)
            break
    finally:
        source.close()
        cv2.destroyAllWindows()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
