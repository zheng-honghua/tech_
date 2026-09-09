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
)
from sorting_vision.camera import DualCameraSource, OpenCVCameraSource, RealSenseSource
from sorting_vision.config import load_config
from sorting_vision.rgbd_dataset import depth_preview


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
    parser.add_argument("--side-camera-index", type=int)
    parser.add_argument("--tag-size-mm", type=float, required=True)
    parser.add_argument("--fixed-tag-a", type=int, default=0)
    parser.add_argument("--fixed-tag-b", type=int, default=1)
    parser.add_argument("--free-tag", type=int, default=2)
    parser.add_argument("--fixed-tag-inset-mm", type=float)
    parser.add_argument("--tray-width-mm", type=float)
    parser.add_argument("--tray-height-mm", type=float)
    parser.add_argument("--dictionary", default="DICT_APRILTAG_36h11")
    parser.add_argument("--required-poses", type=int, default=20)
    parser.add_argument("--discard-frames", type=int, default=30)
    return parser


def _camera_source(args, config):
    camera = config.camera
    dual = config.dual_view
    primary = RealSenseSource(
        depth_width=camera.realsense_depth_width,
        depth_height=camera.realsense_depth_height,
        color_width=camera.realsense_color_width,
        color_height=camera.realsense_color_height,
        fps=camera.realsense_fps,
        camera_model=camera.realsense_model,
        frame_prefix=camera.realsense_frame_prefix,
    )
    try:
        side = OpenCVCameraSource(
            camera_index=dual.side_camera_index if args.side_camera_index is None else args.side_camera_index,
            width=dual.side_width,
            height=dual.side_height,
            fps=dual.side_fps,
            warmup_frames=camera.warmup_frames,
            reconnect_attempts=camera.reconnect_attempts,
            auto_exposure=dual.side_auto_exposure,
            exposure=dual.side_exposure,
            auto_white_balance=dual.side_auto_white_balance,
            white_balance=dual.side_white_balance,
            autofocus=dual.side_autofocus,
            focus=dual.side_focus,
        )
    except Exception:
        primary.close()
        raise
    return DualCameraSource(
        primary,
        side,
        max_pair_delta_ms=dual.max_pair_delta_ms,
        pair_timeout_ms=dual.pair_timeout_ms,
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.required_poses < 20:
        raise ValueError("required-poses must be at least 20")
    fixed_ids = (args.fixed_tag_a, args.fixed_tag_b)
    if args.free_tag in fixed_ids or fixed_ids[0] == fixed_ids[1]:
        raise ValueError("fixed and free AprilTag IDs must be distinct")
    if args.generate_tags_dir:
        paths = generate_three_tag_assets(
            args.generate_tags_dir,
            fixed_tag_ids=fixed_ids,
            free_tag_id=args.free_tag,
            tag_size_mm=args.tag_size_mm,
            dictionary_name=args.dictionary,
        )
        print(json.dumps({"generated": [str(path.resolve()) for path in paths]}, ensure_ascii=False, indent=2))
        return 0
    if not args.output:
        raise ValueError("--output is required unless --generate-tags-dir is used")
    config = load_config(args.config)
    source = _camera_source(args, config)
    session = Path(args.session_dir) / args.platform_id
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
        f"move and tilt tag {args.free_tag}"
    )
    exit_code = 1
    try:
        for _ in range(max(0, args.discard_frames)):
            source.read()
        while True:
            pair = source.read()
            primary_observation = detect_apriltags(pair.primary.color_bgr, args.dictionary)
            side_observation = (
                detect_apriltags(pair.side.color_bgr, args.dictionary)
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
                if not primary_observation.has(args.free_tag) or not side_observation.has(args.free_tag):
                    message = f"Pose rejected: free tag {args.free_tag} must be visible in both views"
                    continue
                signature = np.concatenate(
                    (
                        free_tag_pose_signature(primary_observation.corners_by_id[args.free_tag]),
                        free_tag_pose_signature(side_observation.corners_by_id[args.free_tag]),
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
                            "free_tag_id": args.free_tag,
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
            calibration = calibrate_apriltag_pairs(
                primary_images,
                side_images,
                reference_frame,
                reference_image,
                platform_id=args.platform_id,
                tag_size_mm=args.tag_size_mm,
                fixed_tag_ids=fixed_ids,
                free_tag_id=args.free_tag,
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
