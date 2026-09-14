"""Calibrate one camera from its RGB stream using a rigid checkerboard."""

from __future__ import annotations


import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from sorting_vision.apriltag_calibration import pose_is_diverse
from sorting_vision.camera import OpenCVCameraSource, RGBFrame, RealSenseColorSource
from sorting_vision.intrinsic_calibration import (
    calibrate_camera_intrinsics,
    calibration_view_signature,
    checkerboard_points_from_image,
    generate_checkerboard_asset,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate one RGB camera from live video; no depth stream is used"
    )
    parser.add_argument("--platform-id", choices=("temporary", "competition"), default="temporary")
    parser.add_argument("--source", choices=("uvc", "realsense", "video"), default="uvc")
    parser.add_argument("--camera-id", default="side")
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--realsense-serial")
    parser.add_argument("--video-file")
    parser.add_argument("--video-sample-interval-ms", type=_positive_int, default=500)
    parser.add_argument("--video-max-adjacent-difference", type=float, default=4.0)
    parser.add_argument("--width", type=_positive_int, default=1280)
    parser.add_argument("--height", type=_positive_int, default=720)
    parser.add_argument("--fps", type=_positive_int, default=30)
    parser.add_argument("--backend", default="DSHOW")
    parser.add_argument("--fourcc", default="NV12")
    parser.add_argument("--corners-x", type=_positive_int, default=10)
    parser.add_argument("--corners-y", type=_positive_int, default=7)
    parser.add_argument("--square-size-mm", type=float, default=20.0)
    parser.add_argument("--required-frames", type=_positive_int, default=25)
    parser.add_argument("--detection-width", type=_positive_int, default=960)
    parser.add_argument("--discard-frames", type=int, default=20)
    parser.add_argument("--auto-capture", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--auto-interval-ms", type=_positive_int, default=350)
    parser.add_argument("--session-dir")
    parser.add_argument("--output")
    parser.add_argument("--generate-board-dir")
    return parser


def _resolve_storage_paths(args):
    calibration_root = Path("data") / "calibration" / args.platform_id / "intrinsics"
    if not args.session_dir:
        args.session_dir = str(calibration_root / "sessions")
    if args.source == "video" and not args.video_file:
        args.video_file = str(calibration_root / "videos" / f"{args.camera_id}.mp4")
    if not args.output and not args.generate_board_dir:
        args.output = str(
            Path("config") / "dual" / args.platform_id / f"{args.camera_id}-intrinsics.json"
        )
    return args


class _VideoSource:
    def __init__(self, path: str, sample_interval_ms: int = 500) -> None:
        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            raise RuntimeError(f"cannot open video file {path}")
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self._stride = max(2, int(round(max(fps, 1.0) * sample_interval_ms / 1000.0)))
        self._source_index = 0
        self._first = True
        self.last_adjacent_difference = float("inf")

    def read(self) -> RGBFrame:
        if not self._first:
            for _ in range(max(0, self._stride - 2)):
                if not self._capture.grab():
                    raise EOFError("video ended")
                self._source_index += 1
        self._first = False
        ok, image = self._capture.read()
        if not ok or image is None:
            raise EOFError("video ended")
        self._source_index += 1
        comparison_ok, comparison = self._capture.read()
        if comparison_ok and comparison is not None:
            self._source_index += 1
            first_gray = cv2.resize(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (320, 180)
            )
            second_gray = cv2.resize(
                cv2.cvtColor(comparison, cv2.COLOR_BGR2GRAY), (320, 180)
            )
            self.last_adjacent_difference = float(
                np.mean(cv2.absdiff(first_gray, second_gray))
            )
        else:
            self.last_adjacent_difference = float("inf")
        timestamp_ns = time.monotonic_ns()
        return RGBFrame(image, timestamp_ns, f"video-{self._source_index:09d}")

    def close(self) -> None:
        self._capture.release()


def _camera_source(args):
    if args.source == "video":
        if not args.video_file:
            raise ValueError("--video-file is required for --source video")
        return _VideoSource(args.video_file, args.video_sample_interval_ms)
    if args.source == "realsense":
        return RealSenseColorSource(
            width=args.width,
            height=args.height,
            fps=args.fps,
            serial=args.realsense_serial,
            frame_prefix=args.camera_id,
        )
    return OpenCVCameraSource(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        backend=args.backend,
        fourcc=args.fourcc,
        warmup_frames=max(0, args.discard_frames),
    )


def _target_metadata(args) -> dict[str, object]:
    return {
        "type": "checkerboard",
        "inner_corners_x": args.corners_x,
        "inner_corners_y": args.corners_y,
        "square_size_mm": args.square_size_mm,
    }


def _solve_and_save(args, session: Path, object_sets, image_sets, image_size):
    calibration = calibrate_camera_intrinsics(
        object_sets,
        image_sets,
        image_size,
        camera_id=args.camera_id,
        target=_target_metadata(args),
    )
    calibration.save(args.output)
    (session / "calibration-summary.json").write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2))
    return calibration


def _render(
    image: np.ndarray,
    image_points: np.ndarray,
    detected_count: int,
    captured: int,
    required: int,
    message: str,
) -> np.ndarray:
    canvas = image.copy()
    for point in np.asarray(image_points).reshape(-1, 2):
        cv2.circle(canvas, tuple(np.rint(point).astype(int)), 3, (0, 255, 0), -1)
    scale = min(1.0, 1280.0 / canvas.shape[1])
    if scale < 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale)
    panel = np.full((112, canvas.shape[1], 3), 25, np.uint8)
    output = np.vstack((canvas, panel))
    y = canvas.shape[0]
    lines = (
        f"corners {detected_count} | frames {captured}/{required}",
        "SPACE=capture | C=calibrate | Q=quit",
        message[:150],
    )
    for index, text in enumerate(lines):
        cv2.putText(
            output,
            text,
            (14, y + 28 + index * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return output


def main(argv: list[str] | None = None) -> int:
    args = _resolve_storage_paths(build_parser().parse_args(argv))
    if args.generate_board_dir:
        paths = generate_checkerboard_asset(
            args.generate_board_dir,
            corners_x=args.corners_x,
            corners_y=args.corners_y,
            square_size_mm=args.square_size_mm,
        )
        print(json.dumps({"generated": [str(path.resolve()) for path in paths]}, ensure_ascii=False))
        return 0
    if args.required_frames < 20:
        raise ValueError("--required-frames must be at least 20")
    if args.corners_x < 3 or args.corners_y < 3:
        raise ValueError("checkerboard needs at least 3x3 inner corners")
    if args.square_size_mm <= 0:
        raise ValueError("--square-size-mm must be positive")
    if args.video_max_adjacent_difference <= 0:
        raise ValueError("--video-max-adjacent-difference must be positive")
    if args.headless and not args.auto_capture:
        raise ValueError("--headless requires --auto-capture")

    source = _camera_source(args)
    session = Path(args.session_dir) / args.camera_id
    image_dir = session / "frames"
    image_dir.mkdir(parents=True, exist_ok=True)
    object_sets: list[np.ndarray] = []
    image_sets: list[np.ndarray] = []
    signatures: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    message = "Move the rigid checkerboard across the full image and tilt it in both axes"
    last_auto_capture_ns = 0
    exit_code = 1
    try:
        while True:
            try:
                frame = source.read()
            except EOFError:
                if len(image_sets) >= args.required_frames and image_size is not None:
                    calibration = _solve_and_save(
                        args, session, object_sets, image_sets, image_size
                    )
                    exit_code = 0 if calibration.valid else 2
                else:
                    print(
                        f"Video ended with {len(image_sets)}/{args.required_frames} "
                        "usable frames"
                    )
                break
            image = frame.color_bgr
            current_size = (image.shape[1], image.shape[0])
            if image_size is None:
                image_size = current_size
            elif current_size != image_size:
                raise ValueError("video resolution changed during intrinsic calibration")
            object_points, image_points = checkerboard_points_from_image(
                image,
                corners_x=args.corners_x,
                corners_y=args.corners_y,
                square_size_mm=args.square_size_mm,
                maximum_detection_width=args.detection_width,
            )
            expected_corners = args.corners_x * args.corners_y
            detected_count = len(image_points)
            adjacent_difference = getattr(source, "last_adjacent_difference", None)
            video_is_stable = (
                adjacent_difference is None
                or adjacent_difference <= args.video_max_adjacent_difference
            )
            can_capture = detected_count == expected_corners and video_is_stable
            signature = (
                calibration_view_signature(image_points) if can_capture else None
            )
            now_ns = time.monotonic_ns()
            auto_due = (
                args.auto_capture
                and now_ns - last_auto_capture_ns >= args.auto_interval_ms * 1_000_000
            )
            if args.headless:
                key = -1
            else:
                cv2.imshow(
                    "RGB intrinsic calibration (no depth)",
                    _render(
                        image,
                        image_points,
                        detected_count,
                        len(image_sets),
                        args.required_frames,
                        message,
                    ),
                )
                key = cv2.waitKey(1) & 0xFF
            if ord("A") <= key <= ord("Z"):
                key += ord("a") - ord("A")
            if key == ord("q"):
                break
            capture_requested = key == 32 or auto_due
            if capture_requested:
                if not can_capture or signature is None:
                    if not video_is_stable:
                        message = (
                            "Rejected: checkerboard is moving "
                            f"(adjacent difference {adjacent_difference:.2f})"
                        )
                    else:
                        message = (
                            "Rejected: the complete "
                            f"{args.corners_x}x{args.corners_y} inner-corner checkerboard "
                            "must be visible"
                        )
                elif not pose_is_diverse(signatures, signature):
                    message = "Rejected: move, resize, rotate, or tilt the checkerboard more"
                else:
                    index = len(image_sets)
                    object_sets.append(object_points.copy())
                    image_sets.append(image_points.copy())
                    signatures.append(signature)
                    cv2.imwrite(str(image_dir / f"frame-{index:03d}.png"), image)
                    (session / f"frame-{index:03d}.json").write_text(
                        json.dumps(
                            {
                                "frame_id": frame.frame_id,
                                "timestamp_ns": frame.timestamp_ns,
                                "corner_count": detected_count,
                                "signature": signature.tolist(),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    message = f"Saved frame {len(image_sets)}"
                    last_auto_capture_ns = now_ns
            if key != ord("c"):
                continue
            if len(image_sets) < args.required_frames:
                message = f"Rejected: need {args.required_frames} frames"
                continue
            assert image_size is not None
            calibration = _solve_and_save(
                args, session, object_sets, image_sets, image_size
            )
            message = "Calibration valid" if calibration.valid else "Calibration saved but quality gates failed"
            if not args.headless:
                cv2.imshow(
                    "RGB intrinsic calibration (no depth)",
                    _render(
                        image,
                        image_points,
                        detected_count,
                        len(image_sets),
                        args.required_frames,
                        message,
                    ),
                )
                cv2.waitKey(1200)
            exit_code = 0 if calibration.valid else 2
            break
    finally:
        source.close()
        cv2.destroyAllWindows()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
