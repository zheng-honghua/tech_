"""Interactive paired capture: top RealSense RGB-D plus side UVC RGB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from sorting_vision.camera import (
    DualCameraSource,
    OpenCVCameraSource,
    ProcessOpenCVCameraSource,
    RealSenseSource,
    ThreadedRealSenseSource,
)
from sorting_vision.capture_assistant import (
    CAPTURE_LABELS,
    CaptureAssistantState,
    capture_label_index,
)
from sorting_vision.config import load_config
from sorting_vision.dual_capture_app import (
    DualCaptureQualityTracker,
    load_dual_batch_counts,
    render_dual_capture_assistant,
    save_dual_capture_sample,
)
from sorting_vision.dual_view import DualViewCalibration


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture synchronized top RGB/depth and side RGB samples"
    )
    parser.add_argument("--config", default="config/dual/temporary.yaml")
    parser.add_argument("--dataset-root", default="data/dual")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--platform-id", choices=("temporary", "competition"), required=True)
    parser.add_argument("--side-camera-index", type=int)
    parser.add_argument("--color-width", type=_positive_int, help="top RealSense RGB width")
    parser.add_argument("--color-height", type=_positive_int, help="top RealSense RGB height")
    parser.add_argument("--depth-width", type=_positive_int, help="top RealSense depth width")
    parser.add_argument("--depth-height", type=_positive_int, help="top RealSense depth height")
    parser.add_argument("--fps", type=_positive_int, help="top RealSense RGB/depth FPS")
    parser.add_argument("--side-width", type=_positive_int, help="side RGB width")
    parser.add_argument("--side-height", type=_positive_int, help="side RGB height")
    parser.add_argument("--side-fps", type=_positive_int, help="side RGB FPS")
    parser.add_argument("--start-label", default="empty_tray")
    parser.add_argument("--target-per-label", type=int, default=10)
    parser.add_argument("--dual-calibration")
    parser.add_argument("--discard-frames", type=int, default=30)
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--motion-threshold", type=float, default=2.5)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=0.85)
    parser.add_argument("--max-rgbd-sync-ms", type=float, default=20.0)
    parser.add_argument("--max-pair-delta-ms", type=float, default=50.0)
    parser.add_argument("--min-side-blur", type=float, default=35.0)
    parser.add_argument("--auto-advance", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    camera = config.camera
    dual = config.dual_view
    primary_type = (
        ThreadedRealSenseSource if dual.side_process_isolation else RealSenseSource
    )
    primary = primary_type(
        depth_width=args.depth_width or camera.realsense_depth_width,
        depth_height=args.depth_height or camera.realsense_depth_height,
        color_width=args.color_width or camera.realsense_color_width,
        color_height=args.color_height or camera.realsense_color_height,
        fps=args.fps or camera.realsense_fps,
        camera_model=camera.realsense_model,
        frame_prefix=camera.realsense_frame_prefix,
    )
    try:
        if not dual.side_process_isolation:
            primary.read()
        side_kwargs = dict(
            camera_index=(dual.side_camera_index if args.side_camera_index is None else args.side_camera_index),
            width=args.side_width or dual.side_width,
            height=args.side_height or dual.side_height,
            fps=args.side_fps or dual.side_fps,
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
    source = DualCameraSource(
        primary,
        side,
        max_pair_delta_ms=args.max_pair_delta_ms,
        pair_timeout_ms=dual.pair_timeout_ms,
        acquisition_mode=dual.acquisition_mode,
    )
    state = CaptureAssistantState(
        target_per_label=args.target_per_label,
        selected_index=capture_label_index(args.start_label),
        counts=load_dual_batch_counts(
            args.dataset_root, args.batch_id, args.platform_id
        ),
    )
    quality = DualCaptureQualityTracker(
        stable_frames_required=args.stable_frames,
        motion_threshold=args.motion_threshold,
        minimum_valid_depth_ratio=args.min_valid_depth_ratio,
        maximum_rgbd_sync_ms=args.max_rgbd_sync_ms,
        maximum_pair_delta_ms=args.max_pair_delta_ms,
        minimum_side_blur_variance=args.min_side_blur,
    )
    calibration_path = args.dual_calibration or dual.calibration_path
    calibration_hash = "UNVALIDATED"
    if calibration_path and Path(calibration_path).is_file():
        calibration_hash = DualViewCalibration.load(calibration_path).calibration_hash
    message = "Wait for READY, then press SPACE"
    saved = 0
    print("0=empty tray, 1-9=shape, SPACE=save, F=force, N/P=label, A=next, Q=quit")
    print("labels=" + ", ".join(f"{i}:{name}" for i, (_, name) in enumerate(CAPTURE_LABELS)))
    try:
        for _ in range(max(0, args.discard_frames)):
            source.read()
        while True:
            pair = source.read()
            quality.update(pair)
            cv2.imshow(
                "Dual capture: top RGB + depth + side RGB",
                render_dual_capture_assistant(pair, state, quality, message),
            )
            key = cv2.waitKey(1) & 0xFF
            if ord("A") <= key <= ord("Z"):
                key += ord("a") - ord("A")
            if key == ord("q"):
                break
            if ord("0") <= key <= ord("9"):
                state.select_digit(key - ord("0"))
                message = f"Selected {state.current[0]}"
                continue
            if key == ord("n"):
                state.select_next()
                message = f"Selected {state.current[0]}"
                continue
            if key == ord("p"):
                state.select_next(-1)
                message = f"Selected {state.current[0]}"
                continue
            if key == ord("a"):
                state.select_next_incomplete()
                message = f"Next incomplete: {state.current[0]}"
                continue
            if key not in {32, ord("f")}:
                continue
            forced = key == ord("f")
            if pair.side is None:
                message = "Rejected: side camera missing"
                continue
            if not quality.ready(pair) and not forced:
                message = "Rejected: " + ", ".join(quality.rejection_reasons(pair))
                continue
            label_id, label_name = state.current
            if (
                label_id != "empty_tray"
                and state.count("empty_tray") == 0
                and not forced
            ):
                message = "Rejected: capture empty_tray first (F records an override)"
                continue
            target = save_dual_capture_sample(
                pair,
                args.dataset_root,
                args.batch_id,
                args.platform_id,
                label_id,
                calibration_hash,
                quality,
                forced,
            )
            state.record_saved()
            saved += 1
            message = f"Saved {label_id} #{state.count()}"
            print(f"saved[{saved}] {label_name}={target.resolve()}")
            if args.auto_advance and state.count() >= state.target_per_label:
                state.select_next_incomplete()
                message += f" | next {state.current[0]}"
    finally:
        source.close()
        cv2.destroyAllWindows()
    print(json.dumps({"saved": saved, "counts": state.counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
