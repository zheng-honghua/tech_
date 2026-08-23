from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .calibration import PerspectiveCalibration
from .camera import FileRGBDSource, load_rgbd_frame, save_rgbd_frame
from .config import load_config
from .evaluation3d import run_rgbd_benchmark
from .extensions import QRCodeExtension
from .evaluation import run_synthetic_benchmark
from .pipeline import VisionPipeline
from .pipeline3d import VisionPipeline3D
from .rgbd import RGBDCalibration
from .server import VisionService3D, serve_json_tcp
from .synthetic import competition_demo_scene
from .synthetic3d import competition_rgbd_demo


def _read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return image


def _default_calibration(image: np.ndarray, config) -> PerspectiveCalibration:
    height, width = image.shape[:2]
    return PerspectiveCalibration(
        source_points=np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        ),
        output_width_px=config.tray.rectified_width_px,
        output_height_px=config.tray.rectified_height_px,
        tray_width_mm=config.tray.width_mm,
        tray_height_mm=config.tray.height_mm,
    )


def _write_results(
    output_dir: Path,
    pipeline: VisionPipeline,
    rectified: np.ndarray,
    results,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = []
    for result in results:
        crop_name = f"{result.object_id}-crop.png"
        if result.crop_image is not None:
            cv2.imwrite(str(output_dir / crop_name), result.crop_image)
        payload.append(result.to_dict(crop_path=crop_name))
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cv2.imwrite(str(output_dir / "annotated.png"), pipeline.annotate(rectified, results))


def _run_detect(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    frame = _read_image(args.image)
    background = _read_image(args.background) if args.background else None
    calibration = (
        PerspectiveCalibration.load(args.calibration)
        if args.calibration
        else _default_calibration(frame, config)
    )
    pipeline = VisionPipeline(
        config=config,
        calibration=calibration,
        background=background,
        extensions=[QRCodeExtension()] if args.qrcode else [],
    )
    pipeline.process(frame)
    results = pipeline.process(frame)
    rectified = pipeline.rectify_frame(frame)
    _write_results(Path(args.output_dir), pipeline, rectified, results)
    print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))
    return 0


def _run_demo(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    background, frame, _ = competition_demo_scene()
    calibration = _default_calibration(frame, config)
    pipeline = VisionPipeline(config=config, calibration=calibration, background=background)
    pipeline.process(frame, frame_id="demo")
    results = pipeline.process(frame, frame_id="demo")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "background.png"), background)
    cv2.imwrite(str(output_dir / "scene.png"), frame)
    _write_results(output_dir, pipeline, pipeline.rectify_frame(frame), results)
    selected = next((item for item in results if item.selected), None)
    print(f"detected={len(results)}, selected={selected.class_key if selected else 'none'}")
    print(f"artifacts={output_dir.resolve()}")
    return 0


def _parse_points(values: list[str]) -> np.ndarray:
    if len(values) != 4:
        raise ValueError("exactly four --point values are required")
    points = []
    for value in values:
        x_text, y_text = value.split(",", maxsplit=1)
        points.append([float(x_text), float(y_text)])
    return np.asarray(points, dtype=np.float32)


def _run_calibrate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    calibration = PerspectiveCalibration(
        source_points=_parse_points(args.point),
        output_width_px=config.tray.rectified_width_px,
        output_height_px=config.tray.rectified_height_px,
        tray_width_mm=config.tray.width_mm,
        tray_height_mm=config.tray.height_mm,
    )
    calibration.save(args.output)
    print(Path(args.output).resolve())
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    report = run_synthetic_benchmark(
        load_config(args.config), rounds=args.rounds, seed=args.seed
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _write_rgbd_results(output_dir: Path, frame, pipeline, results) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = []
    for result in results:
        crop_name = f"{result.object_id}-color.png"
        depth_name = f"{result.object_id}-depth.npy"
        if result.crop_image is not None:
            cv2.imwrite(str(output_dir / crop_name), result.crop_image)
        if result.depth_crop is not None:
            np.save(output_dir / depth_name, result.depth_crop)
        payload.append(result.to_dict(crop_name, depth_name))
    (output_dir / "results-v2.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "health.json").write_text(
        json.dumps(pipeline.health(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cv2.imwrite(str(output_dir / "annotated-rgbd.png"), pipeline.annotate(frame, results))
    depth = frame.depth_mm
    valid = depth > 0
    preview = np.zeros(depth.shape, np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [2, 98])
        preview[valid] = np.clip((high - depth[valid]) * 255 / max(high - low, 1.0), 0, 255)
    cv2.imwrite(str(output_dir / "depth-preview.png"), cv2.applyColorMap(preview, cv2.COLORMAP_TURBO))


def _make_rgbd_pipeline(args: argparse.Namespace):
    config = load_config(args.config)
    background = load_rgbd_frame(args.background_dir) if args.background_dir else None
    calibration = RGBDCalibration.load(args.rgbd_calibration) if args.rgbd_calibration else None
    pipeline = VisionPipeline3D(
        config=config,
        calibration=calibration,
        background_frame=background,
    )
    return config, pipeline


def _run_rgbd_detect(args: argparse.Namespace) -> int:
    _, pipeline = _make_rgbd_pipeline(args)
    frame = load_rgbd_frame(args.frame_dir)
    pipeline.process(frame)
    results = pipeline.process(frame)
    _write_rgbd_results(Path(args.output_dir), frame, pipeline, results)
    print(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2))
    return 0


def _run_rgbd_demo(args: argparse.Namespace) -> int:
    background, scene, _ = competition_rgbd_demo()
    pipeline = VisionPipeline3D(config=load_config(args.config), background_frame=background)
    pipeline.process(scene)
    results = pipeline.process(scene)
    output = Path(args.output_dir)
    save_rgbd_frame(background, output / "empty-tray-frame")
    save_rgbd_frame(scene, output / "scene-frame")
    pipeline.calibration.save(output / "rgbd-calibration.json")
    _write_rgbd_results(output, scene, pipeline, results)
    selected = next((item.class_key for item in results if item.selected), "none")
    print(f"detected={len(results)}, selected={selected}, health={pipeline.health()['reason']}")
    print(f"artifacts={output.resolve()}")
    return 0


def _run_rgbd_benchmark(args: argparse.Namespace) -> int:
    report = run_rgbd_benchmark(
        load_config(args.config), rounds=args.rounds, seed=args.seed
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _run_serve(args: argparse.Namespace) -> int:
    config, pipeline = _make_rgbd_pipeline(args)
    source = FileRGBDSource([args.frame_dir], loop=True)
    service = VisionService3D(pipeline, source)
    host = args.host or config.network.host
    port = args.port or config.network.port
    print(f"RGB-D JSON/TCP service listening on {host}:{port}")
    serve_json_tcp(service, host, port)
    return 0


def _run_rgbd_calibrate(args: argparse.Namespace) -> int:
    background = load_rgbd_frame(args.background_dir)
    pipeline = VisionPipeline3D(
        config=load_config(args.config), background_frame=background
    )
    calibration = pipeline.calibration
    if args.camera_to_robot:
        transform = np.asarray(
            json.loads(Path(args.camera_to_robot).read_text(encoding="utf-8")),
            dtype=np.float64,
        )
        calibration = RGBDCalibration(
            background.intrinsics, transform, calibration.tray_plane_camera
        )
    calibration.save(args.output)
    print(Path(args.output).resolve())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Intelligent sorting vision pipeline")
    parser.add_argument("--config", default=None, help="YAML configuration path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="detect objects in one tray image")
    detect.add_argument("--image", required=True)
    detect.add_argument("--background")
    detect.add_argument("--calibration")
    detect.add_argument("--output-dir", default="output/detect")
    detect.add_argument("--qrcode", action="store_true")
    detect.set_defaults(func=_run_detect)

    demo = subparsers.add_parser("demo", help="run the synthetic tray demonstration")
    demo.add_argument("--output-dir", default="output/demo")
    demo.set_defaults(func=_run_demo)

    benchmark = subparsers.add_parser(
        "benchmark", help="run repeatable 12-object synthetic scenes"
    )
    benchmark.add_argument("--rounds", type=int, default=30)
    benchmark.add_argument("--seed", type=int, default=7)
    benchmark.set_defaults(func=_run_benchmark)

    rgbd_demo = subparsers.add_parser("rgbd-demo", help="run the synthetic RGB-D demonstration")
    rgbd_demo.add_argument("--output-dir", default="output/rgbd-demo")
    rgbd_demo.set_defaults(func=_run_rgbd_demo)

    rgbd_benchmark = subparsers.add_parser(
        "rgbd-benchmark", help="run randomised 3-D solid benchmark scenes"
    )
    rgbd_benchmark.add_argument("--rounds", type=int, default=30)
    rgbd_benchmark.add_argument("--seed", type=int, default=11)
    rgbd_benchmark.set_defaults(func=_run_rgbd_benchmark)

    rgbd_detect = subparsers.add_parser("rgbd-detect", help="detect a recorded RGB-D frame")
    rgbd_detect.add_argument("--frame-dir", required=True)
    rgbd_detect.add_argument("--background-dir")
    rgbd_detect.add_argument("--rgbd-calibration")
    rgbd_detect.add_argument("--output-dir", default="output/rgbd-detect")
    rgbd_detect.set_defaults(func=_run_rgbd_detect)

    rgbd_calibrate = subparsers.add_parser(
        "rgbd-calibrate", help="fit the empty-tray plane and save RGB-D calibration"
    )
    rgbd_calibrate.add_argument("--background-dir", required=True)
    rgbd_calibrate.add_argument(
        "--camera-to-robot",
        help="JSON file containing a 4x4 camera-to-robot transform; identity by default",
    )
    rgbd_calibrate.add_argument("--output", default="rgbd-calibration.json")
    rgbd_calibrate.set_defaults(func=_run_rgbd_calibrate)

    serve = subparsers.add_parser("serve", help="serve RGB-D detection over newline JSON/TCP")
    serve.add_argument("--frame-dir", required=True)
    serve.add_argument("--background-dir")
    serve.add_argument("--rgbd-calibration")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(func=_run_serve)

    calibrate = subparsers.add_parser("calibrate", help="save four-point calibration")
    calibrate.add_argument(
        "--point",
        action="append",
        required=True,
        help="corner x,y in TL,TR,BR,BL order; repeat four times",
    )
    calibrate.add_argument("--output", default="calibration.json")
    calibrate.set_defaults(func=_run_calibrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
