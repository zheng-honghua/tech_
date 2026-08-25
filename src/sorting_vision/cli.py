from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .calibration import PerspectiveCalibration
from .camera import (
    FileRGBDSource,
    OpenCVCameraSource,
    RGBFrame,
    RealSenseD415Source,
    load_rgbd_frame,
    save_rgbd_frame,
)
from .config import load_config
from .evaluation3d import run_rgbd_benchmark
from .extensions import QRCodeExtension
from .geometry_rgb import (
    audit_geometry_dataset,
    compare_geometry_models,
    evaluate_geometry_model,
    export_geometry_results,
    train_geometry_model,
)
from .geometry_cnn import (
    benchmark_geometry_backend,
    evaluate_geometry_backend,
    export_geometry_backend_results,
    export_geometry_cnn,
    load_geometry_shape_model,
    train_geometry_cnn,
)
from .geometry_edge_audit import audit_geometry_edges
from .evaluation import run_synthetic_benchmark
from .pipeline import VisionPipeline
from .pipeline3d import VisionPipeline3D
from .rgb_development import RGBDevelopmentPipeline
from .rgbd import RGBDCalibration
from .interlock import MotionInterlock
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
    return PerspectiveCalibration.identity(
        width,
        height,
        config.tray.width_mm,
        config.tray.height_mm,
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
        shape_model=_make_shape_model(args),
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
    config = load_config(args.config)
    if args.source == "file":
        if not args.frame_dir:
            raise ValueError("--frame-dir is required when --source=file")
        _, pipeline = _make_rgbd_pipeline(args)
        source = FileRGBDSource([args.frame_dir], loop=True)
        service = VisionService3D(pipeline, source, _make_interlock(config), "RGBD")
    else:
        service, source = _make_live_service(args, config)
    host = args.host or config.network.host
    port = args.port or config.network.port
    print(f"{service.mode} JSON/TCP service listening on {host}:{port}")
    try:
        serve_json_tcp(service, host, port)
    finally:
        source.close()
    return 0


def _make_interlock(config) -> MotionInterlock:
    return MotionInterlock(config.motion_interlock)


def _make_shape_model(args: argparse.Namespace):
    model_path = getattr(args, "shape_model", None)
    if not model_path:
        return None
    return load_geometry_shape_model(
        getattr(args, "shape_backend", "opencv"),
        model_path,
        getattr(args, "shape_device", "CPU"),
    )


def _make_camera_source(args: argparse.Namespace, config):
    camera = config.camera
    if args.source == "uvc":
        return OpenCVCameraSource(
            camera_index=args.camera_index,
            width=args.width or camera.width,
            height=args.height or camera.height,
            fps=args.fps or camera.fps,
            warmup_frames=camera.warmup_frames,
            reconnect_attempts=camera.reconnect_attempts,
        )
    if args.source == "realsense":
        return RealSenseD415Source(
            width=args.width or 640,
            height=args.height or 480,
            fps=args.fps or 30,
        )
    raise ValueError(f"unsupported live source: {args.source}")


def _make_live_service(args: argparse.Namespace, config):
    source = _make_camera_source(args, config)
    try:
        first = source.read()
        if args.source == "uvc":
            background = _read_image(args.background) if args.background else None
            calibration = (
                PerspectiveCalibration.load(args.calibration)
                if args.calibration
                else _default_calibration(first.color_bgr, config)
            )
            pipeline = RGBDevelopmentPipeline(
                VisionPipeline(
                    config=config,
                    calibration=calibration,
                    background=background,
                    shape_model=_make_shape_model(args),
                )
            )
            return VisionService3D(
                pipeline, source, _make_interlock(config), "RGB_ONLY"
            ), source

        background_frame = (
            load_rgbd_frame(args.background_dir) if args.background_dir else None
        )
        calibration = (
            RGBDCalibration.load(args.rgbd_calibration)
            if args.rgbd_calibration
            else None
        )
        if calibration is None and background_frame is None:
            raise ValueError(
                "D415 mode requires --background-dir or --rgbd-calibration; "
                "capture an empty tray first"
            )
        pipeline = VisionPipeline3D(
            config=config,
            calibration=calibration,
            background_frame=background_frame,
        )
        return VisionService3D(
            pipeline, source, _make_interlock(config), "RGBD"
        ), source
    except Exception:
        source.close()
        raise


def _run_camera_live(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    service, source = _make_live_service(args, config)
    count = 0
    print("keys: s=motion_start, r=motion_stop, q=quit")
    try:
        while args.max_frames <= 0 or count < args.max_frames:
            service.update()
            count += 1
            if not args.headless:
                preview = service.preview_image()
                if preview is not None:
                    cv2.imshow("sorting-vision", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    service.motion_start()
                elif key == ord("r"):
                    service.motion_stop()
    finally:
        source.close()
        if not args.headless:
            cv2.destroyAllWindows()
    print(json.dumps(service.health(), ensure_ascii=False, indent=2))
    return 0


def _run_camera_record(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    source = _make_camera_source(args, config)
    session = Path(args.session)
    color_dir = session / "color"
    depth_dir = session / "depth"
    color_dir.mkdir(parents=True, exist_ok=True)
    if args.source == "realsense":
        depth_dir.mkdir(parents=True, exist_ok=True)
    manifest = session / "manifest.jsonl"
    count = 0
    try:
        with manifest.open("a", encoding="utf-8") as stream:
            while args.max_frames <= 0 or count < args.max_frames:
                frame = source.read()
                count += 1
                color_name = f"{frame.frame_id}.png"
                if not cv2.imwrite(str(color_dir / color_name), frame.color_bgr):
                    raise OSError("failed to save camera frame")
                item = {
                    "frame_id": frame.frame_id,
                    "timestamp_ns": frame.timestamp_ns,
                    "source": args.source,
                    "label": args.label,
                    "color": str(Path("color") / color_name),
                }
                if not isinstance(frame, RGBFrame):
                    depth_name = f"{frame.frame_id}.npy"
                    np.save(depth_dir / depth_name, frame.depth)
                    item["depth"] = str(Path("depth") / depth_name)
                    item["intrinsics"] = frame.intrinsics.to_dict()
                    item["color_timestamp_ns"] = frame.color_timestamp_ns
                    item["depth_timestamp_ns"] = frame.depth_timestamp_ns
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                stream.flush()
                if not args.headless:
                    cv2.imshow("sorting-vision record", frame.color_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        source.close()
        if not args.headless:
            cv2.destroyAllWindows()
    print(f"recorded={count}, session={session.resolve()}")
    return 0


def _write_optional_report(report: dict, output: str | None) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(text)


def _run_geometry_audit(args: argparse.Namespace) -> int:
    _write_optional_report(audit_geometry_dataset(args.data_root), args.output_report)
    return 0


def _run_geometry_train(args: argparse.Namespace) -> int:
    model, report = train_geometry_model(args.data_root, args.feature_set)
    model.save(args.output)
    report["model_path"] = str(Path(args.output).resolve())
    _write_optional_report(report, args.output_report)
    return 0


def _run_geometry_edge_audit(args: argparse.Namespace) -> int:
    report = audit_geometry_edges(args.data_root, args.output_dir)
    _write_optional_report(report, args.output_report)
    return 0


def _run_geometry_edge_compare(args: argparse.Namespace) -> int:
    report = compare_geometry_models(
        args.data_root, args.legacy_model, args.edge_model
    )
    _write_optional_report(report, args.output_report)
    return 0


def _run_geometry_evaluate(args: argparse.Namespace) -> int:
    report = (
        evaluate_geometry_model(args.data_root, args.model)
        if args.backend == "opencv"
        else evaluate_geometry_backend(
            args.data_root, args.backend, args.model, args.device
        )
    )
    _write_optional_report(report, args.output_report)
    return 0


def _run_geometry_cnn_train(args: argparse.Namespace) -> int:
    report = train_geometry_cnn(
        args.data_root,
        args.output,
        epochs=args.epochs,
        seed=args.seed,
        pretrained=not args.no_pretrained,
        cross_validation=not args.skip_cross_validation,
    )
    _write_optional_report(report, args.output_report)
    return 0


def _run_geometry_cnn_export(args: argparse.Namespace) -> int:
    report = export_geometry_cnn(
        args.checkpoint,
        args.output_dir,
        precision=args.precision,
        data_root=args.data_root,
    )
    _write_optional_report(report, args.output_report)
    return 0


def _run_geometry_benchmark(args: argparse.Namespace) -> int:
    report = benchmark_geometry_backend(
        args.data_root,
        args.backend,
        args.model,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device,
    )
    _write_optional_report(report, args.output_report)
    return 0


def _run_geometry_export(args: argparse.Namespace) -> int:
    report = (
        export_geometry_results(args.data_root, args.model, args.output_dir)
        if args.backend == "opencv"
        else export_geometry_backend_results(
            args.data_root,
            args.backend,
            args.model,
            args.output_dir,
            args.device,
        )
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "exported_images": report["exported_images"],
                "backend": args.backend,
                "accuracy": report.get(
                    "accuracy", report.get("training_replay_accuracy")
                ),
                "leave_one_out_accuracy": report.get("leave_one_out_accuracy"),
                "same_batch_only": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
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
    detect.add_argument("--shape-model", help="trained geometry RGB model (.npz)")
    detect.add_argument(
        "--shape-backend", choices=("opencv", "openvino"), default="opencv"
    )
    detect.add_argument("--shape-device", default="CPU")
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

    serve = subparsers.add_parser("serve", help="serve detection over newline JSON/TCP")
    serve.add_argument("--source", choices=("file", "uvc", "realsense"), default="file")
    serve.add_argument("--frame-dir")
    serve.add_argument("--background-dir")
    serve.add_argument("--rgbd-calibration")
    serve.add_argument("--background")
    serve.add_argument("--calibration")
    serve.add_argument("--camera-index", type=int, default=0)
    serve.add_argument("--width", type=int)
    serve.add_argument("--height", type=int)
    serve.add_argument("--fps", type=int)
    serve.add_argument("--shape-model", help="trained geometry RGB model (.npz)")
    serve.add_argument(
        "--shape-backend", choices=("opencv", "openvino"), default="opencv"
    )
    serve.add_argument("--shape-device", default="CPU")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(func=_run_serve)

    camera_live = subparsers.add_parser(
        "camera-live", help="preview a UVC or RealSense camera with motion interlock"
    )
    camera_live.add_argument("--source", choices=("uvc", "realsense"), required=True)
    camera_live.add_argument("--camera-index", type=int, default=0)
    camera_live.add_argument("--width", type=int)
    camera_live.add_argument("--height", type=int)
    camera_live.add_argument("--fps", type=int)
    camera_live.add_argument("--shape-model", help="trained geometry RGB model (.npz)")
    camera_live.add_argument(
        "--shape-backend", choices=("opencv", "openvino"), default="opencv"
    )
    camera_live.add_argument("--shape-device", default="CPU")
    camera_live.add_argument("--background")
    camera_live.add_argument("--calibration")
    camera_live.add_argument("--background-dir")
    camera_live.add_argument("--rgbd-calibration")
    camera_live.add_argument("--headless", action="store_true")
    camera_live.add_argument("--max-frames", type=int, default=0)
    camera_live.set_defaults(func=_run_camera_live)

    camera_record = subparsers.add_parser(
        "camera-record", help="record a UVC or aligned RealSense capture session"
    )
    camera_record.add_argument("--source", choices=("uvc", "realsense"), required=True)
    camera_record.add_argument("--camera-index", type=int, default=0)
    camera_record.add_argument("--width", type=int)
    camera_record.add_argument("--height", type=int)
    camera_record.add_argument("--fps", type=int)
    camera_record.add_argument("--session", required=True)
    camera_record.add_argument("--label")
    camera_record.add_argument("--headless", action="store_true")
    camera_record.add_argument("--max-frames", type=int, default=0)
    camera_record.set_defaults(func=_run_camera_record)

    geometry_audit = subparsers.add_parser(
        "geometry-audit", help="audit a folder-labelled RGB geometry dataset"
    )
    geometry_audit.add_argument("--data-root", required=True)
    geometry_audit.add_argument("--output-report")
    geometry_audit.set_defaults(func=_run_geometry_audit)

    geometry_train = subparsers.add_parser(
        "geometry-train", help="train a lightweight RGB geometry model"
    )
    geometry_train.add_argument("--data-root", required=True)
    geometry_train.add_argument("--output", required=True)
    geometry_train.add_argument(
        "--feature-set",
        choices=("legacy", "edge-topology"),
        default="legacy",
    )
    geometry_train.add_argument("--output-report")
    geometry_train.set_defaults(func=_run_geometry_train)

    geometry_edge_audit = subparsers.add_parser(
        "geometry-edge-audit", help="export internal edge and topology diagnostics"
    )
    geometry_edge_audit.add_argument("--data-root", required=True)
    geometry_edge_audit.add_argument("--output-dir", required=True)
    geometry_edge_audit.add_argument("--output-report")
    geometry_edge_audit.set_defaults(func=_run_geometry_edge_audit)

    geometry_edge_compare = subparsers.add_parser(
        "geometry-edge-compare", help="compare legacy and edge-topology OpenCV models"
    )
    geometry_edge_compare.add_argument("--data-root", required=True)
    geometry_edge_compare.add_argument("--legacy-model", required=True)
    geometry_edge_compare.add_argument("--edge-model", required=True)
    geometry_edge_compare.add_argument("--output-report")
    geometry_edge_compare.set_defaults(func=_run_geometry_edge_compare)

    geometry_evaluate = subparsers.add_parser(
        "geometry-evaluate", help="leave-one-out evaluation of a geometry dataset"
    )
    geometry_evaluate.add_argument("--data-root", required=True)
    geometry_evaluate.add_argument("--model", required=True)
    geometry_evaluate.add_argument(
        "--backend", choices=("opencv", "openvino"), default="opencv"
    )
    geometry_evaluate.add_argument("--device", default="CPU")
    geometry_evaluate.add_argument("--output-report")
    geometry_evaluate.set_defaults(func=_run_geometry_evaluate)

    geometry_cnn_train = subparsers.add_parser(
        "geometry-cnn-train", help="train a MobileNetV3-Small geometry classifier"
    )
    geometry_cnn_train.add_argument("--data-root", required=True)
    geometry_cnn_train.add_argument("--output", required=True)
    geometry_cnn_train.add_argument("--epochs", type=int, default=40)
    geometry_cnn_train.add_argument("--seed", type=int, default=17)
    geometry_cnn_train.add_argument("--no-pretrained", action="store_true")
    geometry_cnn_train.add_argument("--skip-cross-validation", action="store_true")
    geometry_cnn_train.add_argument("--output-report")
    geometry_cnn_train.set_defaults(func=_run_geometry_cnn_train)

    geometry_cnn_export = subparsers.add_parser(
        "geometry-cnn-export", help="export a CNN checkpoint to ONNX and OpenVINO"
    )
    geometry_cnn_export.add_argument("--checkpoint", required=True)
    geometry_cnn_export.add_argument("--output-dir", required=True)
    geometry_cnn_export.add_argument(
        "--precision", choices=("fp32", "fp16", "int8"), default="int8"
    )
    geometry_cnn_export.add_argument("--data-root", help="required calibration data for INT8")
    geometry_cnn_export.add_argument("--output-report")
    geometry_cnn_export.set_defaults(func=_run_geometry_cnn_export)

    geometry_benchmark = subparsers.add_parser(
        "geometry-benchmark", help="measure geometry backend P50/P95 latency"
    )
    geometry_benchmark.add_argument("--data-root", required=True)
    geometry_benchmark.add_argument(
        "--backend", choices=("opencv", "openvino"), required=True
    )
    geometry_benchmark.add_argument("--model", required=True)
    geometry_benchmark.add_argument("--batch-size", type=int, choices=(1, 12), default=1)
    geometry_benchmark.add_argument("--warmup", type=int, default=20)
    geometry_benchmark.add_argument("--iterations", type=int, default=200)
    geometry_benchmark.add_argument("--device", default="CPU")
    geometry_benchmark.add_argument("--output-report")
    geometry_benchmark.set_defaults(func=_run_geometry_benchmark)

    geometry_export = subparsers.add_parser(
        "geometry-export", help="export per-image geometry crops, masks and predictions"
    )
    geometry_export.add_argument("--data-root", required=True)
    geometry_export.add_argument("--model", required=True)
    geometry_export.add_argument("--output-dir", required=True)
    geometry_export.add_argument(
        "--backend", choices=("opencv", "openvino"), default="opencv"
    )
    geometry_export.add_argument("--device", default="CPU")
    geometry_export.set_defaults(func=_run_geometry_export)

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
