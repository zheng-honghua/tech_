from __future__ import annotations

import argparse
import json
import sys
import time
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
from .capture_assistant import (
    CAPTURE_LABELS,
    CaptureAssistantState,
    CaptureQualityTracker,
    capture_label_index,
    load_batch_counts,
    render_capture_assistant,
)
from .multi_capture import (
    MultiCaptureState,
    load_scene_counts,
    parse_scene_composition,
    render_multi_capture,
    resolve_scene_index,
    save_multi_object_sample,
    validate_scene_composition,
)
from .config import load_config
from .evaluation3d import run_rgbd_benchmark
from .extensions import QRCodeExtension
from .geometry_rgb import (
    audit_geometry_dataset,
    compare_geometry_models,
    evaluate_geometry_holdout,
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
from .geometry_structure_audit import (
    audit_geometry_scene_structures,
    audit_geometry_structures,
)
from .evaluation import run_synthetic_benchmark
from .pipeline import VisionPipeline
from .pipeline3d import VisionPipeline3D
from .rgb_development import RGBDevelopmentPipeline
from .rgbd import RGBDCalibration
from .rgbd_dataset import audit_rgbd_dataset, depth_preview, save_rgbd_dataset_sample
from .geometry_rgbd_model import DepthGeometryModel, train_rgbd_geometry_model
from .face_topology3d import extract_face_topology
from .geometry3d import segment_depth_objects
from .interlock import MotionInterlock
from .server import VisionService3D, serve_json_tcp
from .scene_image import GeometryScenePredictor, save_scene_image_result
from .single_image import (
    DEFAULT_GEOMETRY_MODEL,
    GeometryImagePredictor,
    REASON_TEXT_ZH,
    save_single_image_result,
)
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


def _run_predict_image(args: argparse.Namespace) -> int:
    try:
        predictor = GeometryImagePredictor.load(
            args.model, backend=args.backend, device=args.device
        )
        result = predictor.predict_file(args.image)
        artifacts = (
            save_single_image_result(result, args.output_dir)
            if args.output_dir
            else {}
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(f"预测失败：{error}", file=sys.stderr)
        return 2
    payload = result.to_dict(artifacts)
    if args.json_only:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    prediction = result.prediction
    state = "识别成功" if prediction.accepted else "已安全拒识"
    print(f"状态：{state}")
    print(f"类别：{prediction.label_name} ({prediction.label_id})")
    print(f"置信度：{prediction.confidence:.3f}")
    explanation = REASON_TEXT_ZH.get(prediction.reason, "未提供进一步说明")
    print(f"依据：{explanation} ({prediction.reason})")
    print(f"后端：{prediction.backend}")
    print(
        f"耗时：模型 {prediction.inference_ms:.2f} ms，"
        f"端到端 {result.total_ms:.2f} ms"
    )
    print("机械执行：禁止（单RGB图片仅用于开发识别）")
    if args.output_dir:
        print(f"输出目录：{Path(args.output_dir).resolve()}")
    return 0


def _run_predict_scene(args: argparse.Namespace) -> int:
    try:
        predictor = GeometryScenePredictor.load(
            args.model, backend=args.backend, device=args.device
        )
        result = predictor.predict_file(args.image)
        artifacts = save_scene_image_result(result, args.output_dir)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(f"场景预测失败：{error}", file=sys.stderr)
        return 2
    payload = result.to_dict(
        object_artifacts=artifacts["objects"],
        scene_artifacts=artifacts["scene"],
    )
    if args.json_only:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"检测到：{len(result.objects)} 个分离物块")
    for item in result.objects:
        prediction = item.prediction
        state = "接受" if prediction.accepted else "拒识"
        explanation = REASON_TEXT_ZH.get(prediction.reason, prediction.reason)
        print(
            f"{item.object_id}：{prediction.label_name} "
            f"({prediction.label_id})，置信度 {prediction.confidence:.3f}，"
            f"{state}，{explanation}"
        )
    print(f"端到端耗时：{result.total_ms:.2f} ms")
    print("机械执行：禁止（普通RGB多物块图片仅用于开发识别）")
    print(f"输出目录：{Path(args.output_dir).resolve()}")
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
        shape_model=_make_depth_shape_model(args),
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


def _make_depth_shape_model(args: argparse.Namespace):
    model_path = getattr(args, "rgbd_shape_model", None)
    return None if not model_path else DepthGeometryModel.load(model_path)


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
            depth_width=getattr(args, "depth_width", None),
            depth_height=getattr(args, "depth_height", None),
            color_width=getattr(args, "color_width", None),
            color_height=getattr(args, "color_height", None),
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
            shape_model=_make_depth_shape_model(args),
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


def _run_rgbd_capture(args: argparse.Namespace) -> int:
    if args.headless and args.count <= 0:
        raise ValueError("--headless requires --count greater than zero")
    config = load_config(args.config)
    source = _make_camera_source(args, config)
    saved = 0
    frame = None
    print("D415 capture: SPACE=save one RGB-D bundle, q=quit")
    try:
        for _ in range(max(0, args.discard_frames)):
            frame = source.read()
        while args.count <= 0 or saved < args.count:
            frame = source.read()
            should_save = args.headless
            if not args.headless:
                preview = depth_preview(frame.depth_mm)
                preview = cv2.resize(
                    preview,
                    (frame.color_bgr.shape[1], frame.color_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                canvas = np.hstack((frame.color_bgr, preview))
                valid_ratio = float(np.mean(frame.depth_mm > 0))
                cv2.putText(
                    canvas,
                    f"SPACE capture | saved={saved} | valid depth={valid_ratio:.1%}",
                    (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,
                    cv2.LINE_AA,
                )
                cv2.imshow("D415 RGB-D dataset capture", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                should_save = key == 32
            if not should_save:
                continue
            settings = (
                source.capture_metadata()
                if hasattr(source, "capture_metadata")
                else {}
            )
            target = save_rgbd_dataset_sample(
                frame, args.dataset_root, args.batch_id, args.label, settings
            )
            saved += 1
            print(f"saved[{saved}]={target.resolve()}")
            if args.headless and saved < args.count:
                for _ in range(max(1, args.discard_frames)):
                    frame = source.read()
    finally:
        source.close()
        if not args.headless:
            cv2.destroyAllWindows()
    print(f"captured={saved}, dataset={Path(args.dataset_root).resolve()}")
    return 0


def _run_rgbd_capture_assistant(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    source = _make_camera_source(args, config)
    state = CaptureAssistantState(
        target_per_label=args.target_per_label,
        selected_index=capture_label_index(args.start_label),
        counts=load_batch_counts(args.dataset_root, args.batch_id),
    )
    quality = CaptureQualityTracker(
        required_stable_frames=args.stable_frames,
        motion_threshold=args.motion_threshold,
        minimum_valid_depth_ratio=args.min_valid_depth_ratio,
        maximum_sync_delta_ms=args.max_sync_delta_ms,
    )
    saved_this_run = 0
    message = "Wait until READY, then press SPACE"
    print("D415 dataset assistant")
    print("0=empty tray, 1-9=shape, SPACE=save, f=force, n/p=class, a=next incomplete, q=quit")
    print("labels=" + ", ".join(f"{index}:{name}" for index, (_, name) in enumerate(CAPTURE_LABELS)))
    try:
        for _ in range(max(0, args.discard_frames)):
            source.read()
        while True:
            frame = source.read()
            quality.update(frame)
            cv2.imshow(
                "D415 RGB-D capture assistant",
                render_capture_assistant(frame, state, quality, message),
            )
            key = cv2.waitKey(1) & 0xFF
            if ord("A") <= key <= ord("Z"):
                key += ord("a") - ord("A")
            if key == ord("q"):
                break
            if ord("0") <= key <= ord("9"):
                state.select_digit(key - ord("0"))
                label_id, label_name = state.current
                message = f"Selected {label_id} ({label_name})"
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
            label_id, label_name = state.current
            if (
                label_id != "empty_tray"
                and state.count("empty_tray") == 0
                and not forced
            ):
                message = "Rejected: capture empty tray before object classes"
                print(message)
                continue
            if not quality.ready and not forced:
                message = "Rejected: " + ", ".join(quality.rejection_reasons())
                print(message)
                continue
            settings = source.capture_metadata()
            settings["capture_assistant"] = {
                **quality.to_dict(),
                "quality_override": forced,
                "target_per_label": state.target_per_label,
            }
            target = save_rgbd_dataset_sample(
                frame,
                args.dataset_root,
                args.batch_id,
                label_id,
                settings,
            )
            state.record_saved()
            saved_this_run += 1
            message = f"Saved {label_id} #{state.count()}"
            print(f"saved[{saved_this_run}] {label_name}={target.resolve()}")
            if args.auto_advance and state.count() >= state.target_per_label:
                state.select_next_incomplete()
                message += f" | next {state.current[0]}"
    finally:
        source.close()
        cv2.destroyAllWindows()
    print(
        json.dumps(
            {
                "dataset": str(Path(args.dataset_root).resolve()),
                "batch_id": args.batch_id,
                "saved_this_run": saved_this_run,
                "counts": state.counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run_rgbd_multi_capture(args: argparse.Namespace) -> int:
    if args.interval_ms < 0:
        raise ValueError("interval_ms must not be negative")
    if args.max_captures < 0:
        raise ValueError("max_captures must not be negative")
    config = load_config(args.config)
    composition = parse_scene_composition(args.composition)
    counts = load_scene_counts(args.dataset_root, args.batch_id)
    scene_index = resolve_scene_index(counts, args.captures_per_scene, args.scene_index)
    validate_scene_composition(
        args.dataset_root, args.batch_id, scene_index, composition
    )
    state = MultiCaptureState(
        batch_id=args.batch_id,
        composition=composition,
        captures_per_scene=args.captures_per_scene,
        scene_index=scene_index,
        scene_counts=counts,
        auto_capture=args.auto_start or args.headless,
    )
    quality = CaptureQualityTracker(
        required_stable_frames=args.stable_frames,
        motion_threshold=args.motion_threshold,
        minimum_valid_depth_ratio=args.min_valid_depth_ratio,
        maximum_sync_delta_ms=args.max_sync_delta_ms,
    )
    source = _make_camera_source(args, config)
    saved_this_run = 0
    last_saved_at = 0.0
    message = "Arrange objects, wait for READY, then press B or SPACE"
    print("D415 multi-object batch capture")
    print("SPACE=save, b=auto batch, p=pause, n=next layout, f=force, q=quit")
    try:
        for _ in range(max(0, args.discard_frames)):
            source.read()
        while True:
            frame = source.read()
            quality.update(frame)
            key = -1
            if not args.headless:
                cv2.imshow(
                    "D415 multi-object batch capture",
                    render_multi_capture(frame, state, quality, message),
                )
                key = cv2.waitKey(1) & 0xFF
                if ord("A") <= key <= ord("Z"):
                    key += ord("a") - ord("A")
            if key == ord("q"):
                break
            if key == ord("n"):
                state.next_scene()
                message = "New layout: rearrange objects, then start capture"
                continue
            if key == ord("p"):
                state.auto_capture = False
                message = "Automatic capture paused"
                continue
            if key == ord("b"):
                if state.scene_complete:
                    message = "Scene complete; rearrange objects and press N"
                else:
                    state.auto_capture = True
                    message = "Automatic capture started"
                continue

            forced = key == ord("f")
            manual = key == 32 or forced
            now = time.monotonic()
            due = state.auto_capture and (
                (now - last_saved_at) * 1000.0 >= args.interval_ms
            )
            if not manual and not due:
                continue
            if not quality.ready and not forced:
                message = "Waiting: " + ", ".join(quality.rejection_reasons())
                continue

            settings = source.capture_metadata()
            settings["multi_capture"] = {
                **quality.to_dict(),
                "quality_override": forced,
                "captures_per_scene": state.captures_per_scene,
                "interval_ms": args.interval_ms,
            }
            target = save_multi_object_sample(
                frame, args.dataset_root, state, settings
            )
            saved_this_run += 1
            last_saved_at = now
            message = f"Saved scene {state.scene_index:04d}, capture {state.current_count}"
            print(f"saved[{saved_this_run}]={target.resolve()}")
            if state.scene_complete:
                state.auto_capture = False
                message += " | complete; rearrange and press N"
                if args.headless:
                    break
            if args.max_captures and saved_this_run >= args.max_captures:
                break
    finally:
        source.close()
        if not args.headless:
            cv2.destroyAllWindows()
    print(json.dumps({
        "dataset": str(Path(args.dataset_root).resolve()),
        "batch_id": state.batch_id,
        "scene_index": state.scene_index,
        "saved_this_run": saved_this_run,
        "scene_counts": state.scene_counts,
    }, ensure_ascii=False, indent=2))
    return 0


def _run_rgbd_dataset_audit(args: argparse.Namespace) -> int:
    _write_optional_report(audit_rgbd_dataset(args.data_root), args.output_report)
    return 0


def _run_geometry_rgbd_train(args: argparse.Namespace) -> int:
    report = train_rgbd_geometry_model(
        args.data_root, args.output, load_config(args.config),
        set(args.batch_id) if args.batch_id else None,
        getattr(args, "base_model", None),
    )
    report["model_path"] = str(Path(args.output).resolve())
    _write_optional_report(report, args.output_report)
    return 0


def _run_rgbd_face_audit(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    frame = load_rgbd_frame(args.frame_dir)
    background = load_rgbd_frame(args.background_dir)
    pipeline = VisionPipeline3D(config=config, background_frame=background)
    objects, _ = segment_depth_objects(
        frame.color_bgr, frame.depth_mm, frame.intrinsics,
        pipeline.calibration.tray_plane_camera, config.rgbd,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    canvas = frame.color_bgr.copy()
    palette = ((255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 220, 60),
               (220, 60, 255), (60, 220, 255), (180, 180, 60), (60, 180, 180))
    report_objects = []
    for object_index, item in enumerate(objects, start=1):
        topology = extract_face_topology(
            frame.depth_mm, item.mask, frame.intrinsics,
            color_crop_bgr=frame.color_bgr,
        )
        face_items = []
        for face in topology.faces:
            colour = palette[face.face_id % len(palette)]
            overlay = np.full_like(canvas, colour)
            selected = face.mask > 0
            canvas[selected] = (
                0.55 * canvas[selected] + 0.45 * overlay[selected]
            ).astype(np.uint8)
            contours, _ = cv2.findContours(
                face.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, colour, 2)
            if contours:
                moment = cv2.moments(max(contours, key=cv2.contourArea))
                if moment["m00"]:
                    center = (
                        int(moment["m10"] / moment["m00"]),
                        int(moment["m01"] / moment["m00"]),
                    )
                    cv2.putText(
                        canvas, f"O{object_index}F{face.face_id}", center,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA,
                    )
            face_items.append({
                "face_id": face.face_id,
                "area_px": face.area_px,
                "normal": face.plane.normal.tolist(),
                "plane_offset": face.plane.offset,
                "fit_rmse_mm": face.plane.rmse_mm,
                "center_camera_mm": face.center_camera_mm.tolist(),
            })
        report_objects.append({
            "object_id": object_index,
            "bbox_px": list(item.bbox),
            "faces": face_items,
            "adjacency": [list(pair) for pair in topology.adjacency],
            "dihedral_angles_deg": list(topology.angles_deg),
            "triple_junctions": topology.triple_junctions,
            "evidence_ratio": topology.evidence_ratio,
            "rgb_edge_support": topology.rgb_edge_support,
            "quality": topology.quality,
        })
    report = {
        "frame_dir": str(Path(args.frame_dir).resolve()),
        "background_dir": str(Path(args.background_dir).resolve()),
        "object_count": len(objects),
        "objects": report_objects,
        "notes": "Observed depth faces only; no hidden face is used for grasping.",
    }
    cv2.imwrite(str(output / "annotated-faces.png"), canvas)
    (output / "face-topology.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
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
    model, report = train_geometry_model(
        args.data_root, args.feature_set, args.additional_data_root
    )
    model.save(args.output)
    report["model_path"] = str(Path(args.output).resolve())
    _write_optional_report(report, args.output_report)
    return 0


def _run_geometry_edge_audit(args: argparse.Namespace) -> int:
    report = audit_geometry_edges(args.data_root, args.output_dir)
    _write_optional_report(report, args.output_report)
    return 0


def _run_geometry_structure_audit(args: argparse.Namespace) -> int:
    report = audit_geometry_structures(args.data_root, args.output_dir)
    _write_optional_report(report, args.output_report)
    return 0


def _run_geometry_scene_structure_audit(args: argparse.Namespace) -> int:
    report = audit_geometry_scene_structures(args.data_root, args.output_dir)
    _write_optional_report(report, args.output_report)
    return 0


def _run_geometry_edge_compare(args: argparse.Namespace) -> int:
    report = compare_geometry_models(
        args.data_root, args.legacy_model, args.edge_model
    )
    _write_optional_report(report, args.output_report)
    return 0


def _run_geometry_holdout_evaluate(args: argparse.Namespace) -> int:
    report = evaluate_geometry_holdout(
        args.training_data_root, args.test_data_root, args.model
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
        additional_data_roots=args.additional_data_root,
        resume_checkpoint=args.resume,
        fine_tune_backbone=args.fine_tune_backbone,
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

    predict_image = subparsers.add_parser(
        "predict-image",
        help="predict the geometry class of one RGB photograph",
    )
    predict_image.add_argument("image", help="input JPG/PNG image path")
    predict_image.add_argument(
        "--model",
        default=str(DEFAULT_GEOMETRY_MODEL),
        help="NPZ model or OpenVINO model directory",
    )
    predict_image.add_argument(
        "--backend", choices=("auto", "opencv", "openvino"), default="auto"
    )
    predict_image.add_argument("--device", default="CPU")
    predict_image.add_argument(
        "--output-dir",
        help="optional directory for result.json, annotated image, crop and mask",
    )
    predict_image.add_argument(
        "--json-only", action="store_true", help="print machine-readable JSON only"
    )
    predict_image.set_defaults(func=_run_predict_image)

    predict_scene = subparsers.add_parser(
        "predict-scene",
        help="detect and classify multiple separated objects in one RGB image",
    )
    predict_scene.add_argument("image", help="input scene JPG/PNG image path")
    predict_scene.add_argument(
        "--model", default=str(DEFAULT_GEOMETRY_MODEL), help="geometry model path"
    )
    predict_scene.add_argument(
        "--backend", choices=("auto", "opencv", "openvino"), default="auto"
    )
    predict_scene.add_argument("--device", default="CPU")
    predict_scene.add_argument("--output-dir", default="output/predict-scene")
    predict_scene.add_argument("--json-only", action="store_true")
    predict_scene.set_defaults(func=_run_predict_scene)

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
    rgbd_detect.add_argument("--rgbd-shape-model", help="trained RGB-D geometry NPZ model")
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

    rgbd_capture = subparsers.add_parser(
        "rgbd-capture", help="capture labelled D415 samples as self-contained folders"
    )
    rgbd_capture.add_argument("--dataset-root", required=True)
    rgbd_capture.add_argument("--batch-id", required=True)
    rgbd_capture.add_argument("--label", required=True)
    rgbd_capture.add_argument("--color-width", type=int, default=1920)
    rgbd_capture.add_argument("--color-height", type=int, default=1080)
    rgbd_capture.add_argument("--depth-width", type=int, default=640)
    rgbd_capture.add_argument("--depth-height", type=int, default=480)
    rgbd_capture.add_argument("--fps", type=int, default=30)
    rgbd_capture.add_argument("--discard-frames", type=int, default=30)
    rgbd_capture.add_argument(
        "--count", type=int, default=0,
        help="stop after N captures; zero means manual capture until q",
    )
    rgbd_capture.add_argument("--headless", action="store_true")
    rgbd_capture.set_defaults(
        func=_run_rgbd_capture, source="realsense", camera_index=0,
        width=None, height=None,
    )

    rgbd_assistant = subparsers.add_parser(
        "rgbd-capture-assistant",
        help="interactively capture and label a complete D415 geometry batch",
    )
    rgbd_assistant.add_argument("--dataset-root", required=True)
    rgbd_assistant.add_argument("--batch-id", required=True)
    rgbd_assistant.add_argument("--target-per-label", type=int, default=10)
    rgbd_assistant.add_argument("--start-label", default="empty_tray")
    rgbd_assistant.add_argument("--color-width", type=int, default=1920)
    rgbd_assistant.add_argument("--color-height", type=int, default=1080)
    rgbd_assistant.add_argument("--depth-width", type=int, default=640)
    rgbd_assistant.add_argument("--depth-height", type=int, default=480)
    rgbd_assistant.add_argument("--fps", type=int, default=30)
    rgbd_assistant.add_argument("--discard-frames", type=int, default=60)
    rgbd_assistant.add_argument("--stable-frames", type=int, default=3)
    rgbd_assistant.add_argument("--motion-threshold", type=float, default=2.5)
    rgbd_assistant.add_argument("--min-valid-depth-ratio", type=float, default=0.85)
    rgbd_assistant.add_argument("--max-sync-delta-ms", type=float, default=50.0)
    rgbd_assistant.add_argument("--auto-advance", action="store_true")
    rgbd_assistant.set_defaults(
        func=_run_rgbd_capture_assistant,
        source="realsense",
        camera_index=0,
        width=None,
        height=None,
    )

    rgbd_multi = subparsers.add_parser(
        "rgbd-multi-capture",
        help="batch-capture labelled multi-object D415 test scenes",
    )
    rgbd_multi.add_argument("--dataset-root", default="data/rgbd-multi-scenes")
    rgbd_multi.add_argument("--batch-id", required=True)
    rgbd_multi.add_argument(
        "--composition", required=True,
        help='scene object counts, e.g. "三棱柱:2,四棱锥:1,圆锥:1"',
    )
    rgbd_multi.add_argument(
        "--scene-index", type=int, default=0,
        help="scene number; zero resumes the latest incomplete scene",
    )
    rgbd_multi.add_argument("--captures-per-scene", type=int, default=10)
    rgbd_multi.add_argument("--interval-ms", type=int, default=500)
    rgbd_multi.add_argument("--color-width", type=int, default=1920)
    rgbd_multi.add_argument("--color-height", type=int, default=1080)
    rgbd_multi.add_argument("--depth-width", type=int, default=640)
    rgbd_multi.add_argument("--depth-height", type=int, default=480)
    rgbd_multi.add_argument("--fps", type=int, default=30)
    rgbd_multi.add_argument("--discard-frames", type=int, default=60)
    rgbd_multi.add_argument("--stable-frames", type=int, default=3)
    rgbd_multi.add_argument("--motion-threshold", type=float, default=2.5)
    rgbd_multi.add_argument("--min-valid-depth-ratio", type=float, default=0.85)
    rgbd_multi.add_argument("--max-sync-delta-ms", type=float, default=50.0)
    rgbd_multi.add_argument("--auto-start", action="store_true")
    rgbd_multi.add_argument("--headless", action="store_true")
    rgbd_multi.add_argument("--max-captures", type=int, default=0)
    rgbd_multi.set_defaults(
        func=_run_rgbd_multi_capture,
        source="realsense", camera_index=0, width=None, height=None,
    )

    rgbd_audit = subparsers.add_parser(
        "rgbd-dataset-audit", help="validate RGB-D sample bundles and depth quality"
    )
    rgbd_audit.add_argument("--data-root", required=True)
    rgbd_audit.add_argument("--output-report")
    rgbd_audit.set_defaults(func=_run_rgbd_dataset_audit)

    rgbd_train = subparsers.add_parser(
        "geometry-rgbd-train", help="train the metric point-cloud geometry baseline"
    )
    rgbd_train.add_argument("--data-root", required=True)
    rgbd_train.add_argument("--output", required=True)
    rgbd_train.add_argument("--output-report")
    rgbd_train.add_argument(
        "--base-model",
        help="append new samples to exemplars from an existing RGB-D v3 model",
    )
    rgbd_train.add_argument(
        "--batch-id", action="append",
        help="train only this capture batch; repeat to select multiple batches",
    )
    rgbd_train.set_defaults(func=_run_geometry_rgbd_train)

    rgbd_face_audit = subparsers.add_parser(
        "rgbd-face-audit", help="visualise observed 3-D planes and face topology"
    )
    rgbd_face_audit.add_argument("--frame-dir", required=True)
    rgbd_face_audit.add_argument("--background-dir", required=True)
    rgbd_face_audit.add_argument("--output-dir", default="output/rgbd-face-audit")
    rgbd_face_audit.set_defaults(func=_run_rgbd_face_audit)

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
    serve.add_argument("--depth-width", type=int)
    serve.add_argument("--depth-height", type=int)
    serve.add_argument("--color-width", type=int)
    serve.add_argument("--color-height", type=int)
    serve.add_argument("--shape-model", help="trained geometry RGB model (.npz)")
    serve.add_argument("--rgbd-shape-model", help="trained RGB-D geometry NPZ model")
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
    camera_live.add_argument("--depth-width", type=int)
    camera_live.add_argument("--depth-height", type=int)
    camera_live.add_argument("--color-width", type=int)
    camera_live.add_argument("--color-height", type=int)
    camera_live.add_argument("--shape-model", help="trained geometry RGB model (.npz)")
    camera_live.add_argument("--rgbd-shape-model", help="trained RGB-D geometry NPZ model")
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
    geometry_train.add_argument(
        "--additional-data-root",
        action="append",
        default=[],
        help="additional labelled batch; exact duplicate images are skipped",
    )
    geometry_train.add_argument("--output", required=True)
    geometry_train.add_argument(
        "--feature-set",
        choices=("legacy", "edge-topology", "structure-topology"),
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

    geometry_structure_audit = subparsers.add_parser(
        "geometry-structure-audit",
        help="export conservative noise-reduced contour lines and vertices",
    )
    geometry_structure_audit.add_argument("--data-root", required=True)
    geometry_structure_audit.add_argument("--output-dir", required=True)
    geometry_structure_audit.add_argument("--output-report")
    geometry_structure_audit.set_defaults(func=_run_geometry_structure_audit)

    geometry_scene_structure_audit = subparsers.add_parser(
        "geometry-scene-structure-audit",
        help="extract noise-reduced structure for every separated object in scene images",
    )
    geometry_scene_structure_audit.add_argument("--data-root", required=True)
    geometry_scene_structure_audit.add_argument("--output-dir", required=True)
    geometry_scene_structure_audit.add_argument("--output-report")
    geometry_scene_structure_audit.set_defaults(
        func=_run_geometry_scene_structure_audit
    )

    geometry_edge_compare = subparsers.add_parser(
        "geometry-edge-compare", help="compare legacy and edge-topology OpenCV models"
    )
    geometry_edge_compare.add_argument("--data-root", required=True)
    geometry_edge_compare.add_argument("--legacy-model", required=True)
    geometry_edge_compare.add_argument("--edge-model", required=True)
    geometry_edge_compare.add_argument("--output-report")
    geometry_edge_compare.set_defaults(func=_run_geometry_edge_compare)

    geometry_holdout = subparsers.add_parser(
        "geometry-holdout-evaluate",
        help="evaluate a trained OpenCV model on hash-distinct images",
    )
    geometry_holdout.add_argument("--training-data-root", required=True)
    geometry_holdout.add_argument("--test-data-root", required=True)
    geometry_holdout.add_argument("--model", required=True)
    geometry_holdout.add_argument("--output-report")
    geometry_holdout.set_defaults(func=_run_geometry_holdout_evaluate)

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
    geometry_cnn_train.add_argument(
        "--additional-data-root",
        action="append",
        default=[],
        help="additional labelled batch; exact duplicate images are skipped",
    )
    geometry_cnn_train.add_argument("--output", required=True)
    geometry_cnn_train.add_argument("--epochs", type=int, default=40)
    geometry_cnn_train.add_argument("--seed", type=int, default=17)
    geometry_cnn_train.add_argument("--no-pretrained", action="store_true")
    geometry_cnn_train.add_argument(
        "--resume", help="continue from a compatible CNN checkpoint"
    )
    geometry_cnn_train.add_argument(
        "--fine-tune-backbone",
        action="store_true",
        help="unfreeze the last MobileNet feature blocks at a lower learning rate",
    )
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
