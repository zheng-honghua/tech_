from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrayConfig:
    width_mm: float = 160.0
    height_mm: float = 160.0
    rectified_width_px: int = 800
    rectified_height_px: int = 800


@dataclass(frozen=True)
class SegmentationConfig:
    background_delta: float = 22.0
    retry_threshold_scale: float = 0.75
    min_area_px: int = 700
    max_area_px: int = 90000
    morphology_kernel: int = 5
    split_touching: bool = True
    watershed_peak_ratio: float = 0.44
    min_clearance_px: float = 10.0
    border_margin_px: int = 4


@dataclass(frozen=True)
class ClassificationConfig:
    max_color_distance: float = 48.0
    min_color_confidence: float = 0.72
    min_shape_confidence: float = 0.68
    min_pick_confidence: float = 0.72
    colors: dict[str, dict[str, str]] = field(default_factory=dict)
    shapes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionConfig:
    stable_frames: int = 2
    prefer_clearance_weight: float = 0.25


@dataclass(frozen=True)
class RGBDConfig:
    processing_scale: float = 0.75
    min_depth_mm: float = 280.0
    max_depth_mm: float = 1500.0
    foreground_height_mm: float = 3.0
    max_object_height_mm: float = 80.0
    min_valid_depth_ratio: float = 0.85
    plane_ransac_threshold_mm: float = 1.5
    max_plane_shift_mm: float = 3.0
    max_rgb_depth_sync_ms: float = 20.0
    point_sample_stride: int = 2
    min_area_px: int = 250
    max_area_px: int = 120000
    morphology_kernel: int = 5
    border_margin_px: int = 4
    min_clearance_px: float = 8.0


@dataclass(frozen=True)
class GraspConfig:
    cup_diameter_mm: float = 15.0
    edge_margin_mm: float = 2.0
    max_flatness_rmse_mm: float = 1.2
    max_tilt_deg: float = 35.0
    min_patch_valid_ratio: float = 0.9
    candidate_stride_px: int = 4
    max_candidates: int = 3
    min_grasp_score: float = 0.72


@dataclass(frozen=True)
class NetworkConfig:
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass(frozen=True)
class CameraConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    warmup_frames: int = 30
    reconnect_attempts: int = 3
    realsense_model: str = "Intel RealSense D435if"
    realsense_frame_prefix: str = "d435if"
    realsense_color_width: int = 1280
    realsense_color_height: int = 720
    realsense_depth_width: int = 1280
    realsense_depth_height: int = 720
    realsense_fps: int = 30


@dataclass(frozen=True)
class DualViewConfig:
    enabled: bool = False
    acquisition_mode: str = "threaded"
    side_process_isolation: bool = False
    platform_id: str = "temporary"
    calibration_path: str = "config/dual/temporary/calibration.json"
    side_background_path: str = "config/dual/temporary/side-background.png"
    side_model_path: str = "models/side-geometry.npz"
    side_camera_index: int = 1
    side_width: int = 1280
    side_height: int = 720
    side_fps: int = 30
    side_backend: str | None = None
    side_fourcc: str | None = None
    side_auto_exposure: bool = False
    side_exposure: float | None = -6.0
    side_auto_white_balance: bool = False
    side_white_balance: float | None = 4500.0
    side_autofocus: bool = False
    side_focus: float | None = None
    max_pair_delta_ms: float = 50.0
    pair_timeout_ms: float = 250.0
    min_side_quality: float = 0.60
    side_weight: float = 0.30
    conflict_probability: float = 0.75
    fused_probability: float = 0.72
    fused_margin: float = 0.12
    top_only_probability: float = 0.85
    roi_overlap_threshold: float = 0.30
    roi_padding_ratio: float = 0.15
    side_background_delta: float = 22.0
    side_min_area_px: int = 120
    side_min_blur_variance: float = 35.0
    top_temperature: float = 1.0
    side_temperature: float = 1.0


@dataclass(frozen=True)
class MotionInterlockConfig:
    min_settle_ms: float = 300.0
    discard_frames: int = 8
    stable_frames: int = 3
    frame_diff_threshold: float = 2.5
    timeout_ms: float = 2000.0


@dataclass(frozen=True)
class VisionConfig:
    tray: TrayConfig
    segmentation: SegmentationConfig
    classification: ClassificationConfig
    selection: SelectionConfig
    rgbd: RGBDConfig = field(default_factory=RGBDConfig)
    grasp: GraspConfig = field(default_factory=GraspConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    dual_view: DualViewConfig = field(default_factory=DualViewConfig)
    motion_interlock: MotionInterlockConfig = field(default_factory=MotionInterlockConfig)


def _construct(cls: type, data: dict[str, Any]):
    fields = cls.__dataclass_fields__
    return cls(**{key: value for key, value in data.items() if key in fields})


def _load_yaml_with_base(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    visited = set() if seen is None else set(seen)
    if resolved in visited:
        raise ValueError(f"cyclic config inheritance: {resolved}")
    visited.add(resolved)
    with resolved.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    base_name = raw.pop("extends", None)
    if base_name is None:
        return raw
    base = _load_yaml_with_base(resolved.parent / str(base_name), visited)
    merged = {key: dict(value) if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in raw.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> VisionConfig:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    raw = _load_yaml_with_base(Path(path))
    return VisionConfig(
        tray=_construct(TrayConfig, raw.get("tray", {})),
        segmentation=_construct(SegmentationConfig, raw.get("segmentation", {})),
        classification=_construct(
            ClassificationConfig, raw.get("classification", {})
        ),
        selection=_construct(SelectionConfig, raw.get("selection", {})),
        rgbd=_construct(RGBDConfig, raw.get("rgbd", {})),
        grasp=_construct(GraspConfig, raw.get("grasp", {})),
        network=_construct(NetworkConfig, raw.get("network", {})),
        camera=_construct(CameraConfig, raw.get("camera", {})),
        dual_view=_construct(DualViewConfig, raw.get("dual_view", {})),
        motion_interlock=_construct(
            MotionInterlockConfig, raw.get("motion_interlock", {})
        ),
    )
