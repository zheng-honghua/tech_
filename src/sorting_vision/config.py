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
    min_depth_mm: float = 150.0
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
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_frames: int = 30
    reconnect_attempts: int = 3


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
    motion_interlock: MotionInterlockConfig = field(default_factory=MotionInterlockConfig)


def _construct(cls: type, data: dict[str, Any]):
    fields = cls.__dataclass_fields__
    return cls(**{key: value for key, value in data.items() if key in fields})


def load_config(path: str | Path | None = None) -> VisionConfig:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
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
        motion_interlock=_construct(
            MotionInterlockConfig, raw.get("motion_interlock", {})
        ),
    )
