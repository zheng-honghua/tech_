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
class VisionConfig:
    tray: TrayConfig
    segmentation: SegmentationConfig
    classification: ClassificationConfig
    selection: SelectionConfig


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
    )
