from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .geometry_rgb import GEOMETRY_LABELS
from .rgbd import RGBDFrame
from .rgbd_dataset import EMPTY_TRAY_LABEL, depth_preview


CAPTURE_LABELS: tuple[tuple[str, str], ...] = (
    (EMPTY_TRAY_LABEL, "空托盘"),
    *((label_id, label_name) for label_id, label_name in GEOMETRY_LABELS.values()),
)


def capture_label_index(value: str) -> int:
    normalized = value.strip()
    for index, (label_id, label_name) in enumerate(CAPTURE_LABELS):
        if normalized in {label_id, label_name, str(index)}:
            return index
    raise ValueError(f"unsupported capture label: {value}")


def load_batch_counts(dataset_root: str | Path, batch_id: str) -> dict[str, int]:
    """Read existing counts so an interrupted capture session can resume safely."""
    manifest = Path(dataset_root) / "manifest.jsonl"
    counts: Counter[str] = Counter()
    if not manifest.is_file():
        return {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid manifest JSON at line {line_number}: {manifest}"
            ) from error
        if item.get("batch_id") == batch_id:
            counts[str(item.get("label_id", ""))] += 1
    return dict(counts)


@dataclass
class CaptureAssistantState:
    target_per_label: int = 10
    selected_index: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_per_label <= 0:
            raise ValueError("target_per_label must be greater than zero")
        if not 0 <= self.selected_index < len(CAPTURE_LABELS):
            raise ValueError("selected_index is out of range")

    @property
    def current(self) -> tuple[str, str]:
        return CAPTURE_LABELS[self.selected_index]

    @property
    def total_saved(self) -> int:
        return sum(self.counts.get(label_id, 0) for label_id, _ in CAPTURE_LABELS)

    def count(self, label_id: str | None = None) -> int:
        target = label_id or self.current[0]
        return int(self.counts.get(target, 0))

    def select_digit(self, digit: int) -> None:
        if not 0 <= digit < len(CAPTURE_LABELS):
            raise ValueError("label digit is out of range")
        self.selected_index = digit

    def select_next(self, step: int = 1) -> None:
        self.selected_index = (self.selected_index + step) % len(CAPTURE_LABELS)

    def select_next_incomplete(self) -> None:
        for offset in range(1, len(CAPTURE_LABELS) + 1):
            index = (self.selected_index + offset) % len(CAPTURE_LABELS)
            if self.count(CAPTURE_LABELS[index][0]) < self.target_per_label:
                self.selected_index = index
                return

    def record_saved(self) -> None:
        label_id = self.current[0]
        self.counts[label_id] = self.count(label_id) + 1


@dataclass
class CaptureQualityTracker:
    required_stable_frames: int = 3
    motion_threshold: float = 2.5
    minimum_valid_depth_ratio: float = 0.85
    maximum_sync_delta_ms: float = 50.0
    stable_frames: int = 0
    motion_score: float = float("inf")
    valid_depth_ratio: float = 0.0
    sync_delta_ms: float = float("inf")
    _previous_gray: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.required_stable_frames <= 0:
            raise ValueError("required_stable_frames must be greater than zero")
        if self.motion_threshold < 0:
            raise ValueError("motion_threshold must not be negative")
        if not 0 <= self.minimum_valid_depth_ratio <= 1:
            raise ValueError("minimum_valid_depth_ratio must be between zero and one")

    def update(self, frame: RGBDFrame) -> None:
        gray = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        if self._previous_gray is None:
            self.motion_score = float("inf")
            self.stable_frames = 0
        else:
            self.motion_score = float(
                np.mean(cv2.absdiff(gray, self._previous_gray), dtype=np.float64)
            )
            self.stable_frames = (
                self.stable_frames + 1
                if self.motion_score <= self.motion_threshold
                else 0
            )
        self._previous_gray = gray
        valid = np.isfinite(frame.depth_mm) & (frame.depth_mm > 0)
        self.valid_depth_ratio = float(np.mean(valid))
        self.sync_delta_ms = float(frame.sync_delta_ms)

    @property
    def ready(self) -> bool:
        return not self.rejection_reasons()

    def rejection_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.stable_frames < self.required_stable_frames:
            reasons.append("camera_or_object_moving")
        if self.valid_depth_ratio < self.minimum_valid_depth_ratio:
            reasons.append("depth_valid_ratio_low")
        if self.sync_delta_ms > self.maximum_sync_delta_ms:
            reasons.append("rgb_depth_out_of_sync")
        return tuple(reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "stable_frames": self.stable_frames,
            "required_stable_frames": self.required_stable_frames,
            "motion_score": self.motion_score,
            "motion_threshold": self.motion_threshold,
            "valid_depth_ratio": self.valid_depth_ratio,
            "minimum_valid_depth_ratio": self.minimum_valid_depth_ratio,
            "sync_delta_ms": self.sync_delta_ms,
            "maximum_sync_delta_ms": self.maximum_sync_delta_ms,
            "rejection_reasons": list(self.rejection_reasons()),
        }


def render_capture_assistant(
    frame: RGBDFrame,
    state: CaptureAssistantState,
    quality: CaptureQualityTracker,
    message: str = "",
) -> np.ndarray:
    """Render RGB, depth and an English-only overlay compatible with OpenCV."""
    height, width = frame.color_bgr.shape[:2]
    display_scale = min(1.0, 800.0 / width, 500.0 / height)
    display_size = (
        max(1, int(round(width * display_scale))),
        max(1, int(round(height * display_scale))),
    )
    colour_view = cv2.resize(
        frame.color_bgr, display_size, interpolation=cv2.INTER_AREA
    )
    depth = cv2.resize(
        depth_preview(frame.depth_mm),
        display_size,
        interpolation=cv2.INTER_NEAREST,
    )
    views = np.hstack((colour_view, depth))
    panel_height = 390
    panel = np.full((panel_height, views.shape[1], 3), 28, np.uint8)
    canvas = np.vstack((views, panel))
    panel_y = views.shape[0]

    label_id, _ = state.current
    status = "READY" if quality.ready else "WAIT"
    colour = (40, 220, 40) if quality.ready else (0, 190, 255)
    lines = [
        f"{status} | label [{state.selected_index}] {label_id}",
        f"class {state.count()}/{state.target_per_label} | total {state.total_saved}",
        (
            f"depth {quality.valid_depth_ratio:.1%} | motion {quality.motion_score:.2f} "
            f"| sync {quality.sync_delta_ms:.1f} ms"
        ),
        "0-9 label | SPACE save | F force | N/P class | A next incomplete | Q quit",
    ]
    if message:
        lines.append(message[:78])
    for index, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (14, panel_y + 30 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            colour if index == 0 else (235, 235, 235),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )

    start_y = panel_y + 165
    for index, (item_id, _) in enumerate(CAPTURE_LABELS):
        count = state.count(item_id)
        marker = ">" if index == state.selected_index else " "
        complete = "OK" if count >= state.target_per_label else "  "
        cv2.putText(
            canvas,
            f"{marker}[{index}] {item_id:<20} {count:>3}/{state.target_per_label:<3} {complete}",
            (14, start_y + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (60, 255, 255) if index == state.selected_index else (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
    return canvas
