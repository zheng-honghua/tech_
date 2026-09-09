from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .camera import SynchronizedFramePair
from .capture_assistant import CaptureAssistantState
from .dual_view import save_synchronized_pair
from .rgbd_dataset import depth_preview


def load_dual_batch_counts(
    dataset_root: str | Path,
    batch_id: str,
    platform_id: str,
) -> dict[str, int]:
    """Resume counts from the paired-capture manifest for one platform/batch."""
    manifest = Path(dataset_root) / "dual-manifest.jsonl"
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
                f"invalid dual manifest JSON at line {line_number}: {manifest}"
            ) from error
        if (
            item.get("batch_id") == batch_id
            and item.get("platform_id") == platform_id
        ):
            counts[str(item.get("label_id", ""))] += 1
    return dict(counts)


@dataclass
class DualCaptureQualityTracker:
    """Quality gate for top RGB-D plus side RGB dataset capture."""

    stable_frames_required: int = 3
    motion_threshold: float = 2.5
    minimum_valid_depth_ratio: float = 0.85
    maximum_rgbd_sync_ms: float = 20.0
    maximum_pair_delta_ms: float = 50.0
    minimum_side_blur_variance: float = 35.0
    primary_motion: float = float("inf")
    side_motion: float = float("inf")
    valid_depth_ratio: float = 0.0
    side_blur_variance: float = 0.0
    stable_frames: int = 0
    _previous_primary: np.ndarray | None = field(default=None, repr=False)
    _previous_side: np.ndarray | None = field(default=None, repr=False)

    @staticmethod
    def _small_gray(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)

    def update(self, pair: SynchronizedFramePair) -> None:
        primary = self._small_gray(pair.primary.color_bgr)
        side = None if pair.side is None else self._small_gray(pair.side.color_bgr)
        self.primary_motion = (
            float("inf")
            if self._previous_primary is None
            else float(cv2.absdiff(primary, self._previous_primary).mean())
        )
        self.side_motion = (
            float("inf")
            if side is None or self._previous_side is None
            else float(cv2.absdiff(side, self._previous_side).mean())
        )
        self._previous_primary = primary
        self._previous_side = side
        valid = np.isfinite(pair.primary.depth_mm) & (pair.primary.depth_mm > 0)
        self.valid_depth_ratio = float(np.mean(valid))
        self.side_blur_variance = (
            0.0
            if side is None
            else float(cv2.Laplacian(side, cv2.CV_64F).var())
        )
        stable_now = (
            self.primary_motion <= self.motion_threshold
            and self.side_motion <= self.motion_threshold
        )
        self.stable_frames = self.stable_frames + 1 if stable_now else 0

    def rejection_reasons(self, pair: SynchronizedFramePair) -> tuple[str, ...]:
        reasons: list[str] = []
        if pair.side is None:
            reasons.append("side_camera_missing")
        if not pair.synchronized:
            reasons.append("camera_pair_out_of_sync")
        if pair.primary.sync_delta_ms > self.maximum_rgbd_sync_ms:
            reasons.append("primary_rgb_depth_out_of_sync")
        if self.valid_depth_ratio < self.minimum_valid_depth_ratio:
            reasons.append("depth_valid_ratio_low")
        if self.stable_frames < self.stable_frames_required:
            reasons.append("camera_or_object_moving")
        if self.side_blur_variance < self.minimum_side_blur_variance:
            reasons.append("side_image_blurred")
        return tuple(reasons)

    def ready(self, pair: SynchronizedFramePair) -> bool:
        return not self.rejection_reasons(pair)

    def to_dict(self, pair: SynchronizedFramePair) -> dict[str, object]:
        return {
            "ready": self.ready(pair),
            "rejection_reasons": list(self.rejection_reasons(pair)),
            "stable_frames": self.stable_frames,
            "stable_frames_required": self.stable_frames_required,
            "primary_motion": self.primary_motion,
            "side_motion": self.side_motion,
            "motion_threshold": self.motion_threshold,
            "valid_depth_ratio": self.valid_depth_ratio,
            "minimum_valid_depth_ratio": self.minimum_valid_depth_ratio,
            "primary_rgb_depth_sync_ms": pair.primary.sync_delta_ms,
            "camera_pair_delta_ms": pair.pair_delta_ms,
            "maximum_pair_delta_ms": self.maximum_pair_delta_ms,
            "side_blur_variance": self.side_blur_variance,
            "minimum_side_blur_variance": self.minimum_side_blur_variance,
        }


def render_dual_capture_assistant(
    pair: SynchronizedFramePair,
    state: CaptureAssistantState,
    quality: DualCaptureQualityTracker,
    message: str = "",
) -> np.ndarray:
    """Render top RGB, aligned depth and side RGB in one capture window."""
    view_width, view_height = 480, 270
    primary = cv2.resize(pair.primary.color_bgr, (view_width, view_height))
    depth = cv2.resize(
        depth_preview(pair.primary.depth_mm),
        (view_width, view_height),
        interpolation=cv2.INTER_NEAREST,
    )
    side = (
        np.zeros_like(primary)
        if pair.side is None
        else cv2.resize(pair.side.color_bgr, (view_width, view_height))
    )
    for image, title in (
        (primary, "TOP RGB"),
        (depth, "TOP DEPTH"),
        (side, "SIDE RGB"),
    ):
        cv2.rectangle(image, (0, 0), (view_width, 28), (0, 0, 0), -1)
        cv2.putText(
            image, title, (9, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    views = np.hstack((primary, depth, side))
    panel = np.full((220, views.shape[1], 3), 28, np.uint8)
    canvas = np.vstack((views, panel))
    ready = quality.ready(pair)
    label_id, _ = state.current
    pair_delta = pair.pair_delta_ms
    lines = [
        f"{'READY' if ready else 'WAIT'} | label [{state.selected_index}] {label_id} | saved {state.count()}/{state.target_per_label}",
        (
            f"depth {quality.valid_depth_ratio:.1%} | RGB-D {pair.primary.sync_delta_ms:.1f} ms | "
            f"pair {'missing' if pair_delta is None else f'{pair_delta:.1f} ms'}"
        ),
        (
            f"motion top/side {quality.primary_motion:.2f}/{quality.side_motion:.2f} | "
            f"side blur {quality.side_blur_variance:.1f}"
        ),
        "0-9 label | SPACE save | F force | N/P label | A next | Q quit",
    ]
    if message:
        lines.append(message[:110])
    colour = (40, 220, 40) if ready else (0, 190, 255)
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (14, view_height + 32 + index * 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            colour if index == 0 else (235, 235, 235),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
    reasons = ", ".join(quality.rejection_reasons(pair))
    if reasons:
        cv2.putText(
            canvas,
            reasons[:145],
            (14, view_height + 188),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (0, 120, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def save_dual_capture_sample(
    pair: SynchronizedFramePair,
    dataset_root: str | Path,
    batch_id: str,
    platform_id: str,
    label_id: str,
    calibration_hash: str,
    quality: DualCaptureQualityTracker,
    forced: bool = False,
) -> Path:
    root = Path(dataset_root)
    label_dir = root / platform_id / batch_id / label_id
    label_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(label_dir.glob("sample-*"))
    sequence = len(existing) + 1
    while True:
        target = label_dir / f"sample-{sequence:04d}"
        if not target.exists():
            break
        sequence += 1
    save_synchronized_pair(
        pair, target, platform_id, calibration_hash, label_id, reviewed=False
    )
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "batch_id": batch_id,
            "capture_quality": quality.to_dict(pair),
            "quality_override": bool(forced),
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_item = {
        "sample": target.relative_to(root).as_posix(),
        "batch_id": batch_id,
        "platform_id": platform_id,
        "label_id": label_id,
        "primary_frame_id": pair.primary.frame_id,
        "side_frame_id": None if pair.side is None else pair.side.frame_id,
        "pair_delta_ms": pair.pair_delta_ms,
        "quality_override": bool(forced),
        "human_reviewed": False,
    }
    with (root / "dual-manifest.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")
    return target
