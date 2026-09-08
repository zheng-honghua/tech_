from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .camera import save_rgbd_frame
from .capture_assistant import CaptureQualityTracker
from .rgbd import RGBDFrame
from .rgbd_dataset import EMPTY_TRAY_LABEL, depth_preview, normalize_rgbd_label


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_scene_composition(value: str) -> tuple[dict[str, object], ...]:
    """Parse label/count pairs while accepting Chinese or protocol labels."""
    counts: dict[str, int] = {}
    names: dict[str, str] = {}
    for raw_item in re.split(r"[,，;；]", value):
        item = raw_item.strip()
        if not item:
            continue
        parts = re.split(r"[:：=]", item, maxsplit=1)
        label_text = parts[0].strip()
        count = 1 if len(parts) == 1 else int(parts[1].strip())
        label_id, label_name = normalize_rgbd_label(label_text)
        if label_id == EMPTY_TRAY_LABEL:
            raise ValueError("multi-object composition cannot contain empty_tray")
        if count <= 0:
            raise ValueError("object count must be greater than zero")
        counts[label_id] = counts.get(label_id, 0) + count
        names[label_id] = label_name
    if not counts:
        raise ValueError("composition must contain at least one object")
    return tuple(
        {"label_id": label_id, "label_name": names[label_id], "count": count}
        for label_id, count in counts.items()
    )


def composition_summary(composition: Iterable[dict[str, object]]) -> str:
    return ", ".join(
        f"{item['label_id']}x{int(item['count'])}" for item in composition
    )


def _validate_id(value: str, field: str) -> str:
    result = value.strip()
    if not result or not _SAFE_ID.fullmatch(result):
        raise ValueError(f"{field} must use letters, numbers, underscore or hyphen")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_scene_counts(dataset_root: str | Path, batch_id: str) -> dict[int, int]:
    manifest = Path(dataset_root) / "scenes.jsonl"
    counts: dict[int, int] = {}
    if not manifest.is_file():
        return counts
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid scene manifest JSON at line {line_number}: {manifest}"
            ) from error
        if item.get("batch_id") == batch_id:
            index = int(item["scene_index"])
            counts[index] = counts.get(index, 0) + 1
    return counts


def validate_scene_composition(
    dataset_root: str | Path,
    batch_id: str,
    scene_index: int,
    composition: tuple[dict[str, object], ...],
) -> None:
    """Prevent a resumed scene number from silently receiving different labels."""
    manifest = Path(dataset_root) / "scenes.jsonl"
    if not manifest.is_file():
        return
    expected = {
        str(item["label_id"]): int(item["count"]) for item in composition
    }
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("batch_id") != batch_id or int(item["scene_index"]) != scene_index:
            continue
        existing = {
            str(obj["label_id"]): int(obj["count"]) for obj in item["objects"]
        }
        if existing != expected:
            raise ValueError(
                "composition differs from the existing scene; use --scene-index "
                "with a new number"
            )
        return


def resolve_scene_index(counts: dict[int, int], target: int, requested: int) -> int:
    if requested < 0:
        raise ValueError("scene_index must not be negative")
    if requested:
        return requested
    if not counts:
        return 1
    latest = max(counts)
    return latest if counts[latest] < target else latest + 1


@dataclass
class MultiCaptureState:
    batch_id: str
    composition: tuple[dict[str, object], ...]
    captures_per_scene: int = 10
    scene_index: int = 1
    scene_counts: dict[int, int] | None = None
    auto_capture: bool = False

    def __post_init__(self) -> None:
        self.batch_id = _validate_id(self.batch_id, "batch_id")
        if self.captures_per_scene <= 0:
            raise ValueError("captures_per_scene must be greater than zero")
        if self.scene_index <= 0:
            raise ValueError("scene_index must be greater than zero")
        self.scene_counts = dict(self.scene_counts or {})

    @property
    def repeat_index(self) -> int:
        return int(self.scene_counts.get(self.scene_index, 0)) + 1

    @property
    def current_count(self) -> int:
        return int(self.scene_counts.get(self.scene_index, 0))

    @property
    def scene_complete(self) -> bool:
        return self.current_count >= self.captures_per_scene

    @property
    def total_objects(self) -> int:
        return sum(int(item["count"]) for item in self.composition)

    def record_saved(self) -> None:
        self.scene_counts[self.scene_index] = self.current_count + 1

    def next_scene(self) -> None:
        self.scene_index += 1
        self.auto_capture = False


def save_multi_object_sample(
    frame: RGBDFrame,
    dataset_root: str | Path,
    state: MultiCaptureState,
    capture_settings: dict[str, object] | None = None,
) -> Path:
    """Save one aligned RGB-D frame without polluting the single-object manifest."""
    root = Path(dataset_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    sample_id = _validate_id(
        f"capture-{state.repeat_index:04d}_{timestamp}_{frame.frame_id}", "sample_id"
    )
    scene_id = f"scene-{state.scene_index:04d}"
    target = root / state.batch_id / scene_id / sample_id
    if target.exists():
        raise FileExistsError(f"multi-object sample already exists: {target}")
    save_rgbd_frame(frame, target)
    if not cv2.imwrite(str(target / "depth-preview.png"), depth_preview(frame.depth_mm)):
        raise OSError(f"failed to save depth preview to {target}")

    valid = np.isfinite(frame.depth_mm) & (frame.depth_mm > 0)
    scene_metadata = {
        "schema_version": 1,
        "dataset_kind": "multi_object_scene",
        "sample_id": sample_id,
        "batch_id": state.batch_id,
        "scene_id": scene_id,
        "scene_index": state.scene_index,
        "repeat_index": state.repeat_index,
        "objects": list(state.composition),
        "total_objects": state.total_objects,
        "object_annotations": "composition_only",
        "independent_layout": state.repeat_index == 1,
        "valid_depth_ratio": float(np.mean(valid)),
        "sync_delta_ms": frame.sync_delta_ms,
        "capture_settings": capture_settings or {},
    }
    (target / "scene.json").write_text(
        json.dumps(scene_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    entry = {
        "sample_dir": str(target.relative_to(root)),
        **{key: scene_metadata[key] for key in (
            "batch_id", "scene_id", "scene_index", "repeat_index", "objects",
            "total_objects", "independent_layout", "valid_depth_ratio", "sync_delta_ms",
        )},
        "frame_id": frame.frame_id,
        "timestamp_ns": frame.timestamp_ns,
        "color_sha256": _sha256(target / "color.png"),
        "depth_sha256": _sha256(target / "depth.npy"),
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / "scenes.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    state.record_saved()
    return target


def render_multi_capture(
    frame: RGBDFrame,
    state: MultiCaptureState,
    quality: CaptureQualityTracker,
    message: str = "",
) -> np.ndarray:
    height, width = frame.color_bgr.shape[:2]
    scale = min(1.0, 720.0 / width, 440.0 / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    colour = cv2.resize(frame.color_bgr, size, interpolation=cv2.INTER_AREA)
    depth = cv2.resize(depth_preview(frame.depth_mm), size, interpolation=cv2.INTER_NEAREST)
    views = np.hstack((colour, depth))
    panel = np.full((190, views.shape[1], 3), 28, np.uint8)
    canvas = np.vstack((views, panel))
    y0 = views.shape[0]
    status = "READY" if quality.ready else "WAIT"
    mode = "AUTO" if state.auto_capture else "MANUAL"
    lines = [
        f"{status} | {mode} | batch {state.batch_id} | scene {state.scene_index:04d}",
        f"captures {state.current_count}/{state.captures_per_scene} | objects {state.total_objects}",
        composition_summary(state.composition),
        f"depth {quality.valid_depth_ratio:.1%} | motion {quality.motion_score:.2f} | sync {quality.sync_delta_ms:.1f} ms",
        "SPACE save | B auto batch | P pause | N next layout | F force | Q quit",
    ]
    if message:
        lines.append(message[:110])
    for index, line in enumerate(lines):
        cv2.putText(
            canvas, line, (14, y0 + 28 + index * 26), cv2.FONT_HERSHEY_SIMPLEX,
            0.56, (40, 220, 40) if index == 0 and quality.ready else (235, 235, 235),
            2 if index == 0 else 1, cv2.LINE_AA,
        )
    return canvas
