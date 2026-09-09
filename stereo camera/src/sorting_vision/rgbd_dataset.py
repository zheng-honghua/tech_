from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera import load_rgbd_frame, save_rgbd_frame
from .geometry_rgb import GEOMETRY_LABELS
from .rgbd import RGBDFrame


EMPTY_TRAY_LABEL = "empty_tray"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_LABEL_ALIASES = {
    folder: label_id for folder, (label_id, _) in GEOMETRY_LABELS.items()
}
_LABEL_NAMES = {
    label_id: label_name for label_id, label_name in GEOMETRY_LABELS.values()
}
_LABEL_ALIASES.update({label_id: label_id for label_id in _LABEL_NAMES})
_LABEL_ALIASES.update({"空托盘": EMPTY_TRAY_LABEL, EMPTY_TRAY_LABEL: EMPTY_TRAY_LABEL})
_LABEL_NAMES[EMPTY_TRAY_LABEL] = "空托盘"


def normalize_rgbd_label(value: str) -> tuple[str, str]:
    label_id = _LABEL_ALIASES.get(value.strip())
    if label_id is None:
        raise ValueError(f"unsupported RGB-D label: {value}")
    return label_id, _LABEL_NAMES[label_id]


def _validate_id(value: str, field: str) -> str:
    result = value.strip()
    if not result or not _SAFE_ID.fullmatch(result):
        raise ValueError(f"{field} must use letters, numbers, underscore or hyphen")
    return result


def depth_preview(depth_mm: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_mm, np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    gray = np.zeros(depth.shape, np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [2, 98])
        gray[valid] = np.clip(
            (high - depth[valid]) * 255.0 / max(float(high - low), 1.0),
            0,
            255,
        ).astype(np.uint8)
    preview = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_rgbd_dataset_sample(
    frame: RGBDFrame,
    dataset_root: str | Path,
    batch_id: str,
    label: str,
    capture_settings: dict[str, object] | None = None,
    sample_id: str | None = None,
) -> Path:
    root = Path(dataset_root)
    batch = _validate_id(batch_id, "batch_id")
    label_id, label_name = normalize_rgbd_label(label)
    if sample_id is None:
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        sample_id = f"{now}_{frame.frame_id}"
    sample = _validate_id(sample_id, "sample_id")
    target = root / batch / label_id / sample
    if target.exists():
        raise FileExistsError(f"RGB-D sample already exists: {target}")
    save_rgbd_frame(frame, target)
    preview = depth_preview(frame.depth_mm)
    if not cv2.imwrite(str(target / "depth-preview.png"), preview):
        raise OSError(f"failed to save depth preview to {target}")
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    valid = np.isfinite(frame.depth_mm) & (frame.depth_mm > 0)
    metadata.update(
        {
            "schema_version": 1,
            "sample_id": sample,
            "batch_id": batch,
            "label_id": label_id,
            "label_name": label_name,
            "depth_storage": "raw_sensor_values_numpy",
            "depth_dtype": str(frame.depth.dtype),
            "depth_scale_to_mm": frame.intrinsics.depth_scale_to_mm,
            "depth_aligned_to_color": True,
            "valid_depth_ratio": float(np.mean(valid)),
            "sync_delta_ms": frame.sync_delta_ms,
            "capture_settings": capture_settings or {},
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    entry = {
        "sample_dir": str(target.relative_to(root)),
        "batch_id": batch,
        "label_id": label_id,
        "label_name": label_name,
        "frame_id": frame.frame_id,
        "timestamp_ns": frame.timestamp_ns,
        "color_sha256": _sha256(target / "color.png"),
        "depth_sha256": _sha256(target / "depth.npy"),
        "valid_depth_ratio": metadata["valid_depth_ratio"],
        "sync_delta_ms": metadata["sync_delta_ms"],
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / "manifest.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return target


def load_rgbd_dataset_entries(dataset_root: str | Path) -> list[dict[str, Any]]:
    root = Path(dataset_root)
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"RGB-D dataset manifest does not exist: {manifest}")
    entries = []
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        value["line_number"] = line_number
        value["absolute_sample_dir"] = root / value["sample_dir"]
        entries.append(value)
    return entries


def audit_rgbd_dataset(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root)
    entries = load_rgbd_dataset_entries(root)
    errors: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    batches: Counter[str] = Counter()
    color_hashes: Counter[str] = Counter()
    valid_ratios: list[float] = []
    sync_deltas: list[float] = []
    for entry in entries:
        sample_dir = Path(entry["absolute_sample_dir"])
        try:
            frame = load_rgbd_frame(sample_dir)
            metadata = json.loads(
                (sample_dir / "metadata.json").read_text(encoding="utf-8")
            )
            if metadata.get("label_id") != entry.get("label_id"):
                raise ValueError("manifest_label_mismatch")
            counts[str(entry["label_id"])] += 1
            batches[str(entry["batch_id"])] += 1
            color_hashes[str(entry.get("color_sha256", ""))] += 1
            valid_ratios.append(float(np.mean(frame.depth_mm > 0)))
            sync_deltas.append(frame.sync_delta_ms)
            if not (sample_dir / "depth-preview.png").is_file():
                errors.append({"sample_dir": str(sample_dir), "reason": "missing_depth_preview"})
        except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            errors.append({"sample_dir": str(sample_dir), "reason": str(error)})
    duplicate_hashes = [
        value for value, count in color_hashes.items() if value and count > 1
    ]
    return {
        "dataset_root": str(root),
        "samples": len(entries),
        "class_counts": dict(sorted(counts.items())),
        "batch_counts": dict(sorted(batches.items())),
        "mean_valid_depth_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
        "max_sync_delta_ms": max(sync_deltas, default=0.0),
        "duplicate_color_hashes": duplicate_hashes,
        "errors": errors,
    }
