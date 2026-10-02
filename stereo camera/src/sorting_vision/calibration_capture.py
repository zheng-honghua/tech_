"""Capture checks and non-destructive publishing shared by calibration tools."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def prepare_calibration_session(path: str | Path) -> Path:
    """Keep old observations intact when a live session is started again."""
    target = Path(path)
    if target.exists() and any(target.iterdir()):
        target = target / (time.strftime("run-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8])
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_calibration_image(path: str | Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"cannot save calibration image: {path}")


class CalibrationCaptureTracker:
    """Require sharp, fully visible points to remain still over several frames."""

    def __init__(self, stable_frames: int = 3, max_motion_px: float = 1.5,
                 min_sharpness: float = 50.0, border_px: float = 8.0) -> None:
        if stable_frames < 1 or not np.isfinite(max_motion_px) or max_motion_px <= 0:
            raise ValueError("stable frames must be positive and motion limit finite/positive")
        if not np.isfinite(min_sharpness) or min_sharpness < 0:
            raise ValueError("minimum sharpness must be finite and non-negative")
        if not np.isfinite(border_px) or border_px < 0:
            raise ValueError("border margin must be finite and non-negative")
        self.required = stable_frames
        self.max_motion_px = max_motion_px
        self.min_sharpness = min_sharpness
        self.border_px = border_px
        self.previous: np.ndarray | None = None
        self.stable = 0

    def update(self, image: np.ndarray, points: np.ndarray) -> dict[str, Any]:
        values = np.asarray(points, np.float64).reshape(-1, 2)
        height, width = image.shape[:2]
        reasons: list[str] = []
        sharpness = motion = None
        if len(values) < 4 or not np.all(np.isfinite(values)):
            reasons.append("target_missing")
            self.previous = None
            self.stable = 0
        else:
            low, high = values.min(axis=0), values.max(axis=0)
            if np.any(low < self.border_px) or np.any(high >= [width - self.border_px, height - self.border_px]):
                reasons.append("target_near_image_border")
            x0, y0 = np.maximum(np.floor(low).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(high).astype(int) + 1, [width, height])
            roi = image[y0:y1, x0:x1]
            if roi.size:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if sharpness is None or sharpness < self.min_sharpness:
                reasons.append("target_blurred")
            if self.previous is not None and self.previous.shape == values.shape:
                motion = float(np.max(np.linalg.norm(values - self.previous, axis=1)))
            self.stable = self.stable + 1 if motion is not None and motion <= self.max_motion_px else 1
            self.previous = values.copy()
            if reasons:
                self.stable = 0
            if self.stable < self.required:
                reasons.append("target_not_stable")
        return {"ready": not reasons, "reasons": reasons, "sharpness": sharpness,
                "motion_px": motion, "stable_frames": self.stable,
                "required_stable_frames": self.required}


def publish_calibration(calibration: Any, output: str | Path, session: str | Path) -> dict[str, Any]:
    """Save diagnostics for every solve; publish only passing candidates atomically."""
    directory = Path(session)
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / "calibration-summary.json"
    destination = Path(output)
    if candidate.resolve() == destination.resolve():
        raise ValueError("output must differ from the session diagnostic file")
    calibration.save(candidate)
    backup: Path | None = None
    if calibration.valid:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            backup = destination.with_name(destination.name + ".backup-" +
                                           time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8])
            backup.write_bytes(destination.read_bytes())
        temporary = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex[:8])
        try:
            calibration.save(temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    result = {"published": bool(calibration.valid), "requested_output": str(destination.resolve()),
              "diagnostic_file": str(candidate.resolve()),
              "backup_file": None if backup is None else str(backup.resolve())}
    (directory / "publish-status.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
