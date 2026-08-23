from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .rgbd import CameraIntrinsics, RGBDFrame


class RGBDSource(Protocol):
    def read(self) -> RGBDFrame: ...


def save_rgbd_frame(frame: RGBDFrame, directory: str | Path) -> None:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target / "color.png"), frame.color_bgr):
        raise OSError(f"failed to write RGB image to {target}")
    np.save(target / "depth.npy", frame.depth)
    metadata = {
        "intrinsics": frame.intrinsics.to_dict(),
        "timestamp_ns": frame.timestamp_ns,
        "frame_id": frame.frame_id,
        "color_timestamp_ns": frame.color_timestamp_ns,
        "depth_timestamp_ns": frame.depth_timestamp_ns,
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_rgbd_frame(directory: str | Path) -> RGBDFrame:
    source = Path(directory)
    color = cv2.imread(str(source / "color.png"), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(f"missing or invalid {source / 'color.png'}")
    depth = np.load(source / "depth.npy", allow_pickle=False)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    return RGBDFrame(
        color,
        depth,
        CameraIntrinsics(**metadata["intrinsics"]),
        int(metadata["timestamp_ns"]),
        str(metadata["frame_id"]),
        metadata.get("color_timestamp_ns"),
        metadata.get("depth_timestamp_ns"),
    )


class FileRGBDSource:
    """Replays one or more recorded RGB-D frame directories."""

    def __init__(self, directories: list[str | Path], loop: bool = False) -> None:
        if not directories:
            raise ValueError("at least one RGB-D frame directory is required")
        self.directories = [Path(path) for path in directories]
        self.loop = loop
        self.index = 0

    def read(self) -> RGBDFrame:
        if self.index >= len(self.directories):
            if not self.loop:
                raise EOFError("RGB-D file source is exhausted")
            self.index = 0
        frame = load_rgbd_frame(self.directories[self.index])
        self.index += 1
        return frame
