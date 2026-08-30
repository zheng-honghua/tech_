from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from .rgbd import CameraIntrinsics, RGBDFrame


@dataclass(frozen=True)
class RGBFrame:
    color_bgr: np.ndarray
    timestamp_ns: int
    frame_id: str

    def __post_init__(self) -> None:
        image = np.asarray(self.color_bgr)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("RGB frame must be a BGR image with three channels")


class ColorSource(Protocol):
    def read(self) -> RGBFrame: ...

    def close(self) -> None: ...


class RGBDSource(Protocol):
    def read(self) -> RGBDFrame: ...

    def close(self) -> None: ...


class OpenCVCameraSource:
    """USB/UVC colour camera with warm-up and bounded reconnect attempts."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        warmup_frames: int = 30,
        reconnect_attempts: int = 3,
        capture_factory: Callable[[int], Any] = cv2.VideoCapture,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = max(0, warmup_frames)
        self.reconnect_attempts = max(0, reconnect_attempts)
        self._capture_factory = capture_factory
        self._capture: Any | None = None
        self._frame_number = 0
        self._open()

    def _open(self) -> None:
        self.close()
        capture = self._capture_factory(self.camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"cannot open UVC camera index {self.camera_index}")
        self._capture = capture
        for _ in range(self.warmup_frames):
            ok, _ = capture.read()
            if not ok:
                break

    def read(self) -> RGBFrame:
        last_error = "camera read failed"
        for attempt in range(self.reconnect_attempts + 1):
            if self._capture is None:
                self._open()
            ok, image = self._capture.read()
            if ok and image is not None:
                self._frame_number += 1
                timestamp = time.time_ns()
                return RGBFrame(image, timestamp, f"uvc-{self._frame_number:09d}")
            last_error = f"UVC camera read failed (attempt {attempt + 1})"
            if attempt < self.reconnect_attempts:
                self._open()
        raise RuntimeError(last_error)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "OpenCVCameraSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class RealSenseD415Source:
    """Aligned RealSense colour/depth source; pyrealsense2 is loaded lazily."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        depth_width: int | None = None,
        depth_height: int | None = None,
        color_width: int | None = None,
        color_height: int | None = None,
        rs_module: Any | None = None,
    ) -> None:
        if rs_module is None:
            try:
                import pyrealsense2 as rs_module  # type: ignore[import-not-found]
            except ImportError as error:
                raise RuntimeError(
                    "RealSense support is not installed; install sorting-vision[realsense]"
                ) from error
        self._rs = rs_module
        self.depth_width = int(depth_width or width)
        self.depth_height = int(depth_height or height)
        self.color_width = int(color_width or width)
        self.color_height = int(color_height or height)
        self.fps = int(fps)
        self._pipeline = rs_module.pipeline()
        configuration = rs_module.config()
        configuration.enable_stream(
            rs_module.stream.depth,
            self.depth_width,
            self.depth_height,
            rs_module.format.z16,
            self.fps,
        )
        configuration.enable_stream(
            rs_module.stream.color,
            self.color_width,
            self.color_height,
            rs_module.format.bgr8,
            self.fps,
        )
        profile = self._pipeline.start(configuration)
        self._align = rs_module.align(rs_module.stream.color)
        self._depth_scale_mm = (
            float(profile.get_device().first_depth_sensor().get_depth_scale()) * 1000.0
        )
        self._closed = False

    def capture_metadata(self) -> dict[str, object]:
        return {
            "camera_model": "Intel RealSense D415",
            "depth_stream": {
                "width": self.depth_width,
                "height": self.depth_height,
                "fps": self.fps,
                "format": "Z16",
            },
            "color_stream": {
                "width": self.color_width,
                "height": self.color_height,
                "fps": self.fps,
                "format": "BGR8",
            },
            "alignment": "depth_to_color",
            "depth_scale_to_mm": self._depth_scale_mm,
        }

    def read(self) -> RGBDFrame:
        if self._closed:
            raise RuntimeError("RealSense source is closed")
        frames = self._align.process(self._pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("RealSense returned an incomplete aligned frame set")
        depth = np.asanyarray(depth_frame.get_data())
        color = np.asanyarray(color_frame.get_data())
        native = color_frame.profile.as_video_stream_profile().intrinsics
        intrinsics = CameraIntrinsics(
            int(native.width),
            int(native.height),
            float(native.fx),
            float(native.fy),
            float(native.ppx),
            float(native.ppy),
            self._depth_scale_mm,
        )
        color_ns = int(round(float(color_frame.get_timestamp()) * 1_000_000.0))
        depth_ns = int(round(float(depth_frame.get_timestamp()) * 1_000_000.0))
        timestamp_ns = max(color_ns, depth_ns)
        frame_number = int(color_frame.get_frame_number())
        return RGBDFrame(
            color,
            depth,
            intrinsics,
            timestamp_ns,
            f"d415-{frame_number:09d}",
            color_ns,
            depth_ns,
        )

    def close(self) -> None:
        if not self._closed:
            self._pipeline.stop()
            self._closed = True

    def __enter__(self) -> "RealSenseD415Source":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


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

    def close(self) -> None:
        return None
