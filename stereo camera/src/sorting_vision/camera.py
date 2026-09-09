from __future__ import annotations

import json
import threading
import time
from collections import deque
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
    source_timestamp_ns: int | None = None

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


@dataclass(frozen=True)
class SynchronizedFramePair:
    """A software-paired RGB-D primary frame and optional side RGB frame.

    Camera timestamps remain on each frame.  The host monotonic timestamps are
    recorded separately because device timestamp domains are not comparable.
    """

    primary: RGBDFrame
    side: RGBFrame | None
    primary_host_timestamp_ns: int
    side_host_timestamp_ns: int | None
    max_pair_delta_ms: float = 50.0
    side_error: str | None = None

    @property
    def pair_delta_ms(self) -> float | None:
        if self.side_host_timestamp_ns is None:
            return None
        return abs(
            self.primary_host_timestamp_ns - self.side_host_timestamp_ns
        ) / 1_000_000.0

    @property
    def synchronized(self) -> bool:
        delta = self.pair_delta_ms
        return self.side is not None and delta is not None and delta <= self.max_pair_delta_ms

    @property
    def color_bgr(self) -> np.ndarray:
        """Compatibility view used by preview-only callers."""
        return self.primary.color_bgr


@dataclass(frozen=True)
class _TimedFrame:
    frame: RGBDFrame | RGBFrame
    host_timestamp_ns: int


class DualCameraSource:
    """Continuously read two cameras and pair frames in host monotonic time."""

    def __init__(
        self,
        primary_source: RGBDSource,
        side_source: ColorSource,
        max_pair_delta_ms: float = 50.0,
        pair_timeout_ms: float = 250.0,
        buffer_size: int = 12,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if max_pair_delta_ms <= 0 or pair_timeout_ms <= 0:
            raise ValueError("dual-camera timing limits must be positive")
        self.primary_source = primary_source
        self.side_source = side_source
        self.max_pair_delta_ms = float(max_pair_delta_ms)
        self.pair_timeout_ms = float(pair_timeout_ms)
        self._clock = monotonic_ns
        self._primary: deque[_TimedFrame] = deque(maxlen=max(2, buffer_size))
        self._side: deque[_TimedFrame] = deque(maxlen=max(2, buffer_size))
        self._errors: dict[str, str] = {}
        self._condition = threading.Condition()
        self._closed = False
        self._threads = [
            threading.Thread(
                target=self._reader, args=("primary", primary_source, self._primary),
                daemon=True, name="dual-primary-camera",
            ),
            threading.Thread(
                target=self._reader, args=("side", side_source, self._side),
                daemon=True, name="dual-side-camera",
            ),
        ]
        for thread in self._threads:
            thread.start()

    def _reader(self, name: str, source: object, target: deque[_TimedFrame]) -> None:
        while True:
            with self._condition:
                if self._closed:
                    return
            try:
                frame = source.read()  # type: ignore[attr-defined]
                timed = _TimedFrame(frame, self._clock())
                with self._condition:
                    target.append(timed)
                    self._errors.pop(name, None)
                    self._condition.notify_all()
            except Exception as error:
                with self._condition:
                    self._errors[name] = f"{type(error).__name__}: {error}"
                    self._condition.notify_all()
                if name == "primary":
                    return
                # A disconnected side camera must not busy-loop or stop primary use.
                if self._closed:
                    return
                time.sleep(0.02)

    def read(self) -> SynchronizedFramePair:
        deadline_ns = self._clock() + int(self.pair_timeout_ms * 1_000_000)
        with self._condition:
            while not self._primary and not self._closed:
                if "primary" in self._errors:
                    raise RuntimeError(self._errors["primary"])
                remaining = (deadline_ns - self._clock()) / 1_000_000_000.0
                if remaining <= 0:
                    raise TimeoutError("primary camera frame timed out")
                self._condition.wait(min(remaining, 0.05))
            if self._closed:
                raise RuntimeError("dual camera source is closed")
            primary = self._primary.popleft()
            while not self._side and "side" not in self._errors:
                remaining = (deadline_ns - self._clock()) / 1_000_000_000.0
                if remaining <= 0:
                    break
                self._condition.wait(min(remaining, 0.02))
            if not self._side:
                return SynchronizedFramePair(
                    primary.frame, None, primary.host_timestamp_ns, None,
                    self.max_pair_delta_ms, self._errors.get("side"),
                )  # type: ignore[arg-type]
            side_index = min(
                range(len(self._side)),
                key=lambda index: abs(
                    self._side[index].host_timestamp_ns - primary.host_timestamp_ns
                ),
            )
            side = self._side[side_index]
            for _ in range(side_index + 1):
                self._side.popleft()
            return SynchronizedFramePair(
                primary.frame, side.frame, primary.host_timestamp_ns,
                side.host_timestamp_ns, self.max_pair_delta_ms,
                self._errors.get("side"),
            )  # type: ignore[arg-type]

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        self.primary_source.close()
        self.side_source.close()
        for thread in self._threads:
            thread.join(timeout=0.5)

    def __enter__(self) -> "DualCameraSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


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
        auto_exposure: bool | None = None,
        exposure: float | None = None,
        auto_white_balance: bool | None = None,
        white_balance: float | None = None,
        autofocus: bool | None = None,
        focus: float | None = None,
        capture_factory: Callable[[int], Any] = cv2.VideoCapture,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = max(0, warmup_frames)
        self.reconnect_attempts = max(0, reconnect_attempts)
        self.auto_exposure = auto_exposure
        self.exposure = exposure
        self.auto_white_balance = auto_white_balance
        self.white_balance = white_balance
        self.autofocus = autofocus
        self.focus = focus
        self.control_status: dict[str, bool] = {}
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
        controls = (
            (
                "auto_exposure",
                cv2.CAP_PROP_AUTO_EXPOSURE,
                None if self.auto_exposure is None else (0.75 if self.auto_exposure else 0.25),
            ),
            ("exposure", cv2.CAP_PROP_EXPOSURE, self.exposure),
            (
                "auto_white_balance",
                cv2.CAP_PROP_AUTO_WB,
                None if self.auto_white_balance is None else float(self.auto_white_balance),
            ),
            ("white_balance", cv2.CAP_PROP_WB_TEMPERATURE, self.white_balance),
            (
                "autofocus",
                cv2.CAP_PROP_AUTOFOCUS,
                None if self.autofocus is None else float(self.autofocus),
            ),
            ("focus", cv2.CAP_PROP_FOCUS, self.focus),
        )
        self.control_status = {
            name: bool(capture.set(property_id, float(value)))
            for name, property_id, value in controls
            if value is not None
        }
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
                return RGBFrame(
                    image, timestamp, f"uvc-{self._frame_number:09d}", timestamp
                )
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


class RealSenseSource:
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
        camera_model: str = "Intel RealSense D435if",
        frame_prefix: str = "d435if",
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
        self.camera_model = str(camera_model)
        self.frame_prefix = str(frame_prefix).strip()
        if not self.frame_prefix:
            raise ValueError("RealSense frame prefix must not be empty")
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
        try:
            profile = self._pipeline.start(configuration)
        except RuntimeError as error:
            raise RuntimeError(
                "RealSense cannot start the requested streams: "
                f"color={self.color_width}x{self.color_height}@{self.fps}, "
                f"depth={self.depth_width}x{self.depth_height}@{self.fps}. "
                "Choose a color/depth combination available in RealSense Viewer. "
                f"Driver message: {error}"
            ) from error
        self._align = rs_module.align(rs_module.stream.color)
        device = profile.get_device()
        self._depth_scale_mm = float(
            device.first_depth_sensor().get_depth_scale()
        ) * 1000.0
        camera_info = getattr(rs_module, "camera_info", None)
        if camera_info is not None and hasattr(camera_info, "name"):
            try:
                detected_model = str(device.get_info(camera_info.name)).strip()
            except (AttributeError, RuntimeError):
                detected_model = ""
            if detected_model:
                self.camera_model = detected_model
        self._closed = False

    def capture_metadata(self) -> dict[str, object]:
        return {
            "camera_model": self.camera_model,
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
            f"{self.frame_prefix}-{frame_number:09d}",
            color_ns,
            depth_ns,
        )

    def close(self) -> None:
        if not self._closed:
            self._pipeline.stop()
            self._closed = True

    def __enter__(self) -> "RealSenseSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class RealSenseD435IFSource(RealSenseSource):
    """D435if identity defaults with the generic RealSense implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("camera_model", "Intel RealSense D435if")
        kwargs.setdefault("frame_prefix", "d435if")
        super().__init__(*args, **kwargs)


class RealSenseD415Source(RealSenseSource):
    """Backward-compatible D415 source for replay tools and external imports."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("camera_model", "Intel RealSense D415")
        kwargs.setdefault("frame_prefix", "d415")
        super().__init__(*args, **kwargs)


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
