from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale_to_mm: float = 1.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("intrinsic image dimensions must be positive")
        if self.fx <= 0 or self.fy <= 0 or self.depth_scale_to_mm <= 0:
            raise ValueError("focal lengths and depth scale must be positive")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "depth_scale_to_mm": self.depth_scale_to_mm,
        }


@dataclass(frozen=True)
class Plane:
    normal: np.ndarray
    offset: float
    rmse_mm: float = 0.0

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=np.float64)
        if normal.shape != (3,):
            raise ValueError("plane normal must contain three values")
        length = float(np.linalg.norm(normal))
        if length < 1e-9:
            raise ValueError("plane normal cannot be zero")
        normal = normal / length
        offset = float(self.offset) / length
        # For a fixed overhead camera the tray-facing normal points to the camera.
        if normal[2] > 0:
            normal = -normal
            offset = -offset
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "offset", offset)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points) @ self.normal + self.offset

    def to_dict(self) -> dict[str, object]:
        return {
            "normal": self.normal.tolist(),
            "offset": self.offset,
            "rmse_mm": self.rmse_mm,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "Plane":
        return cls(
            np.asarray(value["normal"], dtype=np.float64),
            float(value["offset"]),
            float(value.get("rmse_mm", 0.0)),
        )


@dataclass(frozen=True)
class RGBDCalibration:
    intrinsics: CameraIntrinsics
    camera_to_robot: np.ndarray
    tray_plane_camera: Plane

    def __post_init__(self) -> None:
        transform = np.asarray(self.camera_to_robot, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("camera_to_robot must be a 4x4 matrix")
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError("camera_to_robot has an invalid homogeneous last row")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
            raise ValueError("camera_to_robot rotation must be orthonormal")
        if np.linalg.det(rotation) < 0.99:
            raise ValueError("camera_to_robot rotation must be right-handed")
        object.__setattr__(self, "camera_to_robot", transform)

    def transform_points(self, points_camera: np.ndarray) -> np.ndarray:
        points = np.asarray(points_camera, dtype=np.float64)
        return points @ self.camera_to_robot[:3, :3].T + self.camera_to_robot[:3, 3]

    def transform_vectors(self, vectors_camera: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors_camera, dtype=np.float64)
        return vectors @ self.camera_to_robot[:3, :3].T

    def to_dict(self) -> dict[str, object]:
        return {
            "intrinsics": self.intrinsics.to_dict(),
            "camera_to_robot": self.camera_to_robot.tolist(),
            "tray_plane_camera": self.tray_plane_camera.to_dict(),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "RGBDCalibration":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            CameraIntrinsics(**value["intrinsics"]),
            np.asarray(value["camera_to_robot"], dtype=np.float64),
            Plane.from_dict(value["tray_plane_camera"]),
        )


@dataclass(frozen=True)
class RGBDFrame:
    color_bgr: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    timestamp_ns: int
    frame_id: str
    color_timestamp_ns: int | None = None
    depth_timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        color = np.asarray(self.color_bgr)
        depth = np.asarray(self.depth)
        expected = (self.intrinsics.height, self.intrinsics.width)
        if color.shape != (*expected, 3):
            raise ValueError(f"colour shape must be {(*expected, 3)}, got {color.shape}")
        if depth.shape != expected:
            raise ValueError(f"depth shape must be {expected}, got {depth.shape}")
        if not np.issubdtype(depth.dtype, np.number):
            raise ValueError("depth must be numeric")

    @property
    def depth_mm(self) -> np.ndarray:
        return self.depth.astype(np.float32) * self.intrinsics.depth_scale_to_mm

    @property
    def sync_delta_ms(self) -> float:
        color_time = (
            self.timestamp_ns
            if self.color_timestamp_ns is None
            else self.color_timestamp_ns
        )
        depth_time = (
            self.timestamp_ns
            if self.depth_timestamp_ns is None
            else self.depth_timestamp_ns
        )
        return abs(color_time - depth_time) / 1_000_000.0


def resize_rgbd_frame(frame: RGBDFrame, scale: float) -> RGBDFrame:
    """Resize aligned colour/depth data and keep camera intrinsics consistent."""
    factor = float(scale)
    if not 0 < factor <= 1.0:
        raise ValueError("RGB-D scale must be in the interval (0, 1]")
    if np.isclose(factor, 1.0):
        return frame
    width = max(1, int(round(frame.intrinsics.width * factor)))
    height = max(1, int(round(frame.intrinsics.height * factor)))
    intrinsics = CameraIntrinsics(
        width, height,
        frame.intrinsics.fx * factor,
        frame.intrinsics.fy * factor,
        (frame.intrinsics.cx + 0.5) * factor - 0.5,
        (frame.intrinsics.cy + 0.5) * factor - 0.5,
        frame.intrinsics.depth_scale_to_mm,
    )
    return RGBDFrame(
        cv2.resize(frame.color_bgr, (width, height), interpolation=cv2.INTER_AREA),
        cv2.resize(frame.depth, (width, height), interpolation=cv2.INTER_NEAREST),
        intrinsics, frame.timestamp_ns, frame.frame_id,
        frame.color_timestamp_ns, frame.depth_timestamp_ns,
    )


def backproject_pixels(
    pixels_uv: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    pixels = np.asarray(pixels_uv, dtype=np.float64)
    depth = np.asarray(depth_mm, dtype=np.float64).reshape(-1)
    if pixels.shape != (len(depth), 2):
        raise ValueError("pixels must have shape (N, 2) matching depth")
    x = (pixels[:, 0] - intrinsics.cx) * depth / intrinsics.fx
    y = (pixels[:, 1] - intrinsics.cy) * depth / intrinsics.fy
    return np.column_stack((x, y, depth))


def project_points(points_camera: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    z = points[:, 2]
    if np.any(z <= 0):
        raise ValueError("all projected points must have positive camera Z")
    u = points[:, 0] * intrinsics.fx / z + intrinsics.cx
    v = points[:, 1] * intrinsics.fy / z + intrinsics.cy
    return np.column_stack((u, v))


def depth_to_points(
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: np.ndarray | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_mm, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= np.asarray(mask) > 0
    rows, columns = np.nonzero(valid)
    if stride > 1:
        rows, columns = rows[::stride], columns[::stride]
    values = depth[rows, columns]
    pixels = np.column_stack((columns, rows)).astype(np.float64)
    return backproject_pixels(pixels, values, intrinsics), pixels


def fit_plane_svd(points: np.ndarray) -> Plane:
    values = np.asarray(points, dtype=np.float64)
    if len(values) < 3:
        raise ValueError("at least three points are required to fit a plane")
    center = np.mean(values, axis=0)
    _, _, vectors = np.linalg.svd(values - center, full_matrices=False)
    normal = vectors[-1]
    offset = -float(normal @ center)
    distances = values @ normal + offset
    rmse = float(np.sqrt(np.mean(distances * distances)))
    return Plane(normal, offset, rmse)


def fit_plane_ransac(
    points: np.ndarray,
    threshold_mm: float = 1.5,
    iterations: int = 80,
    seed: int = 0,
) -> Plane:
    values = np.asarray(points, dtype=np.float64)
    if len(values) < 3:
        raise ValueError("at least three valid depth points are required")
    generator = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_count = 0
    for _ in range(iterations):
        sample = values[generator.choice(len(values), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        offset = -normal @ sample[0]
        inliers = np.abs(values @ normal + offset) <= threshold_mm
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_mask = inliers
    if best_mask is None or best_count < 3:
        return fit_plane_svd(values)
    return fit_plane_svd(values[best_mask])
