from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .rgbd import CameraIntrinsics


@dataclass(frozen=True)
class CameraCalibration:
    """RGB camera intrinsics produced independently from stereo geometry."""

    camera_id: str
    intrinsics: CameraIntrinsics
    distortion: np.ndarray
    rms_px: float
    p95_px: float
    coverage_ratio: float
    maximum_tilt_deg: float
    tilt_span_deg: float
    frame_count: int
    target: dict[str, Any]
    version: str

    def __post_init__(self) -> None:
        distortion = np.asarray(self.distortion, np.float64).reshape(-1)
        values = (
            self.rms_px,
            self.p95_px,
            self.coverage_ratio,
            self.maximum_tilt_deg,
            self.tilt_span_deg,
        )
        if not self.camera_id.strip() or not self.version.strip():
            raise ValueError("camera calibration ID and version are required")
        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError("camera calibration metrics must be finite and non-negative")
        if not np.all(np.isfinite(distortion)):
            raise ValueError("camera distortion must be finite")
        if self.frame_count <= 0:
            raise ValueError("camera calibration frame count must be positive")
        object.__setattr__(self, "distortion", distortion)

    @property
    def rejection_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.frame_count < 20:
            reasons.append("fewer_than_20_frames")
        if self.rms_px > 0.8:
            reasons.append("rms_above_0.8_px")
        if self.p95_px > 2.0:
            reasons.append("p95_above_2_px")
        if self.coverage_ratio < 0.35:
            reasons.append("image_coverage_below_35_percent")
        if self.maximum_tilt_deg < 20.0:
            reasons.append("maximum_tilt_below_20_deg")
        if self.tilt_span_deg < 12.0:
            reasons.append("tilt_span_below_12_deg")
        width = self.intrinsics.width
        height = self.intrinsics.height
        if not 0.2 * width <= self.intrinsics.cx <= 0.8 * width:
            reasons.append("principal_point_x_implausible")
        if not 0.2 * height <= self.intrinsics.cy <= 0.8 * height:
            reasons.append("principal_point_y_implausible")
        focal_ratio = self.intrinsics.fx / self.intrinsics.fy
        if not 0.8 <= focal_ratio <= 1.25:
            reasons.append("focal_aspect_ratio_implausible")
        return reasons

    @property
    def valid(self) -> bool:
        return not self.rejection_reasons

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": 1,
            "camera_id": self.camera_id,
            "version": self.version,
            "image_size": [self.intrinsics.width, self.intrinsics.height],
            "intrinsics": self.intrinsics.to_dict(),
            "distortion": self.distortion.tolist(),
            "metrics": {
                "rms_px": self.rms_px,
                "p95_px": self.p95_px,
                "coverage_ratio": self.coverage_ratio,
                "maximum_tilt_deg": self.maximum_tilt_deg,
                "tilt_span_deg": self.tilt_span_deg,
                "frame_count": self.frame_count,
                "valid": self.valid,
                "rejection_reasons": self.rejection_reasons,
            },
            "target": self.target,
        }
        value["calibration_hash"] = self.content_hash(value)
        return value

    @staticmethod
    def content_hash(value: dict[str, Any]) -> str:
        unsigned = {key: item for key, item in value.items() if key != "calibration_hash"}
        payload = json.dumps(unsigned, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalibration":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = str(value.get("calibration_hash", ""))
        if not expected:
            raise ValueError("camera calibration hash is missing")
        actual = cls.content_hash(value)
        if expected and expected != actual:
            raise ValueError("camera calibration hash mismatch")
        metrics = value["metrics"]
        image_size = tuple(map(int, value.get("image_size", ())))
        intrinsic_size = (
            int(value["intrinsics"]["width"]),
            int(value["intrinsics"]["height"]),
        )
        if image_size and image_size != intrinsic_size:
            raise ValueError("camera calibration image size is inconsistent")
        return cls(
            camera_id=str(value["camera_id"]),
            intrinsics=CameraIntrinsics(**value["intrinsics"]),
            distortion=np.asarray(value["distortion"], np.float64),
            rms_px=float(metrics["rms_px"]),
            p95_px=float(metrics["p95_px"]),
            coverage_ratio=float(metrics["coverage_ratio"]),
            maximum_tilt_deg=float(metrics["maximum_tilt_deg"]),
            tilt_span_deg=float(metrics["tilt_span_deg"]),
            frame_count=int(metrics["frame_count"]),
            target=dict(value["target"]),
            version=str(value["version"]),
        )


def checkerboard_object_points(
    corners_x: int,
    corners_y: int,
    square_size_mm: float,
) -> np.ndarray:
    """Return row-major 3-D points for checkerboard inner corners."""
    if corners_x < 3 or corners_y < 3:
        raise ValueError("checkerboard must contain at least 3x3 inner corners")
    if square_size_mm <= 0:
        raise ValueError("checkerboard square size must be positive")
    points = np.zeros((corners_x * corners_y, 1, 3), np.float32)
    grid = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2)
    points[:, 0, :2] = grid.astype(np.float32) * float(square_size_mm)
    return points


def generate_checkerboard_asset(
    output_dir: str | Path,
    *,
    corners_x: int,
    corners_y: int,
    square_size_mm: float,
    pixels_per_mm: float = 10.0,
) -> tuple[Path, Path, Path]:
    checkerboard_object_points(corners_x, corners_y, square_size_mm)
    if pixels_per_mm <= 0:
        raise ValueError("pixels per millimetre must be positive")
    squares_x = corners_x + 1
    squares_y = corners_y + 1
    board_width_mm = squares_x * square_size_mm
    board_height_mm = squares_y * square_size_mm
    margin_mm = square_size_mm * 0.5
    width_mm = board_width_mm + 2 * margin_mm
    height_mm = board_height_mm + 2 * margin_mm
    image_size = (
        int(round(width_mm * pixels_per_mm)),
        int(round(height_mm * pixels_per_mm)),
    )
    image = np.full((image_size[1], image_size[0]), 255, np.uint8)
    margin_px = int(round(margin_mm * pixels_per_mm))
    square_px = int(round(square_size_mm * pixels_per_mm))
    for row in range(squares_y):
        for column in range(squares_x):
            if (row + column) % 2:
                continue
            x0 = margin_px + column * square_px
            y0 = margin_px + row * square_px
            image[y0 : y0 + square_px, x0 : x0 + square_px] = 0
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    image_path = target / "checkerboard-intrinsics.png"
    svg_path = target / "checkerboard-intrinsics.svg"
    metadata_path = target / "checkerboard-intrinsics.json"
    if not cv2.imwrite(str(image_path), image):
        raise OSError(f"cannot write {image_path}")
    rectangles = []
    for row in range(squares_y):
        for column in range(squares_x):
            if (row + column) % 2 == 0:
                rectangles.append(
                    f'<rect x="{margin_mm + column * square_size_mm}" '
                    f'y="{margin_mm + row * square_size_mm}" '
                    f'width="{square_size_mm}" height="{square_size_mm}" fill="black"/>'
                )
    svg_path.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm}mm" '
                f'height="{height_mm}mm" viewBox="0 0 {width_mm} {height_mm}">',
                f'<rect width="{width_mm}" height="{height_mm}" fill="white"/>',
                *rectangles,
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "type": "checkerboard",
                "inner_corners_x": corners_x,
                "inner_corners_y": corners_y,
                "squares_x": squares_x,
                "squares_y": squares_y,
                "square_size_mm": square_size_mm,
                "board_width_mm": board_width_mm,
                "board_height_mm": board_height_mm,
                "page_width_mm": width_mm,
                "page_height_mm": height_mm,
                "print_scale_percent": 100,
                "measurement_rule": "print the SVG at 100% and measure one square edge",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return image_path, svg_path, metadata_path


def checkerboard_points_from_image(
    image: np.ndarray,
    *,
    corners_x: int,
    corners_y: int,
    square_size_mm: float,
    maximum_detection_width: int | None = 1280,
) -> tuple[np.ndarray, np.ndarray]:
    object_points = checkerboard_object_points(
        corners_x, corners_y, square_size_mm
    )
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
    scale = 1.0
    detection = gray
    if maximum_detection_width and gray.shape[1] > maximum_detection_width:
        scale = maximum_detection_width / float(gray.shape[1])
        detection = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )
    found, corners = cv2.findChessboardCornersSB(
        detection, (corners_x, corners_y), flags=flags
    )
    if not found or corners is None:
        return (
            np.empty((0, 1, 3), np.float32),
            np.empty((0, 1, 2), np.float32),
        )
    image_points = np.asarray(corners, np.float32).reshape(-1, 1, 2) / float(scale)
    if scale < 1.0:
        cv2.cornerSubPix(
            gray,
            image_points,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        )
    return object_points, image_points


def calibration_view_signature(image_points: np.ndarray) -> np.ndarray:
    points = np.asarray(image_points, np.float32).reshape(-1, 2)
    if len(points) < 4:
        raise ValueError("at least four image points are required")
    center, size, angle = cv2.minAreaRect(points)
    area = max(float(size[0] * size[1]), 1.0)
    return np.asarray([center[0], center[1], math.log(area), math.radians(angle)])


def calibrate_camera_intrinsics(
    object_points: Iterable[np.ndarray],
    image_points: Iterable[np.ndarray],
    image_size: tuple[int, int],
    *,
    camera_id: str,
    target: dict[str, Any],
) -> CameraCalibration:
    objects = [np.asarray(value, np.float32).reshape(-1, 1, 3) for value in object_points]
    images = [np.asarray(value, np.float32).reshape(-1, 1, 2) for value in image_points]
    if len(objects) != len(images) or len(objects) < 20:
        raise ValueError("at least 20 RGB calibration views are required")
    if any(len(first) != len(second) or len(first) < 16 for first, second in zip(objects, images)):
        raise ValueError("each calibration view needs at least 16 detected corners")
    active_indices = list(range(len(objects)))
    rejected_indices: list[int] = []
    while True:
        active_objects = [objects[index] for index in active_indices]
        active_images = [images[index] for index in active_indices]
        rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
            active_objects,
            active_images,
            image_size,
            None,
            None,
        )
        per_view_errors: list[float] = []
        errors = []
        for obj, img, rotation, translation in zip(
            active_objects, active_images, rotations, translations
        ):
            projected, _ = cv2.projectPoints(
                obj, rotation, translation, matrix, distortion
            )
            residuals = np.linalg.norm(
                projected.reshape(-1, 2) - img.reshape(-1, 2), axis=1
            )
            errors.extend(residuals)
            per_view_errors.append(float(np.sqrt(np.mean(residuals**2))))
        p95_px = float(np.percentile(errors, 95))
        if (rms <= 0.8 and p95_px <= 2.0) or len(active_indices) <= 20:
            break
        values = np.asarray(per_view_errors, np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_limit = max(1.2, median + 3.0 * 1.4826 * mad)
        worst_local_index = int(np.argmax(values))
        if values[worst_local_index] <= robust_limit:
            break
        rejected_indices.append(active_indices.pop(worst_local_index))

    selected_objects = [objects[index] for index in active_indices]
    selected_images = [images[index] for index in active_indices]
    tilts: list[float] = []
    all_pixels: list[np.ndarray] = []
    for obj, img, rotation, translation in zip(
        selected_objects, selected_images, rotations, translations
    ):
        normal = cv2.Rodrigues(rotation)[0][:, 2]
        tilts.append(
            math.degrees(math.acos(float(np.clip(abs(normal[2]), 0.0, 1.0))))
        )
        all_pixels.append(img.reshape(-1, 2))
    pixels = np.concatenate(all_pixels)
    span = np.maximum(pixels.max(axis=0) - pixels.min(axis=0), 0.0)
    coverage_ratio = float(span[0] * span[1] / (image_size[0] * image_size[1]))
    intrinsics = CameraIntrinsics(
        int(image_size[0]),
        int(image_size[1]),
        float(matrix[0, 0]),
        float(matrix[1, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 2]),
        1.0,
    )
    version_payload = np.concatenate((matrix.reshape(-1), distortion.reshape(-1)))
    version = time.strftime("%Y%m%d-%H%M%S") + "-rgb-" + hashlib.sha256(
        version_payload.tobytes()
    ).hexdigest()[:8]
    target_metadata = dict(target)
    target_metadata["view_selection"] = {
        "method": "whole_view_rms_median_mad",
        "input_frame_count": len(objects),
        "retained_frame_count": len(active_indices),
        "rejected_frame_indices": sorted(rejected_indices),
    }
    return CameraCalibration(
        camera_id=camera_id,
        intrinsics=intrinsics,
        distortion=distortion,
        rms_px=float(rms),
        p95_px=p95_px,
        coverage_ratio=min(1.0, coverage_ratio),
        maximum_tilt_deg=float(max(tilts)),
        tilt_span_deg=float(max(tilts) - min(tilts)),
        frame_count=len(active_indices),
        target=target_metadata,
        version=version,
    )
