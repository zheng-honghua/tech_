from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .rgbd import CameraIntrinsics, RGBDFrame
from .synthetic import DEFAULT_BGR


@dataclass(frozen=True)
class SyntheticSolid:
    color_id: str
    shape_id: str
    center_px: tuple[int, int]
    size_px: tuple[int, int] = (40, 40)
    height_mm: float = 25.0
    yaw_deg: float = 0.0
    tilt_x_deg: float = 0.0
    tilt_y_deg: float = 0.0


def default_intrinsics(width: int = 640, height: int = 480) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=700.0,
        fy=700.0,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        depth_scale_to_mm=1.0,
    )


def _regular_polygon(
    center: tuple[int, int], radius: float, vertices: int, rotation_deg: float
) -> np.ndarray:
    start = np.deg2rad(rotation_deg - 90.0)
    return np.round(
        [
            [
                center[0] + radius * np.cos(start + index * 2 * np.pi / vertices),
                center[1] + radius * np.sin(start + index * 2 * np.pi / vertices),
            ]
            for index in range(vertices)
        ]
    ).astype(np.int32)


def _solid_mask(shape_id: str, solid: SyntheticSolid, image_shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, np.uint8)
    width, height = solid.size_px
    if shape_id in {"cylinder", "sphere"}:
        cv2.ellipse(
            mask,
            solid.center_px,
            (width // 2, height // 2),
            solid.yaw_deg,
            0,
            360,
            255,
            -1,
            cv2.LINE_AA,
        )
    elif shape_id in {"cube", "cuboid"}:
        points = np.int32(
            np.round(cv2.boxPoints((solid.center_px, solid.size_px, solid.yaw_deg)))
        )
        cv2.fillConvexPoly(mask, points, 255, cv2.LINE_AA)
    else:
        vertices = {
            "triangular_prism": 3,
            "pentagonal_prism": 5,
            "hexagonal_prism": 6,
        }[shape_id]
        points = _regular_polygon(
            solid.center_px, min(width, height) / 2.0, vertices, solid.yaw_deg
        )
        cv2.fillConvexPoly(mask, points, 255, cv2.LINE_AA)
    return mask


def make_rgbd_scene(
    solids: list[SyntheticSolid],
    width: int = 640,
    height: int = 480,
    tray_depth_mm: float = 700.0,
    noise_std_mm: float = 0.0,
    invalid_fraction: float = 0.0,
    seed: int = 0,
    frame_id: str = "synthetic",
) -> tuple[RGBDFrame, RGBDFrame]:
    intrinsics = default_intrinsics(width, height)
    background_color = np.full((height, width, 3), 238, np.uint8)
    background_depth = np.full((height, width), tray_depth_mm, np.float32)
    color = background_color.copy()
    depth = background_depth.copy()
    rows, columns = np.indices((height, width), dtype=np.float32)

    for solid in solids:
        mask = _solid_mask(solid.shape_id, solid, (height, width)) > 0
        dx = columns - solid.center_px[0]
        dy = rows - solid.center_px[1]
        millimetres_per_pixel = tray_depth_mm / intrinsics.fx
        if solid.shape_id == "sphere":
            radius_px = min(solid.size_px) / 2.0
            radius_mm = radius_px * millimetres_per_pixel
            radial_sq_mm = (dx * millimetres_per_pixel) ** 2 + (dy * millimetres_per_pixel) ** 2
            surface_height = radius_mm + np.sqrt(
                np.maximum(radius_mm * radius_mm - radial_sq_mm, 0.0)
            )
        else:
            slope_x = np.tan(np.deg2rad(solid.tilt_x_deg))
            slope_y = np.tan(np.deg2rad(solid.tilt_y_deg))
            surface_height = (
                solid.height_mm
                - dx * millimetres_per_pixel * slope_x
                - dy * millimetres_per_pixel * slope_y
            )
            surface_height = np.maximum(surface_height, 4.0)
        surface_depth = tray_depth_mm - surface_height
        visible = mask & (surface_depth < depth)
        depth[visible] = surface_depth[visible]
        color[visible] = DEFAULT_BGR[solid.color_id]

    generator = np.random.default_rng(seed)
    if noise_std_mm > 0:
        depth += generator.normal(0.0, noise_std_mm, depth.shape).astype(np.float32)
    if invalid_fraction > 0:
        invalid = generator.random(depth.shape) < invalid_fraction
        depth[invalid] = 0.0
    timestamp = time.time_ns()
    background = RGBDFrame(
        background_color,
        background_depth,
        intrinsics,
        timestamp,
        "empty-tray",
    )
    scene = RGBDFrame(color, depth, intrinsics, timestamp + 1, frame_id)
    return background, scene


def competition_rgbd_demo() -> tuple[RGBDFrame, RGBDFrame, list[SyntheticSolid]]:
    solids = [
        SyntheticSolid("red", "cube", (100, 100), (46, 46), 28, 18, 5, -3),
        SyntheticSolid("yellow", "cuboid", (245, 100), (64, 38), 24, -22, 4, 2),
        SyntheticSolid("blue", "triangular_prism", (390, 100), (50, 50), 26, 12, 3, 0),
        SyntheticSolid("green", "cylinder", (535, 100), (48, 48), 30, 0, 4, -2),
        SyntheticSolid("black", "pentagonal_prism", (100, 240), (50, 50), 26, 8, 2, 3),
        SyntheticSolid("cyan", "hexagonal_prism", (245, 240), (52, 52), 25, 15, -3, 2),
        SyntheticSolid("red", "sphere", (390, 240), (48, 48), 0, 0, 0, 0),
        SyntheticSolid("yellow", "cube", (535, 240), (46, 46), 28, -12, 6, 0),
        SyntheticSolid("blue", "cuboid", (100, 380), (62, 38), 24, 25, -4, 2),
        SyntheticSolid("green", "triangular_prism", (245, 380), (50, 50), 26, -18, 2, -3),
        SyntheticSolid("black", "cylinder", (390, 380), (48, 48), 30, 0, 3, 2),
        SyntheticSolid("cyan", "hexagonal_prism", (535, 380), (52, 52), 25, -8, 2, 1),
    ]
    background, scene = make_rgbd_scene(solids, frame_id="rgbd-demo")
    return background, scene, solids
