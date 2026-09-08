from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SyntheticObject:
    color_id: str
    shape_id: str
    center_px: tuple[int, int]
    size_px: tuple[int, int] = (90, 90)
    angle_deg: float = 0.0


DEFAULT_BGR = {
    "red": (54, 59, 216),
    "yellow": (63, 196, 226),
    "blue": (198, 103, 62),
    "green": (85, 158, 75),
    "black": (37, 37, 37),
    "cyan": (208, 191, 89),
}


def blank_tray(width: int = 800, height: int = 800) -> np.ndarray:
    image = np.full((height, width, 3), 238, dtype=np.uint8)
    cv2.rectangle(image, (3, 3), (width - 4, height - 4), (225, 225, 225), 5)
    return image


def _regular_polygon(
    center: tuple[int, int], radius: float, vertices: int, rotation_deg: float
) -> np.ndarray:
    offset = np.deg2rad(rotation_deg - 90.0)
    points = [
        [
            center[0] + radius * np.cos(offset + index * 2 * np.pi / vertices),
            center[1] + radius * np.sin(offset + index * 2 * np.pi / vertices),
        ]
        for index in range(vertices)
    ]
    return np.round(points).astype(np.int32)


def draw_object(image: np.ndarray, item: SyntheticObject) -> None:
    colour = DEFAULT_BGR[item.color_id]
    center = item.center_px
    width, height = item.size_px
    if item.shape_id == "circle":
        cv2.circle(image, center, min(width, height) // 2, colour, -1, cv2.LINE_AA)
        return
    if item.shape_id in {"square", "rectangle"}:
        rectangle = (center, (width, height), item.angle_deg)
        points = np.int32(np.round(cv2.boxPoints(rectangle)))
    else:
        vertices = {"triangle": 3, "pentagon": 5, "hexagon": 6}[item.shape_id]
        points = _regular_polygon(center, min(width, height) / 2, vertices, item.angle_deg)
    cv2.fillConvexPoly(image, points, colour, cv2.LINE_AA)


def make_scene(
    objects: list[SyntheticObject], width: int = 800, height: int = 800
) -> tuple[np.ndarray, np.ndarray]:
    background = blank_tray(width, height)
    scene = background.copy()
    for item in objects:
        draw_object(scene, item)
    return background, scene


def competition_demo_scene() -> tuple[np.ndarray, np.ndarray, list[SyntheticObject]]:
    objects = [
        SyntheticObject("red", "triangle", (140, 150), (95, 95), 12),
        SyntheticObject("yellow", "square", (390, 145), (90, 90), 22),
        SyntheticObject("blue", "rectangle", (645, 155), (125, 75), -18),
        SyntheticObject("green", "pentagon", (150, 410), (100, 100), 8),
        SyntheticObject("black", "hexagon", (400, 410), (105, 105), 15),
        SyntheticObject("cyan", "circle", (650, 410), (100, 100), 0),
        SyntheticObject("red", "triangle", (160, 665), (95, 95), -20),
        SyntheticObject("yellow", "square", (400, 660), (90, 90), -10),
        SyntheticObject("blue", "rectangle", (650, 665), (125, 75), 25),
    ]
    background, scene = make_scene(objects)
    return background, scene, objects

