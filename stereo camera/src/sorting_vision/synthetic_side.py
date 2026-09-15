from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .shape_registry import ShapeRegistry


def render_synthetic_side(
    class_id: str, *, seed: int = 0, size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Render a deliberately simple side-view invariant-test sample.

    These images are never valid for calibration, thresholds or reported
    accuracy. They exercise preprocessing and model serialization only.
    """
    rng = np.random.default_rng(seed)
    background = np.full((size, size, 3), 232, np.uint8)
    yy, xx = np.indices((size, size))
    shade = (8.0 * xx / size + 5.0 * yy / size).astype(np.uint8)
    image = np.clip(background.astype(np.int16) - shade[:, :, None], 0, 255).astype(np.uint8)
    mask = np.zeros((size, size), np.uint8)
    colour = tuple(map(int, rng.integers(45, 205, size=3)))
    cx = size // 2 + int(rng.integers(-10, 11))
    top = int(size * 0.22 + rng.integers(-5, 6))
    bottom = int(size * 0.80 + rng.integers(-5, 6))
    half = int(size * 0.23 + rng.integers(-5, 6))
    polygon: np.ndarray
    ridge_segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if class_id.endswith("_prism"):
        count = {"triangular_prism": 3, "quadrangular_prism": 4, "pentagonal_prism": 5, "hexagonal_prism": 6}[class_id]
        skew = int(rng.integers(-12, 13))
        polygon = np.asarray(
            [[cx - half + skew, top], [cx + half + skew, top], [cx + half, bottom], [cx - half, bottom]],
            np.int32,
        )
        for index in range(1, count - 1):
            fraction = index / (count - 1)
            x = int(round(cx - half + fraction * 2 * half))
            ridge_segments.append(((x + skew, top), (x, bottom)))
    elif class_id.endswith("_pyramid"):
        count = {"triangular_pyramid": 3, "square_pyramid": 4, "pentagonal_pyramid": 5, "hexagonal_pyramid": 6}[class_id]
        apex = (cx + int(rng.integers(-10, 11)), top)
        polygon = np.asarray([apex, [cx + half, bottom], [cx - half, bottom]], np.int32)
        for index in range(1, count):
            x = int(round(cx - half + index / count * 2 * half))
            ridge_segments.append((apex, (x, bottom)))
    elif class_id == "octahedron":
        polygon = np.asarray([[cx, top], [cx + half, (top + bottom) // 2], [cx, bottom], [cx - half, (top + bottom) // 2]], np.int32)
        ridge_segments.extend([((cx - half, (top + bottom) // 2), (cx + half, (top + bottom) // 2))])
    elif class_id == "cone":
        polygon = np.asarray([[cx, top], [cx + half, bottom - 8], [cx + half - 5, bottom], [cx - half + 5, bottom], [cx - half, bottom - 8]], np.int32)
    elif class_id == "cylinder":
        cv2.rectangle(mask, (cx - half, top + 9), (cx + half, bottom - 9), 255, -1)
        cv2.ellipse(mask, (cx, top + 9), (half, 12), 0, 180, 360, 255, -1)
        cv2.ellipse(mask, (cx, bottom - 9), (half, 12), 0, 0, 180, 255, -1)
        polygon = np.empty((0, 2), np.int32)
    else:
        raise ValueError(f"unsupported synthetic side class: {class_id}")
    if len(polygon):
        cv2.fillConvexPoly(mask, polygon, 255)
    image[mask > 0] = colour
    for first, second in ridge_segments:
        cv2.line(image, first, second, (25, 25, 25), 2, cv2.LINE_AA)
    if class_id in {"cone", "cylinder"}:
        cv2.ellipse(image, (cx, bottom - 8), (half, 11), 0, 0, 180, (35, 35, 35), 2, cv2.LINE_AA)
    noise = rng.normal(0.0, 2.5, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image, mask


def build_synthetic_side_contact_sheet(
    registry: ShapeRegistry, output: str | Path, *, seed: int = 17
) -> Path:
    tiles: list[np.ndarray] = []
    for index, item in enumerate(registry.enabled_classes):
        image, mask = render_synthetic_side(item.class_id, seed=seed + index)
        contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        canvas = image.copy()
        cv2.drawContours(canvas, contour, -1, (0, 255, 0), 2)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(canvas, item.class_id, (7, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)
    columns = 4
    blank = np.full_like(tiles[0], 245)
    while len(tiles) % columns:
        tiles.append(blank.copy())
    rows = [np.hstack(tiles[index : index + columns]) for index in range(0, len(tiles), columns)]
    sheet = np.vstack(rows)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), sheet):
        raise OSError(f"failed to write synthetic side review: {target}")
    return target
