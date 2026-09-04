"""Build reviewed single-object RGB-D samples from multi-02 and multi-03.

This is intentionally dataset-specific: the two scenes only contain composition
labels.  Physical objects were assigned after visual review using their stable
colour family and relative image area.  Frame multi-03/06 is excluded because
one object was split into two components.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sorting_vision.camera import load_rgbd_frame
from sorting_vision.rgbd import RGBDFrame
from sorting_vision.rgbd_dataset import save_rgbd_dataset_sample


def _appearance(item: dict[str, object], output_dir: Path) -> dict[str, object]:
    object_id = str(item["object_id"])
    crop = cv2.imread(str(output_dir / f"{object_id}-color.png"))
    if crop is None:
        raise FileNotFoundError(output_dir / f"{object_id}-color.png")
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    foreground = (hsv[:, :, 1] > 50) & (hsv[:, :, 2] < 245)
    if not np.any(foreground):
        raise ValueError(f"no colour support for {object_id}")
    return {
        "item": item,
        "hue": float(np.median(hsv[:, :, 0][foreground])),
        "area": int(np.count_nonzero(foreground)),
    }


def _assign(batch: str, objects: list[dict[str, object]]) -> list[tuple[dict[str, object], str]]:
    blue = sorted((obj for obj in objects if float(obj["hue"]) >= 100), key=lambda obj: int(obj["area"]))
    teal = sorted((obj for obj in objects if float(obj["hue"]) < 100), key=lambda obj: int(obj["area"]))
    if batch == "multi-02":
        if len(blue) != 1 or len(teal) != 2:
            raise ValueError("multi-02 must contain one blue and two teal objects")
        return [
            (blue[0], "square_pyramid"),
            (teal[0], "triangular_pyramid"),
            (teal[1], "pentagonal_prism"),
        ]
    if len(blue) != 2 or len(teal) != 2:
        raise ValueError("multi-03 must contain two blue and two teal objects")
    return [
        (blue[0], "pentagonal_pyramid"),
        (blue[1], "hexagonal_prism"),
        (teal[0], "triangular_pyramid"),
        (teal[1], "triangular_prism"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="data/rgbd-multi-scenes")
    parser.add_argument("--detect-root", default="output/multi-02-03-test")
    parser.add_argument("--background-dir", required=True)
    parser.add_argument("--output-root", default="data/rgbd-reviewed-multi-02-03")
    parser.add_argument(
        "--batch", action="append", choices=("multi-02", "multi-03"),
        help="prepare only the selected batch; repeat to select both",
    )
    parser.add_argument(
        "--train-through", type=int, default=10,
        help="include frame numbers up to this value (use 8 for a 9-10 holdout)",
    )
    args = parser.parse_args()
    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    background = load_rgbd_frame(args.background_dir)
    source_root = Path(args.source_root)
    detect_root = Path(args.detect_root)
    batches = tuple(args.batch) if args.batch else ("multi-02", "multi-03")
    for batch in batches:
        save_rgbd_dataset_sample(
            background, output_root, batch, "empty_tray",
            {"provenance": "pilot-02 empty tray reference"},
            sample_id="empty-tray-reference",
        )
        frames = sorted((source_root / batch / "scene-0001").iterdir())
        for index, frame_dir in enumerate(frames, start=1):
            if index > args.train_through:
                continue
            if batch == "multi-03" and index == 6:
                continue
            frame = load_rgbd_frame(frame_dir)
            detection_dir = detect_root / batch / f"frame-{index:02d}"
            results = json.loads((detection_dir / "results-v2.json").read_text(encoding="utf-8"))
            expected_count = 3 if batch == "multi-02" else 4
            if len(results) != expected_count:
                print(
                    f"skip {batch}/frame-{index:02d}: "
                    f"expected {expected_count} objects, got {len(results)}"
                )
                continue
            assignments = _assign(batch, [_appearance(item, detection_dir) for item in results])
            for appearance, label in assignments:
                item = appearance["item"]
                assert isinstance(item, dict)
                x, y, width, height = (int(value) for value in item["bbox_px"])
                pad = 8
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1 = min(frame.intrinsics.width, x + width + pad)
                y1 = min(frame.intrinsics.height, y + height + pad)
                color = background.color_bgr.copy()
                depth = background.depth.copy()
                color[y0:y1, x0:x1] = frame.color_bgr[y0:y1, x0:x1]
                depth[y0:y1, x0:x1] = frame.depth[y0:y1, x0:x1]
                object_suffix = str(item["object_id"]).rsplit("-", 1)[-1]
                isolated = RGBDFrame(
                    color, depth, frame.intrinsics, frame.timestamp_ns,
                    f"{frame.frame_id}-{object_suffix}",
                    frame.color_timestamp_ns, frame.depth_timestamp_ns,
                )
                settings = {
                    "provenance": "composition-constrained reviewed extraction",
                    "source_batch": batch,
                    "source_frame_dir": str(frame_dir),
                    "source_object_id": item["object_id"],
                    "source_bbox_px": [x, y, width, height],
                    "assignment_rule": "visually reviewed colour family plus relative area",
                    "median_hue": appearance["hue"],
                    "colour_area_px": appearance["area"],
                    "reviewed": True,
                }
                save_rgbd_dataset_sample(
                    isolated, output_root, batch, label, settings,
                    sample_id=f"frame-{index:02d}-{object_suffix}",
                )
    print(output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
