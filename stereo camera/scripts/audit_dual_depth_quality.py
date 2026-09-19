"""Audit saved aligned depth frames without loading the dataset into memory.

The D435 depth imager has no RGB-style autofocus.  This report therefore
measures missing pixels, local surface noise, alignment upsampling and range;
it deliberately avoids calling a soft-looking colourized preview "defocus".
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sorting_vision.rgbd_dataset import depth_preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a captured dual-view depth dataset")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worst-count", type=int, default=12)
    return parser


def _percentile(values: list[float], level: float) -> float | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], np.float64)
    return None if not len(finite) else round(float(np.percentile(finite, level)), 4)


def _local_noise(depth: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
    height, width = depth.shape
    crop = depth[height // 5 : height * 4 // 5, width // 5 : width * 4 // 5].astype(np.float32)
    crop_valid = valid[height // 5 : height * 4 // 5, width // 5 : width * 4 // 5]
    differences: list[np.ndarray] = []
    stride_three_differences: list[np.ndarray] = []
    equal_count = pair_count = 0
    for first, second, first_valid, second_valid in (
        (crop[:, :-1], crop[:, 1:], crop_valid[:, :-1], crop_valid[:, 1:]),
        (crop[:-1, :], crop[1:, :], crop_valid[:-1, :], crop_valid[1:, :]),
    ):
        pair_valid = first_valid & second_valid
        delta = np.abs(first - second)[pair_valid]
        # Ignore true object/tray discontinuities when estimating flat-surface noise.
        delta = delta[delta <= 10.0]
        if len(delta):
            differences.append(delta)
            equal_count += int(np.count_nonzero(delta == 0))
            pair_count += int(len(delta))
    for first, second, first_valid, second_valid in (
        (crop[:, :-3], crop[:, 3:], crop_valid[:, :-3], crop_valid[:, 3:]),
        (crop[:-3, :], crop[3:, :], crop_valid[:-3, :], crop_valid[3:, :]),
    ):
        pair_valid = first_valid & second_valid
        delta = np.abs(first - second)[pair_valid]
        delta = delta[delta <= 10.0]
        if len(delta):
            stride_three_differences.append(delta)
    if not differences:
        return float("nan"), float("nan"), float("nan")
    joined = np.concatenate(differences)
    stride_three = (
        float(np.percentile(np.concatenate(stride_three_differences), 95))
        if stride_three_differences else float("nan")
    )
    return float(np.percentile(joined, 95)), equal_count / max(pair_count, 1), stride_three


def _sample_metrics(path: Path) -> dict[str, Any]:
    depth = np.load(path, allow_pickle=False)
    if depth.ndim != 2:
        raise ValueError(f"depth array must be 2-D: {path}")
    metadata_path = path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    valid = np.isfinite(depth) & (depth > 0)
    height, width = depth.shape
    center = valid[height // 5 : height * 4 // 5, width // 5 : width * 4 // 5]
    values = depth[valid].astype(np.float32)
    noise_p95, identical_ratio, stride_three_p95 = _local_noise(depth, valid)
    intrinsics = metadata.get("primary_intrinsics", {})
    stored_size_matches_intrinsics = (
        int(intrinsics.get("width", -1)) == width
        and int(intrinsics.get("height", -1)) == height
    )
    record: dict[str, Any] = {
        "sample": path.parent.as_posix(),
        "batch_id": metadata.get("batch_id", path.parents[1].name),
        "label": metadata.get("label_id", metadata.get("label", "unknown")),
        "shape": [height, width],
        "dtype": str(depth.dtype),
        "stored_size_matches_intrinsics": stored_size_matches_intrinsics,
        "valid_percent": round(float(valid.mean() * 100.0), 4),
        "center_valid_percent": round(float(center.mean() * 100.0), 4),
        "local_difference_p95_mm": round(noise_p95, 4) if np.isfinite(noise_p95) else None,
        "three_pixel_difference_p95_mm": (
            round(stride_three_p95, 4) if np.isfinite(stride_three_p95) else None
        ),
        "identical_neighbor_ratio": round(identical_ratio, 4) if np.isfinite(identical_ratio) else None,
        "min_mm": int(values.min()) if len(values) else None,
        "median_mm": round(float(np.median(values)), 3) if len(values) else None,
        "max_mm": int(values.max()) if len(values) else None,
        "pair_delta_ms": metadata.get("pair_delta_ms"),
        "primary_color_depth_delta_ms": round(
            abs(
                int(metadata.get("primary_color_timestamp_ns", 0))
                - int(metadata.get("primary_depth_timestamp_ns", 0))
            ) / 1_000_000.0,
            4,
        ),
    }
    return record


def _tile(color: np.ndarray | None, depth: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    width, height = 480, 270
    if color is None:
        color_view = np.zeros((height, width, 3), np.uint8)
    else:
        color_view = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)
    depth_view = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    combined = np.hstack((color_view, depth_view))
    cv2.rectangle(combined, (0, 0), (combined.shape[1] - 1, combined.shape[0] - 1), (255, 255, 255), 1)
    lines = (
        f"{record['batch_id']} / {record['label']}",
        f"valid={record['valid_percent']:.2f}% center={record['center_valid_percent']:.2f}%",
        f"range={record['min_mm']}-{record['max_mm']} mm noiseP95={record['local_difference_p95_mm']} mm",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            combined, line, (8, 22 + index * 22), cv2.FONT_HERSHEY_SIMPLEX,
            0.52, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            combined, line, (8, 22 + index * 22), cv2.FONT_HERSHEY_SIMPLEX,
            0.52, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return combined


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.dataset_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(root.rglob("primary-depth.npy"))
    if not paths:
        raise ValueError(f"no primary-depth.npy found under {root}")

    records: list[dict[str, Any]] = []
    for path in paths:
        records.append(_sample_metrics(path))

    worst = sorted(
        records,
        key=lambda item: (
            item["center_valid_percent"],
            -(item["local_difference_p95_mm"] or 0.0),
        ),
    )[: max(args.worst_count, 1)]
    tiles: list[np.ndarray] = []
    for record in worst:
        path = Path(record["sample"]) / "primary-depth.npy"
        color = cv2.imread(str(path.with_name("primary-color.png")), cv2.IMREAD_COLOR)
        preview = depth_preview(np.load(path, allow_pickle=False))
        tiles.append(_tile(color, preview, record))
    blank = np.zeros_like(tiles[0])
    while len(tiles) % 2:
        tiles.append(blank)
    sheet = np.vstack([np.hstack(tiles[index : index + 2]) for index in range(0, len(tiles), 2)])
    sheet_path = output / "worst-depth-contact-sheet.jpg"
    if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"failed to write {sheet_path}")

    report = {
        "dataset_root": root.as_posix(),
        "sample_count": len(records),
        "shape_counts": dict(Counter(f"{item['shape'][1]}x{item['shape'][0]}" for item in records)),
        "dtype_counts": dict(Counter(item["dtype"] for item in records)),
        "stored_size_mismatch_count": sum(not item["stored_size_matches_intrinsics"] for item in records),
        "valid_percent": {
            "minimum": _percentile([item["valid_percent"] for item in records], 0),
            "p05": _percentile([item["valid_percent"] for item in records], 5),
            "median": _percentile([item["valid_percent"] for item in records], 50),
            "p95": _percentile([item["valid_percent"] for item in records], 95),
        },
        "center_valid_percent": {
            "minimum": _percentile([item["center_valid_percent"] for item in records], 0),
            "p05": _percentile([item["center_valid_percent"] for item in records], 5),
            "median": _percentile([item["center_valid_percent"] for item in records], 50),
        },
        "local_difference_p95_mm": {
            "median": _percentile([item["local_difference_p95_mm"] for item in records if item["local_difference_p95_mm"] is not None], 50),
            "p95": _percentile([item["local_difference_p95_mm"] for item in records if item["local_difference_p95_mm"] is not None], 95),
        },
        "three_pixel_difference_p95_mm": {
            "median": _percentile([item["three_pixel_difference_p95_mm"] for item in records if item["three_pixel_difference_p95_mm"] is not None], 50),
            "p95": _percentile([item["three_pixel_difference_p95_mm"] for item in records if item["three_pixel_difference_p95_mm"] is not None], 95),
        },
        "identical_neighbor_ratio": {
            "median": _percentile([item["identical_neighbor_ratio"] for item in records if item["identical_neighbor_ratio"] is not None], 50),
        },
        "median_range_mm": {
            "minimum": _percentile([item["median_mm"] for item in records if item["median_mm"] is not None], 0),
            "median": _percentile([item["median_mm"] for item in records if item["median_mm"] is not None], 50),
            "maximum": _percentile([item["median_mm"] for item in records if item["median_mm"] is not None], 100),
        },
        "worst_contact_sheet": sheet_path.as_posix(),
        "worst_samples": worst,
        "interpretation": {
            "depth_has_optical_autofocus": False,
            "saved_depth_is_aligned_to_color": True,
            "soft_preview_warning": "848x480 Z16 is aligned/upsampled to 1920x1080; blockiness is not lens defocus",
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "worst_samples"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
