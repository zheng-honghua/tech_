"""Bounded live acquisition check; no classifier or robot commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dual_apriltag_calibrate import _camera_source, build_parser
from sorting_vision.config import load_config
from sorting_vision.rgbd_dataset import depth_preview
from train_dual_fusion_holdout import _tile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side-camera-index", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    camera_args = build_parser().parse_args(["--platform-id", "temporary", "--tag-size-mm", "24",
                                           "--side-camera-index", str(args.side_camera_index)])
    source = _camera_source(camera_args, load_config("config/dual/temporary.yaml"))
    records = []
    try:
        for index in range(25):
            pair = source.read()
            valid = np.isfinite(pair.primary.depth_mm) & (pair.primary.depth_mm > 0)
            record = {"primary_frame_id": pair.primary.frame_id, "side_frame_id": None if pair.side is None else pair.side.frame_id,
                      "pair_delta_ms": pair.pair_delta_ms, "synchronized": pair.synchronized,
                      "side_error": pair.side_error, "depth_valid_ratio": float(valid.mean()),
                      "primary_mean": float(pair.primary.color_bgr.mean()),
                      "side_mean": None if pair.side is None else float(pair.side.color_bgr.mean())}
            records.append(record)
            if index in (4, 14, 24) and pair.side is not None:
                sheet = np.hstack((_tile(pair.primary.color_bgr, pair.primary.frame_id),
                                   _tile(depth_preview(pair.primary.depth_mm), "TOP DEPTH"),
                                   _tile(pair.side.color_bgr, f"{pair.side.frame_id} {pair.pair_delta_ms:.1f}ms")))
                cv2.imwrite(str(root / f"live-{index:02d}.jpg"), sheet)
    finally:
        source.close()
    summary = {"frames": len(records), "synchronized_frames": sum(item["synchronized"] for item in records),
               "pair_p95_ms": float(np.percentile([item["pair_delta_ms"] for item in records if item["pair_delta_ms"] is not None], 95)) if any(item["pair_delta_ms"] is not None for item in records) else None,
               "records": records, "motion_executed": False}
    (root / "live-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
