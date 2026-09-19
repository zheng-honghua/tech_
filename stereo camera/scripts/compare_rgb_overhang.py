"""Compare frozen application boxes; generate visual evidence, never train."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sorting_vision.rgbd_dataset import depth_preview
from train_dual_fusion_holdout import _load_primary_frame, _tile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def read(path):
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        indexed = {row["sample"]: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError("duplicate regression sample")
        return indexed

    before, after = read(args.before), read(args.after)
    if set(before) != set(after):
        raise ValueError("regression IDs differ")

    def primary(row):
        return max(row["results"], key=lambda x: x["diagnostics"].get("rgb_mask_pixels", 0), default=None)

    changes = []
    for sample, newer in after.items():
        old, new = primary(before[sample]), primary(newer)
        old_box = None if old is None else old["bbox_px"]
        new_box = None if new is None else new["bbox_px"]
        changes.append({"sample": sample, "frame_id": newer["primary_frame_id"],
                        "before_box": old_box, "after_box": new_box,
                        "box_changed": old_box != new_box,
                        "before_rgb_pixels": 0 if old is None else old["diagnostics"]["rgb_mask_pixels"],
                        "after_rgb_pixels": 0 if new is None else new["diagnostics"]["rgb_mask_pixels"],
                        "workspace_support_ratio": None if new is None else new["diagnostics"].get("workspace_support_ratio")})
    summary = {"samples": len(changes),
               "before_missing": sum(not row["results"] for row in before.values()),
               "after_missing": sum(not row["results"] for row in after.values()),
               "box_changed": sum(row["box_changed"] for row in changes),
               "rgb_pixels_increased": sum(row["after_rgb_pixels"] > row["before_rgb_pixels"] for row in changes),
               "rgb_pixels_decreased": sum(row["after_rgb_pixels"] < row["before_rgb_pixels"] for row in changes),
               "newly_detected": [row["frame_id"] for row in changes if row["before_box"] is None and row["after_box"] is not None],
               "newly_missing": [row["frame_id"] for row in changes if row["before_box"] is not None and row["after_box"] is None],
               "limitations": ["V5_MODELS_NOT_RETRAINED", "BOX_AREA_IS_NOT_MASK_IOU", "NO_MANUAL_PIXEL_GROUND_TRUTH", "REUSED_SAME_BATCH_REGRESSION"]}
    (output / "comparison.json").write_text(json.dumps({"summary": summary, "changes": changes}, ensure_ascii=False, indent=2), encoding="utf-8")
    focus = (4253, 3559, 813, 922, 10365, 647, 4560, 11007, 11210, 3521, 7227, 5612)
    rows = []
    focused = []
    sheet = 0
    for frame in focus:
        sample = next((key for key, value in after.items() if int(value["primary_frame_id"].rsplit("-", 1)[-1]) == frame), None)
        if sample is None:
            continue
        loaded = _load_primary_frame(Path(sample))
        views = []
        for name, record in (("before v5", before[sample]), ("after segmentation", after[sample])):
            image = loaded.color_bgr.copy()
            for result in record["results"]:
                x, y, width, height = result["bbox_px"]
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 255), 3)
            views.append(_tile(image, f"{frame} {name} n={len(record['results'])}"))
        views.append(_tile(depth_preview(loaded.depth_mm), "real depth (unmodified)"))
        side = cv2.imread(str(Path(sample) / "side-color.png"))
        views.append(_tile(side, "side RGB (unchanged)"))
        if frame in (647, 813, 10365):
            focused.append(np.hstack(views[:2]))
        rows.append(np.hstack(views))
        if len(rows) == 4:
            sheet += 1
            cv2.imwrite(str(output / f"comparison-{sheet:02d}.jpg"), np.vstack(rows))
            rows = []
    if rows:
        cv2.imwrite(str(output / "comparison-extra.jpg"), np.vstack(rows))
    if focused:
        cv2.imwrite(str(output / "focused-before-after.jpg"), np.vstack(focused))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
