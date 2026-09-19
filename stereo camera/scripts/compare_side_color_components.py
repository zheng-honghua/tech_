"""Audit frozen side masks and render RGB/depth/before/after evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from sorting_vision.dual_view import DualViewFusion
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
        mapping = {row["sample"]: row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError("duplicate sample")
        return mapping

    before, after = read(args.before), read(args.after)
    if set(before) != set(after):
        raise ValueError("frozen sample IDs differ")

    def primary(row):
        return max(row["results"], key=lambda item: item["diagnostics"].get("rgb_mask_pixels", 0), default=None)

    changes = []
    for sample, record in after.items():
        old, new = primary(before[sample]), primary(record)
        old_diag = {} if old is None else old["diagnostics"].get("dual_view", {})
        new_diag = {} if new is None else new["diagnostics"].get("dual_view", {})
        changes.append({"sample": sample, "frame_id": record["primary_frame_id"],
                        "before_side_pixels": old_diag.get("side_mask_pixels") or 0,
                        "after_side_pixels": new_diag.get("side_mask_pixels") or 0,
                        "before_state": old_diag.get("fusion_state"), "after_state": new_diag.get("fusion_state"),
                        "after_segmentation": new_diag.get("side_segmentation"),
                        "before_shape": None if old is None else old["shape_id"],
                        "after_shape": None if new is None else new["shape_id"],
                        "true_label": record["true_label"]})
    summary = {"samples": len(changes),
               "side_pixels_increased": sum(row["after_side_pixels"] > row["before_side_pixels"] for row in changes),
               "side_pixels_decreased": sum(row["after_side_pixels"] < row["before_side_pixels"] for row in changes),
               "shape_corrected": sum(row["before_shape"] != row["true_label"] == row["after_shape"] for row in changes),
               "shape_harmed": sum(row["before_shape"] == row["true_label"] != row["after_shape"] for row in changes),
               "limitations": ["V5_MODELS_NOT_RETRAINED", "REUSED_DEVELOPMENT_HOLDOUT", "MASK_PIXELS_NOT_GROUND_TRUTH_IOU"]}
    (output / "comparison.json").write_text(json.dumps({"summary": summary, "changes": changes}, ensure_ascii=False, indent=2), encoding="utf-8")

    focus = (1946, 4253, 3088, 3974, 3410, 3057, 813, 922, 10365, 647, 4560, 6181, 601, 7085, 4269)
    rows, focused, edge_review, sheet = [], [], [], 0
    for frame in focus:
        sample = next((key for key, value in after.items() if int(value["primary_frame_id"].rsplit("-", 1)[-1]) == frame), None)
        if sample is None:
            continue
        loaded = _load_primary_frame(Path(sample))
        side = cv2.imread(str(Path(sample) / "side-color.png"))
        old_results = [SimpleNamespace(diagnostics=item["diagnostics"]) for item in before[sample]["results"]]
        new_results = [SimpleNamespace(diagnostics=item["diagnostics"]) for item in after[sample]["results"]]
        old_image = DualViewFusion.annotate_side(side, old_results, show_geometry=True, show_edges=True)
        new_image = DualViewFusion.annotate_side(side, new_results)
        views = [_tile(loaded.color_bgr, f"{frame} top RGB"), _tile(depth_preview(loaded.depth_mm), "real depth"),
                 _tile(old_image, "before: projected crop + debug overlays"), _tile(new_image, "after: actual full colour mask")]
        rows.append(np.hstack(views))
        if frame in (3088, 813, 647):
            focused.append(np.hstack(views[2:]))
            edges = DualViewFusion.annotate_side(side, new_results, show_edges=True)
            edge_review.append(np.hstack((_tile(new_image, "observed mask"), _tile(edges, "RGB line candidates; NO OBB"))))
        if len(rows) == 4:
            sheet += 1
            cv2.imwrite(str(output / f"comparison-{sheet:02d}.jpg"), np.vstack(rows))
            rows = []
    if rows:
        cv2.imwrite(str(output / "comparison-extra.jpg"), np.vstack(rows))
    if focused:
        cv2.imwrite(str(output / "focused-before-after.jpg"), np.vstack(focused))
    if edge_review:
        cv2.imwrite(str(output / "observed-edge-review.jpg"), np.vstack(edge_review))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
