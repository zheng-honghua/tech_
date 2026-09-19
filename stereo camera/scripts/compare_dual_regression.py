"""Compare two fixed-holdout dual runs and build visual review overviews."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    before, after, output = Path(args.before), Path(args.after), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def records(root: Path) -> dict[str, dict]:
        return {
            item["sample"]: item
            for item in (json.loads(line) for line in (root / "holdout-fusion-records.jsonl").read_text(encoding="utf-8").splitlines())
        }

    old, new = records(before), records(after)
    if set(old) != set(new):
        raise ValueError("holdout sample keys changed; comparison is not paired")
    old_report = json.loads((before / "report.json").read_text(encoding="utf-8"))
    new_report = json.loads((after / "report.json").read_text(encoding="utf-8"))
    bbox_changes = []
    feature_differences = []
    with np.load(before / "top-feature-cache.npz", allow_pickle=False) as first, np.load(after / "top-feature-cache.npz", allow_pickle=False) as second:
        first_ids = {str(key): i for i, key in enumerate(first["sample_keys"])}
        second_ids = {str(key): i for i, key in enumerate(second["sample_keys"])}
        if set(first_ids) != set(second_ids):
            raise ValueError("valid source sample keys changed")
        for key in sorted(first_ids):
            i, j = first_ids[key], second_ids[key]
            feature_differences.append(float(np.max(np.abs(first["features"][i] - second["features"][j]))))
            old_box, new_box = first["bboxes"][i], second["bboxes"][j]
            if not np.array_equal(old_box, new_box):
                bbox_changes.append({"sample": key, "before": old_box.tolist(), "after": new_box.tolist()})
    report = {
        "paired_holdout_count": len(new),
        "valid_source_count": len(first_ids),
        "top_feature_max_absolute_difference": max(feature_differences),
        "rgb_bbox_changed_count": len(bbox_changes),
        "rgb_bbox_changes": bbox_changes,
        "corrected_vs_before": [key for key in new if old[key]["fused_prediction"] != old[key]["true_label"] and new[key]["fused_prediction"] == new[key]["true_label"]],
        "damaged_vs_before": [key for key in new if old[key]["fused_prediction"] == old[key]["true_label"] and new[key]["fused_prediction"] != new[key]["true_label"]],
        "side_segment_count_median": {
            "before": float(np.median([item["cross_view_topology"]["segment_count_raw"] for item in old.values()])),
            "after": float(np.median([item["cross_view_topology"]["segment_count_raw"] for item in new.values()])),
        },
        "before_fused_metrics": old_report["fused_metrics"],
        "after_fused_metrics": new_report["fused_metrics"],
    }
    sheets = sorted((after / "fusion-contact-sheets").glob("*.jpg"))
    for start in range(0, len(sheets), 6):
        tiles = [cv2.resize(cv2.imread(str(path)), (840, 472), interpolation=cv2.INTER_AREA) for path in sheets[start:start + 6]]
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        overview = np.vstack([np.hstack(tiles[index:index + 2]) for index in range(0, len(tiles), 2)])
        cv2.imwrite(str(output / f"overview-{start // 6 + 1:02d}.jpg"), overview)
    changed = {Path(item["sample"]).parent.name for item in bbox_changes}
    for index, item in enumerate(bbox_changes, start=1):
        rgb = cv2.imread(str(Path(item["sample"]) / "primary-color.png"))
        boxes = [item["before"], item["after"]]
        left = max(0, min(box[0] for box in boxes) - 40)
        top = max(0, min(box[1] for box in boxes) - 40)
        right = min(rgb.shape[1], max(box[0] + box[2] for box in boxes) + 40)
        bottom = min(rgb.shape[0], max(box[1] + box[3] for box in boxes) + 40)
        panels = []
        for name, box, color in zip(("before", "after"), boxes, ((0, 0, 255), (0, 255, 0))):
            panel = rgb.copy()
            x, y, width, height = box
            cv2.rectangle(panel, (x, y), (x + width, y + height), color, 2)
            panel = panel[top:bottom, left:right]
            cv2.putText(panel, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
            panels.append(panel)
        cv2.imwrite(str(output / f"bbox-change-{index:02d}.jpg"), np.hstack(panels))
    # Keep full-resolution before/after group sheets where recovered RGB boxes
    # changed; the overview above covers every holdout frame.
    for path in sheets:
        if any(label in path.stem for label in changed):
            previous = before / "fusion-contact-sheets" / path.name
            if previous.exists():
                cv2.imwrite(str(output / f"paired-{path.name}"), np.hstack((cv2.imread(str(previous)), cv2.imread(str(path)))))
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("rgb_bbox_changes", "before_fused_metrics", "after_fused_metrics", "corrected_vs_before", "damaged_vs_before")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
