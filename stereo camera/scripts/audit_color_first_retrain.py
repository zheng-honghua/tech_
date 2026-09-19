"""Compare a completed retrain on frozen IDs; never fit or tune a model."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from sorting_vision.dual_view import DualViewFusion
from sorting_vision.rgbd_dataset import depth_preview
from train_dual_fusion_holdout import _load_primary_frame, _metrics, _tile


def read_records(path: Path) -> dict[str, dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mapping = {Path(row["sample"]).resolve().as_posix(): row for row in records}
    if len(mapping) != len(records):
        raise ValueError(f"duplicate sample IDs: {path}")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--baseline-application", required=True)
    parser.add_argument("--frozen-records", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    frozen = read_records(Path(args.frozen_records))
    old = read_records(Path(args.baseline_dir) / "holdout-fusion-records.jsonl")
    new = read_records(run / "holdout-fusion-records.jsonl")
    if not set(old) <= set(frozen) or not set(new) <= set(frozen):
        raise ValueError("classification records contain unfrozen IDs")

    def complete(mapping):
        return [mapping.get(key, {"sample": key, "true_label": source["true_label"],
            "top_prediction": "unknown", "classic_side_prediction": "unknown",
            "side_prediction": "unknown", "fused_prediction": "unknown",
            "top_accepted": False, "side_accepted": False, "fused_accepted": False})
            for key, source in frozen.items()]

    old_complete, new_complete = complete(old), complete(new)
    metrics = {}
    for name, records in (("v5", old_complete), ("v6", new_complete)):
        metrics[name] = {
            "top": _metrics(records, "top_prediction", "top_accepted"),
            "classic_side": _metrics(records, "classic_side_prediction"),
            "cross_view": _metrics(records, "side_prediction", "side_accepted"),
            "fused": _metrics(records, "fused_prediction", "fused_accepted"),
        }
    metrics["v6_corrected_vs_v5"] = [after["sample"] for before, after in zip(old_complete, new_complete)
        if before["fused_prediction"] != after["true_label"] == after["fused_prediction"]]
    metrics["v6_harmed_vs_v5"] = [after["sample"] for before, after in zip(old_complete, new_complete)
        if before["fused_prediction"] == after["true_label"] != after["fused_prediction"]]
    metrics["v6_wrong_classification_accepts"] = [row for row in new_complete
        if row["fused_accepted"] and row["fused_prediction"] != row["true_label"]]

    manifest = json.loads((run / "bundle-manifest.json").read_text(encoding="utf-8"))
    training = {Path(key).resolve().as_posix() for key in manifest["training_ids"]}
    if training & set(frozen):
        raise ValueError("frozen sample IDs leaked into training")
    # Exact duplicate checks are not near-pose or unseen-entity guarantees.
    hashes = {key: hashlib.sha256((Path(key) / "side-color.png").read_bytes()).hexdigest()
              for key in training | set(frozen)}
    train_hashes = {hashes[key] for key in training}
    duplicate_leaks = [key for key in frozen if hashes[key] in train_hashes]
    metrics["split_audit"] = {"reserved": len(frozen), "training": len(training),
        "reserved_id_leaks": 0, "exact_side_rgb_hash_leaks": duplicate_leaks,
        "unique_side_rgb_hashes": len(set(hashes.values())),
        "near_duplicate_pose_audit": "NOT_PERFORMED", "unseen_entity_test": False}

    old_app = read_records(Path(args.baseline_application))
    new_app = read_records(run / "application-frozen108-review" / "application-records.jsonl")
    if set(old_app) != set(frozen) or set(new_app) != set(frozen):
        raise ValueError("application replay IDs differ from frozen set")

    def primary(row):
        return max(row["results"], key=lambda item: item["diagnostics"].get("rgb_mask_pixels", 0), default=None)

    changes = []
    for key, record in new_app.items():
        before, after = primary(old_app[key]), primary(record)
        changes.append({"sample": key, "frame_id": record["primary_frame_id"],
            "truth": record["true_label"], "before": None if before is None else before["shape_id"],
            "after": None if after is None else after["shape_id"]})
    metrics["application_vs_v5_same_segmentation"] = {
        "corrected": [row for row in changes if row["before"] != row["truth"] == row["after"]],
        "harmed": [row for row in changes if row["before"] == row["truth"] != row["after"]],
        "before_raw_including_unknown": sum(row["before"] == row["truth"] for row in changes) / len(changes),
        "after_raw_including_unknown": sum(row["after"] == row["truth"] for row in changes) / len(changes),
        "statuses": dict(Counter(item["status"] for row in new_app.values() for item in row["results"])),
    }
    metrics["limitations"] = ["REUSED_SAME_BATCH_DEVELOPMENT_REGRESSION", "NOT_FINAL_ACCEPTANCE",
        "UNKNOWN_RETAINED_AS_FAILURE", "EXACT_HASH_CHECK_NOT_NEAR_POSE_CHECK"]
    (run / "v5-fixed108-audit.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    focus = (1946, 4253, 3088, 3974, 3410, 3057, 813, 922, 10365, 647, 4560, 6181, 601, 7085, 4269)
    wrong = {Path(row["sample"]).resolve().as_posix() for row in metrics["v6_wrong_classification_accepts"]}
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    failed_or_changed = {Path(key).resolve().as_posix() for key in
        [*report["missing_fixed_holdout"], *report["corrected_samples"], *report["damaged_samples"]]}
    failed_or_changed.update(row["sample"] for row in changes if row["before"] != row["after"])
    selected = [row for key, row in new_app.items()
        if int(row["primary_frame_id"].rsplit("-", 1)[-1]) in focus or key in wrong or key in failed_or_changed]
    selected_ids = {Path(row["sample"]).resolve().as_posix() for row in selected}
    selected.extend(row for key, row in new_app.items() if key not in selected_ids and any(
        item["diagnostics"].get("dual_view", {}).get("fusion_state") == "CONFLICT" for item in row["results"]))
    output = run / "focused-visual-audit"
    output.mkdir(exist_ok=True)
    rows = []
    for index, record in enumerate(selected, 1):
        frame = _load_primary_frame(Path(record["sample"]))
        top_image = frame.color_bgr.copy()
        for item in record["results"]:
            box = item["bbox_px"]
            x, y, width, height = map(int, box)
            cv2.rectangle(top_image, (x, y), (x + width, y + height), (0, 220, 255), 2)
        side_image = cv2.imread(str(Path(record["sample"]) / "side-color.png"))
        results = [SimpleNamespace(diagnostics=item["diagnostics"]) for item in record["results"]]
        side_image = DualViewFusion.annotate_side(side_image, results, show_edges=True)
        main_item = primary(record)
        prediction = "unknown" if main_item is None else main_item["shape_id"]
        title = f"{record['primary_frame_id']} true={record['true_label']} pred={prediction}"
        rows.append(np.hstack((_tile(top_image, title), _tile(depth_preview(frame.depth_mm), "REAL measured depth"),
                               _tile(side_image, "Measured colour contour + RGB LSD; no reference OBB"))))
        if len(rows) == 3 or index == len(selected):
            cv2.imwrite(str(output / f"focused-{(index - 1) // 3 + 1:02d}.jpg"), np.vstack(rows))
            rows = []
    print(json.dumps({"reserved": len(frozen), "v5_fused": metrics["v5"]["fused"],
        "v6_fused": metrics["v6"]["fused"], "application": metrics["application_vs_v5_same_segmentation"],
        "split_audit": metrics["split_audit"], "focused_visual_frames": [row["primary_frame_id"] for row in selected]},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
