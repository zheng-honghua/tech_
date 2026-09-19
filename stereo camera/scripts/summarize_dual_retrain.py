"""Summarize an already frozen regression; never selects model parameters."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from train_dual_fusion_holdout import _metrics
from sorting_vision.dual_view import DualViewCalibration


def read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-dir", required=True)
    args = parser.parse_args()
    run, baseline = Path(args.run_dir), Path(args.baseline_dir)
    old = read_records(baseline / "holdout-fusion-records.jsonl")
    new = read_records(run / "holdout-fusion-records.jsonl")
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    by_key = {r["sample"]: r for r in new}
    assert set(by_key) <= {r["sample"] for r in old}, "holdout IDs changed"
    complete = [by_key.get(r["sample"], {"sample": r["sample"], "true_label": r["true_label"],
        "top_prediction": "unknown", "side_prediction": "unknown", "classic_side_prediction": "unknown",
        "fused_prediction": "unknown", "top_accepted": False, "side_accepted": False, "fused_accepted": False}) for r in old]
    common = [r for r in old if r["sample"] in by_key]
    summary = {"reserved_count": len(old), "evaluable_count": len(new), "failed_reserved_count": len(old) - len(new),
        "baseline_common": _metrics(common, "fused_prediction", "fused_accepted"),
        "new_common": _metrics(new, "fused_prediction", "fused_accepted"),
        "top_all_reserved": _metrics(complete, "top_prediction", "top_accepted"),
        "fused_all_reserved": _metrics(complete, "fused_prediction", "fused_accepted"),
        "model_vs_baseline_corrected": [r["sample"] for r in common if r["fused_prediction"] != r["true_label"] and by_key[r["sample"]]["fused_prediction"] == r["true_label"]],
        "model_vs_baseline_damaged": [r["sample"] for r in common if r["fused_prediction"] == r["true_label"] and by_key[r["sample"]]["fused_prediction"] != r["true_label"]],
        "no_parameter_tuning": True, "promotion_eligible": False}
    (run / "fixed108-comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    keys = np.load(run / "top-feature-cache.npz", allow_pickle=False)["sample_keys"].tolist()
    reserved = {r["sample"] for r in old}
    manifest = {"input_contract": "rgb_silhouette_depth_owned_v2", "calibration_path": "output/native-calibration-debug-20260918/calibration-native.json",
        "calibration_hash": DualViewCalibration.load("output/native-calibration-debug-20260918/calibration-native.json").calibration_hash,
        "registry_hash": report["policy"]["registry_hash"], "training_ids": sorted(set(keys) - reserved),
        "reserved_ids": sorted(reserved), "excluded_reserved": report["missing_fixed_holdout"],
        "files": {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                  for name in ("top-rgbd.npz", "side-lsd.npz", "joint-topology.npz", "fusion-policy.json")},
        "promotion_eligible": False}
    assert not set(manifest["training_ids"]) & reserved
    (run / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sheets = sorted((run / "fusion-contact-sheets").glob("*.jpg"))
    for start in range(0, len(sheets), 6):
        tiles = []
        for path in sheets[start:start + 6]:
            tile = cv2.resize(cv2.imread(str(path)), (630, 354))
            tiles.append(tile)
        while len(tiles) < 6:
            tiles.append(np.zeros((354, 630, 3), np.uint8))
        overview = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 6, 2)])
        cv2.imwrite(str(run / f"overview-{start // 6 + 1:02d}.jpg"), overview)
    print(json.dumps({"reserved": len(old), "evaluable": len(new), "top_raw_all108": summary["top_all_reserved"]["raw_accuracy"],
        "fused_raw_all108": summary["fused_all_reserved"]["raw_accuracy"], "corrected_vs_old": len(summary["model_vs_baseline_corrected"]),
        "damaged_vs_old": len(summary["model_vs_baseline_damaged"]), "training_count": len(manifest["training_ids"])}, indent=2))


if __name__ == "__main__":
    main()
