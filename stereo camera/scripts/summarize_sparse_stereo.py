"""Audit a completed sparse reconstruction replay, without selecting thresholds."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np

from sorting_vision.config import load_config
from sorting_vision.dual_view import DualViewCalibration
from sorting_vision.sparse_stereo import camera_poses
from sorting_vision.rgbd_dataset import depth_preview
from train_dual_fusion_holdout import _load_primary_frame, _tile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--baseline-records", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    calibration = DualViewCalibration.load(args.calibration)

    def read(path):
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        mapping = {Path(row["sample"]).resolve().as_posix(): row for row in rows}
        if len(rows) != len(mapping):
            raise ValueError("duplicate replay IDs")
        return mapping

    rows, baseline = read(args.records), read(args.baseline_records)
    if rows.keys() != baseline.keys():
        raise ValueError("frozen replay sample IDs changed")
    entries = [(row, item, item["diagnostics"].get("dual_view", {}).get("cross_view_topology", {}).get("sparse_stereo"))
        for row in rows.values() for item in row["results"]
        if item["diagnostics"].get("dual_view", {}).get("cross_view_topology")]
    entries = [(row, item, sparse) for row, item, sparse in entries if sparse is not None]
    errors = [max(node["primary_error_px"], node["side_error_px"]) for _, _, sparse in entries for node in sparse.get("nodes", [])]
    changed = []
    for key, row in rows.items():
        signature = lambda r: [(item["object_id"], item["shape_id"], item["status"], item["selected"]) for item in r["results"]]
        if signature(row) != signature(baseline[key]):
            changed.append(row["primary_frame_id"])
    summary = {"frames": len(rows), "objects_with_sparse_diagnostics": len(entries),
        "states": dict(Counter(sparse["state"] for _, _, sparse in entries)),
        "triangulated_nodes": sum(len(sparse.get("nodes", [])) for _, _, sparse in entries),
        "triangulated_edges": sum(len(sparse.get("edges", [])) for _, _, sparse in entries),
        "frames_with_nodes": len({row["sample"] for row, _, sparse in entries if sparse.get("nodes")}),
        "frames_with_edges": len({row["sample"] for row, _, sparse in entries if sparse.get("edges")}),
        "rejection_reasons": dict(Counter(node["reason"] for _, _, sparse in entries for node in sparse.get("rejected_corners", []))),
        "accepted_reprojection_p95_px": float(np.percentile(errors, 95)) if errors else None,
        "changed_application_frames": changed,
        "illegal_safety_upgrades": sum(len(row["illegal_safety_upgrades"]) for row in rows.values()),
        "camera_poses": camera_poses(calibration), "promotion_eligible": False,
        "limitations": ["RGB_CANDIDATES_NOT_VERIFIED_PHYSICAL_VERTICES", "SAME_BATCH_REGRESSION",
                        "NO_METRIC_GROUND_TRUTH", "NO_NEW_CLASSIFIER_TRAINING"]}
    ranked = sorted(entries, key=lambda entry: (-len(entry[2].get("edges", [])), -len(entry[2].get("nodes", []))))
    focus = ranked[:8] + [entry for entry in ranked[-2:] if entry not in ranked[:8]]
    visuals = []
    for index, (row, item, sparse) in enumerate(focus, 1):
        primary = _load_primary_frame(Path(row["sample"]))
        top = primary.color_bgr.copy()
        side = cv2.imread(str(Path(row["sample"]) / "side-color.png"))
        nodes = sparse.get("nodes", [])
        for node in nodes:
            a, b = tuple(np.rint(node["primary_uv"]).astype(int)), tuple(np.rint(node["side_uv"]).astype(int))
            for image, pixel in ((top, a), (side, b)):
                cv2.circle(image, pixel, 5, (255, 255, 0), -1)
                cv2.putText(image, str(node["node_id"]), (pixel[0]+5, pixel[1]-4), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 0), 1)
        for edge in sparse.get("edges", []):
            for image, field in ((top, "primary_uv"), (side, "side_uv")):
                endpoints = [np.rint(nodes[node_id][field]).astype(int) for node_id in edge["node_ids"]]
                cv2.line(image, tuple(endpoints[0]), tuple(endpoints[1]), (255, 255, 0), 2)

        def crop(image, box):
            x, y, w, h = map(int, box)
            return image[max(0, y-25):min(image.shape[0], y+h+25), max(0, x-25):min(image.shape[1], x+w+25)]

        side_box = item["diagnostics"]["dual_view"]["side_roi"]
        image = np.hstack((_tile(crop(top, item["bbox_px"]), f"{row['primary_frame_id']} paired corner IDs"),
            _tile(crop(depth_preview(primary.depth_mm), item["bbox_px"]), "Real depth; no stereo depth fill"),
            _tile(crop(side, side_box), f"RGB triangulation: {len(nodes)} nodes {len(sparse.get('edges', []))} edges")))
        path = output / f"matched-corners-{index:02d}.jpg"
        cv2.imwrite(str(path), image)
        visuals.append({"image": str(path), "frame_id": row["primary_frame_id"], "state": sparse["state"],
                        "node_count": len(nodes), "edge_count": len(sparse.get("edges", []))})
    summary["visual_review_candidates"] = visuals
    code_files = [*Path("src/sorting_vision").glob("*.py"), Path(__file__)]
    snapshot = {"effective_config": asdict(cfg), "calibration_hash": calibration.calibration_hash,
        "python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
        "source_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files},
        "visual_review_status": "GENERATED_REQUIRES_ACTUAL_INSPECTION"}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "reproducibility-snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
