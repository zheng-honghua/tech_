"""Replay saved pairs through the real application pipeline, without motion."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from sorting_vision.camera import RGBFrame, SynchronizedFramePair
from sorting_vision.config import load_config
from sorting_vision.dual_view import DualViewCalibration, DualViewFusion
from sorting_vision.fusion_policy import FusionPolicy
from sorting_vision.geometry_rgbd_model import DepthGeometryModel
from sorting_vision.cross_view_topology import CrossViewTopologyModel
from sorting_vision.pipeline3d import VisionPipeline3D
from sorting_vision.side_geometry import SideGeometryModel
from sorting_vision.shape_registry import ShapeRegistry
from sorting_vision.rgbd_dataset import depth_preview
from sorting_vision.rgbd import resize_rgbd_frame
from sorting_vision.face_topology3d import extract_face_topology
from train_dual_fusion_holdout import _load_primary_frame, _tile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="config/dual/temporary-observed10.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dual-calibration", help="defaults to the configured platform calibration")
    parser.add_argument("--show-top-edges", action="store_true", help="extra audit reconstruction, excluded from pipeline timing")
    parser.add_argument("--show-side-geometry", action="store_true", help="show reference boxes/OBB, NOT physical object edges")
    parser.add_argument("--show-side-edges", action="store_true", help="show measured RGB line candidates")
    parser.add_argument("--show-sparse-geometry", action="store_true", help="triangulated candidates only; no reference OBB")
    parser.add_argument("--input-records", help="replay a separately frozen input set, including extraction failures")
    parser.add_argument("--shape-registry", help="override the configured registry for a new platform bundle")
    args = parser.parse_args()
    run, output = Path(args.run_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    registry = ShapeRegistry.load(args.shape_registry or cfg.dual_view.shape_registry_path)
    top = DepthGeometryModel.load(run / "top-rgbd.npz")
    side = SideGeometryModel.load(run / "side-lsd.npz", registry)
    cross = CrossViewTopologyModel.load(run / "joint-topology.npz", registry)
    policy = FusionPolicy.load(run / "fusion-policy.json", registry.registry_hash)
    calibration = DualViewCalibration.load(args.dual_calibration or cfg.dual_view.calibration_path)
    background = cv2.imread(cfg.dual_view.side_background_path)
    if background is None:
        raise ValueError("configured platform side background is unreadable")
    input_path = Path(args.input_records) if args.input_records else run / "holdout-fusion-records.jsonl"
    inputs = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    pipelines = {}
    records, rows, sheets, latencies = [], [], [], []
    safety_failures = {"DEPTH_INVALID", "NO_GRASP_SURFACE", "OCCLUDED"}
    for index, source in enumerate(inputs, 1):
        directory = Path(source["sample"])
        batch = directory.parents[1]
        if str(batch) not in pipelines:
            empty = sorted((batch / "empty_tray").glob("sample-*"))[0]
            pipeline = VisionPipeline3D(config=cfg, background_frame=_load_primary_frame(empty), shape_model=top)
            fusion = DualViewFusion(calibration, background, side, cfg.dual_view,
                                    pipeline.calibration.tray_plane_camera, policy,
                                    registry.registry_hash, shape_registry=registry, cross_view_model=cross)
            pipelines[str(batch)] = pipeline, fusion
        pipeline, fusion = pipelines[str(batch)]
        primary = _load_primary_frame(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        side_frame = RGBFrame(cv2.imread(str(directory / "side-color.png")), metadata["side_timestamp_ns"], metadata["side_frame_id"])
        pair = SynchronizedFramePair(primary, side_frame, metadata["primary_host_timestamp_ns"],
                                     metadata["side_host_timestamp_ns"], metadata.get("max_pair_delta_ms", 50.0))
        pipeline.dual_view_fusion = None
        pipeline.reset_tracking()
        baseline = pipeline.process(primary)
        baseline_status = {item.object_id: item.status.value for item in baseline}
        pipeline.dual_view_fusion = fusion
        pipeline.reset_tracking()
        started = time.perf_counter()
        results = pipeline.process_pair(pair)
        elapsed = (time.perf_counter() - started) * 1000
        latencies.append(elapsed)
        items = [item.to_dict() for item in results]
        illegal = [item.object_id for item in results if baseline_status.get(item.object_id) in safety_failures and
                   (item.status.value == "PICKABLE" or item.selected)]
        accepted = [item for item in results if item.status.value == "PICKABLE"]
        record = {"sample": str(directory), "primary_frame_id": primary.frame_id, "side_frame_id": side_frame.frame_id,
                  "true_label": source["true_label"], "true_color": metadata.get("color_id"), "health": pipeline.health(),
                  "results": items, "baseline_status": baseline_status, "illegal_safety_upgrades": illegal,
                  "wrong_pickable_count": sum(item.shape_id != source["true_label"] for item in accepted),
                  "capture_calibration_hash": metadata.get("calibration_hash"),
                  "runtime_calibration_hash": calibration.calibration_hash, "latency_ms": elapsed,
                  "cold_start": True, "motion_executed": False}
        records.append(record)
        title = f"{primary.frame_id} " + ";".join(f"{item.shape_id}:{item.status.value}" for item in results)
        top_image = pipeline.annotate(primary, results)
        if args.show_top_edges:
            scale = cfg.rgbd.processing_scale
            working = resize_rgbd_frame(primary, scale)
            for result in results:
                if result.crop_image is None or result.rgb_crop_mask is None:
                    continue
                origin = np.rint(np.asarray(result.diagnostics["rgb_crop_origin_uv"]) * scale).astype(int)
                observed = np.where(result.depth_valid_crop_mask > 0, result.depth_crop, 0).astype(np.float32)
                topology = extract_face_topology(observed, result.rgb_crop_mask, working.intrinsics,
                    tuple(origin.tolist()), color_crop_bgr=result.crop_image, extract_fused_edges=True)
                contours, _ = cv2.findContours(result.rgb_crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = max(contours, key=cv2.contourArea)
                corners = cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True).reshape(-1, 2)
                for point in corners:
                    uv = np.rint((point + origin) / scale).astype(int)
                    cv2.circle(top_image, tuple(uv), 5, (0, 220, 255), -1)
                for edge in topology.fused_edges:
                    endpoints = np.rint((edge.points() + origin) / scale).astype(int)
                    cv2.line(top_image, tuple(endpoints[0]), tuple(endpoints[1]), (255, 0, 255), 3)
                for edge in topology.rejected_rgb_edges:
                    endpoints = np.rint((edge.points() + origin) / scale).astype(int)
                    cv2.line(top_image, tuple(endpoints[0]), tuple(endpoints[1]), (0, 140, 255), 1)
                record.setdefault("audit_top_geometry", {})[result.object_id] = {
                    "silhouette_corners_2d": corners.tolist(), "visible_faces": len(topology.faces),
                    "depth_supported_edges": len(topology.fused_edges), "unresolved_rgb_lines": len(topology.rejected_rgb_edges),
                    "extra_audit_time_excluded": True}
        side_image = fusion.annotate_side(side_frame.color_bgr, results,
            show_geometry=args.show_side_geometry, show_edges=args.show_side_edges,
            show_sparse_geometry=args.show_sparse_geometry)
        rows.append(np.hstack((_tile(top_image, title), _tile(depth_preview(primary.depth_mm), f"depth pair={pair.pair_delta_ms:.1f}ms"),
                               _tile(side_image, f"true={source['true_label']} actual application"))))
        if len(rows) == 6 or index == len(inputs):
            path = output / f"application-{len(sheets) + 1:03d}.jpg"
            cv2.imwrite(str(path), np.vstack(rows))
            sheets.append(str(path))
            rows = []
        if index % 20 == 0:
            print(json.dumps({"replayed": index, "total": len(inputs)}), flush=True)
    (output / "application-records.jsonl").write_text("\n".join(json.dumps(record, ensure_ascii=False, allow_nan=False) for record in records) + "\n", encoding="utf-8")
    states = Counter(item.get("diagnostics", {}).get("dual_view", {}).get("fusion_state", "NO_DUAL_RESULT") for record in records for item in record["results"])
    primary_results = [max(record["results"], key=lambda item: item["diagnostics"].get("rgb_mask_pixels", 0)) if record["results"] else None for record in records]
    graph_metrics = [item["diagnostics"].get("dual_view", {}).get("cross_view_topology") or {} for record in records for item in record["results"]]
    graphs = [item["joint_topology_graph"] for item in graph_metrics if "joint_topology_graph" in item]
    summary = {"samples": len(records), "fusion_states": dict(states), "no_object_samples": sum(not record["results"] for record in records),
               "wrong_pickable_count": sum(record["wrong_pickable_count"] for record in records),
               "illegal_safety_upgrades": sum(len(record["illegal_safety_upgrades"]) for record in records),
               "selected_count": sum(item["selected"] for record in records for item in record["results"]),
               "primary_shape_accuracy_including_unknown": sum(item is not None and item["shape_id"] == record["true_label"] for item, record in zip(primary_results, records)) / max(len(records), 1),
               "primary_color_accuracy_including_unknown": sum(item is not None and item["color_id"] == record["true_color"] for item, record in zip(primary_results, records)) / max(len(records), 1),
               "graphs_evaluated": len(graphs),
               "graph_edge_count_median": float(np.median([len(graph["edges"]) for graph in graphs])) if graphs else 0.0,
               "graph_shared_edge_count": sum(len(edge["sources"]) > 1 for graph in graphs for edge in graph["edges"]),
               "pipeline_latency_p95_ms": float(np.percentile(latencies, 95)),
               "calibration_hash_mismatch_samples": sum(record["capture_calibration_hash"] != calibration.calibration_hash for record in records),
               "cold_start": True, "promotion_eligible": False,
               "limitations": ["SAME_BATCH_REUSED_HOLDOUT", "NO_ACQUISITION_OR_ROBOT_TIMING", "NO_MECHANICAL_EXECUTION", "COLD_START_NOT_TEMPORAL_STABILITY_TEST"]}
    (output / "application-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
