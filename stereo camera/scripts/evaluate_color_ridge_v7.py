"""Frozen offline metrics and controlled stage ablations for the v7 experiment."""
from __future__ import annotations
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
from experiment_color_ridge_v7 import ROOT, DEFAULT_OUTPUT, read_features, write_json, bootstrap_config
from sorting_vision.classification import LabColorClassifier
from sorting_vision.color_v7 import CameraColorProfile, ColorFrameContext, canonical_color, color_vector
from sorting_vision.geometry_edges import extract_edge_topology
from sorting_vision.ridge_v7 import extract_ridges
from sorting_vision.side_geometry import extract_side_features
from sorting_vision.sparse_stereo import SparseStereoLimits, reconstruct
from sorting_vision.dual_view import DualViewCalibration
from sorting_vision.visual_contract import V7, LEGACY, visual_contract


def cached(output, identifier):
    path = output / "features" / (hashlib.sha256(identifier.encode()).hexdigest()[:20] + ".npz")
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["contract"][0]) != V7:
            raise ValueError("stale evaluation cache")
        return {key:data[key].copy() for key in data.files}


def macro_color(rows):
    labels = sorted({label for label,_ in rows})
    recalls = {label: sum(true == label and pred == label for true,pred in rows)/max(1,sum(true==label for true,_ in rows)) for label in labels}
    return {"samples":len(rows), "macro_recall":float(np.mean(list(recalls.values()))) if recalls else 0.,
        "recall_per_color":recalls, "wrong_acceptance_count":sum(pred not in {"unknown", true} for true,pred in rows),
        "unknown_count":sum(pred=="unknown" for _,pred in rows)}


def contour_metrics(predicted, annotation):
    truth = np.zeros(predicted.shape, np.uint8)
    cv2.fillPoly(truth, [np.rint(annotation["visible_contour"]).astype(np.int32)], 255)
    union = (truth>0)|(predicted>0)
    intersection = (truth>0)&(predicted>0)
    n = cv2.connectedComponents((predicted>0).astype(np.uint8))[0]-1
    return {"iou":float(intersection.sum()/max(1,union.sum())),
        "missed_fraction":float(((truth>0)&(predicted==0)).sum()/max(1,(truth>0).sum())),
        "excess_fraction":float(((truth==0)&(predicted>0)).sum()/max(1,(predicted>0).sum())),
        "component_count":n, "split_flag":int(n>1), "empty_flag":int(n==0)}


def line_metrics(lines, annotation, shape, internal_only=False):
    true_lines = annotation["true_internal_ridges"] + ([] if internal_only else annotation["straight_boundary"])
    truth = np.zeros(shape,np.uint8)
    for a,b in true_lines:
        cv2.line(truth,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),255,1)
    predicted = np.zeros(shape,np.uint8)
    for line in np.asarray(lines).reshape(-1,4):
        endpoints = np.rint(line.reshape(2,2)).astype(int)
        cv2.line(predicted,tuple(endpoints[0]),tuple(endpoints[1]),255,1)
    # Coarse manual point placement sets an explicit, unchanged tolerance for
    # all methods. This is a 2D development metric, not metric 3D accuracy.
    tolerance = max(4., annotation["placement_uncertainty_native_px"]*2)
    dt_true = cv2.distanceTransform(255-truth,cv2.DIST_L2,3)
    dt_pred = cv2.distanceTransform(255-predicted,cv2.DIST_L2,3)
    tp_pred = int(np.count_nonzero((predicted>0)&(dt_true<=tolerance)))
    tp_true = int(np.count_nonzero((truth>0)&(dt_pred<=tolerance)))
    return {"predicted_support_pixels":int(np.count_nonzero(predicted)),"true_support_pixels":int(np.count_nonzero(truth)),
        "matched_predicted_pixels":tp_pred,"matched_true_pixels":tp_true,"tolerance_px":tolerance,
        "candidate_count":len(lines),"true_line_count":len(true_lines)}


def aggregate_line(rows):
    counts = {key:sum(row[key] for row in rows) for key in ("predicted_support_pixels","true_support_pixels","matched_predicted_pixels","matched_true_pixels","candidate_count","true_line_count")}
    return {**counts,"precision":counts["matched_predicted_pixels"]/max(1,counts["predicted_support_pixels"]),
        "recall":counts["matched_true_pixels"]/max(1,counts["true_support_pixels"]),"metric":"tolerant raster support; candidates, not confirmed 3D"}


def records_from(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def app_metrics(records):
    positions, errors, stages, side_colors, primary_colors, ridge_counts = [], [], defaultdict(list), [], [], []
    for record in records:
        item = max(record["results"], key=lambda value:value["diagnostics"].get("rgb_mask_pixels",0)) if record["results"] else None
        label = canonical_color(record["true_color"])
        primary_colors.append((label,item["color_id"] if item else "unknown"))
        diag = item["diagnostics"] if item else {}
        side_colors.append((label,(diag.get("dual_view",{}).get("cross_view_topology") or {}).get("color",{}).get("label","unknown")))
        sparse = (diag.get("dual_view",{}).get("cross_view_topology") or {}).get("sparse_stereo",{})
        nodes = sparse.get("nodes",[])
        positions.append(len(nodes))
        ridge_counts.append(len(diag.get("confirmed_ridges_3d",[])))
        errors.extend(max(node["primary_error_px"],node["side_error_px"]) for node in nodes)
        for key,value in record.get("health",{}).get("stages_ms",{}).items():
            stages[key].append(value)
        for key,value in diag.get("dual_view",{}).get("stages_ms",{}).items():
            stages["side_"+key].append(value)
        if "matching_and_3d_ms" in sparse:
            stages["matching_and_3d"].append(sparse["matching_and_3d_ms"])
    return {"samples":len(records),"primary_color":macro_color(primary_colors),"side_color":macro_color(side_colors),
        "corner_coverage":sum(value>0 for value in positions)/max(1,len(records)),"confirmed_corner_count":sum(positions),
        "geometry_supported_ridge_coverage":sum(value>0 for value in ridge_counts)/max(1,len(records)),
        "geometry_supported_ridge_count":sum(ridge_counts),
        "reprojection_p95_px":float(np.percentile(errors,95)) if errors else None,
        "latency_p95_ms":float(np.percentile([row["latency_ms"] for row in records],95)) if records else None,
        "illegal_safety_upgrades":sum(len(row["illegal_safety_upgrades"]) for row in records),
        "selected_count":sum(item["selected"] for row in records for item in row["results"]),
        "status_counts":dict(Counter(item["status"] for row in records for item in row["results"])),
        "stage_p95_ms":{key:float(np.percentile(values,95)) for key,values in stages.items()}}


def main():
    cv2.setNumThreads(2)
    output = ROOT / DEFAULT_OUTPUT
    manifest = json.loads((output/"split-manifest.json").read_text(encoding="utf-8"))
    correction = json.loads((output/"camera-corrections.json").read_text(encoding="utf-8"))
    annotations = json.loads((output/"review/manual-annotations.json").read_text(encoding="utf-8"))["records"]
    cfg = bootstrap_config()
    old_color = LabColorClassifier(cfg.classification)
    profiles = {camera:CameraColorProfile.load(output/f"{camera}-color-profile.json",camera) for camera in ("primary","side")}
    color_metrics = {}
    for partition,key in (("calibration","calibration_ids"),("regression","regression_ids")):
        rows = read_features(output,manifest[key])
        by_camera = {}
        for camera in profiles:
            predictions, legacy_predictions = [], []
            for row in rows:
                pred, legacy = "unknown", "unknown"
                if not row.get("failed") and np.isfinite(row[f"{camera}_color"]).all():
                    pred = profiles[camera].predict_vector(row[f"{camera}_color"]).label_id
                true = canonical_color(json.loads((ROOT/row["sample"]/"metadata.json").read_text(encoding="utf-8"))["color_id"])
                predictions.append((true,pred))
                data = cached(output,row["sample"])
                if data is not None:
                    image = cv2.imread(str(ROOT/row["sample"]/f"{camera}-color.png"))
                    legacy = old_color.classify(image,data[f"{camera}_mask"]).label_id
                legacy_predictions.append((true,legacy))
            by_camera[camera] = macro_color(predictions)
            by_camera[camera]["same_mask_legacy_hex"] = macro_color(legacy_predictions)
        color_metrics[partition] = by_camera
    masks, ridges, colors, matching = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
    details = []
    calibration = DualViewCalibration.load(cfg.dual_view.calibration_path)
    for index, record in enumerate(annotations,1):
        identifier = record["sample"]
        data = cached(output,identifier)
        if data is None:
            details.append({"sample":identifier,"extraction_failed":True,"denominator_retained":True})
            continue
        partition = record["partition"]
        for camera in ("primary","side"):
            image = cv2.imread(str(ROOT/identifier/f"{camera}-color.png"))
            mask = data[f"{camera}_mask"]
            ctx = ColorFrameContext.create(image,correction[Path(identifier).parents[1].name][camera])
            hsv = ctx.hsv
            minimum_s = cfg.rgbd.hsv_min_saturation if camera=="primary" else cfg.dual_view.side_hsv_min_saturation
            minimum_v = cfg.rgbd.hsv_min_value if camera=="primary" else cfg.dual_view.side_hsv_min_value
            seed = ((hsv[:,:,1]>=minimum_s)&(hsv[:,:,2]>=minimum_v)&(mask>0)).astype(np.uint8)*255
            annotation = record["cameras"][camera]
            for name,predicted in (("full_v7",mask),("without_mask_growth",seed)):
                masks[partition,camera,name].append(contour_metrics(predicted,annotation))
            measured = extract_ridges(image,mask)
            with visual_contract(LEGACY):
                old_lines = (extract_side_features(image,mask).segments_px if camera=="side" else np.asarray(
                    [line.points().reshape(-1) for line in extract_edge_topology(image,mask).merged_lines],np.float32).reshape(-1,4))
            # LSD ablation retains the same recovered mask and original input.
            for name,lines in (("full_v7",measured.segments),("without_ridge_upgrade",old_lines)):
                ridges[partition,camera,name].append(line_metrics(lines,annotation,mask.shape))
            raw_prediction = old_color.classify(image,mask).label_id
            old_corrected = old_color.classify(ctx.corrected,mask).label_id
            new_prediction = profiles[camera].classify(ctx,mask).label_id
            true = canonical_color(record["raw_color"])
            for name,pred in (("full_v7",new_prediction),("without_color_profile",old_corrected),("legacy_hex_raw",raw_prediction)):
                colors[partition,camera,name].append((true,pred))
        # Matching ablation holds masks, calibration and strict gates fixed,
        # comparing legacy corners/DLT with stable candidates/refinement.
        images = [cv2.imread(str(ROOT/identifier/f"{camera}-color.png")) for camera in ("primary","side")]
        spatial = {}
        for name,contract in (("full_v7",V7),("without_matching_upgrade",LEGACY)):
            with visual_contract(contract):
                graph = reconstruct(images[0],data["primary_mask"],images[1],data["side_mask"],data["points"],calibration,SparseStereoLimits())
            nodes = graph["nodes"]
            gt = record["confirmed_correspondences"]
            confirmed_correct = sum(any(np.linalg.norm(np.asarray(node["primary_uv"])-pair["primary_uv"])<=8 and
                np.linalg.norm(np.asarray(node["side_uv"])-pair["side_uv"])<=8 for pair in gt) for node in nodes)
            spatial[name] = {"sample":identifier,"nodes":len(nodes),"edges":len(graph["edges"]),
                "errors":[max(node["primary_error_px"],node["side_error_px"]) for node in nodes],
                "confirmable_ground_truth_pairs":len(gt),"matched_confirmable_pairs":confirmed_correct,
                "unverifiable_predictions":len(nodes)-confirmed_correct,
                "position_sensitivity_max_mm":[node["position_sensitivity"]["maximum_shift_mm"] for node in nodes if "position_sensitivity" in node]}
            matching[partition,name].append(spatial[name])
        details.append({"sample":identifier,"spatial":spatial})
        if index%12==0:
            print(f"annotated stage evaluation {index}/48",flush=True)
    mask_summary = {"/".join(key):{**{name:float(np.mean([row[name] for row in rows])) for name in ("iou","missed_fraction","excess_fraction")},
        "samples":len(rows),"split_samples":sum(row["split_flag"] for row in rows),"empty_samples":sum(row["empty_flag"] for row in rows),
        "merge_accuracy":None,"merge_limit":"single-object real captures; use synthetic touching tests"} for key,rows in masks.items()}
    spatial_summary = {}
    for key,rows in matching.items():
        errors = [error for row in rows for error in row["errors"]]
        denominator = 24
        spatial_summary["/".join(key)] = {"denominator":denominator,"evaluated":len(rows),"coverage":sum(row["nodes"]>0 for row in rows)/denominator,
            "nodes":sum(row["nodes"] for row in rows),"reprojection_p95_px":float(np.percentile(errors,95)) if errors else None,
            "confirmable_pairs":sum(row["confirmable_ground_truth_pairs"] for row in rows),
            "matched_confirmable_pairs":sum(row["matched_confirmable_pairs"] for row in rows),
            "correspondence_accuracy":None,"reason":"insufficient independent confirmed correspondences; residual is not correctness"}
    app = {partition:app_metrics(records_from(output/f"application-{partition}/application-records.jsonl")) for partition in ("calibration","regression")}
    baseline_path = ROOT/"output/dual-color-first-v6-20260918/application-frozen108-review/application-records.jsonl"
    baseline = app_metrics(records_from(baseline_path))
    full = app["regression"]
    # Preconditions are retained even when a metric cannot yet be established.
    verdict = {"color_macro_improves_both_and_errors_do_not_increase": all(
        color_metrics["regression"][camera]["macro_recall"] > color_metrics["regression"][camera]["same_mask_legacy_hex"]["macro_recall"] and
        color_metrics["regression"][camera]["wrong_acceptance_count"] <= color_metrics["regression"][camera]["same_mask_legacy_hex"]["wrong_acceptance_count"] for camera in ("primary","side")),
        "ridge_recall_improves_without_precision_loss": all(
            aggregate_line(ridges["regression",camera,"full_v7"])["recall"] > aggregate_line(ridges["regression",camera,"without_ridge_upgrade"])["recall"] and
            aggregate_line(ridges["regression",camera,"full_v7"])["precision"] >= aggregate_line(ridges["regression",camera,"without_ridge_upgrade"])["precision"] for camera in ("primary","side")),
        "spatial_coverage_preserved_and_residual_improved": all(
            spatial_summary[partition+"/full_v7"]["coverage"] >= spatial_summary[partition+"/without_matching_upgrade"]["coverage"] and
            spatial_summary[partition+"/full_v7"]["reprojection_p95_px"] is not None and
            spatial_summary[partition+"/without_matching_upgrade"]["reprojection_p95_px"] is not None and
            spatial_summary[partition+"/full_v7"]["reprojection_p95_px"] < spatial_summary[partition+"/without_matching_upgrade"]["reprojection_p95_px"]
            for partition in ("calibration","regression")),
        "spatial_correspondence_correctness": "UNVERIFIED_INSUFFICIENT_CORRESPONDENCE_TRUTH",
        "no_illegal_depth_safety_upgrade":full["samples"]==108 and full["illegal_safety_upgrades"]==0,
        "promotion_eligible":False}
    write_json(output/"evaluation-metrics.json",{"version":7,"visual_contract":V7,"frozen_population_hash":manifest["frozen_population_hash"],
        "color_all_candidates":color_metrics,"application":app,"historical_v6_application":baseline,
        "mask_annotations":mask_summary,"ridge_annotations":{"/".join(key):aggregate_line(rows) for key,rows in ridges.items()},
        "color_ablation":{"/".join(key):macro_color(rows) for key,rows in colors.items()},"matching_ablation":spatial_summary,
        "acceptance":verdict,"limitations":["NO_INDEPENDENT_3D_TRUTH","COARSE_CODEX_VISUAL_ANNOTATIONS_NOT_HUMAN_GROUND_TRUTH",
        "STAGE_ABLATIONS_HOLD_OTHER_INPUTS_FIXED_NOT_RETRAINED_END_TO_END_MODELS","CORRESPONDENCE_IDENTITIES_MOSTLY_AMBIGUOUS","MISSING_RED_CURVES_AND_CYLINDERS"]})
    write_json(output/"evaluation-details.json",details)
    print(json.dumps(verdict,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
