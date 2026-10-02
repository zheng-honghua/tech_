"""Reproducible isolated v7 development run; never promotes or edits captures."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from sorting_vision.color_v7 import CameraColorProfile, ColorFrameContext, canonical_color, color_vector, neutral_correction
from sorting_vision.config import load_config
from sorting_vision.cross_view_topology import CrossViewTopologyModel, extract_cross_view_features
from sorting_vision.dual_view import DualViewCalibration
from sorting_vision.geometry_rgbd_model import DepthGeometryModel, FUSED_FEATURE_NAMES
from sorting_vision.group_holdout import deterministic_group_holdout
from sorting_vision.pipeline3d import VisionPipeline3D
from sorting_vision.ridge_v7 import extract_ridges
from sorting_vision.shape_registry import ShapeRegistry
from sorting_vision.side_geometry import SideGeometryModel, SideTrainingSample, extract_side_features
from sorting_vision.visual_contract import V7, V7_PARAMETERS, visual_contract
from train_dual_fusion_holdout import _RecordingShapeModel, _extract_top, _load_primary_frame, _select_policy

DEFAULT_OUTPUT = "output/dual-color-ridge-v7-20261001"
ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def relative_id(value):
    text = str(value).replace("\\", "/")
    if "data/dual/temporary/" in text:
        return "data/dual/temporary/" + text.split("data/dual/temporary/", 1)[1]
    return Path(value).relative_to(ROOT).as_posix()


def metadata(identifier):
    return json.loads((ROOT / identifier / "metadata.json").read_text(encoding="utf-8"))


def category(identifier):
    label = Path(identifier).parent.name
    return "curved" if label in {"cone", "cylinder"} else "prism" if label.endswith("prism") else "pyramid"


def stratified24(identifiers):
    groups = defaultdict(list)
    for identifier in sorted(identifiers, key=lambda value: (Path(value).parent.name not in {"square_pyramid", "octahedron"}, value)):
        groups[canonical_color(metadata(identifier)["color_id"]), category(identifier)].append(identifier)
    chosen = [identifier for key in sorted(groups) for identifier in groups[key][:2]]
    for identifier in sorted(identifiers):
        if len(chosen) >= 24:
            break
        if identifier not in chosen:
            chosen.append(identifier)
    return chosen


def preview_crop(identifier, camera):
    image = cv2.imread(str(ROOT / identifier / f"{camera}-color.png"))
    if image is None:
        raise ValueError(f"unreadable image: {identifier}/{camera}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    foreground = ((hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 10)).astype(np.uint8) * 255
    if camera == "side":
        # Review thumbnail only: keep the observed tray area rather than the
        # large orange equipment above it. This is never a runtime ROI.
        foreground[:int(image.shape[0] * .35)] = 0
        foreground[:, :int(image.shape[1] * .18)] = 0
        foreground[:, int(image.shape[1] * .85):] = 0
    contours = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = [c for c in contours if 50 < cv2.contourArea(c) < image.shape[0] * image.shape[1] * .30]
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        pad = max(35, int(max(w, h) * .3))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
    else:
        x0, y0, x1, y1 = 0, 0, image.shape[1], image.shape[0]
    return image[y0:y1, x0:x1], (x0, y0, x1, y1)


def contact_sheets(output, name, identifiers, rows_per_sheet=6):
    folder = output / "review"
    folder.mkdir(exist_ok=True)
    sheets, transforms = [], []
    for offset in range(0, len(identifiers), rows_per_sheet):
        rows = []
        for identifier in identifiers[offset:offset + rows_per_sheet]:
            meta = metadata(identifier)
            row = np.full((290, 1280, 3), 245, np.uint8)
            label = f"{Path(identifier).parents[1].name}/{Path(identifier).parent.name}/{Path(identifier).name}"
            cv2.putText(row, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (10, 10, 10), 1, cv2.LINE_AA)
            cv2.putText(row, f"primary {meta['primary_frame_id']} | side {meta['side_frame_id']} | raw color {meta['color_id']}",
                        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, .45, (10, 10, 10), 1, cv2.LINE_AA)
            for column, camera in enumerate(("primary", "side")):
                crop, box = preview_crop(identifier, camera)
                scale = min(620 / crop.shape[1], 230 / crop.shape[0])
                thumb = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                left, top = column * 640 + (640 - thumb.shape[1]) // 2, 54
                row[top:top + thumb.shape[0], left:left + thumb.shape[1]] = thumb
                transforms.append({"sample": identifier, "camera": camera, "sheet": f"{name}-{offset // rows_per_sheet + 1:02d}.jpg",
                    "row": len(rows), "box_native": list(box), "scale": scale, "thumbnail_left": left, "thumbnail_top": top})
            rows.append(row)
        path = folder / f"{name}-{offset // rows_per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 94])
        sheets.append(path.relative_to(output).as_posix())
    write_json(folder / f"{name}-coordinates.json", transforms)
    return sheets


def prepare(output):
    v6 = json.loads((ROOT / "output/dual-color-first-v6-20260918/bundle-manifest.json").read_text(encoding="utf-8"))
    training = [relative_id(value) for value in v6["training_ids"]]
    external = [relative_id(json.loads(line)["sample"]) for line in
                (ROOT / "output/dual-joint-v4-holdout-20260917/holdout-fusion-records.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(training) != 420 or len(external) != 108 or set(training) & set(external):
        raise ValueError("frozen candidate/holdout population changed")
    split = deterministic_group_holdout(training, group_key=lambda value: str(Path(value).parent.parent / Path(value).parent.name),
        item_key=lambda value: value, holdout_per_group=2, seed=20260915)
    colors = defaultdict(list)
    for identifier in split.training:
        colors[canonical_color(metadata(identifier)["color_id"])].append(identifier)
    color_training = []
    for color in sorted(colors):
        selected, shapes = [], set()
        for identifier in sorted(colors[color], key=lambda value: (category(value), value)):
            shape = Path(identifier).parent.name
            if shape not in shapes:
                selected.append(identifier)
                shapes.add(shape)
            if len(selected) == 6:
                break
        if len(selected) != 6:
            raise ValueError("six distinct shapes per colour unavailable")
        color_training.extend(selected)
    manifest = {"version": 7, "visual_contract": V7, "parameters": V7_PARAMETERS,
        "promotion_eligible": False, "training_candidates": training, "fit_ids": list(split.training),
        "calibration_ids": list(split.holdout), "regression_ids": external, "color_training_ids": color_training,
        "annotation_calibration_ids": stratified24(split.holdout), "annotation_regression_ids": stratified24(external),
        "group_counts": split.group_counts, "source_marks_modified": False, "seed": 20260915,
        "coordinate_contract": "native_input_pixels; resized_primary_intrinsics; restored_before_dual_fusion",
        "thresholds": {"epipolar_px": 3., "reprojection_px": 2., "ray_angle_deg": 2., "depth_mm": 5.}}
    manifest["frozen_population_hash"] = hashlib.sha256(json.dumps([training, external], sort_keys=True).encode()).hexdigest()
    write_json(output / "split-manifest.json", manifest)
    sheets = {}
    for name, identifiers in (("color-training", color_training), ("calibration", split.holdout), ("regression", external),
                              ("annotation-calibration", manifest["annotation_calibration_ids"]),
                              ("annotation-regression", manifest["annotation_regression_ids"])):
        sheets[name] = contact_sheets(output, name, identifiers)
    write_json(output / "review/contact-manifest.json", sheets)
    print(json.dumps({"fit": len(split.training), "calibration": len(split.holdout), "regression": len(external), "sheets": sheets}), flush=True)


def bootstrap_config():
    cfg = load_config(ROOT / "config/dual/temporary-color-ridge-v7.yaml")
    return replace(cfg, classification=replace(cfg.classification, color_profile_path=""),
                   dual_view=replace(cfg.dual_view, side_color_profile_path=""))


def prepare_annotations(output):
    manifest = json.loads((output / "split-manifest.json").read_text(encoding="utf-8"))
    contacts = json.loads((output / "review/contact-manifest.json").read_text(encoding="utf-8"))
    for split, identifiers in (("calibration", manifest["calibration_ids"]), ("regression", manifest["regression_ids"])):
        selected = stratified24(identifiers)
        manifest[f"annotation_{split}_ids"] = selected
        contacts[f"annotation-{split}"] = contact_sheets(output, f"annotation-{split}", selected)
    write_json(output / "split-manifest.json", manifest)
    write_json(output / "review/contact-manifest.json", contacts)


def extract(output):
    manifest = json.loads((output / "split-manifest.json").read_text(encoding="utf-8"))
    if manifest["visual_contract"] != V7:
        raise ValueError("stale v7 split contract")
    cfg = bootstrap_config()
    calibration = DualViewCalibration.load(cfg.dual_view.calibration_path)
    base = None  # Recording only; a legacy model never consumes v7 features.
    registry = ShapeRegistry.load(cfg.dual_view.shape_registry_path)
    cache = output / "features"
    cache.mkdir(exist_ok=True)
    pipelines, corrections = {}, {}
    failures = []
    identifiers = manifest["training_candidates"] + manifest["regression_ids"]
    for index, identifier in enumerate(identifiers, 1):
        directory = ROOT / identifier
        target = cache / (hashlib.sha256(identifier.encode()).hexdigest()[:20] + ".npz")
        signature = hashlib.sha256()
        signature.update(V7.encode())
        for name in ("metadata.json", "primary-color.png", "primary-depth.npy", "side-color.png"):
            signature.update((directory / name).read_bytes())
        source_hash = signature.hexdigest()
        if target.exists():
            with np.load(target, allow_pickle=False) as existing:
                if str(existing["source_hash"][0]) == source_hash:
                    continue
        started = time.perf_counter()
        try:
            meta = metadata(identifier)
            batch = directory.parents[1]
            if str(batch) not in pipelines:
                empty = sorted((batch / "empty_tray").glob("sample-*"))[0]
                recorder = _RecordingShapeModel(base, True)
                recorder.edge_parameters = {"feature_contract": V7}
                pipeline = VisionPipeline3D(config=cfg, background_frame=_load_primary_frame(empty), shape_model=recorder)
                background = cv2.imread(str(empty / "side-color.png"))
                correction = neutral_correction(background)
                pipelines[str(batch)] = pipeline, recorder, background, correction
                corrections[batch.name] = {"primary": pipeline._color_correction, "side": correction,
                    "source": empty.relative_to(ROOT).as_posix()}
            pipeline, recorder, background, correction = pipelines[str(batch)]
            image = cv2.imread(str(directory / "side-color.png"))
            # Shared side association uses the same corrected colour, but
            # luminance/LSD and model inputs retain the observed original RGB.
            ctx = ColorFrameContext.create(image, correction)
            sample = SideTrainingSample(directory, image, np.zeros(image.shape[:2], np.uint8),
                registry.resolve(Path(identifier).parent.name), "train", batch.name, str(meta.get("instance_id", "")),
                canonical_color(meta["color_id"]), source_hash)
            with visual_contract(V7):
                pair = _extract_top(sample, pipeline, recorder, calibration, background, color_context=ctx)
                side_features = extract_side_features(image, pair.side.mask)
                cross_features = extract_cross_view_features(image, pair.side.mask, pair.primary_points, calibration)
                side_ridges = extract_ridges(image, pair.side.mask)
            primary_frame = _load_primary_frame(directory)
            # Extracted primary result's visible mask is restored to native
            # coordinates for colour annotation/evaluation only.
            chosen = pair.primary_result
            from sorting_vision.sparse_stereo import restore_native_mask
            primary_mask = restore_native_mask(chosen.rgb_crop_mask, chosen.diagnostics["rgb_crop_origin_uv"],
                chosen.diagnostics.get("processing_scale", 1.), primary_frame.color_bgr.shape)
            primary_ctx = ColorFrameContext.create(primary_frame.color_bgr, pipeline._color_correction)
            vectors = []
            for color_ctx, mask in ((primary_ctx, primary_mask), (ctx, pair.side.mask)):
                try:
                    vectors.append(color_vector(color_ctx, mask))
                except ValueError:
                    vectors.append(np.full(5, np.nan))
            payload = {"sample": identifier, "label": sample.label_id, "raw_color": meta["color_id"],
                "color": sample.color_id, "topology_quality": pair.topology_quality,
                "side_quality": min(pair.side_quality_override if pair.side_quality_override is not None else 0.,
                                    float(cross_features.diagnostics.get("spatial_quality", 0.))),
                "primary_ridges": chosen.diagnostics.get("candidate_ridges_2d", []),
                "side_ridges": side_ridges.evidence, "cross_view_topology": cross_features.diagnostics,
                "primary_frame_id": meta["primary_frame_id"], "side_frame_id": meta["side_frame_id"],
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "source_hash": source_hash, "feature_contract": V7}
            np.savez_compressed(target, source_hash=np.asarray([source_hash]), contract=np.asarray([V7]),
                top=pair.top_features, side=side_features.vector, cross=cross_features.vector,
                side_groups=side_features.group_ids, cross_groups=cross_features.group_ids,
                primary_color=vectors[0], side_color=vectors[1], primary_mask=primary_mask, side_mask=pair.side.mask,
                points=pair.primary_points, payload=np.asarray([json.dumps(payload, ensure_ascii=False, allow_nan=False)]))
        except (ValueError, KeyError, OSError) as error:
            failures.append({"sample": identifier, "reason": str(error), "denominator_retained": True})
        if index % 12 == 0:
            print(f"extracted {index}/{len(identifiers)} failures={len(failures)}", flush=True)
    write_json(output / "extraction-failures.json", failures)
    if corrections:
        write_json(output / "camera-corrections.json", corrections)


def read_features(output, identifiers):
    rows = []
    for identifier in identifiers:
        path = output / "features" / (hashlib.sha256(identifier.encode()).hexdigest()[:20] + ".npz")
        if not path.exists():
            rows.append({"sample": identifier, "failed": True})
            continue
        with np.load(path, allow_pickle=False) as data:
            if str(data["contract"][0]) != V7:
                raise ValueError("cached visual contract mismatch")
            rows.append({**json.loads(str(data["payload"][0])), **{key: data[key].copy() for key in
                ("top", "side", "cross", "side_groups", "cross_groups", "primary_color", "side_color")}})
    return rows


def reconcile_quality(output):
    """Use original-image sharpness in cached calibration, exactly as runtime.

    The initial extraction used corrected RGB for association and sharpness.
    Masks/features are identical; this migration updates only the quality
    gate, records its revision, and never rewrites source captures.
    """
    from sorting_vision.dual_view import associate_side_color_components, measure_side_color_segmentation, projected_roi
    manifest = json.loads((output / "split-manifest.json").read_text(encoding="utf-8"))
    corrections = json.loads((output / "camera-corrections.json").read_text(encoding="utf-8"))
    cfg = bootstrap_config()
    calibration = DualViewCalibration.load(cfg.dual_view.calibration_path)
    backgrounds = {}
    for identifier in manifest["training_candidates"] + manifest["regression_ids"]:
        path = output / "features" / (hashlib.sha256(identifier.encode()).hexdigest()[:20] + ".npz")
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as cached:
            data = {key: cached[key].copy() for key in cached.files}
        payload = json.loads(str(data["payload"][0]))
        batch = Path(identifier).parents[1].name
        if batch not in backgrounds:
            empty = ROOT / corrections[batch]["source"]
            recorder = _RecordingShapeModel(None, True)
            recorder.edge_parameters = {"feature_contract": V7}
            backgrounds[batch] = VisionPipeline3D(config=cfg, background_frame=_load_primary_frame(empty), shape_model=recorder)
        image = cv2.imread(str(ROOT / identifier / "side-color.png"))
        ctx = ColorFrameContext.create(image, corrections[batch]["side"])
        roi = projected_roi(calibration, data["points"], backgrounds[batch].calibration.tray_plane_camera,
                            image.shape, cfg.dual_view.roi_padding_ratio)
        with visual_contract(V7):
            segmentation = associate_side_color_components(image, {"object": roi}, cfg.dual_view, ctx)["object"]
        _, _, quality = measure_side_color_segmentation(image, segmentation, cfg.dual_view)
        full_mask = np.zeros(image.shape[:2], np.uint8)
        x,y,w,h = segmentation.roi.bbox
        full_mask[y:y+h,x:x+w] = segmentation.mask
        if not np.array_equal(full_mask, data["side_mask"]):
            raise ValueError(f"quality reconciliation changed segmentation: {identifier}")
        payload["side_quality"] = min(quality, float(payload["cross_view_topology"].get("spatial_quality", 0.)))
        payload["quality_revision"] = "original_rgb_sharpness_v1"
        data["payload"] = np.asarray([json.dumps(payload, ensure_ascii=False, allow_nan=False)])
        np.savez_compressed(path, **data)
    write_json(output / "quality-reconciliation.json", {"revision": "original_rgb_sharpness_v1", "masks_verified_equal": True})


def normalized_model(rows, key, groups_key, cls, registry):
    raw = np.stack([row[key] for row in rows]).astype(np.float32)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale < 1e-5] = 1.
    groups = rows[0][groups_key]
    labels = np.array([row["label"] for row in rows])
    if set(labels) != set(registry.class_ids):
        raise ValueError("fit partition does not cover registry")
    args = ((raw - mean) / scale, labels, mean, scale)
    model = (cls(*args, groups, registry, method="rtrees") if cls is CrossViewTopologyModel
             else cls(*args, registry, group_ids=groups, method="rtrees"))
    distances = []
    for index, value in enumerate(model.features):
        all_distances = model._distances(value).astype(float)
        all_distances[index] = np.inf
        own = all_distances[labels == labels[index]]
        if np.isfinite(own).any():
            distances.append(float(own.min()))
    model.distance_threshold = max(.25, float(np.percentile(distances, 99)) * 1.35)
    return model


def score_features(row, top, side, cross):
    with visual_contract(V7):
        label, confidence, reason = top.predict_features(row["top"])
    scores = dict(top.last_class_scores)
    records = {}
    for prefix, model, raw in (("classic_side", side, row["side"]), ("side", cross, row["cross"])):
        with visual_contract(V7):
            _, probability, diagnostics = model.predict_features(raw)
        probabilities = dict(model.last_class_scores)
        winner, rejection = diagnostics["nearest_label"], diagnostics["reason"]
        records.update({f"{prefix}_prediction": winner, f"{prefix}_class_scores": probabilities,
                        f"{prefix}_reason": rejection, f"{prefix}_confidence": probability})
    return {"sample": str(ROOT / row["sample"]), "true_label": row["label"], "top_prediction": label,
        "top_reason": reason, "top_confidence": confidence, "top_class_scores": scores,
        "side_quality": row["side_quality"], "topology_quality": row["topology_quality"], **records}


def calibrate_models(rows, top, side, cross):
    """Acceptance thresholds use only the held-out internal calibration IDs."""
    observations = []
    for row in rows:
        distances = top._class_distances((row["top"]-top.mean)/top.scale)
        order = np.argsort(distances)
        best, second = int(order[0]), int(order[1])
        margin = float((distances[second]-distances[best])/max(float(distances[best]),.05))
        observations.append((row["label"],top.labels[best],float(distances[best]),margin))
    best_option = None
    for margin in (.04,.08,.12,.2,.3):
        thresholds, wrong_total, correct_total = [],0,0
        for label in top.labels:
            local = [value for value in observations if value[1]==label]
            candidates = sorted({.001,*[max(.001,value[2]+1e-6) for value in local]})
            scored = []
            for threshold in candidates:
                accepted = [value for value in local if value[2]<=threshold and value[3]>=margin]
                wrong = sum(true!=pred for true,pred,_,_ in accepted)
                correct = sum(true==pred for true,pred,_,_ in accepted)
                scored.append((wrong,-correct,threshold))
            wrong,minus_correct,threshold = min(scored)
            thresholds.append(threshold)
            wrong_total += wrong
            correct_total -= minus_correct
        option = (wrong_total,-correct_total,margin,thresholds)
        if best_option is None or option[:3]<best_option[:3]:
            best_option=option
    top.min_margin = best_option[2]
    top.thresholds = np.asarray(best_option[3],np.float32)
    report = {"top":{"wrong_acceptance":best_option[0],"correct_acceptance":-best_option[1],
        "margin":top.min_margin,"thresholds":dict(zip(top.labels,top.thresholds.tolist()))}}
    for key,model in (("side",side),("cross",cross)):
        observations=[]
        for row in rows:
            with visual_contract(V7):
                _,_,diag=model.predict_features(row[key])
            observations.append((row["label"],diag["nearest_label"],diag["distance"],diag["margin"]))
        distances=sorted({.001,*[value[2]+1e-6 for value in observations]})
        candidates=[]
        for distance in distances:
            for margin in (.04,.08,.12,.2,.3):
                accepted=[value for value in observations if value[2]<=distance and value[3]>=margin]
                wrong=sum(true!=pred for true,pred,_,_ in accepted)
                correct=sum(true==pred for true,pred,_,_ in accepted)
                candidates.append((wrong,-correct,distance,margin))
        best=min(candidates)
        model.distance_threshold,model.margin_threshold=best[2:]
        report[key]={"wrong_acceptance":best[0],"correct_acceptance":-best[1],
            "distance_threshold":best[2],"margin_threshold":best[3]}
    return report


def fit(output):
    manifest = json.loads((output / "split-manifest.json").read_text(encoding="utf-8"))
    review = json.loads((output / "review/visual-review.json").read_text(encoding="utf-8"))
    if not set(manifest["color_training_ids"]).issubset(review.get("reviewed_ids", [])):
        raise ValueError("colour training images require actual visual review first")
    fit_rows = [row for row in read_features(output, manifest["fit_ids"]) if not row.get("failed")]
    cal_rows = [row for row in read_features(output, manifest["calibration_ids"]) if not row.get("failed")]
    registry = ShapeRegistry.load(bootstrap_config().dual_view.shape_registry_path)
    with visual_contract(V7):
        top = DepthGeometryModel.fit(np.stack([row["top"] for row in fit_rows]), [row["label"] for row in fit_rows], feature_names=FUSED_FEATURE_NAMES)
        top.edge_parameters = {"input_contract":"rgb_silhouette_depth_owned_v2", "feature_contract":V7, "parameters":V7_PARAMETERS}
        side = normalized_model(fit_rows, "side", "side_groups", SideGeometryModel, registry)
        cross = normalized_model(fit_rows, "cross", "cross_groups", CrossViewTopologyModel, registry)
    model_calibration = calibrate_models(cal_rows, top, side, cross)
    records = [score_features(row, top, side, cross) for row in cal_rows]
    policy, policy_fit = _select_policy(records, registry.registry_hash, strict=True)
    top.save(output / "top-rgbd.npz", metadata={"fit_ids": manifest["fit_ids"], "calibration_not_refitted": True})
    side.save(output / "side-lsd.npz")
    cross.save(output / "joint-topology.npz")
    policy.save(output / "fusion-policy.json")
    color_rows = read_features(output, manifest["color_training_ids"])
    for camera in ("primary", "side"):
        vectors = [(row["raw_color"], row[f"{camera}_color"]) for row in color_rows if not row.get("failed")]
        profile = CameraColorProfile.fit(camera, vectors, {"fit_ids": manifest["color_training_ids"],
            "review_file": "review/visual-review.json", "calibration_ids": manifest["calibration_ids"], "calibration_not_refitted": True})
        # Independent calibration chooses the smallest threshold giving no
        # wrong colour acceptance; neither regression labels nor vectors enter.
        candidates = []
        for distance in (.25, .4, .55, .65, .8):
            for margin in (.08, .12, .18, .25):
                profile.distance_threshold, profile.margin_threshold = distance, margin
                predictions = [(row["color"], profile.predict_vector(row[f"{camera}_color"]).label_id)
                               for row in cal_rows if np.isfinite(row[f"{camera}_color"]).all()]
                wrong = sum(pred != "unknown" and pred != label for label, pred in predictions)
                correct = sum(pred == label for label, pred in predictions)
                candidates.append((wrong, -correct, distance, margin))
        chosen = min(candidates)
        profile.distance_threshold, profile.margin_threshold = chosen[2:]
        profile.save(output / f"{camera}-color-profile.json")
    (output / "holdout-fusion-records.jsonl").write_text("".join(json.dumps({"sample": str(ROOT / identifier),
        "true_label": registry.resolve(Path(identifier).parent.name)}, ensure_ascii=False) + "\n" for identifier in manifest["regression_ids"]), encoding="utf-8")
    (output / "calibration-replay-inputs.jsonl").write_text("".join(json.dumps({"sample": str(ROOT / identifier),
        "true_label": registry.resolve(Path(identifier).parent.name)}, ensure_ascii=False) + "\n" for identifier in manifest["calibration_ids"]), encoding="utf-8")
    write_json(output / "fit-report.json", {"fit_candidates": len(manifest["fit_ids"]), "fit_usable": len(fit_rows),
        "calibration_candidates": len(manifest["calibration_ids"]), "calibration_usable": len(cal_rows),
        "policy_fit": policy_fit, "registry_hash": registry.registry_hash, "calibration_hash": DualViewCalibration.load(bootstrap_config().dual_view.calibration_path).calibration_hash,
        "model_acceptance_calibration": model_calibration,
        "calibration_samples_refitted": False, "promotion_eligible": False})
    print(f"fit {len(fit_rows)}; calibrate {len(cal_rows)}; regression 108 untouched", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "extract", "fit", "prepare-annotations", "reconcile-quality"))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "extract": extract, "fit": fit, "prepare-annotations": prepare_annotations,
     "reconcile-quality": reconcile_quality}[args.phase](output)


if __name__ == "__main__":
    raise SystemExit(main())
