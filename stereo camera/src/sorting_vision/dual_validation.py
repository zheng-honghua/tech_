from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .dual_view import temperature_scale_scores
from .fusion_policy import FusionPolicy, fuse_top2_scores, top2_fusion_features
from .shape_registry import ShapeRegistry


def _depth_preview(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, np.float32)
    valid = np.isfinite(values) & (values > 0)
    normalized = np.zeros(values.shape, np.uint8)
    if np.any(valid):
        low, high = np.percentile(values[valid], [2, 98])
        if high <= low:
            high = low + 1.0
        normalized[valid] = np.clip(
            (values[valid] - low) * 255.0 / (high - low), 0, 255
        ).astype(np.uint8)
    return cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)


def _tile(image: np.ndarray, width: int, height: int, title: str) -> np.ndarray:
    canvas = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(canvas, (0, 0), (width, 28), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def build_dual_contact_sheets(
    samples_root: str | Path,
    output_dir: str | Path,
    columns: int = 1,
    tile_width: int = 360,
    tile_height: int = 240,
    annotations_root: str | Path | None = None,
) -> dict[str, Any]:
    """Render primary RGB, side RGB and depth for mandatory human review."""
    root = Path(samples_root)
    targets = sorted(
        path for path in root.rglob("metadata.json")
        if (path.parent / "primary-color.png").is_file()
        and (path.parent / "side-color.png").is_file()
        and (path.parent / "primary-depth.npy").is_file()
    )
    if not targets:
        raise ValueError("no complete dual-view samples found")
    output = Path(output_dir)
    annotations = None if annotations_root is None else Path(annotations_root)
    output.mkdir(parents=True, exist_ok=True)
    rows_per_sheet = 4
    samples_per_sheet = max(1, columns * rows_per_sheet)
    sheets: list[str] = []
    reviewed_frames: list[dict[str, Any]] = []
    for sheet_index in range(0, len(targets), samples_per_sheet):
        subset = targets[sheet_index : sheet_index + samples_per_sheet]
        row_tiles: list[np.ndarray] = []
        for metadata_path in subset:
            directory = metadata_path.parent
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            primary = cv2.imread(str(directory / "primary-color.png"), cv2.IMREAD_COLOR)
            side = cv2.imread(str(directory / "side-color.png"), cv2.IMREAD_COLOR)
            depth = np.load(directory / "primary-depth.npy", allow_pickle=False)
            if primary is None or side is None:
                continue
            sample_name = directory.relative_to(root).as_posix()
            if annotations is not None:
                annotated_directory = annotations / directory.relative_to(root)
                annotated_primary = cv2.imread(
                    str(annotated_directory / "primary-annotated.png"), cv2.IMREAD_COLOR
                )
                annotated_side = cv2.imread(
                    str(annotated_directory / "side-annotated.png"), cv2.IMREAD_COLOR
                )
                if annotated_primary is not None:
                    primary = annotated_primary
                if annotated_side is not None:
                    side = annotated_side
            delta = metadata.get("pair_delta_ms")
            row_tiles.append(
                np.hstack(
                    (
                        _tile(primary, tile_width, tile_height, f"{sample_name} primary"),
                        _tile(side, tile_width, tile_height, f"side dt={delta}ms"),
                        _tile(_depth_preview(depth), tile_width, tile_height, "aligned depth"),
                    )
                )
            )
            reviewed_frames.append(
                {
                    "sample": sample_name,
                    "primary_frame_id": metadata.get("primary_frame_id"),
                    "side_frame_id": metadata.get("side_frame_id"),
                    "pair_delta_ms": delta,
                    "human_reviewed": bool(metadata.get("human_reviewed", False)),
                }
            )
        if not row_tiles:
            continue
        max_width = max(tile.shape[1] for tile in row_tiles)
        padded = [
            cv2.copyMakeBorder(tile, 0, 0, 0, max_width - tile.shape[1], cv2.BORDER_CONSTANT)
            for tile in row_tiles
        ]
        sheet = np.vstack(padded)
        name = f"dual-contact-sheet-{len(sheets) + 1:03d}.jpg"
        if not cv2.imwrite(str(output / name), sheet):
            raise OSError(f"failed to write {output / name}")
        sheets.append(name)
    report = {
        "schema_version": 1,
        "samples_root": str(root.resolve()),
        "sample_count": len(reviewed_frames),
        "contact_sheets": sheets,
        "frames": reviewed_frames,
        "human_visual_review_complete": False,
        "projected_roi_overlays": annotations is not None,
        "required_checks": [
            "projected ROI",
            "side occlusion and cross-object overlap",
            "wrong pairing",
            "reflection and shadow",
            "missing depth",
            "merged or split instances",
            "cross-view class conflict",
        ],
    }
    (output / "dual-review-manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def apply_dual_review_records(
    samples_root: str | Path,
    reviews: list[dict[str, Any]],
    registry: ShapeRegistry,
) -> dict[str, Any]:
    """Apply explicit per-frame review decisions; no blanket approval is allowed."""
    root = Path(samples_root).resolve()
    required = (
        "roi_ok", "association_ok", "shadow_ok", "depth_ok", "instance_split_ok"
    )
    updated: list[str] = []
    rejected: list[dict[str, str]] = []
    for review in reviews:
        sample = (root / str(review.get("sample", ""))).resolve()
        try:
            sample.relative_to(root)
        except ValueError as error:
            raise ValueError(f"review sample escapes dataset root: {sample}") from error
        metadata_path = sample / "metadata.json"
        if not metadata_path.is_file():
            rejected.append({"sample": str(review.get("sample", "")), "reason": "metadata_missing"})
            continue
        missing = [key for key in required if not isinstance(review.get(key), bool)]
        if missing:
            rejected.append({"sample": str(review.get("sample", "")), "reason": f"missing_boolean_checks:{','.join(missing)}"})
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        is_multi = bool(metadata.get("scene_split"))
        normalized_objects: list[dict[str, Any]] = []
        if is_multi:
            raw_objects = review.get("objects")
            if not isinstance(raw_objects, list):
                rejected.append({"sample": str(review.get("sample", "")), "reason": "multi_objects_missing"})
                continue
            try:
                for item in raw_objects:
                    if not isinstance(item, dict):
                        raise ValueError("multi_object_entry_not_mapping")
                    object_id = str(item.get("object_id", ""))
                    if not object_id:
                        raise ValueError("multi_object_id_missing")
                    if not isinstance(item.get("association_ok"), bool):
                        raise ValueError(f"multi_association_check_missing:{object_id}")
                    if not isinstance(item.get("side_occluded"), bool):
                        raise ValueError(f"multi_occlusion_check_missing:{object_id}")
                    visible_ratio = float(item.get("side_visible_ratio"))
                    if not 0.0 <= visible_ratio <= 1.0:
                        raise ValueError(f"multi_visible_ratio_out_of_range:{object_id}")
                    normalized_objects.append(
                        {
                            "object_id": object_id,
                            "true_label": registry.resolve(str(item.get("true_label", ""))),
                            "side_visible_ratio": visible_ratio,
                            "side_occluded": bool(item["side_occluded"]),
                            "association_ok": bool(item["association_ok"]),
                            "notes": str(item.get("notes", "")),
                        }
                    )
                identifiers = [item["object_id"] for item in normalized_objects]
                if len(set(identifiers)) != len(identifiers):
                    raise ValueError("multi_object_ids_not_unique")
                expected_count = int(metadata.get("object_count", len(metadata.get("composition", ()))))
                if len(normalized_objects) != expected_count:
                    raise ValueError("multi_object_review_count_mismatch")
                expected_labels = Counter(registry.resolve(str(item)) for item in metadata.get("composition", ()))
                if Counter(item["true_label"] for item in normalized_objects) != expected_labels:
                    raise ValueError("multi_object_labels_do_not_match_composition")
            except (TypeError, ValueError) as error:
                rejected.append({"sample": str(review.get("sample", "")), "reason": str(error)})
                continue
            label_value = None
        else:
            label_value = review.get("true_label") or metadata.get("label_id") or metadata.get("label")
            if label_value and label_value != "empty_tray":
                label_value = registry.resolve(str(label_value))
        approved = all(bool(review[key]) for key in required) and bool(review.get("approved", False))
        if is_multi:
            approved = approved and all(item["association_ok"] for item in normalized_objects)
        metadata.update(
            {
                "label_id": label_value,
                "human_reviewed": approved,
                "review": {
                    key: bool(review[key]) for key in required
                } | {
                    "approved": bool(review.get("approved", False)),
                    "notes": str(review.get("notes", "")),
                    "objects": normalized_objects,
                },
                "shape_registry_hash": registry.registry_hash,
            }
        )
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        updated.append(sample.relative_to(root).as_posix())
    return {
        "schema_version": 1,
        "registry_hash": registry.registry_hash,
        "updated_count": len(updated),
        "updated": updated,
        "rejected": rejected,
    }


def _macro_recall(records: list[dict[str, Any]], prediction_key: str) -> float:
    labels = sorted({str(record["true_label"]) for record in records})
    recalls = []
    for label in labels:
        selected = [record for record in records if str(record["true_label"]) == label]
        recalls.append(
            sum(str(record.get(prediction_key)) == label for record in selected)
            / max(len(selected), 1)
        )
    return float(np.mean(recalls)) if recalls else 0.0


def evaluate_promotion(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the locked independent-holdout promotion gates."""
    eligible = [
        record
        for record in records
        if record.get("split") == "final_holdout"
        and bool(record.get("human_reviewed", False))
    ]
    if not eligible:
        raise ValueError("no human-reviewed final_holdout records")
    mono_recall = _macro_recall(eligible, "mono_label")
    fused_recall = _macro_recall(eligible, "fused_label")

    def wrong_accept(prefix: str) -> float:
        wrong = sum(
            bool(record.get(f"{prefix}_accepted", False))
            and str(record.get(f"{prefix}_label")) != str(record["true_label"])
            for record in eligible
        )
        return wrong / len(eligible)

    mono_wrong = wrong_accept("mono")
    fused_wrong = wrong_accept("fused")
    multi_final = [
        record for record in records
        if record.get("split") == "final_acceptance"
        and bool(record.get("human_reviewed", False))
    ]
    multi_development = [
        record for record in records
        if record.get("split") == "development"
        and bool(record.get("human_reviewed", False))
    ]
    measured = eligible + multi_final
    projection = [float(record["projection_error_px"]) for record in measured]
    pair_delta = [float(record["pair_delta_ms"]) for record in measured]
    latency = [float(record["latency_ms"]) for record in measured]
    safety_violations = sum(bool(record.get("safety_state_upgraded", False)) for record in measured)
    platform_ids = sorted({str(record.get("platform_id", "")) for record in measured})
    association_errors = sum(
        bool(record.get("association_error", False))
        and not bool(record.get("side_occluded", False))
        for record in measured
    )
    deployment_latencies: dict[str, float] = {}
    for target in ("orin_nano", "n100"):
        values = [
            float(record["latency_ms"])
            for record in measured
            if str(record.get("deployment_target", "")).lower() == target
        ]
        if values:
            deployment_latencies[target] = float(np.percentile(values, 95))
    deployment_latency_ok = (
        set(deployment_latencies) == {"orin_nano", "n100"}
        and all(value <= 1000.0 for value in deployment_latencies.values())
    )
    multi_wrong_accepts = sum(
        bool(record.get("fused_accepted", False))
        and str(record.get("fused_label")) != str(record.get("true_label"))
        for record in multi_final
    )
    final_scenes = {str(record.get("scene_id")) for record in multi_final if record.get("scene_id")}
    development_scenes = {
        str(record.get("scene_id")) for record in multi_development if record.get("scene_id")
    }
    missing_multi_objects = sum(bool(record.get("missed_object", False)) for record in multi_final)
    gates = {
        "macro_recall_gain_ge_0_03": fused_recall - mono_recall >= 0.03,
        "wrong_accept_zero": fused_wrong == 0.0,
        "multi_object_wrong_accept_zero": multi_wrong_accepts == 0,
        "multi_scene_counts_30_30": len(development_scenes) >= 30 and len(final_scenes) >= 30,
        "no_missed_final_multi_object": missing_multi_objects == 0,
        "no_safety_upgrade": safety_violations == 0,
        "no_unoccluded_association_error": association_errors == 0,
        "projection_p95_le_3_px": float(np.percentile(projection, 95)) <= 3.0,
        "pair_delta_p95_le_50_ms": float(np.percentile(pair_delta, 95)) <= 50.0,
        "orin_and_n100_latency_p95_le_1000_ms": deployment_latency_ok,
        "competition_platform_only": platform_ids == ["competition"],
    }
    return {
        "schema_version": 1,
        "record_count": len(eligible),
        "platform_ids": platform_ids,
        "mono_macro_recall": mono_recall,
        "fused_macro_recall": fused_recall,
        "macro_recall_gain": fused_recall - mono_recall,
        "mono_wrong_accept_rate": mono_wrong,
        "fused_wrong_accept_rate": fused_wrong,
        "multi_object_wrong_accept_count": multi_wrong_accepts,
        "multi_development_scene_count": len(development_scenes),
        "multi_final_scene_count": len(final_scenes),
        "multi_final_missed_object_count": missing_multi_objects,
        "safety_upgrade_violations": safety_violations,
        "unoccluded_association_errors": association_errors,
        "projection_error_p95_px": float(np.percentile(projection, 95)),
        "pair_delta_p95_ms": float(np.percentile(pair_delta, 95)),
        "latency_p95_ms": float(np.percentile(latency, 95)),
        "deployment_latency_p95_ms": deployment_latencies,
        "gates": gates,
        "promote_dual_view": all(gates.values()),
    }


def fit_probability_temperatures(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit top/side temperatures on a dedicated, human-reviewed split."""
    eligible = [
        record
        for record in records
        if record.get("split") == "probability_calibration"
        and bool(record.get("human_reviewed", False))
    ]
    if not eligible:
        raise ValueError("no human-reviewed probability_calibration records")

    def fit(key: str) -> tuple[float, int, float]:
        samples = [
            (str(record["true_label"]), record.get(key, {}))
            for record in eligible
            if isinstance(record.get(key), dict) and record.get(key)
        ]
        if len(samples) < 2:
            raise ValueError(f"insufficient {key} records for temperature fitting")
        candidates = np.geomspace(0.2, 5.0, 161)
        losses = []
        for temperature in candidates:
            negative_log_likelihood = 0.0
            for label, scores in samples:
                scaled = temperature_scale_scores(
                    {str(name): float(value) for name, value in scores.items()},
                    float(temperature),
                )
                negative_log_likelihood -= np.log(max(scaled.get(label, 0.0), 1e-9))
            losses.append(negative_log_likelihood / len(samples))
        best = int(np.argmin(losses))
        return float(candidates[best]), len(samples), float(losses[best])

    top_temperature, top_count, top_nll = fit("top_class_scores")
    side_temperature, side_count, side_nll = fit("side_class_scores")
    return {
        "schema_version": 1,
        "split": "probability_calibration",
        "top_temperature": top_temperature,
        "side_temperature": side_temperature,
        "top_sample_count": top_count,
        "side_sample_count": side_count,
        "top_nll": top_nll,
        "side_nll": side_nll,
    }


def _fit_top2_logistic(
    records: list[dict[str, Any]], top_temperature: float, side_temperature: float,
    l2: float = 0.10,
) -> tuple[tuple[float, ...], float]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    for record in records:
        top = temperature_scale_scores(record["top_class_scores"], top_temperature)
        side = temperature_scale_scores(record["side_class_scores"], side_temperature)
        candidates = tuple(
            item for item, _ in sorted(top.items(), key=lambda item: item[1], reverse=True)[:2]
        )
        truth = str(record["true_label"])
        if len(candidates) != 2 or truth not in candidates:
            continue
        features.append(top2_fusion_features(top, side, candidates, float(record.get("side_quality", 1.0))))
        targets.append(float(truth == candidates[0]))
    if len(features) < 4 or len(set(targets)) < 2:
        # A generic top/side log-odds initializer remains deterministic, but a
        # promotion report must disclose that the stacker was not trainable.
        return (1.0, 0.30, 0.0, 0.0, 0.0), 0.0
    matrix = np.stack(features)
    target = np.asarray(targets, np.float64)
    weights = np.zeros(matrix.shape[1], np.float64)
    bias = 0.0
    for iteration in range(2500):
        logits = np.clip(matrix @ weights + bias, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        error = probability - target
        rate = 0.08 / (1.0 + iteration / 600.0)
        weights -= rate * (matrix.T @ error / len(matrix) + l2 * weights)
        bias -= rate * float(error.mean())
    return tuple(map(float, weights)), float(bias)


def fit_fusion_policy(
    records: list[dict[str, Any]], registry_hash: str
) -> tuple[FusionPolicy, dict[str, Any]]:
    """Fit temperatures and select one of the three locked Top-2 fusion rules."""
    eligible = [
        record for record in records
        if record.get("split") == "probability_calibration"
        and bool(record.get("human_reviewed", False))
        and isinstance(record.get("top_class_scores"), dict)
        and isinstance(record.get("side_class_scores"), dict)
    ]
    if not eligible:
        raise ValueError("no reviewed probability_calibration records with both score vectors")
    temperatures = fit_probability_temperatures(eligible)
    top_temperature = float(temperatures["top_temperature"])
    side_temperature = float(temperatures["side_temperature"])
    logistic_weights, logistic_bias = _fit_top2_logistic(
        eligible, top_temperature, side_temperature
    )
    labels = sorted({str(item["true_label"]) for item in eligible})
    candidates: list[dict[str, Any]] = []
    for method in ("probability_sum", "log_product", "top2_logistic"):
        weights_to_try = (0.15, 0.30, 0.45) if method != "top2_logistic" else (0.30,)
        for side_weight in weights_to_try:
            for probability_threshold in np.arange(0.55, 0.86, 0.03):
                for margin_threshold in np.arange(0.02, 0.25, 0.03):
                    predictions: list[str | None] = []
                    accepted = 0
                    wrong = 0
                    elapsed = 0.0
                    for record in eligible:
                        top = temperature_scale_scores(record["top_class_scores"], top_temperature)
                        side = temperature_scale_scores(record["side_class_scores"], side_temperature)
                        started = time.perf_counter()
                        fused, _ = fuse_top2_scores(
                            top, side, float(record.get("side_quality", 1.0)),
                            method=method, side_weight=side_weight,
                            logistic_weights=logistic_weights,
                            logistic_bias=logistic_bias,
                        )
                        elapsed += (time.perf_counter() - started) * 1000.0
                        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
                        probability = ordered[0][1] if ordered else 0.0
                        margin = probability - (ordered[1][1] if len(ordered) > 1 else 0.0)
                        prediction = ordered[0][0] if ordered and probability >= probability_threshold and margin >= margin_threshold else None
                        predictions.append(prediction)
                        if prediction is not None:
                            accepted += 1
                            wrong += int(prediction != str(record["true_label"]))
                    recalls = []
                    for label in labels:
                        indices = [index for index, item in enumerate(eligible) if str(item["true_label"]) == label]
                        recalls.append(sum(predictions[index] == label for index in indices) / max(len(indices), 1))
                    candidates.append(
                        {
                            "method": method,
                            "side_weight": float(side_weight),
                            "fused_probability": float(probability_threshold),
                            "fused_margin": float(margin_threshold),
                            "wrong_accept_count": wrong,
                            "macro_recall": float(np.mean(recalls)),
                            "coverage": accepted / len(eligible),
                            "mean_fusion_latency_ms": elapsed / len(eligible),
                        }
                    )
    candidates.sort(
        key=lambda item: (
            item["wrong_accept_count"] == 0,
            -item["wrong_accept_count"],
            item["macro_recall"],
            item["coverage"],
            -item["mean_fusion_latency_ms"],
        ),
        reverse=True,
    )
    selected = candidates[0]
    policy = FusionPolicy(
        registry_hash=registry_hash,
        method=selected["method"],
        top_temperature=top_temperature,
        side_temperature=side_temperature,
        side_weight=selected["side_weight"],
        fused_probability=selected["fused_probability"],
        fused_margin=selected["fused_margin"],
        logistic_weights=logistic_weights if selected["method"] == "top2_logistic" else (),
        logistic_bias=logistic_bias if selected["method"] == "top2_logistic" else 0.0,
    )
    return policy, {
        "schema_version": 1,
        "split": "probability_calibration",
        "record_count": len(eligible),
        "selection_priority": ["zero_wrong_accept", "macro_recall", "coverage", "latency"],
        "temperature_fit": temperatures,
        "selected": selected,
        "candidates": candidates,
        "policy": policy.to_dict(),
    }
