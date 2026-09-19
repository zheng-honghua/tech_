"""Train and replay the actual top RGB-D + side RGB Top-2 fusion chain.

The outer holdout is fixed at three samples per batch/shape group.  A second
holdout inside the remaining data is used for temperature, fusion weight and
acceptance-threshold selection.  The outer samples are never used to tune the
models or policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sorting_vision.config import load_config
from sorting_vision.fusion_policy import FusionPolicy, fuse_top2_scores
from sorting_vision.geometry_rgbd_model import (
    DepthGeometryModel,
    FEATURE_NAMES,
    FUSED_FEATURE_NAMES,
    extract_rgbd_geometry_features,
)
from sorting_vision.group_holdout import GroupHoldoutSplit, deterministic_group_holdout
from sorting_vision.pipeline3d import VisionPipeline3D
from sorting_vision.rgbd import CameraIntrinsics, RGBDFrame
from sorting_vision.rgbd_dataset import depth_preview
from sorting_vision.shape_registry import ShapeRegistry, load_shape_registry
from sorting_vision.side_geometry import SideGeometryModel, SideTrainingSample, load_side_training_samples
from sorting_vision.dual_view import temperature_scale_scores
from sorting_vision.dual_view import DualViewCalibration, projected_roi, _side_mask
from sorting_vision.dual_view import associate_side_color_components, measure_side_color_segmentation
from sorting_vision.cross_view_topology import (
    CrossViewTopologyModel,
    annotate_cross_view_topology,
    extract_cross_view_features,
)


@dataclass(frozen=True)
class PairedFeatures:
    side: SideTrainingSample
    top_features: np.ndarray
    top_bbox: tuple[int, int, int, int]
    topology_quality: float
    primary_points: np.ndarray
    side_quality_override: float | None = None


class _RecordingShapeModel:
    """Delegate to the stable mono model while retaining its exact inputs."""

    def __init__(self, base: DepthGeometryModel, full_rgb_instance: bool = False) -> None:
        self.base = base
        self.input_contract = "rgb_silhouette_depth_owned_v2" if full_rgb_instance else "depth_owned_v1"
        self.full_rgb_instance = full_rgb_instance
        self.features: list[np.ndarray] = []
        self.last_diagnostics: dict[str, float] = {}
        self.last_class_scores: dict[str, float] = {}
        self.last_rejection_reason = "not_run"

    def reset(self) -> None:
        self.features.clear()

    def classify(
        self, points_camera_mm: np.ndarray, color_crop_bgr: np.ndarray,
        depth_crop_mm: np.ndarray, crop_mask: np.ndarray,
        intrinsics: CameraIntrinsics | None = None,
        crop_origin_uv: tuple[int, int] = (0, 0),
    ) -> tuple[str, float]:
        feature = extract_rgbd_geometry_features(
            points_camera_mm, depth_crop_mm, crop_mask, intrinsics,
            crop_origin_uv, color_crop_bgr, include_fused_edges=self.full_rgb_instance,
        )
        self.features.append(feature)
        if self.full_rgb_instance:
            # The old model's normalization is not valid for this new contract.
            self.last_diagnostics, self.last_class_scores = {}, {}
            self.last_rejection_reason = "recording_only"
            return "unknown", 0.0
        label, confidence, reason = self.base.predict_features(feature)
        self.last_diagnostics = dict(self.base.last_diagnostics)
        self.last_class_scores = dict(self.base.last_class_scores)
        self.last_rejection_reason = reason
        return label, confidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nested-holdout RGB-D/side fusion training")
    parser.add_argument("--samples-root", required=True)
    parser.add_argument("--side-background", required=True)
    parser.add_argument("--config", default="config/dual/temporary.yaml")
    parser.add_argument("--shape-registry", default="config/shapes/competition-11.yaml")
    parser.add_argument("--base-top-model", default="models/stable/rgbd/geometry-rgbd-multipose-v4.npz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-model-output", required=True)
    parser.add_argument("--side-model-output", required=True)
    parser.add_argument("--policy-output", required=True)
    parser.add_argument("--cross-model-output", required=True)
    parser.add_argument("--dual-calibration", required=True)
    parser.add_argument("--holdout-per-group", type=int, default=3)
    parser.add_argument("--calibration-per-group", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--full-rgb-instance", action="store_true")
    parser.add_argument("--fixed-holdout-records")
    parser.add_argument("--resume-features", action="store_true")
    return parser


def _key(item: SideTrainingSample | PairedFeatures) -> str:
    sample = item.side if isinstance(item, PairedFeatures) else item
    return sample.directory.resolve().as_posix()


def _group(item: SideTrainingSample | PairedFeatures) -> str:
    sample = item.side if isinstance(item, PairedFeatures) else item
    return f"{sample.batch_id}/{sample.label_id}"


def _subset_registry(registry: ShapeRegistry, observed: set[str]) -> ShapeRegistry:
    return ShapeRegistry(
        version=registry.version,
        classes=tuple(item for item in registry.classes if item.enabled and item.class_id in observed),
    )


def _adaptive_calibration_holdout(
    items: tuple[PairedFeatures, ...], requested_per_group: int, seed: int,
) -> GroupHoldoutSplit[PairedFeatures]:
    """Hold out up to N per group while always retaining one training sample."""
    grouped: dict[str, list[PairedFeatures]] = defaultdict(list)
    for item in items:
        grouped[_group(item)].append(item)
    training: list[PairedFeatures] = []
    holdout: list[PairedFeatures] = []
    counts: dict[str, dict[str, int]] = {}
    for group, group_items in sorted(grouped.items()):
        count = min(requested_per_group, len(group_items) - 1)
        if count < 1:
            training.extend(group_items)
            counts[group] = {"total": len(group_items), "training": len(group_items), "holdout": 0}
            continue
        split = deterministic_group_holdout(
            group_items, group_key=lambda _item: group, item_key=_key,
            holdout_per_group=count, seed=seed,
        )
        training.extend(split.training)
        holdout.extend(split.holdout)
        counts.update(split.group_counts)
    return GroupHoldoutSplit(tuple(training), tuple(holdout), counts)


def _load_primary_frame(directory: Path) -> RGBDFrame:
    image = cv2.imread(str(directory / "primary-color.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("primary_image_unreadable")
    depth = np.load(directory / "primary-depth.npy", allow_pickle=False)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    intrinsics = CameraIntrinsics(**metadata["primary_intrinsics"])
    return RGBDFrame(
        image, depth, intrinsics,
        int(metadata.get("primary_timestamp_ns", 0)),
        str(metadata.get("primary_frame_id", directory.name)),
        metadata.get("primary_color_timestamp_ns"), metadata.get("primary_depth_timestamp_ns"),
    )


def _mono_pipelines_by_batch(
    source: tuple[SideTrainingSample, ...],
    cfg: Any,
    base: DepthGeometryModel,
    full_rgb_instance: bool = False,
) -> dict[str, tuple[VisionPipeline3D, _RecordingShapeModel]]:
    """Build the production monocular pipeline with each batch's own empty tray."""
    pipelines: dict[str, tuple[VisionPipeline3D, _RecordingShapeModel]] = {}
    for sample in source:
        if sample.batch_id in pipelines:
            continue
        batch_root = sample.directory.parents[1]
        empty_dirs = sorted((batch_root / "empty_tray").glob("sample-*"))
        if not empty_dirs:
            raise ValueError(f"missing_batch_empty_tray:{sample.batch_id}")
        recorder = _RecordingShapeModel(base, full_rgb_instance)
        pipeline = VisionPipeline3D(
            config=cfg,
            background_frame=_load_primary_frame(empty_dirs[0]),
            shape_model=recorder,
        )
        pipelines[sample.batch_id] = (pipeline, recorder)
    return pipelines


def _extract_top(
    sample: SideTrainingSample,
    pipeline: VisionPipeline3D,
    recorder: _RecordingShapeModel,
    calibration: DualViewCalibration | None = None,
    background: np.ndarray | None = None,
) -> PairedFeatures:
    frame = _load_primary_frame(sample.directory)
    recorder.reset()
    pipeline.reset_tracking()
    results = pipeline.process(frame)
    if not pipeline.health().get("ok", False):
        raise ValueError(f"mono_pipeline_unhealthy:{pipeline.health().get('reason')}")
    if not results:
        raise ValueError("mono_pipeline_no_object")
    if len(recorder.features) != len(results):
        raise ValueError("mono_pipeline_feature_alignment_failed")
    # This is association only: segmentation, fragment repair and complete RGB
    # contour recovery are owned by VisionPipeline3D.  A single-object capture
    # is represented by the result with the largest recovered RGB support.
    selected_index = max(
        range(len(results)),
        key=lambda index: int(results[index].diagnostics.get("rgb_mask_pixels", 0)),
    )
    result = results[selected_index]
    primary_points = pipeline.last_object_points(result.object_id)
    quality_override = None
    if recorder.full_rgb_instance:
        roi = projected_roi(calibration, primary_points, pipeline.calibration.tray_plane_camera,
                            sample.image_bgr.shape, pipeline.config.dual_view.roi_padding_ratio)
        if roi is None:
            raise ValueError("side_projection_missing")
        if pipeline.config.dual_view.side_complete_color_components:
            rois = {}
            for item in results:
                candidate = projected_roi(calibration, pipeline.last_object_points(item.object_id),
                    pipeline.calibration.tray_plane_camera, sample.image_bgr.shape,
                    pipeline.config.dual_view.roi_padding_ratio)
                if candidate is not None:
                    rois[item.object_id] = candidate
            segmentation = associate_side_color_components(sample.image_bgr, rois, pipeline.config.dual_view)[result.object_id]
            if segmentation.reason != "accepted":
                raise ValueError(f"side_color_segmentation:{segmentation.reason}")
            roi, local_mask = segmentation.roi, segmentation.mask
            pixels, _, quality_override = measure_side_color_segmentation(sample.image_bgr, segmentation, pipeline.config.dual_view)
        else:
            local_mask, pixels, _, quality_override = _side_mask(sample.image_bgr, background, roi, pipeline.config.dual_view)
        if pixels < pipeline.config.dual_view.side_min_area_px:
            raise ValueError("side_projected_foreground_too_small")
        full_mask = np.zeros(sample.image_bgr.shape[:2], np.uint8)
        rx, ry, rw, rh = roi.bbox
        full_mask[ry:ry + rh, rx:rx + rw] = local_mask
        sample = replace(sample, mask=full_mask)
    feature = recorder.features[selected_index]
    quality_index = FEATURE_NAMES.index("plane_quality")
    return PairedFeatures(
        sample,
        feature,
        result.bbox_px,
        float(feature[quality_index]),
        primary_points,
        quality_override,
    )


def _fit_temperature(records: list[dict[str, Any]], key: str) -> float:
    candidates = np.geomspace(0.25, 4.0, 121)
    losses = []
    for temperature in candidates:
        loss = 0.0
        for record in records:
            scores = temperature_scale_scores(record[key], float(temperature))
            loss -= np.log(max(scores.get(record["true_label"], 0.0), 1e-9))
        losses.append(loss / max(len(records), 1))
    return float(candidates[int(np.argmin(losses))])


def _score_pair(
    top_model: DepthGeometryModel,
    side_model: SideGeometryModel,
    cross_model: CrossViewTopologyModel,
    calibration: DualViewCalibration,
    sample: PairedFeatures,
) -> dict[str, Any]:
    top_label, top_confidence, top_reason = top_model.predict_features(sample.top_features)
    top_scores = dict(top_model.last_class_scores)
    side_label, side_confidence, side_diag = side_model.predict(sample.side.image_bgr, sample.side.mask)
    side_scores = dict(side_model.last_class_scores)
    cross_label, cross_confidence, cross_diag = cross_model.predict(
        sample.side.image_bgr,
        sample.side.mask,
        sample.primary_points,
        calibration,
    )
    cross_scores = dict(cross_model.last_class_scores)
    top_raw = max(top_scores.items(), key=lambda item: item[1])[0] if top_scores else "unknown"
    side_raw = max(side_scores.items(), key=lambda item: item[1])[0] if side_scores else "unknown"
    cross_raw = max(cross_scores.items(), key=lambda item: item[1])[0] if cross_scores else "unknown"
    feature_diag = dict(side_diag.get("feature_groups", {}))
    blur = float(cv2.Laplacian(cv2.cvtColor(sample.side.image_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    mask_ratio = float(cv2.countNonZero(sample.side.mask)) / max(sample.side.mask.size, 1)
    area_quality = float(np.clip(mask_ratio / 0.003, 0.0, 1.0))
    blur_quality = float(np.clip(blur / 120.0, 0.0, 1.0))
    geometry_quality = float(np.clip(feature_diag.get("solidity", 1.0), 0.0, 1.0))
    side_quality = float(np.clip(0.45 * area_quality + 0.35 * blur_quality + 0.20 * geometry_quality, 0, 1))
    if sample.side_quality_override is not None:
        side_quality = sample.side_quality_override
    return {
        "sample": _key(sample), "group": _group(sample),
        "true_label": sample.side.label_id,
        "top_prediction": top_raw, "top_decision": top_label,
        "top_confidence": float(top_confidence),
        "top_reason": top_reason, "top_class_scores": top_scores,
        "classic_side_prediction": side_raw,
        "classic_side_decision": side_label,
        "classic_side_confidence": float(side_confidence),
        "classic_side_reason": str(side_diag.get("reason", "unknown")),
        "classic_side_class_scores": side_scores,
        "side_prediction": cross_raw, "side_decision": cross_label,
        "side_confidence": float(cross_confidence),
        "side_reason": str(cross_diag.get("reason", "unknown")),
        "side_class_scores": cross_scores,
        "side_quality": float(
            min(side_quality, cross_diag.get("feature_groups", {}).get("spatial_quality", 0.0))
        ),
        "cross_view_topology": cross_diag.get("feature_groups", {}),
        "topology_quality": sample.topology_quality,
    }


def _metrics(records: list[dict[str, Any]], field: str, accepted_field: str | None = None) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    accepted = wrong = raw_correct = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        truth = record["true_label"]
        prediction = record[field]
        is_accepted = True if accepted_field is None else bool(record[accepted_field])
        totals[truth] += 1
        raw_correct += int(prediction == truth)
        correct[truth] += int(is_accepted and prediction == truth)
        accepted += int(is_accepted)
        wrong += int(is_accepted and prediction != truth)
        confusion[truth][prediction] += 1
    recalls = [correct[label] / totals[label] for label in sorted(totals)]
    return {
        "sample_count": len(records), "raw_accuracy": raw_correct / max(len(records), 1),
        "macro_recall_with_rejection": float(np.mean(recalls)) if recalls else 0.0,
        "coverage": accepted / max(len(records), 1),
        "accepted_accuracy": (accepted - wrong) / max(accepted, 1),
        "wrong_accept_count": wrong,
        "confusion": {label: dict(sorted(values.items())) for label, values in sorted(confusion.items())},
    }


def _apply_candidate(
    records: list[dict[str, Any]], *, method: str, side_weight: float,
    same_family_scale: float, topology_guard: float,
    top_temperature: float, side_temperature: float,
    probability_threshold: float = 0.0, margin_threshold: float = 0.0,
    minimum_side_quality: float = 0.60,
) -> list[dict[str, Any]]:
    output = []
    for source in records:
        record = dict(source)
        top = temperature_scale_scores(source["top_class_scores"], top_temperature)
        side = temperature_scale_scores(source["side_class_scores"], side_temperature)
        if source["side_quality"] < minimum_side_quality:
            ordered_top = sorted(top.items(), key=lambda item: item[1], reverse=True)
            candidates = tuple(label for label, _ in ordered_top[:2])
            total = sum(top.get(label, 0.0) for label in candidates)
            fused = {
                label: top.get(label, 0.0) / max(total, 1e-12) for label in candidates
            }
        else:
            fused, candidates = fuse_top2_scores(
                top, side, source["side_quality"], method=method, side_weight=side_weight,
                topology_quality=source["topology_quality"],
                same_family_side_scale=same_family_scale,
                topology_guard_strength=topology_guard,
            )
        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        winner = ordered[0][0] if ordered else source["top_prediction"]
        probability = ordered[0][1] if ordered else 0.0
        margin = probability - (ordered[1][1] if len(ordered) > 1 else 0.0)
        top_max = max(top.values(), default=0.0)
        side_max_label, side_max = max(side.items(), key=lambda item: item[1]) if side else ("unknown", 0.0)
        top_max_label = max(top.items(), key=lambda item: item[1])[0] if top else "unknown"
        conflict = (
            source["side_quality"] >= minimum_side_quality
            and top_max >= 0.75 and side_max >= 0.75 and top_max_label != side_max_label
        )
        accepted = (
            not conflict and source["top_reason"] != "distance_rejected"
            and probability >= probability_threshold and margin >= margin_threshold
        )
        record.update({
            "top2_candidates": list(candidates), "fused_prediction": winner,
            "fused_scores": fused, "fused_probability": probability,
            "fused_margin": margin, "fused_accepted": accepted,
            "fusion_conflict": conflict,
            "side_quality_gated": source["side_quality"] < minimum_side_quality,
        })
        output.append(record)
    return output


def _macro_raw(records: list[dict[str, Any]], field: str) -> float:
    labels = sorted({record["true_label"] for record in records})
    return float(np.mean([
        np.mean([record[field] == label for record in records if record["true_label"] == label])
        for label in labels
    ]))


def _select_policy(records: list[dict[str, Any]], registry_hash: str, strict: bool = False) -> tuple[FusionPolicy, dict[str, Any]]:
    top_temperature = _fit_temperature(records, "top_class_scores")
    side_temperature = _fit_temperature(records, "side_class_scores")
    candidates = []
    for method in ("probability_sum", "log_product", "evidence_adaptive"):
        for side_weight in (0.15, 0.25, 0.35, 0.45, 0.60):
            family_scales = (0.25, 0.35, 0.50) if method == "evidence_adaptive" else (0.35,)
            guards = (0.35, 0.65, 0.85) if method == "evidence_adaptive" else (0.65,)
            for family_scale in family_scales:
                for guard in guards:
                    replay = _apply_candidate(
                        records, method=method, side_weight=side_weight,
                        same_family_scale=family_scale, topology_guard=guard,
                        top_temperature=top_temperature, side_temperature=side_temperature,
                    )
                    raw_accuracy = float(np.mean([r["fused_prediction"] == r["true_label"] for r in replay]))
                    corrected = sum(
                        r["top_prediction"] != r["true_label"] and r["fused_prediction"] == r["true_label"]
                        for r in replay
                    )
                    damaged = sum(
                        r["top_prediction"] == r["true_label"] and r["fused_prediction"] != r["true_label"]
                        for r in replay
                    )
                    candidates.append({
                        "method": method, "side_weight": side_weight,
                        "same_family_side_scale": family_scale,
                        "topology_guard_strength": guard,
                        "raw_accuracy": raw_accuracy,
                        "macro_raw_recall": _macro_raw(replay, "fused_prediction"),
                        "corrected": corrected, "damaged": damaged,
                    })
    candidates.sort(key=lambda item: (
        item["raw_accuracy"], item["macro_raw_recall"],
        item["corrected"] - item["damaged"], -item["side_weight"],
    ), reverse=True)
    selected = candidates[0]
    if strict:
        # Choose rule and thresholds jointly, prioritizing zero false accepts.
        ranked = []
        for rule in candidates:
            for probability in (.72, .78, .84, .90, .96, .999):
                for margin in (.12, .18, .24):
                    replay = _apply_candidate(records, method=rule["method"], side_weight=rule["side_weight"],
                        same_family_scale=rule["same_family_side_scale"], topology_guard=rule["topology_guard_strength"],
                        top_temperature=top_temperature, side_temperature=side_temperature,
                        probability_threshold=probability, margin_threshold=margin)
                    metrics = _metrics(replay, "fused_prediction", "fused_accepted")
                    ranked.append((metrics["wrong_accept_count"] == 0, -metrics["wrong_accept_count"],
                        metrics["macro_recall_with_rejection"], metrics["coverage"], rule, probability, margin, metrics))
        best = max(ranked, key=lambda item: item[:4])
        rule, probability, margin, metrics = best[4:]
        policy = FusionPolicy(registry_hash=registry_hash, method=rule["method"], top_temperature=top_temperature,
            side_temperature=side_temperature, side_weight=rule["side_weight"], fused_probability=probability,
            fused_margin=margin, same_family_side_scale=rule["same_family_side_scale"],
            topology_guard_strength=rule["topology_guard_strength"])
        return policy, {"temperature": {"top": top_temperature, "side": side_temperature},
                        "selected_score_rule": rule, "selected_thresholds": {"probability": probability, "margin": margin, **metrics},
                        "strict_zero_error_priority": True, "candidate_count": len(ranked)}
    threshold_candidates = []
    for probability in np.arange(0.50, 0.86, 0.03):
        for margin in np.arange(0.00, 0.25, 0.03):
            replay = _apply_candidate(
                records, method=selected["method"], side_weight=selected["side_weight"],
                same_family_scale=selected["same_family_side_scale"],
                topology_guard=selected["topology_guard_strength"],
                top_temperature=top_temperature, side_temperature=side_temperature,
                probability_threshold=float(probability), margin_threshold=float(margin),
            )
            metrics = _metrics(replay, "fused_prediction", "fused_accepted")
            threshold_candidates.append({"probability": float(probability), "margin": float(margin), **metrics})
    viable = [item for item in threshold_candidates if item["coverage"] >= 0.15]
    viable.sort(key=lambda item: (
        item["wrong_accept_count"] == 0, -item["wrong_accept_count"],
        item["macro_recall_with_rejection"], item["coverage"],
    ), reverse=True)
    threshold = viable[0]
    policy = FusionPolicy(
        registry_hash=registry_hash, method=selected["method"],
        top_temperature=top_temperature, side_temperature=side_temperature,
        side_weight=selected["side_weight"], fused_probability=threshold["probability"],
        fused_margin=threshold["margin"],
        same_family_side_scale=selected["same_family_side_scale"],
        topology_guard_strength=selected["topology_guard_strength"],
    )
    return policy, {
        "temperature": {"top": top_temperature, "side": side_temperature},
        "selected_score_rule": selected, "selected_thresholds": threshold,
        "score_rule_candidates": candidates, "threshold_candidates": threshold_candidates,
    }


def _tile(image: np.ndarray, title: str, width: int = 420, height: int = 236) -> np.ndarray:
    canvas = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(canvas, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.putText(canvas, title[:72], (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _contact_sheets(
    output_dir: Path,
    samples: list[PairedFeatures],
    records: list[dict[str, Any]],
    calibration: DualViewCalibration,
) -> list[str]:
    by_key = {record["sample"]: record for record in records}
    grouped: dict[str, list[PairedFeatures]] = defaultdict(list)
    for sample in samples:
        grouped[_group(sample)].append(sample)
    target = output_dir / "fusion-contact-sheets"
    target.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, (group, items) in enumerate(sorted(grouped.items()), 1):
        rows = []
        for sample in sorted(items, key=_key):
            record = by_key[_key(sample)]
            primary = cv2.imread(str(sample.side.directory / "primary-color.png"), cv2.IMREAD_COLOR)
            x, y, width, height = sample.top_bbox
            colour = (0, 220, 0) if record["fused_prediction"] == record["true_label"] else (0, 0, 255)
            cv2.rectangle(primary, (x, y), (x + width, y + height), colour, 4)
            depth = np.load(sample.side.directory / "primary-depth.npy", allow_pickle=False)
            try:
                cross = extract_cross_view_features(
                    sample.side.image_bgr,
                    sample.side.mask,
                    sample.primary_points,
                    calibration,
                )
                side = annotate_cross_view_topology(sample.side.image_bgr, cross)
            except ValueError:
                side = sample.side.image_bgr.copy()
            contours, _ = cv2.findContours(sample.side.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(side, contours, -1, (0, 255, 0), 3)
            summary = f"T={record['true_label']} top={record['top_prediction']} fused={record['fused_prediction']}"
            rows.append(np.hstack((
                _tile(primary, summary), _tile(depth_preview(depth), f"TOP RGB-D q={record['topology_quality']:.2f}"),
                _tile(side, f"cross={record['side_prediction']} q={record['side_quality']:.2f}"),
            )))
        sheet = np.vstack(rows)
        name = f"{index:03d}-{group.replace('/', '__')}.jpg"
        if not cv2.imwrite(str(target / name), sheet):
            raise OSError(f"cannot write {name}")
        outputs.append((target / name).relative_to(output_dir).as_posix())
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    if args.full_rgb_instance:
        cv2.setNumThreads(2)
    registry = load_shape_registry(args.shape_registry)
    background = cv2.imread(args.side_background, cv2.IMREAD_COLOR)
    if background is None:
        raise ValueError("side background is unreadable")
    calibration = DualViewCalibration.load(args.dual_calibration)
    side_samples, side_errors = load_side_training_samples(
        args.samples_root, registry, background, require_reviewed=False,
        defer_segmentation=args.full_rgb_instance,
        image_cache_directory=output_dir / "image-cache" if args.full_rgb_instance else None,
    )
    source = tuple(item for item in side_samples if item.split == "train")
    base = DepthGeometryModel.load(args.base_top_model)
    pipelines = _mono_pipelines_by_batch(source, cfg, base, args.full_rgb_instance)
    paired: dict[str, PairedFeatures] = {}
    top_errors = []
    started = time.perf_counter()
    cache_root = output_dir / "paired-cache"
    cache_root.mkdir(exist_ok=True)
    fingerprint_files = [Path(args.config), Path(args.dual_calibration), Path(args.shape_registry), Path(__file__),
        *Path("src/sorting_vision").glob("*.py"), *Path("config/dual").glob("*.yaml")]
    fingerprint = hashlib.sha256()
    for path in sorted(fingerprint_files):
        fingerprint.update(path.as_posix().encode())
        fingerprint.update(path.read_bytes())
    fingerprint.update(background.tobytes())
    cache_manifest = {"contract": "rgb_silhouette_depth_owned_v2" if args.full_rgb_instance else "depth_owned_v1",
                      "fingerprint": fingerprint.hexdigest()}
    manifest_path = cache_root / "manifest.json"
    if args.resume_features:
        if not manifest_path.exists() or json.loads(manifest_path.read_text(encoding="utf-8")) != cache_manifest:
            raise ValueError("feature cache configuration/code fingerprint mismatch; use a new output directory")
    else:
        manifest_path.write_text(json.dumps(cache_manifest, indent=2), encoding="utf-8")
    for index, sample in enumerate(source, 1):
        try:
            pipeline, recorder = pipelines[sample.batch_id]
            sample_fingerprint = hashlib.sha256((_key(sample) + sample.sha256).encode())
            sample_fingerprint.update((sample.directory / "primary-depth.npy").read_bytes())
            sample_fingerprint.update((sample.directory / "primary-color.png").read_bytes())
            sample_fingerprint.update((sample.directory / "metadata.json").read_bytes())
            cache_path = cache_root / (sample_fingerprint.hexdigest() + ".npz")
            if args.resume_features and cache_path.exists():
                with np.load(cache_path, allow_pickle=False) as cached:
                    paired[_key(sample)] = PairedFeatures(replace(sample, mask=cached["side_mask"]), cached["features"],
                        tuple(cached["bbox"].tolist()), float(cached["topology_quality"]), cached["points"],
                        float(cached["side_quality"]) if args.full_rgb_instance else None)
            else:
                item = _extract_top(sample, pipeline, recorder, calibration, background)
                paired[_key(sample)] = item
                np.savez_compressed(cache_path, side_mask=item.side.mask, features=item.top_features,
                    bbox=item.top_bbox, topology_quality=item.topology_quality, points=item.primary_points,
                    side_quality=item.side_quality_override if item.side_quality_override is not None else 0.)
        except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
            top_errors.append({"sample": _key(sample), "reason": str(error)})
            if len(top_errors) <= 5:
                print(json.dumps(top_errors[-1], ensure_ascii=False), flush=True)
        if index % 50 == 0:
            print(json.dumps({"extracted": index, "valid": len(paired), "failed": len(top_errors)}), flush=True)
    ordered_features = [paired[key] for key in sorted(paired)]
    np.savez_compressed(
        output_dir / "top-feature-cache.npz",
        sample_keys=np.asarray([_key(item) for item in ordered_features]),
        features=np.stack([item.top_features for item in ordered_features]),
        bboxes=np.asarray([item.top_bbox for item in ordered_features], np.int32),
        topology_quality=np.asarray([item.topology_quality for item in ordered_features], np.float32),
        point_counts=np.asarray([len(item.primary_points) for item in ordered_features], np.int32),
    )
    # Split after production-pipeline extraction so every group contributes
    # exactly the requested number of valid holdout frames.
    outer = deterministic_group_holdout(
        tuple(paired.values()), group_key=_group, item_key=_key,
        holdout_per_group=args.holdout_per_group, seed=args.seed,
    )
    missing_fixed_holdout = []
    if args.fixed_holdout_records:
        reserved = {Path(json.loads(line)["sample"]).resolve().as_posix()
                    for line in Path(args.fixed_holdout_records).read_text(encoding="utf-8").splitlines()}
        missing_fixed_holdout = sorted(reserved - set(paired))
        counts = {}
        for item in paired.values():
            count = counts.setdefault(_group(item), {"total": 0, "training": 0, "holdout": 0})
            count["total"] += 1
            count["holdout" if _key(item) in reserved else "training"] += 1
        outer = GroupHoldoutSplit(tuple(item for key, item in paired.items() if key not in reserved),
                                 tuple(item for key, item in paired.items() if key in reserved), counts)
    outer_training = outer.training
    outer_holdout = outer.holdout
    inner = _adaptive_calibration_holdout(
        outer_training, args.calibration_per_group, args.seed + 1,
    )
    observed = {item.side.label_id for item in paired.values()}
    model_registry = _subset_registry(registry, observed)
    base_features, base_labels = base.training_data()
    if args.full_rgb_instance:
        # Do not mix historical depth-mask features with RGB-silhouette inputs.
        base_features = np.empty((0, len(FUSED_FEATURE_NAMES)), np.float32)
        base_labels = []

    def train_models(
        items: tuple[PairedFeatures, ...]
    ) -> tuple[DepthGeometryModel, SideGeometryModel, CrossViewTopologyModel]:
        top_features = np.vstack([base_features, np.vstack([item.top_features for item in items])])
        top_labels = [*base_labels, *[item.side.label_id for item in items]]
        top_model = DepthGeometryModel.fit(
            top_features, top_labels, feature_names=FUSED_FEATURE_NAMES if args.full_rgb_instance else base.feature_names
        )
        if args.full_rgb_instance:
            top_model.edge_parameters["input_contract"] = "rgb_silhouette_depth_owned_v2"
        side_model = SideGeometryModel.train(
            [(item.side.image_bgr, item.side.mask, item.side.label_id) for item in items],
            model_registry, method="rtrees",
        )
        cross_model = CrossViewTopologyModel.train(
            [
                (
                    item.side.image_bgr,
                    item.side.mask,
                    item.primary_points,
                    calibration,
                    item.side.label_id,
                )
                for item in items
            ],
            model_registry,
            method="rtrees",
        )
        return top_model, side_model, cross_model

    calibration_top, calibration_side, calibration_cross = train_models(inner.training)
    calibration_records = [
        _score_pair(calibration_top, calibration_side, calibration_cross, calibration, item)
        for item in inner.holdout
    ]
    policy, policy_fit = _select_policy(calibration_records, model_registry.registry_hash, strict=args.full_rgb_instance)
    final_top, final_side, final_cross = train_models(outer_training)
    final_top.save(args.top_model_output, metadata={
        "purpose": "experimental_dual_fusion", "outer_seed": args.seed,
        "base_model": str(Path(args.base_top_model).resolve()),
    })
    final_side.save(args.side_model_output)
    final_cross.save(args.cross_model_output)
    policy.save(args.policy_output)
    holdout_records = [
        _score_pair(final_top, final_side, final_cross, calibration, item)
        for item in outer_holdout
    ]
    fused_records = _apply_candidate(
        holdout_records, method=policy.method, side_weight=policy.side_weight,
        same_family_scale=policy.same_family_side_scale,
        topology_guard=policy.topology_guard_strength,
        top_temperature=policy.top_temperature, side_temperature=policy.side_temperature,
        probability_threshold=policy.fused_probability, margin_threshold=policy.fused_margin,
    )
    for record in fused_records:
        record["top_accepted"] = record["top_reason"] == "accepted"
        record["side_accepted"] = record["side_reason"] == "accepted"
    sheets = _contact_sheets(output_dir, list(outer_holdout), fused_records, calibration)
    top_metrics = _metrics(fused_records, "top_prediction", "top_accepted")
    side_metrics = _metrics(fused_records, "side_prediction", "side_accepted")
    classic_side_metrics = _metrics(fused_records, "classic_side_prediction")
    fused_metrics = _metrics(fused_records, "fused_prediction", "fused_accepted")
    corrected = [r["sample"] for r in fused_records if r["top_prediction"] != r["true_label"] and r["fused_prediction"] == r["true_label"]]
    damaged = [r["sample"] for r in fused_records if r["top_prediction"] == r["true_label"] and r["fused_prediction"] != r["true_label"]]
    top2_misses = [r["sample"] for r in fused_records if r["true_label"] not in r["top2_candidates"]]
    (output_dir / "holdout-fusion-records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in fused_records), encoding="utf-8",
    )
    report = {
        "schema_version": 1, "evaluation_kind": "nested_same_batch_group_holdout",
        "full_rgb_instance": args.full_rgb_instance,
        "fixed_holdout_records": args.fixed_holdout_records,
        "missing_fixed_holdout": missing_fixed_holdout,
        "outer_seed": args.seed, "inner_seed": args.seed + 1,
        "outer_holdout_per_group": args.holdout_per_group,
        "inner_calibration_per_group": args.calibration_per_group,
        "source_samples": len(source), "valid_paired_samples": len(paired),
        "model_training_samples": len(outer_training), "inner_training_samples": len(inner.training),
        "inner_calibration_samples": len(inner.holdout), "outer_holdout_samples": len(outer_holdout),
        "outer_group_counts": outer.group_counts, "inner_group_counts": inner.group_counts,
        "stable_top_base_samples": len(base_labels), "observed_classes": list(model_registry.class_ids),
        "missing_classes": sorted(set(registry.class_ids) - observed),
        "side_load_errors": side_errors, "top_extraction_errors": top_errors,
        "policy": policy.to_dict(), "policy_fit": policy_fit,
        "top_rgbd_metrics": top_metrics,
        "classic_side_metrics": classic_side_metrics,
        "cross_view_topology_metrics": side_metrics,
        "side_metrics": side_metrics,
        "fused_metrics": fused_metrics,
        "raw_accuracy_gain": fused_metrics["raw_accuracy"] - top_metrics["raw_accuracy"],
        "macro_recall_gain_with_rejection": fused_metrics["macro_recall_with_rejection"] - top_metrics["macro_recall_with_rejection"],
        "corrected_count": len(corrected), "damaged_count": len(damaged),
        "corrected_samples": corrected, "damaged_samples": damaged,
        "top2_miss_count": len(top2_misses), "top2_miss_samples": top2_misses,
        "contact_sheets": sheets, "runtime_seconds": time.perf_counter() - started,
        "top_segmentation_backend": "VisionPipeline3D production RGB-D segmentation plus RGB contour recovery",
        "promotion_eligible": False,
        "limitations": [
            "temporary platform and same-batch random holdout; not an unseen-instance competition test",
            "source metadata is not human-reviewed",
            "cylinder is absent from this first dataset",
            "single-object frames do not validate projected multi-object association or occlusion",
            "temporary calibration passes only the relaxed development limits, not strict competition limits",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str((output_dir / "report.json").resolve()),
        "valid_paired_samples": report["valid_paired_samples"],
        "outer_holdout_samples": report["outer_holdout_samples"],
        "policy": report["policy"],
        "top_rgbd_metrics": report["top_rgbd_metrics"],
        "fused_metrics": report["fused_metrics"],
        "corrected_count": report["corrected_count"],
        "damaged_count": report["damaged_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
