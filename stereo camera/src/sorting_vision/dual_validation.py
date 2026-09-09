from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .dual_view import temperature_scale_scores


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
    projection = [float(record["projection_error_px"]) for record in eligible]
    pair_delta = [float(record["pair_delta_ms"]) for record in eligible]
    latency = [float(record["latency_ms"]) for record in eligible]
    safety_violations = sum(bool(record.get("safety_state_upgraded", False)) for record in eligible)
    platform_ids = sorted({str(record.get("platform_id", "")) for record in eligible})
    gates = {
        "macro_recall_gain_ge_0_03": fused_recall - mono_recall >= 0.03,
        "wrong_accept_not_worse": fused_wrong <= mono_wrong,
        "no_safety_upgrade": safety_violations == 0,
        "projection_p95_le_3_px": float(np.percentile(projection, 95)) <= 3.0,
        "pair_delta_p95_le_50_ms": float(np.percentile(pair_delta, 95)) <= 50.0,
        "latency_p95_le_1000_ms": float(np.percentile(latency, 95)) <= 1000.0,
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
        "safety_upgrade_violations": safety_violations,
        "projection_error_p95_px": float(np.percentile(projection, 95)),
        "pair_delta_p95_ms": float(np.percentile(pair_delta, 95)),
        "latency_p95_ms": float(np.percentile(latency, 95)),
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
