"""Compare legacy and fused-edge KNN models on identical batch-held-out samples."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from sorting_vision.geometry_rgbd_model import (
    DepthGeometryModel,
    FEATURE_NAMES,
    FUSED_FEATURE_NAMES,
)


def _metrics(truth: np.ndarray, predicted: np.ndarray, labels: list[str]) -> dict:
    recalls = {}
    for label in labels:
        selected = truth == label
        recalls[label] = float(np.mean(predicted[selected] == label)) if np.any(selected) else 0.0
    return {
        "correct": int(np.sum(predicted == truth)),
        "wrong": int(np.sum((predicted != truth) & (predicted != "unknown"))),
        "unknown": int(np.sum(predicted == "unknown")),
        "total": int(len(truth)),
        "accuracy": float(np.mean(predicted == truth)),
        "macro_recall": float(np.mean(list(recalls.values()))),
        "class_recall": recalls,
    }


def _edge_weights(value: float) -> np.ndarray:
    weights = np.ones(len(FUSED_FEATURE_NAMES), np.float32)
    weights[len(FEATURE_NAMES):] = value
    return weights


def _select_weight_inside_training_batches(
    features: np.ndarray,
    truth: np.ndarray,
    batches: np.ndarray,
    outer_train: np.ndarray,
    labels: list[str],
    candidates: tuple[float, ...],
) -> tuple[float, list[dict]]:
    training_batches = sorted(set(batches[outer_train].tolist()))
    reports = []
    for weight in candidates:
        predictions = np.full(int(np.sum(outer_train)), "unknown", dtype=object)
        local_features = features[outer_train]
        local_truth = truth[outer_train]
        local_batches = batches[outer_train]
        for held_out in training_batches:
            train = local_batches != held_out
            test = ~train
            model = DepthGeometryModel.fit(
                local_features[train], local_truth[train].tolist(),
                feature_weights=_edge_weights(weight),
                feature_names=FUSED_FEATURE_NAMES,
            )
            predictions[test] = [
                model.predict_features(row)[0] for row in local_features[test]
            ]
        metrics = _metrics(local_truth, predictions, labels)
        reports.append({"edge_group_weight": weight, **metrics})
    selected = max(
        reports,
        key=lambda item: (item["macro_recall"], -item["wrong"], item["correct"]),
    )
    return float(selected["edge_group_weight"]), reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--candidate-output", required=True)
    parser.add_argument("--baseline-output", required=True)
    args = parser.parse_args()
    for value in (args.report, args.candidate_output, args.baseline_output):
        if Path(value).exists():
            raise FileExistsError(value)

    cache = Path(args.cache)
    rows = [
        row for row in json.loads((cache / "frames.json").read_text(encoding="utf-8"))
        if "feature_index" in row
    ]
    features = np.load(cache / "features.npz")["features"].astype(np.float32)
    if features.shape != (len(rows), len(FUSED_FEATURE_NAMES)):
        raise ValueError("cache does not contain fused-edge feature vectors")
    truth = np.asarray([row["label_id"] for row in rows])
    batches = np.asarray([row["batch_id"] for row in rows])
    labels = sorted(set(truth.tolist()))
    fold_reports = []
    legacy_predictions = np.full(len(rows), "unknown", dtype=object)
    fused_predictions = np.full(len(rows), "unknown", dtype=object)
    weight_candidates = (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
    selected_weights: list[float] = []
    for held_out in sorted(set(batches.tolist())):
        train = batches != held_out
        test = ~train
        legacy = DepthGeometryModel.fit(
            features[train, :len(FEATURE_NAMES)], truth[train].tolist(),
            feature_names=FEATURE_NAMES,
        )
        selected_weight, inner_reports = _select_weight_inside_training_batches(
            features, truth, batches, train, labels, weight_candidates,
        )
        selected_weights.append(selected_weight)
        fused = DepthGeometryModel.fit(
            features[train], truth[train].tolist(),
            feature_weights=_edge_weights(selected_weight),
            feature_names=FUSED_FEATURE_NAMES,
        )
        legacy_predictions[test] = [legacy.predict_features(row)[0] for row in features[test]]
        fused_predictions[test] = [fused.predict_features(row)[0] for row in features[test]]
        fold_reports.append({
            "held_out_batch": held_out,
            "selected_edge_group_weight": selected_weight,
            "inner_training_only_selection": inner_reports,
            "legacy": _metrics(truth[test], legacy_predictions[test], labels),
            "fused": _metrics(truth[test], fused_predictions[test], labels),
        })

    legacy_metrics = _metrics(truth, legacy_predictions, labels)
    fused_metrics = _metrics(truth, fused_predictions, labels)
    class_deltas = {
        label: fused_metrics["class_recall"][label] - legacy_metrics["class_recall"][label]
        for label in labels
    }
    macro_delta = fused_metrics["macro_recall"] - legacy_metrics["macro_recall"]
    gate = macro_delta >= 0.03 and min(class_deltas.values(), default=0.0) >= -0.05
    report = {
        "evaluation": "nested leave-one-capture-batch-out",
        "cache": str(cache),
        "samples": len(rows),
        "legacy": legacy_metrics,
        "fused": fused_metrics,
        "macro_recall_delta": macro_delta,
        "class_recall_delta": class_deltas,
        "cross_batch_gate_passed": gate,
        "promotion_status": "candidate_only; multi-scene and visual gates not evaluated",
        "folds": fold_reports,
    }
    final_weight = Counter(selected_weights).most_common(1)[0][0]
    report["final_edge_group_weight"] = final_weight
    baseline = DepthGeometryModel.fit(
        features[:, :len(FEATURE_NAMES)], truth.tolist(), feature_names=FEATURE_NAMES
    )
    candidate = DepthGeometryModel.fit(
        features, truth.tolist(), feature_weights=_edge_weights(final_weight),
        feature_names=FUSED_FEATURE_NAMES,
    )
    baseline.save(args.baseline_output, {**report, "paired_role": "legacy baseline"})
    candidate.save(args.candidate_output, {**report, "paired_role": "fused candidate"})
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
