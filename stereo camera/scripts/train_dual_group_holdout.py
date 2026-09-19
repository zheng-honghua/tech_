"""Train a provisional side-view model with fixed random holdout per batch/label.

This tool does not mutate source metadata. It is intended for first-pass dataset
diagnosis; a model trained from incomplete or unreviewed data is marked as not
deployable in its report.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sorting_vision.group_holdout import deterministic_group_holdout
from sorting_vision.rgbd_dataset import depth_preview
from sorting_vision.shape_registry import ShapeRegistry, load_shape_registry
from sorting_vision.side_geometry import (
    SideGeometryModel,
    SideTrainingSample,
    load_side_training_samples,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Provisional dual side training with N random holdouts per batch/shape group"
    )
    parser.add_argument("--samples-root", required=True)
    parser.add_argument("--side-background", required=True)
    parser.add_argument("--shape-registry", default="config/shapes/competition-11.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--holdout-per-group", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser


def _sample_key(sample: SideTrainingSample) -> str:
    return sample.directory.resolve().as_posix()


def _group_key(sample: SideTrainingSample) -> str:
    return f"{sample.batch_id}/{sample.label_id}"


def _subset_registry(registry: ShapeRegistry, observed: set[str]) -> ShapeRegistry:
    return ShapeRegistry(
        version=registry.version,
        classes=tuple(item for item in registry.classes if item.enabled and item.class_id in observed),
    )


def _evaluate(
    model: SideGeometryModel,
    samples: tuple[SideTrainingSample, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    class_totals: Counter[str] = Counter()
    class_raw_correct: Counter[str] = Counter()
    class_accepted_correct: Counter[str] = Counter()
    latencies: list[float] = []
    wrong_accepts = 0
    accepted = 0
    raw_correct = 0
    for sample in samples:
        started = time.perf_counter()
        prediction, confidence, diagnostics = model.predict(sample.image_bgr, sample.mask)
        latency = (time.perf_counter() - started) * 1000.0
        latencies.append(latency)
        nearest = str(diagnostics.get("nearest_label", "unknown"))
        reason = str(diagnostics.get("reason", "unknown"))
        is_accepted = reason == "accepted"
        is_raw_correct = nearest == sample.label_id
        is_accepted_correct = is_accepted and prediction == sample.label_id
        class_totals[sample.label_id] += 1
        class_raw_correct[sample.label_id] += int(is_raw_correct)
        class_accepted_correct[sample.label_id] += int(is_accepted_correct)
        confusion[sample.label_id][nearest] += 1
        raw_correct += int(is_raw_correct)
        accepted += int(is_accepted)
        wrong_accepts += int(is_accepted and prediction != sample.label_id)
        records.append(
            {
                "sample": _sample_key(sample),
                "group": _group_key(sample),
                "batch_id": sample.batch_id,
                "true_label": sample.label_id,
                "color_id": sample.color_id,
                "instance_id": sample.instance_id,
                "prediction": prediction,
                "nearest_label": nearest,
                "accepted": is_accepted,
                "reason": reason,
                "confidence": float(confidence),
                "distance": float(diagnostics.get("distance", float("inf"))),
                "margin": float(diagnostics.get("margin", 0.0)),
                "raw_correct": is_raw_correct,
                "accepted_correct": is_accepted_correct,
                "latency_ms": latency,
            }
        )
    recalls = [
        class_accepted_correct[label] / class_totals[label]
        for label in sorted(class_totals)
    ]
    metrics = {
        "method": model.method,
        "sample_count": len(samples),
        "raw_accuracy": raw_correct / max(len(samples), 1),
        "accepted_accuracy": (accepted - wrong_accepts) / max(accepted, 1),
        "macro_recall_with_rejection": float(np.mean(recalls)) if recalls else 0.0,
        "coverage": accepted / max(len(samples), 1),
        "wrong_accept_count": wrong_accepts,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "per_class": {
            label: {
                "count": class_totals[label],
                "raw_accuracy": class_raw_correct[label] / class_totals[label],
                "accepted_recall": class_accepted_correct[label] / class_totals[label],
            }
            for label in sorted(class_totals)
        },
        "raw_confusion": {
            label: dict(sorted(values.items())) for label, values in sorted(confusion.items())
        },
    }
    return metrics, records


def _tile(image: np.ndarray, title: str, width: int = 420, height: int = 236) -> np.ndarray:
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(resized, (0, 0), (width, 28), (0, 0, 0), -1)
    cv2.putText(
        resized, title[:68], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return resized


def _write_contact_sheets(
    output_dir: Path,
    samples: tuple[SideTrainingSample, ...],
    prediction_records: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    contact_dir = output_dir / "holdout-contact-sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[SideTrainingSample]] = defaultdict(list)
    for sample in samples:
        groups[_group_key(sample)].append(sample)
    outputs: list[str] = []
    for index, (group, group_samples) in enumerate(sorted(groups.items()), start=1):
        rows: list[np.ndarray] = []
        for sample in sorted(group_samples, key=_sample_key):
            primary = cv2.imread(str(sample.directory / "primary-color.png"), cv2.IMREAD_COLOR)
            depth = np.load(sample.directory / "primary-depth.npy", allow_pickle=False)
            side = sample.image_bgr.copy()
            contours, _ = cv2.findContours(
                (sample.mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(side, contours, -1, (0, 255, 0), 2)
            sample_id = _sample_key(sample)
            knn = prediction_records["knn"][sample_id]
            forest = prediction_records["rtrees"][sample_id]
            summary = (
                f"truth={sample.label_id} K={knn['nearest_label']} "
                f"R={forest['nearest_label']}"
            )
            rows.append(
                np.hstack(
                    (
                        _tile(primary, f"{sample.directory.name} {summary}"),
                        _tile(depth_preview(depth), "TOP DEPTH"),
                        _tile(side, f"SIDE + MASK color={sample.color_id}"),
                    )
                )
            )
        sheet = np.vstack(rows)
        safe_group = group.replace("/", "__").replace("\\", "__")
        name = f"{index:03d}-{safe_group}.jpg"
        if not cv2.imwrite(str(contact_dir / name), sheet):
            raise OSError(f"failed to write contact sheet {name}")
        outputs.append((contact_dir / name).relative_to(output_dir).as_posix())
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = load_shape_registry(args.shape_registry)
    background = cv2.imread(str(args.side_background), cv2.IMREAD_COLOR)
    if background is None:
        raise ValueError(f"cannot read side background: {args.side_background}")
    samples, load_errors = load_side_training_samples(
        args.samples_root, registry, background, require_reviewed=False
    )
    source_samples = tuple(sample for sample in samples if sample.split == "train")
    split = deterministic_group_holdout(
        source_samples,
        group_key=_group_key,
        item_key=_sample_key,
        holdout_per_group=args.holdout_per_group,
        seed=args.seed,
    )
    observed = {sample.label_id for sample in source_samples}
    missing = sorted(set(registry.class_ids) - observed)
    model_registry = registry if not missing else _subset_registry(registry, observed)
    triples = [
        (sample.image_bgr, sample.mask, sample.label_id) for sample in split.training
    ]
    candidates: list[tuple[SideGeometryModel, dict[str, Any], list[dict[str, Any]]]] = []
    for method in ("knn", "rtrees"):
        model = SideGeometryModel.train(triples, model_registry, method=method)
        metrics, records = _evaluate(model, split.holdout)
        candidates.append((model, metrics, records))
    candidates.sort(
        key=lambda item: (
            item[1]["wrong_accept_count"] == 0,
            -item[1]["wrong_accept_count"],
            item[1]["macro_recall_with_rejection"],
            item[1]["coverage"],
            item[1]["raw_accuracy"],
            -item[1]["latency_p95_ms"],
        ),
        reverse=True,
    )
    selected_model, selected_metrics, _ = candidates[0]
    selected_model.save(args.model_output)
    (output_dir / "observed-shape-registry.json").write_text(
        json.dumps(model_registry.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    prediction_records = {
        model.method: {_sample_key(sample): record for sample, record in zip(split.holdout, records)}
        for model, _, records in candidates
    }
    sheets = _write_contact_sheets(output_dir, split.holdout, prediction_records)
    holdout_records = []
    for sample in split.holdout:
        key = _sample_key(sample)
        holdout_records.append(
            {
                "sample": key,
                "group": _group_key(sample),
                "label_id": sample.label_id,
                "batch_id": sample.batch_id,
                "instance_id": sample.instance_id,
                "color_id": sample.color_id,
                "predictions": {
                    method: prediction_records[method][key]
                    for method in sorted(prediction_records)
                },
            }
        )
    (output_dir / "holdout-records.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in holdout_records),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "evaluation_kind": "same_batch_group_random_holdout",
        "seed": args.seed,
        "holdout_per_batch_label_group": args.holdout_per_group,
        "samples_root": str(Path(args.samples_root).resolve()),
        "side_background": str(Path(args.side_background).resolve()),
        "source_sample_count": len(source_samples),
        "training_sample_count": len(split.training),
        "holdout_sample_count": len(split.holdout),
        "group_counts": split.group_counts,
        "full_registry_class_ids": list(registry.class_ids),
        "observed_class_ids": list(model_registry.class_ids),
        "missing_class_ids": missing,
        "all_source_samples_human_reviewed": all(
            json.loads((sample.directory / "metadata.json").read_text(encoding="utf-8")).get(
                "human_reviewed", False
            )
            for sample in source_samples
        ),
        "load_errors": load_errors,
        "candidates": [metrics for _, metrics, _ in candidates],
        "selected_method": selected_model.method,
        "selected_metrics": selected_metrics,
        "model_output": str(Path(args.model_output).resolve()),
        "model_registry_hash": model_registry.registry_hash,
        "full_registry_hash": registry.registry_hash,
        "deployable_with_full_registry": not missing and not load_errors,
        "promotion_eligible": False,
        "limitations": [
            "same-batch random holdout is a development regression test, not unseen-instance validation",
            "source samples were loaded before mandatory human review",
            *(["enabled classes are missing; the saved diagnostic model cannot load with the full registry"] if missing else []),
        ],
        "contact_sheets": sheets,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not missing and not load_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())

