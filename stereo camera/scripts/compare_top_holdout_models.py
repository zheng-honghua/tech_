"""Compare frozen and augmented RGB-D models on one cached dual holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sorting_vision.geometry_rgbd_model import DepthGeometryModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache = np.load(args.cache, allow_pickle=False)
    features = {
        str(key).replace("\\", "/"): feature
        for key, feature in zip(cache["sample_keys"], cache["features"], strict=True)
    }
    records = [
        json.loads(line)
        for line in Path(args.records).read_text(encoding="utf-8").splitlines()
        if line
    ]
    reports = []
    for model_path in args.model:
        model = DepthGeometryModel.load(model_path)
        raw_correct = accepted = accepted_correct = 0
        reasons: dict[str, int] = {}
        per_class: dict[str, list[int]] = {}
        for record in records:
            key = str(record["sample"]).replace("\\", "/")
            _label, _confidence, reason = model.predict_features(features[key])
            scores = dict(model.last_class_scores)
            prediction = max(scores, key=scores.get) if scores else "unknown"
            truth = str(record["true_label"])
            raw_correct += int(prediction == truth)
            accepted += int(reason == "accepted")
            accepted_correct += int(reason == "accepted" and prediction == truth)
            reasons[reason] = reasons.get(reason, 0) + 1
            counts = per_class.setdefault(truth, [0, 0])
            counts[0] += int(prediction == truth)
            counts[1] += 1
        reports.append({
            "model": str(Path(model_path).resolve()),
            "feature_count": len(model.feature_names),
            "training_samples": len(model.training_data()[1]),
            "labels": list(model.labels),
            "samples": len(records),
            "raw_accuracy": raw_correct / max(len(records), 1),
            "coverage": accepted / max(len(records), 1),
            "accepted_accuracy": accepted_correct / max(accepted, 1),
            "wrong_accept_count": accepted - accepted_correct,
            "reasons": reasons,
            "per_class_raw_accuracy": {
                label: correct / total
                for label, (correct, total) in sorted(per_class.items())
            },
        })
    output = {"schema_version": 1, "models": reports}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
