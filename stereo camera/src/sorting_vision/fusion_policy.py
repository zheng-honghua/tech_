from __future__ import annotations

import importlib.util
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np



def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    positive = {str(key): max(0.0, float(value)) for key, value in scores.items()}
    total = sum(positive.values())
    return {} if total <= 0 else {key: value / total for key, value in positive.items()}


FUSION_METHODS = ("probability_sum", "log_product", "top2_logistic")
FUSION_BACKENDS = ("auto", "opencv", "openvino", "tensorrt")


@dataclass(frozen=True)
class FusionPolicy:
    registry_hash: str
    method: str = "log_product"
    top_temperature: float = 1.0
    side_temperature: float = 1.0
    side_weight: float = 0.30
    fused_probability: float = 0.72
    fused_margin: float = 0.12
    logistic_weights: tuple[float, ...] = ()
    logistic_bias: float = 0.0
    version: int = 1

    def __post_init__(self) -> None:
        if self.method not in FUSION_METHODS:
            raise ValueError(f"unsupported fusion method: {self.method}")
        if not self.registry_hash:
            raise ValueError("fusion policy needs a shape registry hash")
        if self.top_temperature <= 0 or self.side_temperature <= 0:
            raise ValueError("fusion temperatures must be positive")
        if not 0.0 <= self.side_weight <= 1.0:
            raise ValueError("fusion side weight must be in [0, 1]")
        if self.method == "top2_logistic" and len(self.logistic_weights) != 5:
            raise ValueError("top2_logistic needs five generic feature weights")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "version": self.version,
            "registry_hash": self.registry_hash,
            "method": self.method,
            "top_temperature": self.top_temperature,
            "side_temperature": self.side_temperature,
            "side_weight": self.side_weight,
            "fused_probability": self.fused_probability,
            "fused_margin": self.fused_margin,
            "logistic_weights": list(self.logistic_weights),
            "logistic_bias": self.logistic_bias,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, expected_registry_hash: str | None = None) -> "FusionPolicy":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        policy = cls(
            registry_hash=str(value["registry_hash"]),
            method=str(value.get("method", "log_product")),
            top_temperature=float(value.get("top_temperature", 1.0)),
            side_temperature=float(value.get("side_temperature", 1.0)),
            side_weight=float(value.get("side_weight", 0.30)),
            fused_probability=float(value.get("fused_probability", 0.72)),
            fused_margin=float(value.get("fused_margin", 0.12)),
            logistic_weights=tuple(map(float, value.get("logistic_weights", ()))),
            logistic_bias=float(value.get("logistic_bias", 0.0)),
            version=int(value.get("version", 1)),
        )
        if expected_registry_hash is not None and policy.registry_hash != expected_registry_hash:
            raise ValueError("fusion policy shape registry hash mismatch")
        return policy


def top2_fusion_features(
    top_scores: dict[str, float], side_scores: dict[str, float], candidates: tuple[str, str], quality: float
) -> np.ndarray:
    first, second = candidates
    epsilon = 1e-8
    top_first = max(top_scores.get(first, 0.0), epsilon)
    top_second = max(top_scores.get(second, 0.0), epsilon)
    side_first = max(side_scores.get(first, 0.0), epsilon)
    side_second = max(side_scores.get(second, 0.0), epsilon)
    return np.asarray(
        [
            math.log(top_first / top_second),
            math.log(side_first / side_second),
            float(np.clip(quality, 0.0, 1.0)),
            top_first - top_second,
            side_first - side_second,
        ],
        np.float64,
    )


def fuse_top2_scores(
    top_scores: dict[str, float],
    side_scores: dict[str, float],
    quality: float,
    *,
    method: str = "log_product",
    side_weight: float = 0.30,
    logistic_weights: tuple[float, ...] = (),
    logistic_bias: float = 0.0,
) -> tuple[dict[str, float], tuple[str, ...]]:
    if method not in FUSION_METHODS:
        raise ValueError(f"unsupported fusion method: {method}")
    ordered = sorted(top_scores.items(), key=lambda item: item[1], reverse=True)
    candidates = tuple(label for label, _ in ordered[:2])
    if not candidates:
        return {}, ()
    if len(candidates) == 1:
        return {candidates[0]: 1.0}, candidates
    first, second = candidates
    top = _normalize_scores({item: top_scores.get(item, 0.0) for item in candidates})
    side = _normalize_scores({item: side_scores.get(item, 0.0) for item in candidates})
    if not side:
        return top, candidates
    effective_side = float(np.clip(side_weight * quality, 0.0, 0.95))
    if method == "probability_sum":
        fused = _normalize_scores(
            {item: (1.0 - effective_side) * top[item] + effective_side * side[item] for item in candidates}
        )
    elif method == "log_product":
        raw = {
            item: (1.0 - effective_side) * math.log(max(top[item], 1e-9))
            + effective_side * math.log(max(side[item], 1e-9))
            for item in candidates
        }
        maximum = max(raw.values())
        fused = _normalize_scores({item: math.exp(value - maximum) for item, value in raw.items()})
    else:
        if len(logistic_weights) != 5:
            raise ValueError("top2_logistic needs five weights")
        features = top2_fusion_features(top, side, (first, second), quality)
        logit = float(np.dot(np.asarray(logistic_weights), features) + logistic_bias)
        probability = 1.0 / (1.0 + math.exp(-float(np.clip(logit, -40.0, 40.0))))
        fused = {first: probability, second: 1.0 - probability}
    return fused, candidates


def resolve_fusion_backend(
    requested: str,
    *,
    machine: str | None = None,
    openvino_available: bool | None = None,
    tensorrt_available: bool | None = None,
) -> str:
    if requested not in FUSION_BACKENDS:
        raise ValueError(f"unsupported fusion backend: {requested}")
    architecture = (machine or platform.machine()).lower()
    if openvino_available is None:
        openvino_available = importlib.util.find_spec("openvino") is not None
    if tensorrt_available is None:
        tensorrt_available = importlib.util.find_spec("tensorrt") is not None
    if requested == "auto":
        if architecture in {"aarch64", "arm64"} and tensorrt_available:
            return "tensorrt"
        if architecture in {"amd64", "x86_64", "x64"} and openvino_available:
            return "openvino"
        return "opencv"
    if requested == "openvino" and not openvino_available:
        raise RuntimeError("OpenVINO backend was requested but openvino is not installed")
    if requested == "tensorrt" and not tensorrt_available:
        raise RuntimeError("TensorRT backend was requested but tensorrt is not installed")
    return requested
