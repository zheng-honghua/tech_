from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry_cnn import load_geometry_shape_model
from .geometry_models import GeometryPrediction, GeometryShapeModel
from .geometry_rgb import (
    GeometryPreprocessed,
    GeometryRGBModel,
    preprocess_geometry_object,
)


DEFAULT_GEOMETRY_MODEL = Path("models/geometry-rgb-edges-expanded.npz")
REASON_TEXT_ZH = {
    "accepted": "证据充分，类别已接受",
    "object_not_found": "没有找到清晰的单个物块",
    "multiple_objects": "检测到多个主体，请每张图片只放一个物块",
    "edge_evidence_low": "可见内部棱线不足",
    "topology_conflict": "棱线拓扑互相矛盾",
    "margin_rejected": "前两种类别过于接近",
    "distance_rejected": "图片与所有已知类别差异过大",
}


@dataclass(frozen=True)
class SingleImageResult:
    input_path: str
    model_path: str
    image_width: int
    image_height: int
    prediction: GeometryPrediction
    bbox_px: tuple[int, int, int, int] | None
    candidate_count: int
    analysis_scale: float
    total_ms: float
    annotated_bgr: np.ndarray
    normalized_bgr: np.ndarray | None = None
    mask: np.ndarray | None = None

    def to_dict(self, artifacts: dict[str, str] | None = None) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task": "single_image_geometry_prediction",
            "input_image": self.input_path,
            "model": self.model_path,
            "image_size_px": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "object": {
                "bbox_px": list(self.bbox_px) if self.bbox_px else None,
                "candidate_count": self.candidate_count,
            },
            "analysis_scale": self.analysis_scale,
            "prediction": self.prediction.to_dict(),
            "total_ms": self.total_ms,
            "safe_for_robot": False,
            "artifacts": artifacts or {},
        }


class GeometryImagePredictor:
    """Small public API for predicting one labelled-object photograph."""

    def __init__(
        self,
        model: GeometryShapeModel,
        model_path: str | Path,
    ) -> None:
        self.model = model
        self.model_path = str(Path(model_path).resolve())

    @classmethod
    def load(
        cls,
        model_path: str | Path = DEFAULT_GEOMETRY_MODEL,
        backend: str = "auto",
        device: str = "CPU",
    ) -> "GeometryImagePredictor":
        path = Path(model_path)
        if not path.exists() and path == DEFAULT_GEOMETRY_MODEL:
            repository_model = Path(__file__).resolve().parents[2] / path
            if repository_model.exists():
                path = repository_model
        selected_backend = backend
        if backend == "auto":
            selected_backend = "openvino" if path.is_dir() or path.suffix == ".xml" else "opencv"
        if selected_backend not in {"opencv", "openvino"}:
            raise ValueError(f"unsupported geometry backend: {selected_backend}")
        if not path.exists():
            raise FileNotFoundError(f"geometry model does not exist: {path}")
        return cls(
            load_geometry_shape_model(selected_backend, path, device=device),
            path,
        )

    def predict(self, image_bgr: np.ndarray, input_path: str = "<array>") -> SingleImageResult:
        started = time.perf_counter()
        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError("input image must be a non-empty BGR image")
        height, width = image.shape[:2]
        analysis_scale = min(1.0, 1280.0 / max(width, height))
        analysis_image = (
            cv2.resize(
                image,
                (int(round(width * analysis_scale)), int(round(height * analysis_scale))),
                interpolation=cv2.INTER_AREA,
            )
            if analysis_scale < 1.0
            else image
        )
        prepared = preprocess_geometry_object(analysis_image, output_size=256)
        if prepared is not None and analysis_scale < 1.0:
            x, y, box_width, box_height = prepared.bbox_px
            prepared = GeometryPreprocessed(
                prepared.image_bgr,
                prepared.mask,
                tuple(
                    int(round(value / analysis_scale))
                    for value in (x, y, box_width, box_height)
                ),
                prepared.candidate_count,
            )
        if prepared is not None and prepared.candidate_count == 1:
            prediction = (
                self.model.predict_preprocessed_geometry(prepared)
                if isinstance(self.model, GeometryRGBModel)
                else self.model.predict_geometry(prepared.image_bgr, prepared.mask)
            )
        else:
            prediction = self.model.predict_geometry(image)
        annotated = _annotate(image, prepared, prediction)
        return SingleImageResult(
            input_path=input_path,
            model_path=self.model_path,
            image_width=width,
            image_height=height,
            prediction=prediction,
            bbox_px=prepared.bbox_px if prepared else None,
            candidate_count=prepared.candidate_count if prepared else 0,
            analysis_scale=analysis_scale,
            total_ms=(time.perf_counter() - started) * 1000.0,
            annotated_bgr=annotated,
            normalized_bgr=prepared.image_bgr if prepared else None,
            mask=prepared.mask if prepared else None,
        )

    def predict_file(self, image_path: str | Path) -> SingleImageResult:
        path = Path(image_path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read input image: {path}")
        return self.predict(image, str(path.resolve()))


def _annotate(
    image_bgr: np.ndarray,
    prepared: GeometryPreprocessed | None,
    prediction: GeometryPrediction,
) -> np.ndarray:
    canvas = image_bgr.copy()
    accepted = prediction.accepted and prediction.label_id != "unknown"
    colour = (0, 180, 0) if accepted else (0, 165, 255)
    if prepared is not None:
        x, y, width, height = prepared.bbox_px
        cv2.rectangle(canvas, (x, y), (x + width, y + height), colour, 3)
        origin = (max(8, x), max(30, y - 10))
    else:
        origin = (12, 34)
    label = prediction.label_id if accepted else "unknown"
    text = f"{label} conf={prediction.confidence:.3f} {prediction.reason}"
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        colour,
        2,
        cv2.LINE_AA,
    )
    return canvas


def save_single_image_result(
    result: SingleImageResult, output_dir: str | Path
) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    artifacts = {"annotated_image": "annotated.jpg", "result_json": "result.json"}
    if not cv2.imwrite(str(target / artifacts["annotated_image"]), result.annotated_bgr):
        raise OSError("failed to write annotated image")
    if result.normalized_bgr is not None and result.mask is not None:
        artifacts["normalized_crop"] = "normalized.png"
        artifacts["object_mask"] = "mask.png"
        if not cv2.imwrite(str(target / artifacts["normalized_crop"]), result.normalized_bgr):
            raise OSError("failed to write normalized crop")
        if not cv2.imwrite(str(target / artifacts["object_mask"]), result.mask):
            raise OSError("failed to write object mask")
    payload = result.to_dict(artifacts)
    (target / artifacts["result_json"]).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return artifacts
