from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry_edges import EdgeTopology, extract_edge_topology, render_edge_lines
from .geometry_models import GeometryPrediction
from .geometry_rgb import GeometryPreprocessed, GeometryRGBModel, preprocess_geometry_object
from .single_image import DEFAULT_GEOMETRY_MODEL, GeometryImagePredictor


@dataclass(frozen=True)
class SceneObjectResult:
    object_id: str
    bbox_px: tuple[int, int, int, int]
    prediction: GeometryPrediction
    normalized_bgr: np.ndarray
    mask: np.ndarray
    topology: EdgeTopology | None = None
    complete_in_frame: bool = True

    def to_dict(self, artifacts: dict[str, str] | None = None) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "bbox_px": list(self.bbox_px),
            "prediction": self.prediction.to_dict(),
            "complete_in_frame": self.complete_in_frame,
            "safe_for_robot": False,
            "artifacts": artifacts or {},
        }


@dataclass(frozen=True)
class SceneImageResult:
    input_path: str
    model_path: str
    image_width: int
    image_height: int
    analysis_scale: float
    total_ms: float
    objects: tuple[SceneObjectResult, ...]
    annotated_bgr: np.ndarray
    foreground_mask: np.ndarray

    def to_dict(
        self,
        object_artifacts: dict[str, dict[str, str]] | None = None,
        scene_artifacts: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        object_artifacts = object_artifacts or {}
        return {
            "schema_version": 1,
            "task": "multi_object_geometry_prediction",
            "input_image": self.input_path,
            "model": self.model_path,
            "image_size_px": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "analysis_scale": self.analysis_scale,
            "detected_count": len(self.objects),
            "accepted_count": sum(item.prediction.accepted for item in self.objects),
            "rejected_count": sum(not item.prediction.accepted for item in self.objects),
            "objects": [
                item.to_dict(object_artifacts.get(item.object_id))
                for item in self.objects
            ],
            "total_ms": self.total_ms,
            "safe_for_robot": False,
            "artifacts": scene_artifacts or {},
        }


class GeometryScenePredictor:
    """Detect and classify multiple separated geometry objects in one RGB image."""

    def __init__(self, image_predictor: GeometryImagePredictor) -> None:
        self.image_predictor = image_predictor
        self.model = image_predictor.model
        self.model_path = image_predictor.model_path

    @classmethod
    def load(
        cls,
        model_path: str | Path = DEFAULT_GEOMETRY_MODEL,
        backend: str = "auto",
        device: str = "CPU",
    ) -> "GeometryScenePredictor":
        return cls(GeometryImagePredictor.load(model_path, backend, device))

    def predict(self, image_bgr: np.ndarray, input_path: str = "<array>") -> SceneImageResult:
        started = time.perf_counter()
        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError("input image must be a non-empty BGR image")
        height, width = image.shape[:2]
        scale = min(1.0, 1280.0 / max(width, height))
        analysis = (
            cv2.resize(
                image,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            if scale < 1.0
            else image
        )
        components, combined = _separate_object_masks(analysis)
        items: list[SceneObjectResult] = []
        for index, component in enumerate(components, start=1):
            prepared = preprocess_geometry_object(
                analysis, component.mask, output_size=256
            )
            if prepared is None:
                continue
            topology = None
            if not component.complete_in_frame:
                prediction = GeometryPrediction(
                    label_id="unknown",
                    label_name="未知形状",
                    confidence=0.0,
                    accepted=False,
                    backend=self.model.backend,
                    reason="object_out_of_frame",
                )
            elif isinstance(self.model, GeometryRGBModel):
                if self.model.feature_set == "edge-topology":
                    topology = extract_edge_topology(
                        prepared.image_bgr,
                        prepared.mask,
                        enhanced_faces=(
                            self.model.feature_version >= 3
                        ),
                    )
                prediction = self.model.predict_preprocessed_geometry(
                    prepared, topology
                )
            else:
                prediction = self.model.predict_geometry(
                    prepared.image_bgr, prepared.mask
                )
            bbox = tuple(
                int(round(value / scale)) for value in prepared.bbox_px
            )
            items.append(
                SceneObjectResult(
                    object_id=f"object-{index:03d}",
                    bbox_px=bbox,
                    prediction=prediction,
                    normalized_bgr=prepared.image_bgr,
                    mask=prepared.mask,
                    topology=topology,
                    complete_in_frame=component.complete_in_frame,
                )
            )
        annotated = _annotate_scene(image, items)
        foreground = (
            cv2.resize(combined, (width, height), interpolation=cv2.INTER_NEAREST)
            if scale < 1.0
            else combined
        )
        return SceneImageResult(
            input_path=input_path,
            model_path=self.model_path,
            image_width=width,
            image_height=height,
            analysis_scale=scale,
            total_ms=(time.perf_counter() - started) * 1000.0,
            objects=tuple(items),
            annotated_bgr=annotated,
            foreground_mask=foreground,
        )

    def predict_file(self, image_path: str | Path) -> SceneImageResult:
        path = Path(image_path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read input image: {path}")
        return self.predict(image, str(path.resolve()))


@dataclass(frozen=True)
class _SceneComponent:
    mask: np.ndarray
    complete_in_frame: bool


def _separate_object_masks(
    image_bgr: np.ndarray,
) -> tuple[list[_SceneComponent], np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    combined = ((hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 30)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    if cv2.countNonZero(combined) == 0:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        border = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]), axis=0)
        delta = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
        combined = (delta >= 24.0).astype(np.uint8) * 255
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(combined, 8)
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    minimum_area = max(250, int(round(image_area * 0.0004)))
    maximum_area = int(round(image_area * 0.25))
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    masks: list[tuple[tuple[int, int], _SceneComponent]] = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        touches_border = (
            x <= 3
            or y <= 3
            or x + width >= image_bgr.shape[1] - 3
            or y + height >= image_bgr.shape[0] - 3
        )
        component = labels == label
        fill_ratio = float(area) / max(width * height, 1)
        median_saturation = float(np.median(hsv[:, :, 1][component]))
        median_value = float(np.median(hsv[:, :, 2][component]))
        # The current competition samples are strongly coloured. These quality
        # gates remove dark cables and thin tray accessories that occasionally
        # pass the permissive foreground threshold.
        foreground_quality_ok = (
            fill_ratio >= 0.45
            and median_saturation >= 100.0
            and median_value >= 55.0
        )
        if not minimum_area <= area <= maximum_area or not foreground_quality_ok:
            continue
        masks.append(
            (
                (y, x),
                _SceneComponent(
                    component.astype(np.uint8) * 255,
                    complete_in_frame=not touches_border,
                ),
            )
        )
    masks.sort(key=lambda item: item[0])
    return [component for _, component in masks], combined


def _annotate_scene(
    image_bgr: np.ndarray, objects: list[SceneObjectResult]
) -> np.ndarray:
    canvas = image_bgr.copy()
    for item in objects:
        x, y, width, height = item.bbox_px
        accepted = item.prediction.accepted and item.prediction.label_id != "unknown"
        colour = (0, 180, 0) if accepted else (0, 165, 255)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), colour, 3)
        label = item.prediction.label_id if accepted else "unknown"
        text = f"{item.object_id} {label} {item.prediction.confidence:.2f}"
        cv2.putText(
            canvas,
            text,
            (max(5, x), max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            cv2.LINE_AA,
        )
    return canvas


def save_scene_image_result(
    result: SceneImageResult, output_dir: str | Path
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    scene_artifacts = {
        "annotated_image": "annotated.jpg",
        "foreground_mask": "foreground-mask.png",
        "result_json": "result.json",
    }
    if not cv2.imwrite(str(target / "annotated.jpg"), result.annotated_bgr):
        raise OSError("failed to write scene annotation")
    if not cv2.imwrite(str(target / "foreground-mask.png"), result.foreground_mask):
        raise OSError("failed to write scene foreground mask")

    object_artifacts: dict[str, dict[str, str]] = {}
    for item in result.objects:
        directory = target / item.object_id
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "normalized_crop": f"{item.object_id}/normalized.png",
            "object_mask": f"{item.object_id}/mask.png",
            "result_json": f"{item.object_id}/result.json",
        }
        if not cv2.imwrite(str(directory / "normalized.png"), item.normalized_bgr):
            raise OSError(f"failed to write crop for {item.object_id}")
        if not cv2.imwrite(str(directory / "mask.png"), item.mask):
            raise OSError(f"failed to write mask for {item.object_id}")
        if item.topology is not None:
            artifacts.update(
                {
                    "edge_map": f"{item.object_id}/edge-map.png",
                    "topology_image": f"{item.object_id}/topology.png",
                    "topology_json": f"{item.object_id}/topology.json",
                }
            )
            if not cv2.imwrite(
                str(directory / "edge-map.png"), item.topology.edge_map
            ):
                raise OSError(f"failed to write edge map for {item.object_id}")
            if not cv2.imwrite(
                str(directory / "topology.png"),
                render_edge_lines(item.normalized_bgr, item.topology),
            ):
                raise OSError(f"failed to write topology image for {item.object_id}")
            (directory / "topology.json").write_text(
                json.dumps(item.topology.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        object_artifacts[item.object_id] = artifacts
        (directory / "result.json").write_text(
            json.dumps(item.to_dict(artifacts), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    payload = result.to_dict(object_artifacts, scene_artifacts)
    (target / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"scene": scene_artifacts, "objects": object_artifacts}
