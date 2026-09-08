from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry_rgb import load_geometry_samples, preprocess_geometry_object
from .geometry_structure import extract_structural_contour, render_structural_contour
from .scene_image import _separate_object_masks


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def _write_contact_sheets(
    target: Path, items: dict[str, list[tuple[str, np.ndarray]]]
) -> None:
    overview = target / "总览"
    overview.mkdir(parents=True, exist_ok=True)
    for label_name, labelled_images in items.items():
        tiles = []
        for stem, image in labelled_images:
            tile = cv2.copyMakeBorder(
                image, 34, 2, 2, 2, cv2.BORDER_CONSTANT, value=(245, 245, 245)
            )
            cv2.putText(
                tile, stem[-24:], (7, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (20, 20, 20), 1, cv2.LINE_AA,
            )
            tiles.append(tile)
        if not tiles:
            continue
        columns = min(4, len(tiles))
        rows = (len(tiles) + columns - 1) // columns
        height, width = tiles[0].shape[:2]
        sheet = np.full((rows * height, columns * width, 3), 235, np.uint8)
        for index, tile in enumerate(tiles):
            row, column = divmod(index, columns)
            sheet[row * height:(row + 1) * height,
                  column * width:(column + 1) * width] = tile
        _write_image(overview / f"{label_name}.jpg", sheet)


def audit_geometry_structures(data_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    target = Path(output_root)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"structure audit output directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    samples, errors = load_geometry_samples(data_root)
    rows: list[dict[str, Any]] = []
    durations: list[float] = []
    overview_items: dict[str, list[tuple[str, np.ndarray]]] = {}
    for sample in samples:
        prepared = preprocess_geometry_object(sample.image_bgr, output_size=256)
        if prepared is None:
            errors.append({"path": str(sample.path), "reason": "object_not_found"})
            continue
        started = time.perf_counter()
        result = extract_structural_contour(prepared.image_bgr, prepared.mask)
        durations.append((time.perf_counter() - started) * 1000.0)
        sample_dir = target / "按真实类别" / sample.label_name / sample.path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        _write_image(sample_dir / "clean_mask.png", result.clean_mask)
        _write_image(sample_dir / "raw_edge_map.png", result.raw_edge_map)
        _write_image(sample_dir / "clean_line_map.png", result.clean_line_map)
        _write_image(sample_dir / "vertex_heatmap.png", result.vertex_heatmap)
        overlay = render_structural_contour(prepared.image_bgr, result)
        _write_image(sample_dir / "structure_overlay.png", overlay)
        overview_items.setdefault(sample.label_name, []).append(
            (sample.path.stem, overlay)
        )
        payload = {
            "source_path": str(sample.path),
            "true_label": sample.label_id,
            "true_name": sample.label_name,
            **result.to_dict(),
        }
        (sample_dir / "structure.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.append(payload)
    _write_contact_sheets(target, overview_items)
    values = np.asarray(durations, np.float64)
    summary = {
        "data_root": str(Path(data_root)),
        "images": len(rows),
        "accepted": sum(row["reason"] == "accepted" for row in rows),
        "clean_structure_rate": sum(row["reason"] == "accepted" for row in rows) / max(len(rows), 1),
        "zero_dangling_rate": sum(row["dangling_endpoint_count"] == 0 for row in rows) / max(len(rows), 1),
        "mean_outer_iou": float(np.mean([row["outer_iou"] for row in rows])) if rows else 0.0,
        "mean_rejected_candidate_ratio": float(np.mean([
            row["rejected_candidate_ratio"] for row in rows
        ])) if rows else 0.0,
        "mean_internal_lines": float(np.mean([
            len(row["internal_lines"]) for row in rows
        ])) if rows else 0.0,
        "images_with_internal_lines": sum(bool(row["internal_lines"]) for row in rows),
        "internal_line_coverage_rate": sum(bool(row["internal_lines"]) for row in rows) / max(len(rows), 1),
        "extraction_p50_ms": float(np.percentile(values, 50)) if len(values) else 0.0,
        "extraction_p95_ms": float(np.percentile(values, 95)) if len(values) else 0.0,
        "errors": errors,
        "same_batch_only": True,
    }
    (target / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (target / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary


def audit_geometry_scene_structures(
    data_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    root = Path(data_root)
    target = Path(output_root)
    if not root.is_dir():
        raise FileNotFoundError(f"scene data root does not exist: {root}")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"scene structure output directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    overview_items: dict[str, list[tuple[str, np.ndarray]]] = {}
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    scene_count = 0
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            errors.append({"path": str(path), "reason": "unreadable_image"})
            continue
        scene_count += 1
        scale = min(1.0, 1280.0 / max(image.shape[:2]))
        analysis = (
            cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0 else image
        )
        components, foreground = _separate_object_masks(analysis)
        scene_dir = target / path.stem
        scene_dir.mkdir(parents=True, exist_ok=True)
        _write_image(scene_dir / "foreground_mask.png", foreground)
        for index, component in enumerate(components, start=1):
            prepared = preprocess_geometry_object(
                analysis, component.mask, output_size=256
            )
            if prepared is None:
                errors.append({
                    "path": str(path), "reason": f"object_{index:03d}_preprocess_failed"
                })
                continue
            result = extract_structural_contour(prepared.image_bgr, prepared.mask)
            object_id = f"object-{index:03d}"
            object_dir = scene_dir / object_id
            object_dir.mkdir(parents=True, exist_ok=True)
            overlay = render_structural_contour(prepared.image_bgr, result)
            _write_image(object_dir / "normalized.png", prepared.image_bgr)
            _write_image(object_dir / "clean_mask.png", result.clean_mask)
            _write_image(object_dir / "raw_edge_map.png", result.raw_edge_map)
            _write_image(object_dir / "clean_line_map.png", result.clean_line_map)
            _write_image(object_dir / "vertex_heatmap.png", result.vertex_heatmap)
            _write_image(object_dir / "structure_overlay.png", overlay)
            payload = {
                "source_path": str(path), "object_id": object_id,
                "complete_in_frame": component.complete_in_frame,
                **result.to_dict(),
            }
            (object_dir / "structure.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            rows.append(payload)
            overview_items.setdefault(path.stem, []).append((object_id, overlay))
    _write_contact_sheets(target, overview_items)
    summary = {
        "data_root": str(root),
        "scenes": scene_count,
        "objects": len(rows),
        "objects_with_internal_lines": sum(bool(row["internal_lines"]) for row in rows),
        "zero_dangling_rate": sum(row["dangling_endpoint_count"] == 0 for row in rows) / max(len(rows), 1),
        "mean_outer_iou": float(np.mean([row["outer_iou"] for row in rows])) if rows else 0.0,
        "errors": errors,
        "unlabelled_visual_audit": True,
    }
    (target / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
