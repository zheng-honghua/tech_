from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry_edges import extract_edge_topology, render_edge_lines
from .geometry_rgb import load_geometry_samples, preprocess_geometry_object


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def audit_geometry_edges(
    data_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    target = Path(output_root)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"edge audit output directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    samples, errors = load_geometry_samples(data_root)
    rows: list[dict[str, Any]] = []
    durations: list[float] = []
    accepted = 0
    for sample in samples:
        prepared = preprocess_geometry_object(sample.image_bgr, output_size=256)
        if prepared is None:
            errors.append({"path": str(sample.path), "reason": "object_not_found"})
            continue
        started = time.perf_counter()
        topology = extract_edge_topology(prepared.image_bgr, prepared.mask)
        durations.append((time.perf_counter() - started) * 1000.0)
        accepted += int(topology.reason == "accepted")
        sample_dir = target / "按真实类别" / sample.label_name / sample.path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        _write_image(sample_dir / "enhanced_gray.png", topology.enhanced_gray)
        _write_image(sample_dir / "edge_map.png", topology.edge_map)
        if topology.color_blocks is not None:
            _write_image(sample_dir / "color_blocks.png", topology.color_blocks)
        _write_image(
            sample_dir / "line_segments.png",
            render_edge_lines(prepared.image_bgr, topology, merged=False),
        )
        _write_image(
            sample_dir / "topology.png",
            render_edge_lines(prepared.image_bgr, topology, merged=True),
        )
        annotated = sample.image_bgr.copy()
        x, y, width, height = prepared.bbox_px
        colour = (0, 180, 0) if topology.reason == "accepted" else (0, 190, 255)
        cv2.rectangle(annotated, (x, y), (x + width, y + height), colour, 3)
        cv2.putText(
            annotated,
            f"edges={len(topology.merged_lines)} junctions={len(topology.junctions)} quality={topology.quality:.2f}",
            (max(5, x), max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            colour,
            2,
            cv2.LINE_AA,
        )
        _write_image(sample_dir / "annotated.jpg", annotated)
        payload = {
            "source_path": str(sample.path),
            "true_label": sample.label_id,
            "true_name": sample.label_name,
            **topology.to_dict(),
        }
        (sample_dir / "topology.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.append(payload)

    values = np.asarray(durations, np.float64)
    summary = {
        "data_root": str(Path(data_root)),
        "images": len(rows),
        "accepted": accepted,
        "coverage_rate": accepted / max(len(rows), 1),
        "at_least_two_edges": sum(
            len(row["merged_lines"]) >= 2 for row in rows
        ),
        "at_least_two_edges_rate": sum(
            len(row["merged_lines"]) >= 2 for row in rows
        ) / max(len(rows), 1),
        "mean_merged_edges": float(np.mean([
            len(row["merged_lines"]) for row in rows
        ])) if rows else 0.0,
        "edge_extraction_p50_ms": float(np.percentile(values, 50)) if len(values) else 0.0,
        "edge_extraction_p95_ms": float(np.percentile(values, 95)) if len(values) else 0.0,
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
