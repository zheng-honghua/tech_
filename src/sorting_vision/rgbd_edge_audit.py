from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera import load_rgbd_frame
from .config import VisionConfig, load_config
from .face_topology3d import FaceTopology3D, extract_face_topology
from .geometry3d import segment_depth_objects
from .geometry_rgbd_model import (
    _fit_background_plane,
    detect_rgb_object_support,
    detect_tray_roi_mask,
)
from .rgbd import resize_rgbd_frame
from .rgbd_dataset import EMPTY_TRAY_LABEL, depth_preview, load_rgbd_dataset_entries


def _line_payload(line) -> dict[str, Any]:
    return {
        "points": [[round(line.x1, 3), round(line.y1, 3)],
                   [round(line.x2, 3), round(line.y2, 3)]],
        "length_px": round(line.length_px, 3),
        "rgb_support": round(line.rgb_support, 5),
        "depth_support": round(line.depth_support, 5),
        "confidence": round(line.confidence, 5),
        "source": line.source,
    }


def render_fused_edge_audit(
    color_bgr: np.ndarray, topology: FaceTopology3D
) -> dict[str, np.ndarray]:
    image = np.asarray(color_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("edge audit requires a BGR image")
    if topology.rgb_candidate_map is None or topology.depth_candidate_map is None:
        raise ValueError("topology does not contain fused-edge diagnostics")

    face_map = np.zeros_like(image)
    palette = (
        (225, 90, 70), (70, 210, 110), (80, 110, 235), (230, 190, 60),
        (190, 70, 220), (60, 210, 220), (180, 170, 70), (90, 170, 180),
    )
    for face in topology.faces:
        face_map[face.mask > 0] = palette[face.face_id % len(palette)]
    face_overlay = cv2.addWeighted(image, 0.55, face_map, 0.45, 0)

    def map_overlay(edge_map: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
        canvas = image.copy()
        canvas[edge_map > 0] = colour
        return canvas

    fused = image.copy()
    for line in topology.fused_edges:
        colour = (0, 220, 0) if line.source == "both" else (0, 170, 255)
        cv2.line(
            fused, tuple(np.rint(line.points()[0]).astype(int)),
            tuple(np.rint(line.points()[1]).astype(int)), colour, 2, cv2.LINE_AA,
        )
    rejected = image.copy()
    for line in topology.rejected_rgb_edges:
        cv2.line(
            rejected, tuple(np.rint(line.points()[0]).astype(int)),
            tuple(np.rint(line.points()[1]).astype(int)), (0, 0, 230), 1,
            cv2.LINE_AA,
        )
    return {
        "faces": face_overlay,
        "rgb-candidates": map_overlay(topology.rgb_candidate_map, (0, 220, 255)),
        "depth-candidates": map_overlay(topology.depth_candidate_map, (255, 150, 0)),
        "fused": fused,
        "rejected": rejected,
    }


def _label_tile(image: np.ndarray, label: str, size: tuple[int, int]) -> np.ndarray:
    tile = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (size[0], 24), (20, 20, 20), -1)
    cv2.putText(tile, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def audit_rgbd_edges(
    data_root: str | Path,
    output_dir: str | Path,
    config: VisionConfig | None = None,
    batch_ids: set[str] | None = None,
    limit_per_class: int = 10,
) -> dict[str, Any]:
    cfg = config or load_config()
    entries = load_rgbd_dataset_entries(data_root)
    selected = None if batch_ids is None else {str(value) for value in batch_ids}
    if selected is not None:
        entries = [entry for entry in entries if str(entry["batch_id"]) in selected]
    scale = float(cfg.rgbd.processing_scale)
    backgrounds: dict[str, tuple[Any, Any]] = {}
    for entry in entries:
        if entry["label_id"] != EMPTY_TRAY_LABEL:
            continue
        frame = resize_rgbd_frame(load_rgbd_frame(entry["absolute_sample_dir"]), scale)
        roi = detect_tray_roi_mask(frame.color_bgr)
        backgrounds[str(entry["batch_id"])] = (frame, _fit_background_plane(frame, cfg, roi))

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    contact_tiles: list[np.ndarray] = []
    for entry in entries:
        label = str(entry["label_id"])
        batch = str(entry["batch_id"])
        if label == EMPTY_TRAY_LABEL or batch not in backgrounds:
            continue
        key = (batch, label)
        if limit_per_class > 0 and counts[key] >= limit_per_class:
            continue
        try:
            frame = resize_rgbd_frame(load_rgbd_frame(entry["absolute_sample_dir"]), scale)
            background, plane = backgrounds[batch]
            if frame.intrinsics != background.intrinsics:
                raise ValueError("intrinsics_mismatch_with_empty_tray")
            roi = detect_tray_roi_mask(frame.color_bgr)
            support = detect_rgb_object_support(frame.color_bgr, roi)
            objects, _ = segment_depth_objects(
                frame.color_bgr, frame.depth_mm, frame.intrinsics, plane,
                cfg.rgbd, roi_mask=roi, support_mask=support,
                split_touching_objects=False,
            )
            if not objects:
                raise ValueError("object_not_found")
            item = max(objects, key=lambda candidate: candidate.area)
            x, y, width, height = item.bbox
            color = frame.color_bgr[y:y + height, x:x + width].copy()
            depth = frame.depth_mm[y:y + height, x:x + width].copy()
            mask = item.mask[y:y + height, x:x + width]
            color[mask == 0] = 245
            topology = extract_face_topology(
                depth, mask, frame.intrinsics, (x, y), color_crop_bgr=color,
                extract_fused_edges=True,
            )
            rendered = render_fused_edge_audit(color, topology)
            sample_name = Path(str(entry["sample_dir"])).name
            sample_dir = target / batch / label / sample_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            depth_image = depth_preview(depth)
            depth_image[mask == 0] = 245
            images = {"rgb": color, "depth": depth_image, **rendered}
            for name, image in images.items():
                if not cv2.imwrite(str(sample_dir / f"{name}.png"), image):
                    raise OSError(f"cannot write {name} audit image")
            payload = {
                "batch_id": batch,
                "label_id": label,
                "frame_id": frame.frame_id,
                "sample_dir": str(entry["sample_dir"]),
                "face_count": len(topology.faces),
                "adjacency": [list(pair) for pair in topology.adjacency],
                "fused_edges": [_line_payload(line) for line in topology.fused_edges],
                "rejected_rgb_edges": [
                    _line_payload(line) for line in topology.rejected_rgb_edges
                ],
            }
            (sample_dir / "topology.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            records.append(payload)
            counts[key] += 1
            contact_tiles.append(_label_tile(
                rendered["fused"],
                f"{batch} {label} {frame.frame_id} E={len(topology.fused_edges)}",
                (320, 240),
            ))
        except (ValueError, OSError, FileNotFoundError) as error:
            rejected.append({"sample_dir": str(entry["sample_dir"]), "reason": str(error)})

    if contact_tiles:
        columns = 4
        blank = np.full_like(contact_tiles[0], 235)
        while len(contact_tiles) % columns:
            contact_tiles.append(blank.copy())
        rows = [np.hstack(contact_tiles[index:index + columns])
                for index in range(0, len(contact_tiles), columns)]
        cv2.imwrite(str(target / "fused-edge-contact-sheet.jpg"), np.vstack(rows))
    report = {
        "data_root": str(Path(data_root)),
        "selected_batches": sorted(selected) if selected is not None else "all",
        "reviewed_samples": len(records),
        "limit_per_class": limit_per_class,
        "counts": {f"{batch}/{label}": count for (batch, label), count in sorted(counts.items())},
        "fused_edges": sum(len(record["fused_edges"]) for record in records),
        "rejected_rgb_edges": sum(len(record["rejected_rgb_edges"]) for record in records),
        "frame_ids": [record["frame_id"] for record in records],
        "rejected": rejected,
    }
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
