from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read_json(path: Path, expected_type: type) -> tuple[Any, str | None]:
    if not path.is_file():
        return expected_type(), f"missing: {path.name}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return expected_type(), f"invalid {path.name}: {exc}"
    if not isinstance(value, expected_type):
        return expected_type(), f"invalid type: {path.name}"
    return value, None


def _review_tile(
    image: np.ndarray | None,
    title: str,
    summary: str,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    header_height = 48
    if image is None:
        cv2.putText(
            canvas, "IMAGE MISSING/CORRUPT", (12, header_height + 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 255), 2, cv2.LINE_AA,
        )
    else:
        available_height = height - header_height
        scale = min(width / image.shape[1], available_height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)),
             max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        x = (width - resized.shape[1]) // 2
        y = header_height + (available_height - resized.shape[0]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(
        canvas, title[:65], (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (245, 245, 245), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, summary[:78], (8, 39), cv2.FONT_HERSHEY_SIMPLEX,
        0.38, (100, 220, 255), 1, cv2.LINE_AA,
    )
    return canvas


def _write_contact_sheet(
    tiles: list[np.ndarray], output: Path, columns: int, tile_width: int,
    tile_height: int,
) -> None:
    rows = (len(tiles) + columns - 1) // columns
    blank = np.full((tile_height, tile_width, 3), 24, dtype=np.uint8)
    padded = tiles + [blank] * (rows * columns - len(tiles))
    sheet = np.vstack([
        np.hstack(padded[row * columns:(row + 1) * columns])
        for row in range(rows)
    ])
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"cannot write review sheet: {output}")


def build_rgbd_review(
    results_root: str | Path,
    output_dir: str | Path,
    columns: int = 4,
    tile_width: int = 480,
    tile_height: int = 320,
) -> dict[str, Any]:
    """Build visual review sheets and a machine-readable anomaly inventory.

    This prepares evidence for a real visual inspection. It deliberately leaves
    review_status pending: generating a montage is not itself visual review.
    """
    root = Path(results_root)
    if not root.is_dir():
        raise FileNotFoundError(f"results root does not exist: {root}")
    if columns < 1 or tile_width < 160 or tile_height < 120:
        raise ValueError("invalid contact-sheet dimensions")
    annotated_paths = sorted(
        root.rglob("annotated-rgbd.png"),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    if not annotated_paths:
        raise FileNotFoundError(f"no annotated-rgbd.png under: {root}")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    annotated_tiles: list[np.ndarray] = []
    depth_tiles: list[np.ndarray] = []
    frames: list[dict[str, Any]] = []
    all_issues: list[str] = []
    total_candidates = 0
    total_unknown = 0
    total_selected = 0
    health_failures = 0

    for annotated_path in annotated_paths:
        frame_dir = annotated_path.parent
        relative_dir = frame_dir.relative_to(root).as_posix()
        results, result_issue = _read_json(frame_dir / "results-v2.json", list)
        health, health_issue = _read_json(frame_dir / "health.json", dict)
        issues = [issue for issue in (result_issue, health_issue) if issue]
        statuses = Counter(str(item.get("status", "MISSING")) for item in results)
        unknown_count = sum(
            str(item.get("shape_id", "unknown")) == "unknown" for item in results
        )
        selected_count = sum(bool(item.get("selected", False)) for item in results)
        health_ok = bool(health.get("ok", False)) if health else False
        if not health_ok:
            health_failures += 1
            issues.append(f"health: {health.get('reason', 'missing')}" if health else "health missing")
        total_candidates += len(results)
        total_unknown += unknown_count
        total_selected += selected_count

        annotated = cv2.imread(str(annotated_path), cv2.IMREAD_COLOR)
        if annotated is None:
            issues.append("annotated image unreadable")
        depth_path = frame_dir / "depth-preview.png"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_COLOR) if depth_path.is_file() else None
        if depth is None:
            issues.append("depth preview missing/unreadable")
        summary = (
            f"objects={len(results)} unknown={unknown_count} selected={selected_count} "
            f"health={'OK' if health_ok else 'FAIL'}"
        )
        annotated_tiles.append(
            _review_tile(annotated, relative_dir, summary, tile_width, tile_height)
        )
        depth_tiles.append(
            _review_tile(depth, relative_dir, summary, tile_width, tile_height)
        )
        frames.append({
            "frame": relative_dir,
            "candidate_count": len(results),
            "unknown_shape_count": unknown_count,
            "selected_count": selected_count,
            "statuses": dict(sorted(statuses.items())),
            "health_ok": health_ok,
            "health_reason": health.get("reason") if health else None,
            "issues": issues,
        })
        all_issues.extend(f"{relative_dir}: {issue}" for issue in issues)

    by_batch: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        batch = str(frame["frame"]).split("/", 1)[0]
        by_batch.setdefault(batch, []).append(frame)
    batch_modes: dict[str, int] = {}
    for batch, batch_frames in by_batch.items():
        counts = Counter(int(frame["candidate_count"]) for frame in batch_frames)
        modal_count = counts.most_common(1)[0][0]
        batch_modes[batch] = modal_count
        for frame in batch_frames:
            if int(frame["candidate_count"]) == modal_count:
                continue
            issue = f"candidate count differs from batch mode ({modal_count})"
            frame["issues"].append(issue)
            all_issues.append(f"{frame['frame']}: {issue}")

    annotated_sheet = target / "annotated-contact-sheet.jpg"
    depth_sheet = target / "depth-contact-sheet.jpg"
    _write_contact_sheet(annotated_tiles, annotated_sheet, columns, tile_width, tile_height)
    _write_contact_sheet(depth_tiles, depth_sheet, columns, tile_width, tile_height)
    report: dict[str, Any] = {
        "schema_version": 1,
        "results_root": str(root.resolve()),
        "frame_count": len(frames),
        "candidate_count": total_candidates,
        "unknown_shape_count": total_unknown,
        "selected_count": total_selected,
        "health_failure_count": health_failures,
        "batch_modal_candidate_count": batch_modes,
        "issues": all_issues,
        "contact_sheets": {
            "annotated": str(annotated_sheet.resolve()),
            "depth": str(depth_sheet.resolve()),
        },
        "visual_review_required": True,
        "review_status": "PENDING_HUMAN_VISUAL_REVIEW",
        "review_checklist": [
            "tray ROI covers the white tray interior and rim without outside objects",
            "every physical object has one segmentation region and no tray fragments are objects",
            "RGB boxes, shape labels, and depth regions refer to the same object",
            "unknown, merged, split, clipped, and low-depth cases are inspected individually",
            "findings and reviewed frame IDs are recorded in a Markdown report",
        ],
        "frames": frames,
    }
    report_path = target / "review-manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    return report
