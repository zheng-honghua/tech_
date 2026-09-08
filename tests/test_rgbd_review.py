import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sorting_vision.rgbd_review import build_rgbd_review


def _write_frame(root: Path, name: str, shape_id: str, health_ok: bool = True) -> None:
    target = root / name
    target.mkdir(parents=True)
    image = np.full((90, 160, 3), 180, dtype=np.uint8)
    assert cv2.imwrite(str(target / "annotated-rgbd.png"), image)
    assert cv2.imwrite(str(target / "depth-preview.png"), image)
    (target / "results-v2.json").write_text(json.dumps([{
        "shape_id": shape_id, "status": "UNCERTAIN", "selected": False,
    }]), encoding="utf-8")
    (target / "health.json").write_text(json.dumps({
        "ok": health_ok, "reason": "ok" if health_ok else "depth_invalid",
    }), encoding="utf-8")


def test_build_rgbd_review_generates_sheets_and_manifest(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_frame(root, "batch-b/frame-02", "unknown", health_ok=False)
    _write_frame(root, "batch-a/frame-01", "hexagonal_prism")

    report = build_rgbd_review(root, tmp_path / "review", columns=2)

    assert report["frame_count"] == 2
    assert report["candidate_count"] == 2
    assert report["unknown_shape_count"] == 1
    assert report["health_failure_count"] == 1
    assert report["batch_modal_candidate_count"] == {"batch-a": 1, "batch-b": 1}
    assert report["review_status"] == "PENDING_HUMAN_VISUAL_REVIEW"
    assert [item["frame"] for item in report["frames"]] == [
        "batch-a/frame-01", "batch-b/frame-02",
    ]
    assert Path(report["contact_sheets"]["annotated"]).is_file()
    assert Path(report["contact_sheets"]["depth"]).is_file()
    persisted = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert persisted["visual_review_required"] is True


def test_build_rgbd_review_rejects_empty_result_tree(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="annotated-rgbd"):
        build_rgbd_review(root, tmp_path / "review")
