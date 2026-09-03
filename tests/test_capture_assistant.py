import json
from argparse import Namespace

import numpy as np
import pytest

from sorting_vision.capture_assistant import (
    CAPTURE_LABELS,
    CaptureAssistantState,
    CaptureQualityTracker,
    capture_label_index,
    load_batch_counts,
    render_capture_assistant,
)
from sorting_vision.rgbd import CameraIntrinsics, RGBDFrame
from sorting_vision import cli


def _frame(value=100, valid=True, sync_delta_ms=1.0):
    intrinsics = CameraIntrinsics(32, 24, 30, 30, 16, 12, 1.0)
    depth = np.full((24, 32), 420 if valid else 0, np.uint16)
    return RGBDFrame(
        np.full((24, 32, 3), value, np.uint8),
        depth,
        intrinsics,
        1_000_000,
        "d415-test",
        1_000_000,
        1_000_000 + int(sync_delta_ms * 1_000_000),
    )


def test_capture_labels_have_stable_digit_mapping():
    assert len(CAPTURE_LABELS) == 10
    assert CAPTURE_LABELS[0] == ("empty_tray", "空托盘")
    assert capture_label_index("三棱柱") == 1
    assert capture_label_index("triangular_prism") == 1
    assert capture_label_index("9") == 9
    with pytest.raises(ValueError, match="unsupported capture label"):
        capture_label_index("not-a-shape")


def test_capture_state_resumes_counts_and_selects_incomplete_class(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"batch_id": "batch-01", "label_id": "empty_tray"},
        {"batch_id": "batch-01", "label_id": "empty_tray"},
        {"batch_id": "batch-02", "label_id": "triangular_prism"},
    ]
    manifest.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    state = CaptureAssistantState(
        target_per_label=2,
        counts=load_batch_counts(tmp_path, "batch-01"),
    )
    assert state.count("empty_tray") == 2
    state.select_next_incomplete()
    assert state.current[0] == "triangular_prism"
    state.record_saved()
    assert state.count() == 1


def test_quality_tracker_requires_stability_depth_and_sync():
    tracker = CaptureQualityTracker(required_stable_frames=2)
    tracker.update(_frame())
    assert not tracker.ready
    tracker.update(_frame())
    assert not tracker.ready
    tracker.update(_frame())
    assert tracker.ready

    tracker.update(_frame(valid=False))
    assert "depth_valid_ratio_low" in tracker.rejection_reasons()

    unsynchronised = CaptureQualityTracker(required_stable_frames=1)
    unsynchronised.update(_frame(sync_delta_ms=80))
    unsynchronised.update(_frame(sync_delta_ms=80))
    assert "rgb_depth_out_of_sync" in unsynchronised.rejection_reasons()


def test_capture_overlay_contains_rgb_and_depth_views():
    frame = _frame()
    state = CaptureAssistantState()
    tracker = CaptureQualityTracker(required_stable_frames=1)
    tracker.update(frame)
    tracker.update(frame)
    canvas = render_capture_assistant(frame, state, tracker, "ready")
    assert canvas.shape == (24 + 390, 64, 3)
    assert canvas.dtype == np.uint8


def test_capture_assistant_cli_saves_selected_label(monkeypatch, tmp_path):
    class Source:
        def __init__(self):
            self.closed = False

        def read(self):
            return _frame()

        def capture_metadata(self):
            return {"camera_model": "fake D415"}

        def close(self):
            self.closed = True

    source = Source()
    keys = iter((-1, 32, ord("q")))
    monkeypatch.setattr(cli, "_make_camera_source", lambda args, config: source)
    monkeypatch.setattr(cli.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cli.cv2, "waitKey", lambda delay: next(keys))
    monkeypatch.setattr(cli.cv2, "destroyAllWindows", lambda: None)
    args = Namespace(
        config=None,
        dataset_root=str(tmp_path),
        batch_id="pilot-01",
        target_per_label=2,
        start_label="empty_tray",
        stable_frames=1,
        motion_threshold=2.5,
        min_valid_depth_ratio=0.85,
        max_sync_delta_ms=50.0,
        discard_frames=0,
        auto_advance=False,
    )
    assert cli._run_rgbd_capture_assistant(args) == 0
    assert load_batch_counts(tmp_path, "pilot-01") == {"empty_tray": 1}
    assert source.closed
