from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pytest

from sorting_vision import cli
from sorting_vision.capture_assistant import CaptureQualityTracker
from sorting_vision.multi_capture import (
    MultiCaptureState,
    load_scene_counts,
    parse_scene_composition,
    render_multi_capture,
    resolve_scene_index,
    save_multi_object_sample,
    validate_scene_composition,
)
from sorting_vision.rgbd import CameraIntrinsics, RGBDFrame


def _frame(value: int = 100) -> RGBDFrame:
    intrinsics = CameraIntrinsics(32, 24, 30, 30, 16, 12, 1.0)
    return RGBDFrame(
        np.full((24, 32, 3), value, np.uint8),
        np.full((24, 32), 420, np.uint16),
        intrinsics,
        1_000_000,
        "multi-test",
        1_000_000,
        1_500_000,
    )


def test_parse_scene_composition_accepts_chinese_and_merges_duplicates():
    result = parse_scene_composition("三棱柱:2，square_pyramid=1;三棱柱")
    assert result == (
        {"label_id": "triangular_prism", "label_name": "三棱柱", "count": 3},
        {"label_id": "square_pyramid", "label_name": "四棱锥", "count": 1},
    )
    with pytest.raises(ValueError, match="empty_tray"):
        parse_scene_composition("空托盘:1")
    with pytest.raises(ValueError, match="greater than zero"):
        parse_scene_composition("圆锥:0")


def test_scene_index_resumes_latest_incomplete_scene():
    assert resolve_scene_index({}, 10, 0) == 1
    assert resolve_scene_index({1: 4}, 10, 0) == 1
    assert resolve_scene_index({1: 10}, 10, 0) == 2
    assert resolve_scene_index({1: 10, 2: 3}, 10, 7) == 7


def test_multi_capture_defaults_match_verified_d415_stream_profile():
    args = cli.build_parser().parse_args([
        "rgbd-multi-capture", "--batch-id", "multi-01",
        "--composition", "三棱柱:1",
    ])
    assert (args.color_width, args.color_height) == (1920, 1080)
    assert (args.depth_width, args.depth_height, args.fps) == (640, 480, 30)


def test_multi_capture_saves_self_contained_scene_bundle(tmp_path):
    state = MultiCaptureState(
        "multi-01", parse_scene_composition("三棱柱:2,圆锥:1"),
        captures_per_scene=2,
    )
    target = save_multi_object_sample(_frame(), tmp_path, state, {"exposure": 50})
    assert (target / "color.png").is_file()
    assert (target / "depth.npy").is_file()
    assert (target / "depth-preview.png").is_file()
    scene = json.loads((target / "scene.json").read_text(encoding="utf-8"))
    assert scene["dataset_kind"] == "multi_object_scene"
    assert scene["total_objects"] == 3
    assert scene["object_annotations"] == "composition_only"
    assert scene["independent_layout"] is True
    assert state.current_count == 1
    assert load_scene_counts(tmp_path, "multi-01") == {1: 1}
    assert (tmp_path / "scenes.jsonl").is_file()
    assert not (tmp_path / "manifest.jsonl").exists()
    validate_scene_composition(tmp_path, "multi-01", 1, state.composition)
    with pytest.raises(ValueError, match="composition differs"):
        validate_scene_composition(
            tmp_path, "multi-01", 1, parse_scene_composition("四棱锥:3")
        )


def test_multi_capture_overlay_shows_two_views():
    frame = _frame()
    state = MultiCaptureState("multi-01", parse_scene_composition("三棱柱:2"))
    tracker = CaptureQualityTracker(required_stable_frames=1)
    tracker.update(frame)
    tracker.update(frame)
    canvas = render_multi_capture(frame, state, tracker, "ready")
    assert canvas.shape == (24 + 190, 64, 3)


def test_headless_multi_capture_batches_stable_frames(monkeypatch, tmp_path):
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
    monkeypatch.setattr(cli, "_make_camera_source", lambda args, config: source)
    args = Namespace(
        config=None,
        dataset_root=str(tmp_path),
        batch_id="multi-01",
        composition="三棱柱:2,圆锥:1",
        captures_per_scene=2,
        scene_index=0,
        auto_start=False,
        headless=True,
        stable_frames=1,
        motion_threshold=2.5,
        min_valid_depth_ratio=0.85,
        max_sync_delta_ms=50.0,
        discard_frames=0,
        interval_ms=0,
        max_captures=0,
    )
    assert cli._run_rgbd_multi_capture(args) == 0
    assert load_scene_counts(tmp_path, "multi-01") == {1: 2}
    assert source.closed
