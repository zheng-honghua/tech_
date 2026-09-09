import json
import threading

import cv2
import numpy as np

from sorting_vision.camera import DualCameraSource, RGBFrame, SynchronizedFramePair
from sorting_vision.config import DualViewConfig, load_config
from sorting_vision.dual_view import (
    DualCalibrationMetrics,
    DualViewCalibration,
    DualViewFusion,
    FusionState,
    ProjectedROI,
    _side_mask,
    load_synchronized_pair,
    projected_roi,
    roi_overlap_ratio,
    save_synchronized_pair,
)
from sorting_vision.dual_validation import evaluate_promotion, fit_probability_temperatures
from sorting_vision.rgbd import CameraIntrinsics, Plane, RGBDFrame
from sorting_vision.types import Confidence3D, DetectionStatus, VisionResult3D


class SideModel:
    def __init__(self, scores, label=None, reason="accepted"):
        self.last_class_scores = scores
        self.label = label or max(scores, key=scores.get)
        self.reason = reason

    def predict(self, crop, mask):
        return self.label, self.last_class_scores[self.label], {
            "nearest_label": self.label,
            "reason": self.reason,
        }


def calibration(valid=True):
    intrinsics = CameraIntrinsics(100, 100, 100, 100, 50, 50, 1.0)
    metrics = DualCalibrationMetrics(
        0.5 if valid else 1.2, 0.5, 1.0, 0.005
    )
    return DualViewCalibration(
        intrinsics,
        np.zeros(5),
        intrinsics,
        np.zeros(5),
        np.eye(4),
        metrics,
        "temporary",
        "test-v1",
        {"type": "ChArUco"},
        (100, 100),
        None,
        np.eye(4),
    )


def pair(side=True, delta_ms=10.0):
    intrinsics = CameraIntrinsics(100, 100, 100, 100, 50, 50, 1.0)
    primary = RGBDFrame(
        np.zeros((100, 100, 3), np.uint8),
        np.full((100, 100), 500, np.uint16),
        intrinsics,
        10,
        "primary-1",
    )
    side_frame = None
    if side:
        image = np.zeros((100, 100, 3), np.uint8)
        cv2.rectangle(image, (47, 47), (53, 53), (255, 255, 255), -1)
        side_frame = RGBFrame(image, 20, "side-1", 20)
    return SynchronizedFramePair(
        primary,
        side_frame,
        1_000_000_000,
        None if not side else 1_000_000_000 + int(delta_ms * 1_000_000),
        50.0,
    )


def result(status=DetectionStatus.PICKABLE, probability=0.9, label="cube"):
    return VisionResult3D(
        frame_id="primary-1",
        object_id="object-1",
        color_id="red",
        color_name="红色",
        shape_id=label,
        shape_name=label,
        class_key=f"red:{label}",
        pose_3d=None,
        grasp=None,
        confidence=Confidence3D(0.9, 0.9, probability, 0.9, 0.9),
        status=status,
        bbox_px=(40, 40, 20, 20),
        diagnostics={
            "top_shape_scores": {label: probability, "hexagonal_prism": 1 - probability},
            "top_shape_rejection_reason": "accepted",
            "shape_names": {"cube": "立方体", "hexagonal_prism": "六棱柱"},
            "dual_shape_upgrade_allowed": False,
        },
    )


def points(x=0.0):
    return np.asarray(
        [[x - 10, -10, 480], [x + 10, -10, 480], [x + 10, 10, 480], [x - 10, 10, 480]],
        np.float64,
    )


def test_calibration_round_trip_hash_and_validity(tmp_path):
    original = calibration()
    path = tmp_path / "dual.json"
    original.save(path)
    loaded = DualViewCalibration.load(path)
    assert loaded.valid is True
    assert loaded.calibration_hash == original.calibration_hash
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["platform_id"] = "competition"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        DualViewCalibration.load(path)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("tampered calibration was accepted")


def test_platform_profiles_inherit_default_and_remain_separate():
    temporary = load_config("config/dual/temporary.yaml")
    competition = load_config("config/dual/competition.yaml")
    assert temporary.camera.realsense_color_width == 1920
    assert temporary.tray.rectified_width_px == 640
    assert temporary.dual_view.platform_id == "temporary"
    assert competition.dual_view.platform_id == "competition"
    assert "competition" in competition.dual_view.calibration_path


def test_dual_camera_source_pairs_threaded_frames_on_monotonic_clock():
    expected = pair()

    class OnceSource:
        def __init__(self, frame):
            self.frame = frame
            self.sent = False
            self.closed = threading.Event()

        def read(self):
            if not self.sent:
                self.sent = True
                return self.frame
            self.closed.wait(1.0)
            raise EOFError("closed")

        def close(self):
            self.closed.set()

    primary = OnceSource(expected.primary)
    side = OnceSource(expected.side)
    source = DualCameraSource(primary, side, pair_timeout_ms=500)
    actual = source.read()
    source.close()
    assert actual.primary.frame_id == "primary-1"
    assert actual.side is not None and actual.side.frame_id == "side-1"
    assert actual.synchronized is True
    assert actual.pair_delta_ms is not None and actual.pair_delta_ms <= 50.0


def test_synchronized_pair_dataset_round_trip(tmp_path):
    original = pair(delta_ms=12.5)
    target = tmp_path / "sample"
    save_synchronized_pair(
        original, target, "temporary", calibration().calibration_hash, "cube"
    )
    loaded = load_synchronized_pair(target)
    assert loaded.primary.frame_id == original.primary.frame_id
    assert loaded.side is not None and loaded.side.frame_id == original.side.frame_id
    assert loaded.pair_delta_ms == 12.5
    np.testing.assert_array_equal(loaded.primary.depth, original.primary.depth)


def test_3d_projection_builds_clipped_side_roi_and_overlap_threshold():
    plane = Plane(np.array([0, 0, -1.0]), 500.0)
    first = projected_roi(calibration(), points(), plane, (100, 100, 3))
    second = projected_roi(calibration(), points(3), plane, (100, 100, 3))
    far = ProjectedROI((80, 80, 10, 10), np.zeros((3, 2), np.int32), 1.0)
    assert first is not None and second is not None
    assert first.visible_ratio == 1.0
    assert roi_overlap_ratio(first, second) > 0.30
    assert roi_overlap_ratio(first, far) == 0.0


def test_side_mask_rejects_blur_and_accepts_textured_foreground():
    cfg = DualViewConfig(side_min_blur_variance=20.0, side_min_area_px=20)
    background = np.zeros((100, 100, 3), np.uint8)
    roi = ProjectedROI(
        (20, 20, 60, 60),
        np.asarray([[20, 20], [79, 20], [79, 79], [20, 79]], np.int32),
        1.0,
    )
    blurred = np.full_like(background, 50)
    _, _, _, blurred_quality = _side_mask(blurred, background, roi, cfg)
    textured = background.copy()
    cv2.rectangle(textured, (30, 30), (70, 70), (255, 255, 255), -1)
    cv2.line(textured, (30, 30), (70, 70), (0, 0, 0), 2)
    _, pixels, _, textured_quality = _side_mask(textured, background, roi, cfg)
    assert blurred_quality == 0.0
    assert pixels > 20
    assert textured_quality >= 0.6


def test_missing_side_requires_high_confidence_top_only_fallback():
    cfg = DualViewConfig(enabled=True, top_only_probability=0.85)
    fusion = DualViewFusion(
        calibration(), np.zeros((100, 100, 3), np.uint8), SideModel({"cube": 1.0}), cfg,
        Plane(np.array([0, 0, -1.0]), 500.0),
    )
    accepted = result(probability=0.9)
    rejected = result(probability=0.8)
    fusion.apply([accepted], {accepted.object_id: points()}, pair(side=False))
    fusion.apply([rejected], {rejected.object_id: points()}, pair(side=False))
    assert accepted.status == DetectionStatus.PICKABLE
    assert rejected.status == DetectionStatus.UNCERTAIN
    assert accepted.diagnostics["dual_view"]["fusion_state"] == FusionState.SIDE_MISSING.value


def test_unsynchronised_pair_never_uses_side(monkeypatch):
    cfg = DualViewConfig(enabled=True)
    model = SideModel({"hexagonal_prism": 0.95, "cube": 0.05})
    fusion = DualViewFusion(
        calibration(), np.zeros((100, 100, 3), np.uint8), model, cfg,
        Plane(np.array([0, 0, -1.0]), 500.0),
    )
    item = result(probability=0.9)
    fusion.apply([item], {item.object_id: points()}, pair(delta_ms=80))
    assert item.shape_id == "cube"
    assert item.diagnostics["dual_view"]["fusion_state"] == FusionState.UNSYNCED.value


def test_high_confidence_conflict_rejects(monkeypatch):
    monkeypatch.setattr(
        "sorting_vision.dual_view._side_mask",
        lambda *args: (
            np.full((args[2].bbox[3], args[2].bbox[2]), 255, np.uint8),
            args[2].area,
            100.0,
            1.0,
        ),
    )
    cfg = DualViewConfig(enabled=True)
    model = SideModel({"hexagonal_prism": 0.95, "cube": 0.05})
    fusion = DualViewFusion(
        calibration(), np.zeros((100, 100, 3), np.uint8), model, cfg,
        Plane(np.array([0, 0, -1.0]), 500.0),
    )
    item = result(probability=0.9)
    fusion.apply([item], {item.object_id: points()}, pair())
    assert item.status == DetectionStatus.UNCERTAIN
    assert item.selected is False
    assert item.diagnostics["dual_view"]["fusion_state"] == FusionState.CONFLICT.value


def test_only_margin_rejection_can_be_upgraded_by_side_top_two(monkeypatch):
    monkeypatch.setattr(
        "sorting_vision.dual_view._side_mask",
        lambda *args: (
            np.full((args[2].bbox[3], args[2].bbox[2]), 255, np.uint8),
            args[2].area,
            100.0,
            1.0,
        ),
    )
    fusion = DualViewFusion(
        calibration(),
        np.zeros((100, 100, 3), np.uint8),
        SideModel({"cube": 0.99, "hexagonal_prism": 0.01}),
        DualViewConfig(enabled=True),
        Plane(np.array([0, 0, -1.0]), 500.0),
    )
    margin_item = result(status=DetectionStatus.UNCERTAIN, probability=0.55)
    margin_item.diagnostics.update(
        {
            "top_shape_scores": {"cube": 0.55, "hexagonal_prism": 0.45},
            "top_shape_rejection_reason": "margin_rejected",
            "dual_shape_upgrade_allowed": True,
        }
    )
    fusion.apply([margin_item], {margin_item.object_id: points()}, pair())
    assert margin_item.status == DetectionStatus.PICKABLE

    distance_item = result(status=DetectionStatus.UNCERTAIN, probability=0.55)
    distance_item.diagnostics.update(
        {
            "top_shape_scores": {"cube": 0.55, "hexagonal_prism": 0.45},
            "top_shape_rejection_reason": "distance_rejected",
            "dual_shape_upgrade_allowed": True,
        }
    )
    fusion.apply([distance_item], {distance_item.object_id: points()}, pair())
    assert distance_item.status == DetectionStatus.UNCERTAIN


def test_side_evidence_cannot_upgrade_depth_failure(monkeypatch):
    monkeypatch.setattr(
        "sorting_vision.dual_view._side_mask",
        lambda *args: (
            np.full((args[2].bbox[3], args[2].bbox[2]), 255, np.uint8),
            args[2].area,
            100.0,
            1.0,
        ),
    )
    cfg = DualViewConfig(enabled=True)
    model = SideModel({"cube": 0.99, "hexagonal_prism": 0.01})
    fusion = DualViewFusion(
        calibration(), np.zeros((100, 100, 3), np.uint8), model, cfg,
        Plane(np.array([0, 0, -1.0]), 500.0),
    )
    item = result(status=DetectionStatus.DEPTH_INVALID, probability=0.9)
    fusion.apply([item], {item.object_id: points()}, pair())
    assert item.status == DetectionStatus.DEPTH_INVALID
    assert item.diagnostics["dual_view"]["fusion_state"] == FusionState.FUSED.value


def test_side_conflict_cannot_replace_depth_failure_with_uncertain(monkeypatch):
    monkeypatch.setattr(
        "sorting_vision.dual_view._side_mask",
        lambda *args: (
            np.full((args[2].bbox[3], args[2].bbox[2]), 255, np.uint8),
            args[2].area,
            100.0,
            1.0,
        ),
    )
    fusion = DualViewFusion(
        calibration(),
        np.zeros((100, 100, 3), np.uint8),
        SideModel({"hexagonal_prism": 0.99, "cube": 0.01}),
        DualViewConfig(enabled=True),
        Plane(np.array([0, 0, -1.0]), 500.0),
    )
    item = result(status=DetectionStatus.DEPTH_INVALID, probability=0.9)
    fusion.apply([item], {item.object_id: points()}, pair())
    assert item.status == DetectionStatus.DEPTH_INVALID
    assert item.diagnostics["dual_view"]["fusion_state"] == FusionState.CONFLICT.value


def test_roi_overlap_marks_both_side_views_occluded():
    cfg = DualViewConfig(enabled=True, top_only_probability=0.85)
    fusion = DualViewFusion(
        calibration(), np.zeros((100, 100, 3), np.uint8), SideModel({"cube": 1.0}), cfg,
        Plane(np.array([0, 0, -1.0]), 500.0),
    )
    first = result()
    second = result()
    second.object_id = "object-2"
    fusion.apply(
        [first, second],
        {first.object_id: points(), second.object_id: points(2)},
        pair(),
    )
    assert first.diagnostics["dual_view"]["fusion_state"] == FusionState.SIDE_OCCLUDED.value
    assert second.diagnostics["dual_view"]["fusion_state"] == FusionState.SIDE_OCCLUDED.value


def test_dual_diagnostics_preserve_schema_v2():
    cfg = DualViewConfig(enabled=True)
    fusion = DualViewFusion(
        calibration(valid=False), None, None, cfg,
        Plane(np.array([0, 0, -1.0]), 500.0),
    )
    item = result()
    fusion.apply([item], {item.object_id: points()}, pair())
    payload = item.to_dict()
    assert payload["schema_version"] == 2
    assert "dual_view" in payload["diagnostics"]
    assert payload["diagnostics"]["dual_view"]["fusion_state"] == "CALIBRATION_INVALID"


def test_promotion_gate_requires_gain_and_no_safety_regression():
    records = []
    for index in range(40):
        label = "cube" if index < 20 else "prism"
        mono_correct = index not in {0, 1, 20, 21}
        fused_correct = index not in {0, 20}
        records.append(
            {
                "split": "final_holdout",
                "human_reviewed": True,
                "platform_id": "competition",
                "true_label": label,
                "mono_label": label if mono_correct else "wrong",
                "fused_label": label if fused_correct else None,
                "mono_accepted": mono_correct,
                "fused_accepted": fused_correct,
                "safety_state_upgraded": False,
                "projection_error_px": 2.0,
                "pair_delta_ms": 20.0,
                "latency_ms": 300.0,
            }
        )
    report = evaluate_promotion(records)
    assert report["macro_recall_gain"] >= 0.03
    assert report["promote_dual_view"] is True


def test_temperature_fit_uses_only_reviewed_calibration_split():
    records = [
        {
            "split": "probability_calibration",
            "human_reviewed": True,
            "true_label": label,
            "top_class_scores": scores,
            "side_class_scores": scores,
        }
        for label, scores in (
            ("cube", {"cube": 0.7, "prism": 0.3}),
            ("prism", {"cube": 0.2, "prism": 0.8}),
            ("cube", {"cube": 0.6, "prism": 0.4}),
            ("prism", {"cube": 0.35, "prism": 0.65}),
        )
    ]
    report = fit_probability_temperatures(records)
    assert report["top_sample_count"] == 4
    assert 0.2 <= report["top_temperature"] <= 5.0
    assert 0.2 <= report["side_temperature"] <= 5.0
