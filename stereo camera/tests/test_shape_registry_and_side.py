import json

import cv2
import numpy as np
import pytest

from sorting_vision.fusion_policy import (
    FusionPolicy,
    fuse_top2_scores,
    resolve_fusion_backend,
)
from sorting_vision.dual_validation import apply_dual_review_records, fit_fusion_policy
from sorting_vision.shape_registry import ShapeClass, ShapeRegistry, load_shape_registry
from sorting_vision.side_geometry import (
    SideGeometryModel,
    SideTrainingSample,
    audit_side_dataset,
    calibrate_side_acceptance,
    extract_side_features,
    side_foreground_mask,
    _deduplicate_side_segments,
)
from sorting_vision.side_cnn import calibrate_side_cnn_acceptance, compare_side_cnn_backends
from sorting_vision.synthetic_side import render_synthetic_side


def test_registry_has_locked_11_classes_and_normalizes_quadrangular_aliases():
    registry = load_shape_registry("config/shapes/competition-11.yaml")
    assert len(registry.class_ids) == 11
    assert registry.resolve("cube") == "quadrangular_prism"
    assert registry.resolve("长方体") == "quadrangular_prism"
    assert registry.resolve("square_prism") == "quadrangular_prism"


def test_registry_rejects_alias_collisions():
    with pytest.raises(ValueError, match="belongs to both"):
        ShapeRegistry(1, (ShapeClass("a", "甲", "x", ("same",)), ShapeClass("b", "乙", "x", ("same",))))


def test_side_features_are_finite_and_scale_normalized():
    image, mask = render_synthetic_side("square_pyramid", seed=3)
    first = extract_side_features(image, mask)
    larger_image = cv2.resize(image, (384, 384))
    larger_mask = cv2.resize(mask, (384, 384), interpolation=cv2.INTER_NEAREST)
    second = extract_side_features(larger_image, larger_mask)
    assert np.all(np.isfinite(first.vector))
    assert len(first.vector) == len(first.group_ids)
    assert np.mean(np.abs(first.vector[:31] - second.vector[:31])) < 0.12


def test_side_features_do_not_call_hough_lines(monkeypatch):
    monkeypatch.setattr(
        cv2, "HoughLinesP", lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("side features must not use Hough lines")
        )
    )
    image, mask = render_synthetic_side("square_pyramid", seed=33)
    result = extract_side_features(image, mask)
    assert np.all(np.isfinite(result.vector))
    assert result.diagnostics["lsd_segment_count"] >= 3


def test_side_collinear_fragment_removed_but_parallel_ridge_retained():
    segments = np.asarray([[10, 10, 110, 10], [20, 11, 45, 11], [20, 20, 100, 20]], np.float32)
    merged = _deduplicate_side_segments(segments, 100.0)
    assert len(merged) == 2


def test_side_feature_segments_follow_image_translation():
    image, mask = render_synthetic_side("square_pyramid", seed=33)
    first = extract_side_features(image, mask)
    larger = np.full((image.shape[0] + 60, image.shape[1] + 80, 3), 225, np.uint8)
    larger_mask = np.zeros(larger.shape[:2], np.uint8)
    larger[30:30 + image.shape[0], 40:40 + image.shape[1]] = image
    larger_mask[30:30 + image.shape[0], 40:40 + image.shape[1]] = mask
    second = extract_side_features(larger, larger_mask)
    np.testing.assert_allclose(first.vector, second.vector, atol=1e-4)
    np.testing.assert_allclose(first.segments_px + np.asarray([40, 30, 40, 30]), second.segments_px, atol=1e-4)


def test_side_training_mask_prefers_coloured_object_over_border_change():
    background = np.full((360, 640, 3), 210, np.uint8)
    image = background.copy()
    cv2.rectangle(image, (0, 0), (639, 359), (120, 120, 120), 18)
    cv2.rectangle(image, (420, 205), (500, 295), (0, 0, 210), -1)
    mask = side_foreground_mask(image, background)
    assert np.mean(mask[220:280, 435:485]) > 240
    assert np.count_nonzero(mask[:25]) == 0


def test_side_training_mask_prefers_smaller_coloured_object_over_neutral_component():
    background = np.full((360, 640, 3), 210, np.uint8)
    image = background.copy()
    cv2.rectangle(image, (40, 80), (300, 310), (150, 150, 150), -1)
    cv2.rectangle(image, (410, 205), (490, 295), (20, 90, 20), -1)
    mask = side_foreground_mask(image, background)
    assert np.mean(mask[220:280, 425:475]) > 240
    assert np.count_nonzero(mask[100:290, 60:280]) == 0


def test_side_training_mask_removes_tray_edge_joined_by_thin_shadow_bridge():
    background = np.full((360, 640, 3), 210, np.uint8)
    image = background.copy()
    cv2.rectangle(image, (415, 180), (495, 270), (20, 90, 20), -1)
    cv2.rectangle(image, (493, 223), (590, 229), (150, 150, 150), -1)
    cv2.rectangle(image, (560, 165), (590, 285), (150, 150, 150), -1)
    mask = side_foreground_mask(image, background)
    assert np.mean(mask[195:255, 430:480]) > 240
    assert np.count_nonzero(mask[170:280, 550:600]) == 0


def test_side_model_round_trip_and_registry_hash_guard(tmp_path):
    registry = load_shape_registry("config/shapes/competition-11.yaml")
    samples = []
    for class_index, class_id in enumerate(registry.class_ids):
        for repeat in range(2):
            image, mask = render_synthetic_side(class_id, seed=class_index * 10 + repeat)
            samples.append((image, mask, class_id))
    model = SideGeometryModel.train(samples, registry, method="knn")
    path = tmp_path / "side.npz"
    model.save(path)
    loaded = SideGeometryModel.load(path, registry)
    assert loaded.registry_hash == registry.registry_hash
    incompatible = ShapeRegistry(2, registry.classes)
    with pytest.raises(ValueError, match="hash mismatch"):
        SideGeometryModel.load(path, incompatible)


def test_rtrees_side_model_emits_complete_registry_scores(tmp_path):
    registry = load_shape_registry("config/shapes/competition-11.yaml")
    samples = []
    for class_index, class_id in enumerate(registry.class_ids):
        for repeat in range(3):
            image, mask = render_synthetic_side(class_id, seed=100 + class_index * 10 + repeat)
            samples.append((image, mask, class_id))
    model = SideGeometryModel.train(samples, registry, method="rtrees")
    image, mask = render_synthetic_side("cylinder", seed=999)
    model.predict(image, mask)
    assert set(model.last_class_scores) == set(registry.class_ids)
    assert sum(model.last_class_scores.values()) == pytest.approx(1.0)
    expected = dict(model.last_class_scores)
    path = tmp_path / "forest.npz"
    model.save(path)
    cv2.setRNGSeed(99)
    loaded = SideGeometryModel.load(path, registry)
    loaded.predict(image, mask)
    assert loaded.last_class_scores == expected


def test_all_fusion_methods_are_restricted_to_top_two():
    top = {"a": 0.55, "b": 0.35, "c": 0.10}
    side = {"a": 0.01, "b": 0.09, "c": 0.90}
    for method, weights in (
        ("probability_sum", ()),
        ("log_product", ()),
        ("evidence_adaptive", ()),
        ("top2_logistic", (1.0, 0.3, 0.0, 0.0, 0.0)),
    ):
        fused, candidates = fuse_top2_scores(top, side, 1.0, method=method, logistic_weights=weights)
        assert candidates == ("a", "b")
        assert set(fused) == {"a", "b"}


def test_evidence_adaptive_protects_strong_topology_for_same_family():
    top = {"pentagonal_prism": 0.62, "hexagonal_prism": 0.38}
    side = {"pentagonal_prism": 0.05, "hexagonal_prism": 0.95}
    weak, _ = fuse_top2_scores(
        top, side, 1.0, method="evidence_adaptive", side_weight=0.6,
        topology_quality=0.0, same_family_side_scale=0.35,
    )
    strong, _ = fuse_top2_scores(
        top, side, 1.0, method="evidence_adaptive", side_weight=0.6,
        topology_quality=1.0, same_family_side_scale=0.35,
    )
    assert strong["pentagonal_prism"] > weak["pentagonal_prism"]
    assert strong["pentagonal_prism"] > strong["hexagonal_prism"]


def test_evidence_adaptive_keeps_cross_family_side_taper_useful():
    top = {"triangular_prism": 0.52, "triangular_pyramid": 0.48}
    side = {"triangular_prism": 0.05, "triangular_pyramid": 0.95}
    fused, _ = fuse_top2_scores(
        top, side, 1.0, method="evidence_adaptive", side_weight=0.6,
        topology_quality=0.0, same_family_side_scale=0.25,
    )
    assert fused["triangular_pyramid"] > fused["triangular_prism"]


def test_fusion_policy_hash_guard_and_backend_resolution(tmp_path):
    policy = FusionPolicy("abc", method="log_product")
    path = tmp_path / "policy.json"
    policy.save(path)
    assert FusionPolicy.load(path, "abc") == policy
    with pytest.raises(ValueError, match="hash mismatch"):
        FusionPolicy.load(path, "different")
    assert resolve_fusion_backend("auto", machine="AMD64", openvino_available=False) == "opencv"
    assert resolve_fusion_backend("auto", machine="aarch64", tensorrt_available=True) == "tensorrt"


def test_version_one_fusion_policy_loads_with_conservative_adaptive_defaults(tmp_path):
    path = tmp_path / "policy-v1.json"
    path.write_text(
        json.dumps({
            "registry_hash": "legacy", "version": 1,
            "method": "log_product", "side_weight": 0.3,
        }),
        encoding="utf-8",
    )
    policy = FusionPolicy.load(path, "legacy")
    assert policy.version == 1
    assert policy.same_family_side_scale == 0.35
    assert policy.topology_guard_strength == 0.65


def test_fusion_policy_fitter_compares_three_top2_methods():
    registry = load_shape_registry("config/shapes/competition-11.yaml")
    labels = registry.class_ids[:2]
    records = []
    for index in range(12):
        truth = labels[index % 2]
        other = labels[(index + 1) % 2]
        records.append(
            {
                "split": "probability_calibration",
                "human_reviewed": True,
                "true_label": truth,
                "top_class_scores": {truth: 0.56, other: 0.44},
                "side_class_scores": {truth: 0.86, other: 0.14},
                "side_quality": 0.9,
            }
        )
    policy, report = fit_fusion_policy(records, registry.registry_hash)
    assert policy.registry_hash == registry.registry_hash
    assert {item["method"] for item in report["candidates"]} == {
        "probability_sum", "log_product", "top2_logistic"
    }
    assert report["selected"]["wrong_accept_count"] == 0


def test_review_application_requires_each_explicit_visual_check(tmp_path):
    registry = load_shape_registry("config/shapes/competition-11.yaml")
    sample = tmp_path / "batch" / "sample-0001"
    sample.mkdir(parents=True)
    (sample / "metadata.json").write_text(
        json.dumps({"label": "cube", "human_reviewed": False}), encoding="utf-8"
    )
    incomplete = apply_dual_review_records(
        tmp_path, [{"sample": "batch/sample-0001", "approved": True}], registry
    )
    assert incomplete["updated_count"] == 0
    complete = apply_dual_review_records(
        tmp_path,
        [
            {
                "sample": "batch/sample-0001",
                "approved": True,
                "true_label": "长方体",
                "roi_ok": True,
                "association_ok": True,
                "shadow_ok": True,
                "depth_ok": True,
                "instance_split_ok": True,
                "notes": "clear",
            }
        ],
        registry,
    )
    metadata = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
    assert complete["updated_count"] == 1
    assert metadata["human_reviewed"] is True
    assert metadata["label_id"] == "quadrangular_prism"


def test_multi_review_requires_per_object_correspondence_and_matching_composition(tmp_path):
    registry = load_shape_registry("config/shapes/competition-11.yaml")
    sample = tmp_path / "multi" / "scene-0001"
    sample.mkdir(parents=True)
    (sample / "metadata.json").write_text(
        json.dumps(
            {
                "scene_split": "final_acceptance",
                "composition": ["cone", "cylinder"],
                "object_count": 2,
            }
        ),
        encoding="utf-8",
    )
    review = {
        "sample": "multi/scene-0001",
        "approved": True,
        "roi_ok": True,
        "association_ok": True,
        "shadow_ok": True,
        "depth_ok": True,
        "instance_split_ok": True,
        "objects": [
            {
                "object_id": "object-1",
                "true_label": "圆锥",
                "side_visible_ratio": 0.9,
                "side_occluded": False,
                "association_ok": True,
            },
            {
                "object_id": "object-2",
                "true_label": "圆柱体",
                "side_visible_ratio": 0.5,
                "side_occluded": True,
                "association_ok": True,
            },
        ],
    }
    report = apply_dual_review_records(tmp_path, [review], registry)
    metadata = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
    assert report["updated_count"] == 1
    assert metadata["human_reviewed"] is True
    assert [item["true_label"] for item in metadata["review"]["objects"]] == [
        "cone", "cylinder"
    ]


def test_dataset_audit_requires_five_colors_in_each_split(tmp_path):
    registry = load_shape_registry("config/shapes/competition-11.yaml")
    image, mask = render_synthetic_side("cylinder", seed=31)
    samples = [
        SideTrainingSample(
            tmp_path / str(index), image, mask, "cylinder", split, "batch",
            "C" if split == "final_holdout" else "A", "red", str(index),
        )
        for index, split in enumerate(
            ["train"] * 50 + ["probability_calibration"] * 15 + ["final_holdout"] * 15
        )
    ]
    report = audit_side_dataset(samples, registry)
    assert report["per_class_targets"]["cylinder"]["counts_50_15_15"] is True
    assert report["per_class_targets"]["cylinder"]["five_colors_each_split"] is False
    assert report["ready_for_acceptance"] is False


def test_side_acceptance_thresholds_are_fitted_on_calibration_samples(tmp_path):
    registry = load_shape_registry("config/shapes/competition-11.yaml")
    triples = []
    holdout = []
    for class_index, class_id in enumerate(registry.class_ids):
        for repeat in range(2):
            image, mask = render_synthetic_side(class_id, seed=400 + class_index * 10 + repeat)
            triples.append((image, mask, class_id))
        image, mask = render_synthetic_side(class_id, seed=500 + class_index)
        holdout.append(
            SideTrainingSample(
                tmp_path / class_id, image, mask, class_id, "probability_calibration",
                "batch", "A", "red", class_id,
            )
        )
    model = SideGeometryModel.train(triples, registry)
    report = calibrate_side_acceptance(model, holdout)
    assert report["split"] == "probability_calibration"
    assert model.distance_threshold == report["selected"]["distance_threshold"]
    assert model.margin_threshold == report["selected"]["margin_threshold"]


def test_backend_comparison_requires_parity_and_all_requested_backends(monkeypatch, tmp_path):
    registry = load_shape_registry("config/shapes/competition-11.yaml")
    image, mask = render_synthetic_side("cylinder", seed=21)
    sample = SideTrainingSample(
        tmp_path / "sample", image, mask, "cylinder", "final_holdout",
        "batch", "C", "red", "abc",
    )

    class FakeModel:
        def __init__(self, backend):
            self.backend = backend

        def predict(self, image_bgr, object_mask):
            label = "cone" if self.backend == "tensorrt" else "cylinder"
            return label, 0.9, {"reason": "accepted"}

    monkeypatch.setattr(
        "sorting_vision.side_cnn.SideCNNModel.load",
        lambda _path, backend, _registry, device="CPU": FakeModel(backend),
    )
    report = compare_side_cnn_backends(
        tmp_path, [sample], registry, iter(("opencv", "openvino", "tensorrt"))
    )
    assert report["all_requested_backends_tested"] is True
    assert report["classes_and_acceptance_consistent"] is False
    assert report["passed"] is False
    assert report["mismatches"][0]["backend"] == "tensorrt"


def test_side_cnn_acceptance_calibration_uses_real_calibration_split(monkeypatch, tmp_path):
    registry = load_shape_registry("config/shapes/competition-11.yaml")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "class_ids": list(registry.class_ids),
                "registry_hash": registry.registry_hash,
                "confidence_threshold": 0.65,
                "margin_threshold": 0.12,
            }
        ),
        encoding="utf-8",
    )
    expected = iter(registry.class_ids)

    class FakeModel:
        last_class_scores = {}

        def predict(self, image_bgr, object_mask):
            truth = next(expected)
            self.last_class_scores = {item: 0.01 for item in registry.class_ids}
            self.last_class_scores[truth] = 0.90
            return truth, 0.90, {"reason": "accepted"}

    monkeypatch.setattr(
        "sorting_vision.side_cnn.SideCNNModel.load",
        lambda _path, _backend, _registry, device="CPU": FakeModel(),
    )
    image, mask = render_synthetic_side("cylinder", seed=71)
    samples = [
        SideTrainingSample(
            tmp_path / label, image, mask, label, "probability_calibration",
            "batch", "A", "red", label,
        )
        for label in registry.class_ids
    ]
    report = calibrate_side_cnn_acceptance(model_dir, samples, registry)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    assert report["real_images_only"] is True
    assert metadata["acceptance_calibration"]["split"] == "probability_calibration"
    assert metadata["acceptance_calibration"]["selected"]["wrong_accept_count"] == 0
