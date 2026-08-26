import json

import cv2
import numpy as np
import pytest

from sorting_vision.geometry_edge_audit import audit_geometry_edges
from sorting_vision.geometry_edges import edge_topology_vector, extract_edge_topology
from sorting_vision.geometry_rgb import (
    GeometryRGBModel,
    extract_geometry_features,
    preprocess_geometry_object,
    train_geometry_model,
)


def _pyramid(angle_deg: float = 0.0):
    image = np.full((256, 256, 3), 210, np.uint8)
    mask = np.zeros((256, 256), np.uint8)
    corners = np.asarray([[40, 210], [128, 35], [216, 210]], np.int32)
    cv2.fillPoly(mask, [corners], 255)
    image[mask > 0] = (150, 170, 190)
    apex = (128, 125)
    for point in corners:
        cv2.line(image, apex, tuple(point), (60, 70, 80), 4, cv2.LINE_AA)
    if not angle_deg:
        return image, mask
    transform = cv2.getRotationMatrix2D((128, 128), angle_deg, 1.0)
    return (
        cv2.warpAffine(image, transform, (256, 256), borderValue=(210, 210, 210)),
        cv2.warpAffine(mask, transform, (256, 256), flags=cv2.INTER_NEAREST),
    )


def test_radial_edges_form_rotation_stable_topology():
    image, mask = _pyramid()
    rotated, rotated_mask = _pyramid(37.0)
    first = extract_edge_topology(image, mask)
    second = extract_edge_topology(rotated, rotated_mask)
    assert first.reason == second.reason == "accepted"
    assert len(first.merged_lines) >= 3
    assert max(node.degree for node in first.junctions) >= 3
    first_vector = edge_topology_vector(first)
    second_vector = edge_topology_vector(second)
    cosine = np.dot(first_vector, second_vector) / (
        np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
    )
    assert cosine > 0.95
    assert all(line.support >= 0.25 for line in first.merged_lines)


def test_flat_object_and_external_shadow_are_not_internal_edges():
    image = np.full((256, 256, 3), 210, np.uint8)
    mask = np.zeros((256, 256), np.uint8)
    cv2.circle(mask, (128, 128), 75, 255, -1)
    image[mask > 0] = (130, 160, 190)
    cv2.line(image, (5, 230), (250, 230), (20, 20, 20), 9)
    topology = extract_edge_topology(image, mask)
    assert topology.reason == "edge_evidence_low"
    assert len(topology.merged_lines) < 2
    for line in topology.merged_lines:
        for x, y in line.points():
            assert mask[int(round(y)), int(round(x))] > 0


def _edge_dataset(tmp_path):
    root = tmp_path / "geometry"
    for folder, rotation in (("三棱锥", 0), ("四棱锥", 25)):
        target = root / folder
        target.mkdir(parents=True)
        for index, delta in enumerate((0, 5, 10)):
            image, _ = _pyramid(rotation + delta)
            assert cv2.imwrite(str(target / f"{index}.png"), image)
    return root


def test_v3_model_round_trip_preserves_grouped_features(tmp_path):
    root = _edge_dataset(tmp_path)
    model, report = train_geometry_model(root, feature_set="edge-topology")
    path = tmp_path / "edges.npz"
    model.save(path)
    loaded = GeometryRGBModel.load(path)
    image, _ = _pyramid()
    prepared = preprocess_geometry_object(image, output_size=256)
    feature = extract_geometry_features(prepared, "edge-topology")
    original = model.predict_feature(feature)
    restored = loaded.predict_feature(feature)
    assert original[0] == restored[0]
    assert original[1] == pytest.approx(restored[1])
    assert original[2]["reason"] == restored[2]["reason"]
    assert loaded.model_version == 3
    assert loaded.feature_version == 3
    assert loaded.feature_set == "edge-topology"
    assert np.allclose(
        loaded.feature_group_weights, [0.15, 0.15, 0.05, 0.45, 0.20]
    )
    assert loaded.edge_parameters["input_size"] == 256
    assert loaded.edge_parameters["merge_angle_deg"] == 8.0
    assert loaded.margin_threshold == pytest.approx(0.075)
    assert loaded.class_margin_thresholds["pentagonal_prism"] == pytest.approx(0.045)
    assert report["feature_count"] == 1857


def test_edge_audit_writes_complete_diagnostic_bundle(tmp_path):
    root = _edge_dataset(tmp_path)
    output = tmp_path / "audit"
    report = audit_geometry_edges(root, output)
    assert report["images"] == 6
    assert "at_least_two_edges_rate" in report
    directories = list((output / "按真实类别").glob("*/*"))
    assert len(directories) == 6
    for directory in directories:
        for name in (
            "enhanced_gray.png",
            "edge_map.png",
            "line_segments.png",
            "topology.png",
            "topology.json",
            "annotated.jpg",
        ):
            assert (directory / name).is_file()
        payload = json.loads((directory / "topology.json").read_text(encoding="utf-8"))
        assert "merged_lines" in payload
        assert "junctions" in payload
        assert "face_vertices" in payload
