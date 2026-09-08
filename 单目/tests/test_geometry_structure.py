import json

import cv2
import numpy as np
import pytest

from sorting_vision.geometry_structure import extract_structural_contour
from sorting_vision.geometry_structure_audit import (
    audit_geometry_scene_structures,
    audit_geometry_structures,
)
from sorting_vision.geometry_rgb import GeometryRGBModel, train_geometry_model


def _pyramid(angle_deg: float = 0.0, noisy: bool = False):
    image = np.full((256, 256, 3), 225, np.uint8)
    mask = np.zeros((256, 256), np.uint8)
    corners = np.asarray([[38, 214], [128, 30], [220, 214]], np.int32)
    cv2.fillPoly(mask, [corners], 255)
    image[mask > 0] = (145, 165, 190)
    apex = (128, 128)
    for point in corners:
        cv2.line(image, apex, tuple(point), (55, 68, 82), 4, cv2.LINE_AA)
    if noisy:
        rng = np.random.default_rng(91)
        for x, y in rng.integers(25, 230, size=(160, 2)):
            if mask[y, x]:
                cv2.circle(image, (int(x), int(y)), 1, (45, 45, 45), -1)
        # Mask burrs and holes model imperfect segmentation.
        for x, y in rng.integers(35, 220, size=(40, 2)):
            cv2.circle(mask, (int(x), int(y)), 1, 0, -1)
    if angle_deg:
        transform = cv2.getRotationMatrix2D((128, 128), angle_deg, 1.0)
        image = cv2.warpAffine(image, transform, (256, 256), borderValue=(225, 225, 225))
        mask = cv2.warpAffine(mask, transform, (256, 256), flags=cv2.INTER_NEAREST)
    return image, mask


def test_structural_contour_removes_speckles_and_has_no_dangling_vertices():
    image, mask = _pyramid(noisy=True)
    result = extract_structural_contour(image, mask)
    assert result.reason == "accepted"
    assert result.outer_iou >= 0.95
    assert 3 <= len(result.outer_polygon) <= 5
    assert result.dangling_endpoint_count == 0
    assert len(result.internal_lines) <= 5
    assert result.rejected_candidate_ratio > 0.45
    assert cv2.connectedComponents(result.clean_mask)[0] == 2


def test_uniform_object_outputs_only_fitted_outer_structure():
    image = np.full((256, 256, 3), 225, np.uint8)
    mask = np.zeros((256, 256), np.uint8)
    cv2.rectangle(mask, (42, 55), (214, 205), 255, -1)
    image[mask > 0] = (110, 150, 190)
    rng = np.random.default_rng(3)
    for x, y in rng.integers(55, 200, size=(100, 2)):
        cv2.circle(image, (int(x), int(y)), 1, (60, 60, 60), -1)
    result = extract_structural_contour(image, mask)
    assert len(result.outer_polygon) == 4
    assert len(result.internal_lines) == 0
    assert len(result.vertices) == 4
    assert result.dangling_endpoint_count == 0


def test_smooth_round_shading_is_not_converted_to_straight_ridges():
    yy, xx = np.mgrid[:256, :256]
    radius = np.hypot(xx - 128, yy - 128)
    mask = (radius <= 82).astype(np.uint8) * 255
    shading = np.clip(175 + 0.28 * (xx - 128) - 0.10 * (yy - 128), 60, 220)
    image = np.full((256, 256, 3), 230, np.uint8)
    image[mask > 0] = np.stack((
        shading * 0.65, shading * 0.82, shading,
    ), axis=2)[mask > 0].astype(np.uint8)
    result = extract_structural_contour(image, mask)
    assert len(result.outer_polygon) >= 9
    assert result.internal_lines == ()


def test_disconnected_ridge_fragments_reconstruct_a_shared_apex():
    image, mask = _pyramid()
    apex = np.asarray([128, 128], np.float32)
    corners = np.asarray([[38, 214], [128, 30], [220, 214]], np.float32)
    shades = ((112, 137, 170), (137, 157, 184), (88, 118, 158))
    for index, colour in enumerate(shades):
        face = np.asarray((apex, corners[index], corners[(index + 1) % 3]), np.int32)
        cv2.fillPoly(image, [face], colour)
    for corner in corners:
        first = tuple(np.rint(apex * 0.82 + corner * 0.18).astype(int))
        second = tuple(np.rint(apex * 0.08 + corner * 0.92).astype(int))
        cv2.line(image, first, second, (55, 68, 82), 4, cv2.LINE_AA)
    result = extract_structural_contour(image, mask)
    assert len(result.internal_lines) >= 3
    junctions = [item for item in result.vertices if item.kind == "junction"]
    assert junctions
    assert max(item.degree for item in junctions) >= 3
    assert result.dangling_endpoint_count == 0


def test_rotated_structure_keeps_clean_graph_constraints():
    first = extract_structural_contour(*_pyramid(0))
    second = extract_structural_contour(*_pyramid(31))
    assert abs(len(first.outer_polygon) - len(second.outer_polygon)) <= 1
    assert first.outer_iou > 0.94
    assert second.outer_iou > 0.94
    assert first.dangling_endpoint_count == second.dangling_endpoint_count == 0
    for result in (first, second):
        scale = np.sqrt(cv2.countNonZero(result.clean_mask))
        assert all(
            np.linalg.norm(line.points()[1] - line.points()[0]) >= 0.12 * scale
            for line in result.internal_lines
        )


def test_structure_audit_writes_cnn_ready_artifacts(tmp_path):
    root = tmp_path / "geometry"
    target = root / "三棱锥"
    target.mkdir(parents=True)
    for index, angle in enumerate((0, 20)):
        image, _ = _pyramid(angle)
        assert cv2.imwrite(str(target / f"{index}.png"), image)
    output = tmp_path / "structure-audit"
    report = audit_geometry_structures(root, output)
    assert report["images"] == 2
    assert report["zero_dangling_rate"] == 1.0
    directories = list((output / "按真实类别").glob("*/*"))
    assert len(directories) == 2
    for directory in directories:
        for name in (
            "clean_mask.png", "raw_edge_map.png", "clean_line_map.png",
            "vertex_heatmap.png", "structure_overlay.png", "structure.json",
        ):
            assert (directory / name).is_file()
        payload = json.loads((directory / "structure.json").read_text(encoding="utf-8"))
        assert payload["dangling_endpoint_count"] == 0


def test_scene_structure_audit_processes_each_separated_object(tmp_path):
    root = tmp_path / "scenes"
    root.mkdir()
    scene = np.full((480, 720, 3), 230, np.uint8)
    first, first_mask = _pyramid()
    second, second_mask = _pyramid(20)
    for item, item_mask in ((first, first_mask), (second, second_mask)):
        hsv = cv2.cvtColor(item, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1][item_mask > 0] = np.maximum(hsv[:, :, 1][item_mask > 0], 165)
        item[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    scene[80:336, 45:301][first_mask > 0] = first[first_mask > 0]
    scene[110:366, 390:646][second_mask > 0] = second[second_mask > 0]
    assert cv2.imwrite(str(root / "two-objects.jpg"), scene)
    output = tmp_path / "scene-audit"
    report = audit_geometry_scene_structures(root, output)
    assert report["scenes"] == 1
    assert report["objects"] == 2
    assert report["zero_dangling_rate"] == 1.0
    assert (output / "总览" / "two-objects.jpg").is_file()


def test_structure_topology_model_trains_saves_and_loads(tmp_path):
    root = tmp_path / "geometry"
    for folder, angles in (("三棱锥", (0, 2, 4)), ("四棱锥", (1, 3, 5))):
        directory = root / folder
        directory.mkdir(parents=True)
        for index, angle in enumerate(angles):
            image, _ = _pyramid(angle)
            assert cv2.imwrite(str(directory / f"{index}.png"), image)
    model, report = train_geometry_model(root, feature_set="structure-topology")
    path = tmp_path / "structure-model.npz"
    model.save(path)
    loaded = GeometryRGBModel.load(path)
    assert loaded.feature_set == "structure-topology"
    assert loaded.feature_version == 5
    assert report["feature_count"] == 1868
    image, _ = _pyramid()
    first = model.predict(image)
    second = loaded.predict(image)
    assert first[0] == second[0]
    assert first[1] == pytest.approx(second[1])
