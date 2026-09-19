from __future__ import annotations

import json

import numpy as np
import pytest

from sorting_vision.joint_topology import build_joint_topology, _fit_edge
from test_cross_view_topology import _calibration, _box_points


def test_line_fit_rejects_plane_strip_and_short_line():
    x, y = np.meshgrid(np.linspace(-20, 20, 20), np.linspace(-5, 5, 8))
    assert _fit_edge(np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 200))), 4) is None
    assert _fit_edge(np.column_stack((np.linspace(0, 1, 20), np.zeros(20), np.full(20, 200))), 4) is None


def test_supported_side_line_is_partial_metric_evidence():
    points = np.column_stack((np.linspace(-25, 25, 100), np.zeros(100), np.full(100, 200)))
    graph = build_joint_topology(points, np.asarray([[96.75, 128, 159.25, 128]]), _calibration())
    assert graph["complete_mesh"] is False
    assert len(graph["edges"]) == 1
    assert graph["edges"][0]["sources"] == ["SIDE_DEPTH_SUPPORTED"]
    assert not graph["unresolved_side_segments"]
    assert np.all(np.isfinite(list(graph["features"].values())))
    json.dumps(graph, allow_nan=False)


def test_missing_side_support_never_creates_metric_edge():
    points = _box_points()
    graph = build_joint_topology(points, np.asarray([[0, 0, 40, 0]]), _calibration())
    assert graph["unresolved_side_segments"]
    assert not any("SIDE_DEPTH_SUPPORTED" in edge["sources"] for edge in graph["edges"])


def test_graph_is_deterministic_and_does_not_modify_cloud():
    points = _box_points()
    before = points.copy()
    first = build_joint_topology(points, np.empty((0, 4)), _calibration())
    second = build_joint_topology(points, np.empty((0, 4)), _calibration())
    assert first == second
    np.testing.assert_array_equal(points, before)
    assert first["features"]["graph_unresolved_ratio"] == 0


def test_graph_rejects_insufficient_depth():
    with pytest.raises(ValueError):
        build_joint_topology(np.zeros((50, 3)), np.empty((0, 4)), _calibration())
