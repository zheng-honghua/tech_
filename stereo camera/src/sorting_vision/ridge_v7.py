"""Shared two-dimensional ridge evidence. A candidate is never a metric edge."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import copy
import hashlib
import time
import cv2
import numpy as np

from .visual_contract import V7, frame_cache


@dataclass
class RidgeObservation:
    segments: np.ndarray
    evidence: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    gray: np.ndarray
    edge_map: np.ndarray
    internal: np.ndarray


def combine_recall_candidates(verified: list[dict], legacy_segments) -> list[dict]:
    """Retain recall without claiming legacy candidates passed v7 image gates."""
    result = [{**copy.deepcopy(row), "image_verified": row.get("state") == "CANDIDATE_2D"} for row in verified]
    for raw in legacy_segments:
        points = np.asarray(raw, float).reshape(2, 2)
        if not np.isfinite(points).all() or np.linalg.norm(points[1]-points[0]) < 4:
            continue
        duplicate = any(min(np.linalg.norm(points-np.asarray(row["endpoints_px"])),
                            np.linalg.norm(points-np.asarray(row["endpoints_px"])[::-1])) <= 3
                        for row in result if row.get("state") == "CANDIDATE_2D")
        if duplicate:
            continue
        result.append({"endpoints_px":points.tolist(),"source":"LEGACY_RECALL_CANDIDATE", "state":"CANDIDATE_2D",
            "image_verified":False,"image_support":None,"geometry_support":None,
            "reason":"AWAITING_V7_IMAGE_OR_DEPTH_SUPPORT","model_input":False})
    return result


def _samples(line, count=32):
    return np.linspace(line[:2], line[2:], count)


def _difference(line, lab, mask, offset):
    samples = _samples(line)
    direction = line[2:] - line[:2]
    normal = np.array([-direction[1], direction[0]]) / max(np.linalg.norm(direction), 1e-9)
    a, b = np.rint(samples + normal * offset).astype(int), np.rint(samples - normal * offset).astype(int)
    valid = ((a[:, 0] >= 0) & (a[:, 0] < mask.shape[1]) & (a[:, 1] >= 0) & (a[:, 1] < mask.shape[0])
             & (b[:, 0] >= 0) & (b[:, 0] < mask.shape[1]) & (b[:, 1] >= 0) & (b[:, 1] < mask.shape[0]))
    a, b = a[valid], b[valid]
    if not len(a):
        return 0., 0.
    inside = (mask[a[:, 1], a[:, 0]] > 0) & (mask[b[:, 1], b[:, 0]] > 0)
    delta = lab[a[inside, 1], a[inside, 0]] - lab[b[inside, 1], b[inside, 0]]
    if len(delta) < 8:
        return 0., 0.
    center = np.median(delta, axis=0)
    norms = np.linalg.norm(delta, axis=1)
    consistent = (delta @ center) / np.maximum(norms * np.linalg.norm(center), 1.)
    return float(np.median(norms)), float(np.mean(consistent >= .5))


def _supported_merge(lines, edges, maximum_gap):
    kept = []
    for line in sorted(lines, key=lambda x: -np.linalg.norm(x[2:] - x[:2])):
        joined = False
        for i, other in enumerate(kept):
            direction = other[2:] - other[:2]
            length = np.linalg.norm(direction)
            unit = direction / max(length, 1.)
            candidate_unit = (line[2:] - line[:2]) / max(np.linalg.norm(line[2:] - line[:2]), 1.)
            if abs(unit @ candidate_unit) < np.cos(np.deg2rad(5)):
                continue
            endpoints = line.reshape(2, 2)
            along = (endpoints - other[:2]) @ unit
            relative = endpoints - other[:2]
            perpendicular = np.abs(unit[0] * relative[:, 1] - unit[1] * relative[:, 0])
            if perpendicular.max() > 1.5:
                continue
            gap = max(float(along.min() - length), float(-along.max()), 0.)
            if gap > maximum_gap:
                continue
            minimum, maximum = min(0., float(along.min())), max(length, float(along.max()))
            merged = np.r_[other[:2] + unit * minimum, other[:2] + unit * maximum]
            uv = np.rint(_samples(merged, max(32, int(maximum - minimum)))).astype(int)
            valid = (uv[:, 0] >= 0) & (uv[:, 0] < edges.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < edges.shape[0])
            if valid.all() and np.mean(edges[uv[:, 1], uv[:, 0]] > 0) >= .65:
                kept[i] = merged
                joined = True
                break
        if not joined:
            kept.append(line)
    return np.asarray(kept, np.float32).reshape(-1, 4)


def extract_ridges(image: np.ndarray, mask: np.ndarray) -> RidgeObservation:
    cache = frame_cache()
    key = ("ridges", image.__array_interface__["data"][0], image.shape,
           hashlib.blake2b(np.asarray(mask).tobytes(), digest_size=12).hexdigest())
    if cache is not None and key in cache:
        return copy.deepcopy(cache[key])
    started = time.perf_counter()
    result = _extract_ridges(image, mask)
    if cache is not None:
        cache[key] = result
        timings = cache.setdefault("stage_times", {})
        timings["ridge_lsd_ms"] = timings.get("ridge_lsd_ms", 0.) + (time.perf_counter() - started) * 1000
    return copy.deepcopy(result)


def _extract_ridges(image: np.ndarray, mask: np.ndarray) -> RidgeObservation:
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if image.shape[:2] != binary.shape or cv2.countNonZero(binary) < 20:
        raise ValueError("ridge image/mask invalid")
    x, y, width, height = cv2.boundingRect(binary)
    x0, y0 = max(0, x - 4), max(0, y - 4)
    x1, y1 = min(binary.shape[1], x + width + 4), min(binary.shape[0], y + height + 4)
    factor = min(1., 256. / max(x1 - x0, y1 - y0))
    local = cv2.resize(image[y0:y1, x0:x1], None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    local_mask = cv2.resize(binary[y0:y1, x0:x1], (local.shape[1], local.shape[0]), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(local, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(local, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.bilateralFilter(gray, 5, 20, 20)
    # No neutral-image cutout before LSD: artificial cutout boundaries would
    # otherwise look like measured image edges.
    distance = cv2.distanceTransform(local_mask, cv2.DIST_L2, 5)
    scale = np.sqrt(cv2.countNonZero(local_mask))
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    high = max(12., float(np.percentile(grad[local_mask > 0], 80)))
    edges = cv2.Canny(gray, int(max(5., high * .35)), int(high))
    support_map = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(gray)[0]
    blurred = cv2.GaussianBlur(gray, (0, 0), .8)
    second = detector.detect(blurred)[0]
    second = np.empty((0, 4)) if second is None else second.reshape(-1, 4)
    candidates, rejected = [], []
    for line in ([] if detected is None else detected.reshape(-1, 4)):
        line = line.astype(float)
        length = np.linalg.norm(line[2:] - line[:2])
        if length < max(8., .06 * scale):
            continue
        uv = np.rint(_samples(line)).astype(int)
        uv[:, 0] = np.clip(uv[:, 0], 0, gray.shape[1] - 1)
        uv[:, 1] = np.clip(uv[:, 1], 0, gray.shape[0] - 1)
        inside = float(np.mean(local_mask[uv[:, 1], uv[:, 0]] > 0))
        boundary = float(np.mean(distance[uv[:, 1], uv[:, 0]] < max(2., .025 * scale)))
        support = float(np.mean(support_map[uv[:, 1], uv[:, 0]] > 0))
        delta, consistency = _difference(line, lab, local_mask, max(2., .02 * scale))
        unit = (line[2:] - line[:2]) / length
        stable = False
        for alternate in second:
            direction = alternate[2:] - alternate[:2]
            if abs(unit @ (direction / max(np.linalg.norm(direction), 1.))) < np.cos(np.deg2rad(6)):
                continue
            positions = alternate.reshape(2, 2) - line[:2]
            along = positions @ unit
            overlap = max(0., min(length, float(along.max())) - max(0., float(along.min())))
            if np.max(np.abs(unit[0] * positions[:, 1] - unit[1] * positions[:, 0])) <= 2. and overlap >= .5 * min(length, np.linalg.norm(direction)):
                stable = True
                break
        reason = ("OUTSIDE_OBJECT" if inside < .88 else "SILHOUETTE_BAND" if boundary > .35
                  else "WEAK_SUPPORT" if support < .35 else "UNSTABLE_SCALE" if not stable
                  else "INCONSISTENT_FACE_DIFFERENCE" if delta < 3.5 or consistency < .65 else "candidate")
        record = {"endpoints_px": (line.reshape(2, 2) / factor + [x0, y0]).tolist(),
                  "source": "INTERNAL_LUMINANCE_LSD", "image_support": support,
                  "side_delta_lab": delta, "side_consistency": consistency,
                  "multiscale_stable": stable, "geometry_support": None,
                  "state": "CANDIDATE_2D", "reason": reason, "contract": V7}
        if reason == "candidate":
            candidates.append(line)
        else:
            rejected.append(record)
    internal_local = _supported_merge(candidates[:64], support_map, max(3., .04 * scale))[:32]
    internal = internal_local.reshape(-1, 2, 2) / factor + [x0, y0]
    evidence = []
    for line in internal_local:
        delta, consistency = _difference(line, lab, local_mask, max(2., .02 * scale))
        pixels = np.rint(_samples(line)).astype(int)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, local_mask.shape[1] - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, local_mask.shape[0] - 1)
        evidence.append({"endpoints_px": (line.reshape(2, 2) / factor + [x0, y0]).tolist(),
                         "source": "INTERNAL_LUMINANCE_LSD", "state": "CANDIDATE_2D",
                         "side_delta_lab": delta, "side_consistency": consistency,
                         "image_support": float(np.mean(support_map[pixels[:, 1], pixels[:, 0]] > 0)),
                         "multiscale_stable": True, "contract": V7,
                         "geometry_support": None, "reason": "awaiting_geometric_validation"})
    # Straight contour spans need a low residual, not just polygon chords.
    contour = max(cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    polygon = cv2.approxPolyDP(contour, .008 * cv2.arcLength(contour, True), True).reshape(-1, 2)
    contour_points = contour.reshape(-1, 2)
    for i, first in enumerate(polygon):
        second_point = polygon[(i + 1) % len(polygon)]
        direction = second_point.astype(float) - first
        length = np.linalg.norm(direction)
        if length < max(8., .08 * scale):
            continue
        unit = direction / length
        relative = contour_points - first
        along = relative @ unit
        perpendicular = np.abs(unit[0] * relative[:, 1] - unit[1] * relative[:, 0])
        near = (along >= 0) & (along <= length) & (perpendicular <= max(4., .035 * scale))
        previous = first.astype(float) - polygon[i - 1]
        following = polygon[(i + 2) % len(polygon)].astype(float) - second_point
        turns = [np.degrees(np.arccos(np.clip(v @ direction / max(np.linalg.norm(v) * length, 1e-6), -1, 1)))
                 for v in (previous, following)]
        straight = (max(turns) >= 30 and near.sum() >= max(8, length * .5)
                    and np.percentile(perpendicular[near], 90) <= 1.25)
        native = np.stack((first, second_point)).astype(float) / factor + [x0, y0]
        boundary_pixels = np.rint(_samples(np.r_[first, second_point])).astype(int)
        boundary_pixels[:, 0] = np.clip(boundary_pixels[:, 0], 0, edges.shape[1] - 1)
        boundary_pixels[:, 1] = np.clip(boundary_pixels[:, 1], 0, edges.shape[0] - 1)
        evidence.append({"endpoints_px": native.tolist(), "source": "SILHOUETTE" if straight else "CURVED_BOUNDARY",
                         "state": "CANDIDATE_2D" if straight else "CURVED_2D", "geometry_support": None,
                         "image_support": float(np.mean(support_map[boundary_pixels[:, 1], boundary_pixels[:, 0]] > 0)),
                         "side_delta_lab": None, "side_consistency": None, "contract": V7,
                         "reason": "awaiting_geometric_validation" if straight else "not_a_straight_ridge"})
    segments = np.asarray([record["endpoints_px"] for record in evidence if record["state"] == "CANDIDATE_2D"], np.float32).reshape(-1, 4)
    return RidgeObservation(segments, evidence, rejected, gray, edges, internal.reshape(-1, 4).astype(np.float32))


def extract_topology(image: np.ndarray, mask: np.ndarray):
    from .geometry_edges import EdgeTopology, _line_from_points, _junctions, _angle_difference
    result = extract_ridges(image, mask)
    scale = max(1., np.sqrt(cv2.countNonZero(mask)))
    lines = tuple(_line_from_points(line.reshape(2, 2), 1., evidence.get("side_delta_lab", 3.5))
                  for line, evidence in zip(result.internal, result.evidence))
    junctions = _junctions(list(lines), mask, scale)
    parallel = sum(_angle_difference(a.angle_deg, b.angle_deg) < 8 for i, a in enumerate(lines) for b in lines[i + 1:])
    quality = float(min(1., len(lines) / 4))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = np.zeros(mask.shape, np.uint8)
    for line in result.internal:
        points = np.rint(line.reshape(2, 2)).astype(int)
        cv2.line(edges, tuple(points[0]), tuple(points[1]), 255)
    return EdgeTopology(gray, edges, lines, lines, tuple(junctions), parallel, 0, (), 0.,
                        quality, "candidate_ridges_v7", scale)
