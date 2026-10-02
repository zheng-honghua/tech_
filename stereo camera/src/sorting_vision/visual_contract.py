"""Request-local visual feature contracts; legacy callers remain unchanged."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json

LEGACY = "legacy"
V7_PARAMETERS = {
    "version": 7, "seed": "hsv", "grow": "circular_hue_lab_geodesic",
    "edges": "luminance_lsd_multiscale", "minimum_seed_pixels": 20,
    "weak_saturation": 20, "weak_value": 8, "maximum_hue_delta": 25,
    "minimum_inside": .88, "maximum_boundary_overlap": .35,
    "minimum_support": .35, "maximum_lines": 32,
    "metric_edge_merge": "parallel_preserving_overlap_only_v1",
    "mask_padding": "0.75_sqrt_seed_clamp_12_160",
    "instance_split": "geometry_only_ambiguous_concavity",
    "side_quality": "original_rgb_sharpness_v1",
}
V7 = "color_ridge_v7:" + hashlib.sha256(
    json.dumps(V7_PARAMETERS, sort_keys=True).encode()
).hexdigest()[:16]
_ACTIVE = ContextVar("sorting_vision_visual_contract", default=LEGACY)
_CACHE = ContextVar("sorting_vision_visual_cache", default=None)


def frame_cache() -> dict | None:
    return _CACHE.get()


def active_contract() -> str:
    return _ACTIVE.get()


@contextmanager
def visual_contract(contract: str):
    if contract not in {LEGACY, V7}:
        raise ValueError("unsupported visual feature contract")
    token = _ACTIVE.set(contract)
    cache_token = _CACHE.set({} if contract == V7 else None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        _CACHE.reset(cache_token)


def require_contract(contract: str) -> None:
    if contract != active_contract():
        raise ValueError(f"visual feature contract mismatch: model={contract}, input={active_contract()}")
