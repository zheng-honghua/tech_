import importlib
from pathlib import Path

import numpy as np

from sorting_vision.rgbd import CameraIntrinsics


def test_factory_ray_provenance_and_gate(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    factory = importlib.import_module("debug_native_calibration").FactoryRays
    intrinsics = CameraIntrinsics(1920, 1080, 1368, 1368, 973, 568, 1)
    rays = factory(intrinsics, np.zeros(5), "device", "inverse_brown_conrady")
    assert rays.valid
    assert rays.to_dict()["checkerboard_metrics"] is None
    assert rays.to_dict()["source"] == "REALSENSE_SDK_FACTORY"
    assert rays.to_dict()["calibration_hash"] == rays.to_dict()["calibration_hash"]
    assert not factory(intrinsics, np.ones(5), "device", "inverse_brown_conrady").valid
    assert not factory(intrinsics, np.zeros(5), "", "inverse_brown_conrady").valid
    assert not factory(intrinsics, np.zeros(4), "device", "inverse_brown_conrady").valid
    invalid = CameraIntrinsics(1920, 1080, float("nan"), 1368, 973, 568, 1)
    assert not factory(invalid, np.zeros(5), "device", "inverse_brown_conrady").valid
