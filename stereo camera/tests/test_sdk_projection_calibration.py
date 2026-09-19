import hashlib
import json

import numpy as np
import pytest

from sorting_vision.intrinsic_calibration import SDKProjectionCalibration, load_projection_calibration
from sorting_vision.rgbd import CameraIntrinsics


def factory():
    return SDKProjectionCalibration(CameraIntrinsics(640, 480, 500., 500., 320., 240.),
                                    np.zeros(5), "test-serial", "inverse_brown_conrady")


def test_factory_roundtrip_and_historical_payload(tmp_path):
    value = factory().to_dict()
    path = tmp_path / "sdk.json"
    path.write_text(json.dumps(value))
    assert load_projection_calibration(path).valid
    value.pop("camera_id")
    value.pop("calibration_hash")
    value["calibration_hash"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(value))
    assert load_projection_calibration(path).valid
    value["serial"] = "changed"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_projection_calibration(path)


def test_nonzero_factory_distortion_and_nonfinite_rays_rejected():
    original = factory()
    assert not SDKProjectionCalibration(original.intrinsics, np.ones(5), original.serial, original.model).valid
    assert not SDKProjectionCalibration(CameraIntrinsics(640, 480, float("nan"), 500, 320, 240),
                                        np.zeros(5), original.serial, original.model).valid
