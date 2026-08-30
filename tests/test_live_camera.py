from types import SimpleNamespace
from argparse import Namespace

import numpy as np

from sorting_vision.camera import OpenCVCameraSource, RealSenseD415Source
from sorting_vision import cli
from sorting_vision.config import load_config


class FakeCapture:
    def __init__(self, frames, opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False

    def set(self, *_):
        return True

    def isOpened(self):
        return self.opened

    def read(self):
        return self.frames.pop(0) if self.frames else (False, None)

    def release(self):
        self.released = True


def test_uvc_source_reconnects_and_releases():
    failed = FakeCapture([(False, None)])
    image = np.full((10, 12, 3), 70, np.uint8)
    recovered = FakeCapture([(True, image)])
    captures = iter([failed, recovered])
    source = OpenCVCameraSource(
        warmup_frames=0,
        reconnect_attempts=1,
        capture_factory=lambda _: next(captures),
    )
    frame = source.read()
    assert frame.color_bgr.shape == (10, 12, 3)
    assert failed.released is True
    source.close()
    assert recovered.released is True


def test_default_rgb_calibration_preserves_widescreen_aspect_ratio():
    image = np.zeros((720, 1280, 3), np.uint8)
    calibration = cli._default_calibration(image, load_config())
    assert calibration.output_width_px == 1280
    assert calibration.output_height_px == 720
    assert np.array_equal(calibration.rectify(image), image)


class FakeVideoFrame:
    def __init__(self, data, timestamp, number, intrinsics):
        self._data = data
        self._timestamp = timestamp
        self._number = number
        self.profile = SimpleNamespace(
            as_video_stream_profile=lambda: SimpleNamespace(intrinsics=intrinsics)
        )

    def get_data(self):
        return self._data

    def get_timestamp(self):
        return self._timestamp

    def get_frame_number(self):
        return self._number

    def __bool__(self):
        return True


class FakeRS:
    stream = SimpleNamespace(depth="depth", color="color")
    format = SimpleNamespace(z16="z16", bgr8="bgr8")

    def __init__(self):
        native = SimpleNamespace(width=4, height=3, fx=10, fy=11, ppx=2, ppy=1)
        color = FakeVideoFrame(np.zeros((3, 4, 3), np.uint8), 100.0, 7, native)
        depth = FakeVideoFrame(np.full((3, 4), 500, np.uint16), 99.5, 7, native)
        self.frames = SimpleNamespace(
            get_color_frame=lambda: color,
            get_depth_frame=lambda: depth,
        )
        self.enabled = []
        self.stopped = False

    def config(self):
        return SimpleNamespace(enable_stream=lambda *args: self.enabled.append(args))

    def pipeline(self):
        device = SimpleNamespace(
            first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: 0.001)
        )
        profile = SimpleNamespace(get_device=lambda: device)
        return SimpleNamespace(
            start=lambda _: profile,
            wait_for_frames=lambda: self.frames,
            stop=lambda: setattr(self, "stopped", True),
        )

    def align(self, stream):
        assert stream == "color"
        return SimpleNamespace(process=lambda frames: frames)


def test_realsense_source_aligns_and_builds_intrinsics():
    rs = FakeRS()
    source = RealSenseD415Source(width=4, height=3, rs_module=rs)
    frame = source.read()
    assert len(rs.enabled) == 2
    assert frame.depth_mm[0, 0] == 500
    assert frame.sync_delta_ms == 0.5
    assert frame.frame_id == "d415-000000007"
    source.close()
    assert rs.stopped is True


def test_realsense_source_accepts_separate_depth_and_color_stream_sizes():
    rs = FakeRS()
    source = RealSenseD415Source(
        depth_width=640, depth_height=360,
        color_width=1280, color_height=720, fps=30, rs_module=rs,
    )
    assert rs.enabled[0][1:3] == (640, 360)
    assert rs.enabled[1][1:3] == (1280, 720)
    metadata = source.capture_metadata()
    assert metadata["alignment"] == "depth_to_color"
    source.close()


def test_camera_record_writes_manifest(monkeypatch, tmp_path):
    image = np.full((10, 12, 3), 70, np.uint8)

    class Source:
        def __init__(self):
            self.index = 0
            self.closed = False

        def read(self):
            from sorting_vision.camera import RGBFrame

            self.index += 1
            return RGBFrame(image, self.index, f"capture-{self.index}")

        def close(self):
            self.closed = True

    source = Source()
    monkeypatch.setattr(cli, "_make_camera_source", lambda args, config: source)
    session = tmp_path / "session"
    args = Namespace(
        config=None,
        source="uvc",
        camera_index=0,
        width=None,
        height=None,
        fps=None,
        session=str(session),
        label="red-block",
        headless=True,
        max_frames=2,
    )
    assert cli._run_camera_record(args) == 0
    lines = (session / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"label": "red-block"' in lines[0]
    assert source.closed is True
