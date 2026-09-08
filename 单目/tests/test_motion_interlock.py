import numpy as np

from sorting_vision.camera import RGBFrame
from sorting_vision.config import MotionInterlockConfig
from sorting_vision.interlock import MotionInterlock, RunState
from sorting_vision.server import VisionService3D


class Clock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        return self.value

    def advance_ms(self, value):
        self.value += int(value * 1_000_000)


class Source:
    def __init__(self):
        self.reads = 0

    def read(self):
        self.reads += 1
        image = np.full((48, 64, 3), 100, np.uint8)
        return RGBFrame(image, self.reads, f"frame-{self.reads}")

    def close(self):
        pass


class Pipeline:
    def __init__(self):
        self.calls = 0
        self.resets = 0

    def process(self, frame):
        self.calls += 1
        return []

    def reset_tracking(self):
        self.resets += 1

    acknowledge_pick = reset_tracking

    def health(self):
        return {"ok": True, "reason": "ok"}

    def annotate(self, frame, results):
        return frame.color_bgr.copy()


def _service(clock, **overrides):
    values = dict(
        min_settle_ms=300,
        discard_frames=8,
        stable_frames=3,
        frame_diff_threshold=2.5,
        timeout_ms=2000,
    )
    values.update(overrides)
    source = Source()
    pipeline = Pipeline()
    interlock = MotionInterlock(MotionInterlockConfig(**values), clock)
    return VisionService3D(pipeline, source, interlock, "RGBD"), source, pipeline


def test_moving_reads_preview_but_never_computes_and_clears_tracking():
    clock = Clock()
    service, source, pipeline = _service(clock)
    response = service.handle({"type": "motion_start"})
    assert response["health"]["run_state"] == "MOVING"
    assert pipeline.resets == 1

    response = service.handle({"type": "detect"})
    assert source.reads == 1
    assert pipeline.calls == 0
    assert response["status"] == "BUSY_MOVING"
    assert response["results"] == []
    assert service.preview_image() is not None


def test_stop_discards_frames_and_requires_three_stable_frames():
    clock = Clock()
    service, source, pipeline = _service(clock)
    service.motion_start()
    service.motion_stop()
    clock.advance_ms(300)

    for _ in range(11):
        response = service.handle({"type": "detect"})
        assert response["results"] == []
        assert pipeline.calls == 0
    response = service.handle({"type": "detect"})
    assert response["status"] == "OK"
    assert response["health"]["run_state"] == "READY"
    assert pipeline.calls == 1
    assert source.reads == 12


def test_settling_timeout_never_computes():
    clock = Clock()
    service, _, pipeline = _service(clock)
    service.motion_start()
    service.motion_stop()
    clock.advance_ms(2001)
    response = service.handle({"type": "detect"})
    assert response["status"] == "MOTION_UNSTABLE"
    assert response["health"]["ok"] is False
    assert pipeline.calls == 0


def test_motion_messages_are_idempotent():
    clock = Clock()
    service, _, _ = _service(clock, discard_frames=0, stable_frames=1, min_settle_ms=0)
    service.motion_start()
    service.motion_start()
    service.motion_stop()
    service.motion_stop()
    assert service.interlock.state == RunState.SETTLING


def test_camera_error_clears_results_and_disables_compute():
    clock = Clock()
    service, source, pipeline = _service(clock)
    source.read = lambda: (_ for _ in ()).throw(RuntimeError("unplugged"))
    response = service.handle({"type": "detect"})
    assert response["status"] == "CAMERA_ERROR"
    assert response["results"] == []
    assert response["health"]["can_compute"] is False
    assert response["health"]["can_pick"] is False
    assert response["health"]["reason"] == "camera_error"
    assert pipeline.resets == 1
