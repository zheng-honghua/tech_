from __future__ import annotations

import time
from enum import Enum
from typing import Callable

import cv2
import numpy as np

from .config import MotionInterlockConfig


class RunState(str, Enum):
    MOVING = "MOVING"
    SETTLING = "SETTLING"
    READY = "READY"


class MotionInterlock:
    """Prevents vision computation until the tool has stopped and the image is stable."""

    def __init__(
        self,
        config: MotionInterlockConfig | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.config = config or MotionInterlockConfig()
        self._clock = monotonic_ns
        self.state = RunState.READY
        self._stop_ns = 0
        self._discarded = 0
        self._stable = 0
        self._previous_gray: np.ndarray | None = None
        self._timed_out = False

    def motion_start(self) -> None:
        self.state = RunState.MOVING
        self._reset_settling()

    def motion_stop(self) -> None:
        if self.state != RunState.MOVING:
            return
        self.state = RunState.SETTLING
        self._stop_ns = self._clock()
        self._reset_settling(keep_stop_time=True)

    def observe(self, color_bgr: np.ndarray) -> RunState:
        if self.state != RunState.SETTLING:
            return self.state
        elapsed_ms = (self._clock() - self._stop_ns) / 1_000_000.0
        if elapsed_ms > self.config.timeout_ms:
            self._timed_out = True
            return self.state

        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
        if self._discarded < self.config.discard_frames:
            self._discarded += 1
            self._previous_gray = None
            return self.state
        if elapsed_ms < self.config.min_settle_ms:
            self._previous_gray = gray
            return self.state
        if self._previous_gray is None:
            self._previous_gray = gray
            return self.state

        difference = float(cv2.absdiff(gray, self._previous_gray).mean())
        self._previous_gray = gray
        if difference <= self.config.frame_diff_threshold:
            self._stable += 1
        else:
            self._stable = 0
        if self._stable >= self.config.stable_frames:
            self.state = RunState.READY
            self._timed_out = False
        return self.state

    @property
    def status(self) -> str:
        if self.state == RunState.MOVING:
            return "BUSY_MOVING"
        if self.state == RunState.SETTLING:
            return "MOTION_UNSTABLE" if self._timed_out else "SETTLING"
        return "OK"

    @property
    def reason(self) -> str:
        return {
            "BUSY_MOVING": "tool_moving",
            "SETTLING": "settling",
            "MOTION_UNSTABLE": "motion_unstable",
            "OK": "ready",
        }[self.status]

    def _reset_settling(self, keep_stop_time: bool = False) -> None:
        if not keep_stop_time:
            self._stop_ns = 0
        self._discarded = 0
        self._stable = 0
        self._previous_gray = None
        self._timed_out = False
