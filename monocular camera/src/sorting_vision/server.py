from __future__ import annotations

import json
import socketserver
import threading
from typing import Any

import cv2
import numpy as np

from .camera import ColorSource, RGBDSource
from .interlock import MotionInterlock, RunState
from .pipeline3d import VisionPipeline3D
from .rgb_development import RGBDevelopmentPipeline


VisionPipelineType = VisionPipeline3D | RGBDevelopmentPipeline
VisionSourceType = RGBDSource | ColorSource


class VisionService3D:
    def __init__(
        self,
        pipeline: VisionPipelineType,
        source: VisionSourceType,
        interlock: MotionInterlock | None = None,
        mode: str | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.source = source
        self.interlock = interlock or MotionInterlock()
        self.mode = mode or (
            "RGB_ONLY" if isinstance(pipeline, RGBDevelopmentPipeline) else "RGBD"
        )
        self._latest_frame: Any | None = None
        self._latest_results: list[Any] = []
        self._source_error: str | None = None
        self._lock = threading.RLock()

    def update(self) -> list[Any]:
        """Read one frame for preview and compute only when the interlock is ready."""
        with self._lock:
            try:
                frame = self.source.read()
            except Exception as error:
                self._source_error = f"{type(error).__name__}: {error}"
                self._latest_results = []
                self.pipeline.reset_tracking()
                return []
            self._source_error = None
            self._latest_frame = frame
            state = self.interlock.observe(frame.color_bgr)
            if state != RunState.READY:
                self._latest_results = []
                return []
            self._latest_results = self.pipeline.process(frame)
            return list(self._latest_results)

    def motion_start(self) -> None:
        with self._lock:
            self.interlock.motion_start()
            self.pipeline.reset_tracking()
            self._latest_results = []

    def motion_stop(self) -> None:
        with self._lock:
            self.interlock.motion_stop()

    def preview_image(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_frame is None:
                return None
            if self.interlock.state == RunState.READY and self._latest_results:
                canvas = self.pipeline.annotate(self._latest_frame, self._latest_results)
            else:
                canvas = self._latest_frame.color_bgr.copy()
            if self.interlock.state != RunState.READY:
                cv2.putText(
                    canvas,
                    f"{self.interlock.state.value} - PREVIEW ONLY",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            return canvas

    def health(self) -> dict[str, Any]:
        pipeline_health = self.pipeline.health()
        ready = self.interlock.state == RunState.READY
        if self._source_error is not None:
            reason = "camera_error"
        elif not ready:
            reason = self.interlock.reason
        elif self.mode == "RGB_ONLY":
            reason = "depth_required"
        else:
            reason = str(pipeline_health.get("reason", "ready"))
        health = dict(pipeline_health)
        health.update(
            {
                "ok": False
                if self._source_error is not None
                else (
                    bool(pipeline_health.get("ok", True))
                    if ready
                    else self.interlock.status != "MOTION_UNSTABLE"
                ),
                "mode": self.mode,
                "run_state": self.interlock.state.value,
                "can_compute": ready and self._source_error is None,
                "can_pick": ready
                and self.mode == "RGBD"
                and self._source_error is None
                and bool(pipeline_health.get("ok", False)),
                "reason": reason,
            }
        )
        if self._source_error is not None:
            health["camera_error"] = self._source_error
        return health

    def response_status(self) -> str:
        return "CAMERA_ERROR" if self._source_error is not None else self.interlock.status

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = request.get("type")
        request_id = request.get("request_id")
        if request_type == "detect":
            results = self.update()
            return {
                "schema_version": 2,
                "type": "detect_result",
                "request_id": request_id,
                "status": self.response_status(),
                "health": self.health(),
                "results": [item.to_dict() for item in results],
            }
        if request_type == "motion_start":
            self.motion_start()
            return {
                "schema_version": 2,
                "type": "motion_start_result",
                "request_id": request_id,
                "ok": True,
                "health": self.health(),
            }
        if request_type == "motion_stop":
            self.motion_stop()
            return {
                "schema_version": 2,
                "type": "motion_stop_result",
                "request_id": request_id,
                "ok": True,
                "health": self.health(),
            }
        if request_type == "ack_pick":
            with self._lock:
                self.pipeline.acknowledge_pick()
                self._latest_results = []
            return {
                "schema_version": 2,
                "type": "ack_pick_result",
                "request_id": request_id,
                "ok": True,
            }
        if request_type == "health":
            return {
                "schema_version": 2,
                "type": "health_result",
                "request_id": request_id,
                "health": self.health(),
            }
        return {
            "schema_version": 2,
            "type": "error",
            "request_id": request_id,
            "error": "unsupported_request_type",
        }


class _ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve_json_tcp(
    service: VisionService3D,
    host: str,
    port: int,
) -> None:
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            for raw_line in self.rfile:
                try:
                    request = json.loads(raw_line.decode("utf-8"))
                    response = service.handle(request)
                except Exception as error:
                    response = {
                        "schema_version": 2,
                        "type": "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                self.wfile.write(
                    (json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
                )

    with _ThreadingServer((host, port), Handler) as server:
        server.serve_forever()
