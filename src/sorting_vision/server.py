from __future__ import annotations

import json
import socketserver
from typing import Any

from .camera import RGBDSource
from .pipeline3d import VisionPipeline3D


class VisionService3D:
    def __init__(self, pipeline: VisionPipeline3D, source: RGBDSource) -> None:
        self.pipeline = pipeline
        self.source = source

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = request.get("type")
        request_id = request.get("request_id")
        if request_type == "detect":
            frame = self.source.read()
            results = self.pipeline.process(frame)
            return {
                "schema_version": 2,
                "type": "detect_result",
                "request_id": request_id,
                "health": self.pipeline.health(),
                "results": [item.to_dict() for item in results],
            }
        if request_type == "ack_pick":
            self.pipeline.acknowledge_pick()
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
                "health": self.pipeline.health(),
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

