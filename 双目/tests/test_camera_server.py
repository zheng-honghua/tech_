from sorting_vision.camera import FileRGBDSource, load_rgbd_frame, save_rgbd_frame
from sorting_vision.pipeline3d import VisionPipeline3D
from sorting_vision.server import VisionService3D
from sorting_vision.synthetic3d import competition_rgbd_demo


def test_file_rgbd_source_round_trip(tmp_path):
    _, scene, _ = competition_rgbd_demo()
    directory = tmp_path / "frame"
    save_rgbd_frame(scene, directory)
    loaded = load_rgbd_frame(directory)
    assert loaded.frame_id == scene.frame_id
    assert loaded.color_bgr.shape == scene.color_bgr.shape
    assert loaded.depth.dtype == scene.depth.dtype


def test_json_service_detect_health_and_ack(tmp_path):
    background, scene, _ = competition_rgbd_demo()
    directory = tmp_path / "frame"
    save_rgbd_frame(scene, directory)
    pipeline = VisionPipeline3D(background_frame=background)
    service = VisionService3D(pipeline, FileRGBDSource([directory], loop=True))

    first = service.handle({"type": "detect", "request_id": "1"})
    second = service.handle({"type": "detect", "request_id": "2"})
    assert first["schema_version"] == 2
    assert second["health"]["ok"] is True
    assert sum(item["selected"] for item in second["results"]) == 1
    assert service.handle({"type": "ack_pick", "request_id": "3"})["ok"] is True
    assert service.handle({"type": "health"})["type"] == "health_result"

