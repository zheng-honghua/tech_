import json

import cv2
import pytest

from scripts import dual_apriltag_calibrate as app
from sorting_vision.apriltag_calibration import detect_apriltags
from sorting_vision.config import load_config


def arguments(*extra):
    return app.build_parser().parse_args(["--platform-id", "temporary", "--tag-size-mm", "30", *extra])


def test_grouped_ids_are_shared_by_generation_and_calibration():
    args = arguments("--tag-ids", "10", "21", "35")
    assert app._generation_tag_ids(args) == ((10, 21), 35)
    assert app._resolved_tag_ids(args, load_config("config/dual/temporary.yaml")) == ((10, 21), 35)


def test_interactive_input_retries_invalid_ids_and_accepts_chinese_commas(capsys):
    answers = iter(["10 10 35", "10，21，35"])
    result = app._generation_tag_ids(arguments(), input_fn=lambda prompt: next(answers))
    assert result == ((10, 21), 35)
    assert "ID无效" in capsys.readouterr().out


def test_interactive_generation_can_be_cancelled():
    with pytest.raises(ValueError, match="cancelled"):
        app._generation_tag_ids(arguments(), input_fn=lambda prompt: "q")


@pytest.mark.parametrize("extra", [
    ["--fixed-tag-a", "10"],
    ["--tag-ids", "10", "10", "35"],
    ["--tag-ids", "10", "21", "999999"],
    ["--tag-ids", "10", "21", "35", "--free-tag", "50"],
])
def test_invalid_or_conflicting_id_selection_creates_no_assets(tmp_path, extra):
    target = tmp_path / "assets"
    assert app.main(["--platform-id", "temporary", "--tag-size-mm", "30",
                     "--generate-tags-dir", str(target), *extra]) == 1
    assert not target.exists()


def test_noninteractive_generation_requires_explicit_ids(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(app.sys.stdin, "isatty", lambda: False)
    assert app.main(["--platform-id", "temporary", "--tag-size-mm", "30",
                     "--generate-tags-dir", str(tmp_path / "assets")]) == 1
    assert "--tag-ids" in capsys.readouterr().out


def test_custom_assets_are_detectable_without_config_or_cameras(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("asset generation tried to load camera config or open a camera")
    monkeypatch.setattr(app, "load_config", forbidden)
    monkeypatch.setattr(app, "_camera_source", forbidden)
    assert app.main(["--platform-id", "temporary", "--tag-size-mm", "30", "--tag-ids", "10", "21", "35",
                     "--generate-tags-dir", str(tmp_path)]) == 0
    metadata = json.loads((tmp_path / "three-apriltag-printing.json").read_text(encoding="utf-8"))
    assert metadata["fixed_tag_ids"] == [10, 21] and metadata["free_tag_id"] == 35
    for tag_id in (10, 21, 35):
        path = tmp_path / f"apriltag-{tag_id}-dict_apriltag_36h11.png"
        assert detect_apriltags(cv2.imread(str(path))).has(tag_id)
