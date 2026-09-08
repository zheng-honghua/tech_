from sorting_vision.config import load_config
from sorting_vision.segmentation import segment_objects
from sorting_vision.synthetic import SyntheticObject, make_scene


def test_empty_tray_has_no_objects():
    config = load_config()
    background, scene = make_scene([])
    assert segment_objects(scene, background, config.segmentation) == []


def test_separated_objects_are_independent_instances():
    config = load_config()
    background, scene = make_scene(
        [
            SyntheticObject("red", "circle", (250, 400), (100, 100)),
            SyntheticObject("blue", "square", (550, 400), (100, 100)),
        ]
    )
    objects = segment_objects(scene, background, config.segmentation)
    assert len(objects) == 2
    assert all(item.clearance_px > config.segmentation.min_clearance_px for item in objects)


def test_touching_round_objects_are_split_and_marked_without_clearance():
    config = load_config()
    background, scene = make_scene(
        [
            SyntheticObject("red", "circle", (350, 400), (100, 100)),
            SyntheticObject("blue", "circle", (435, 400), (100, 100)),
        ]
    )
    objects = segment_objects(scene, background, config.segmentation)
    assert len(objects) == 2
    assert all(item.clearance_px < config.segmentation.min_clearance_px for item in objects)
