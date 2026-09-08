from sorting_vision.config import load_config
from sorting_vision.evaluation3d import run_rgbd_benchmark


def test_rgbd_benchmark_is_repeatable_and_safe():
    report = run_rgbd_benchmark(load_config(), rounds=2, seed=11)
    assert report.objects == 24
    assert report.detected == 24
    assert report.correct == 24
    assert report.selected_rounds == 2
    assert report.unsafe_selections == 0
