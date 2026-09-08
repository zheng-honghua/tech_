from sorting_vision.config import load_config
from sorting_vision.evaluation import run_synthetic_benchmark


def test_repeatable_synthetic_benchmark():
    report = run_synthetic_benchmark(load_config(), rounds=2, seed=7)
    assert report.objects == 24
    assert report.detected == 24
    assert report.correct == 24
    assert report.selected_rounds == 2
