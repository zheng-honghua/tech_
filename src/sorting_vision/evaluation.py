from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np

from .calibration import PerspectiveCalibration
from .config import VisionConfig
from .pipeline import VisionPipeline
from .synthetic import SyntheticObject, make_scene


BENCHMARK_CLASSES = [
    ("red", "triangle", (72, 72)),
    ("yellow", "square", (72, 72)),
    ("blue", "rectangle", (94, 58)),
    ("green", "pentagon", (72, 72)),
    ("black", "hexagon", (72, 72)),
    ("cyan", "circle", (72, 72)),
]


@dataclass(frozen=True)
class BenchmarkReport:
    rounds: int
    objects: int
    detected: int
    correct: int
    selected_rounds: int
    median_ms: float
    p95_ms: float

    @property
    def classification_accuracy(self) -> float:
        return self.correct / max(1, self.objects)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "rounds": self.rounds,
            "objects": self.objects,
            "detected": self.detected,
            "correct": self.correct,
            "classification_accuracy": round(self.classification_accuracy, 6),
            "selected_rounds": self.selected_rounds,
            "median_ms": round(self.median_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
        }


def run_synthetic_benchmark(
    config: VisionConfig,
    rounds: int = 30,
    seed: int = 7,
) -> BenchmarkReport:
    """Repeatable smoke benchmark; it does not replace real-sample acceptance."""
    generator = random.Random(seed)
    correct = detected = selected_rounds = 0
    timings: list[float] = []
    source_size = 800
    calibration = PerspectiveCalibration(
        source_points=np.array(
            [[0, 0], [799, 0], [799, 799], [0, 799]], dtype=np.float32
        ),
        output_width_px=config.tray.rectified_width_px,
        output_height_px=config.tray.rectified_height_px,
        tray_width_mm=config.tray.width_mm,
        tray_height_mm=config.tray.height_mm,
    )

    for _ in range(rounds):
        truth: list[SyntheticObject] = []
        for index in range(12):
            color_id, shape_id, size = BENCHMARK_CLASSES[index % len(BENCHMARK_CLASSES)]
            column, row = index % 4, index // 4
            truth.append(
                SyntheticObject(
                    color_id,
                    shape_id,
                    (
                        105 + column * 195 + generator.randint(-8, 8),
                        125 + row * 260 + generator.randint(-8, 8),
                    ),
                    size,
                    generator.uniform(-35.0, 35.0),
                )
            )
        background, image = make_scene(truth, source_size, source_size)
        pipeline = VisionPipeline(config, calibration, background)
        started = time.perf_counter()
        pipeline.process(image)
        results = pipeline.process(image)
        timings.append((time.perf_counter() - started) * 500.0)  # per frame

        expected = sorted(f"{item.color_id}:{item.shape_id}" for item in truth)
        actual = sorted(result.class_key for result in results)
        detected += len(actual)
        if len(actual) == len(expected):
            correct += sum(left == right for left, right in zip(expected, actual))
        selected_rounds += int(sum(result.selected for result in results) == 1)

    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)
    return BenchmarkReport(
        rounds=rounds,
        objects=rounds * 12,
        detected=detected,
        correct=correct,
        selected_rounds=selected_rounds,
        median_ms=float(np.median(timings)),
        p95_ms=float(ordered[p95_index]),
    )

