from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np

from .config import VisionConfig
from .pipeline3d import VisionPipeline3D
from .synthetic3d import SyntheticSolid, make_rgbd_scene


SOLID_CLASSES = [
    ("red", "cube", (46, 46)),
    ("yellow", "cuboid", (64, 38)),
    ("blue", "triangular_prism", (50, 50)),
    ("green", "cylinder", (48, 48)),
    ("black", "pentagonal_prism", (50, 50)),
    ("cyan", "sphere", (48, 48)),
]


@dataclass(frozen=True)
class BenchmarkReport3D:
    rounds: int
    objects: int
    detected: int
    correct: int
    selected_rounds: int
    unsafe_selections: int
    median_ms: float
    p95_ms: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "rounds": self.rounds,
            "objects": self.objects,
            "detected": self.detected,
            "correct": self.correct,
            "classification_accuracy": round(self.correct / max(1, self.objects), 6),
            "selected_rounds": self.selected_rounds,
            "unsafe_selections": self.unsafe_selections,
            "median_ms": round(self.median_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "synthetic_only": True,
        }


def run_rgbd_benchmark(
    config: VisionConfig,
    rounds: int = 30,
    seed: int = 11,
) -> BenchmarkReport3D:
    generator = random.Random(seed)
    detected = correct = selected_rounds = unsafe_selections = 0
    timings: list[float] = []
    for round_index in range(rounds):
        solids: list[SyntheticSolid] = []
        for index in range(12):
            color, shape, size = SOLID_CLASSES[index % len(SOLID_CLASSES)]
            column, row = index % 4, index // 4
            solids.append(
                SyntheticSolid(
                    color,
                    shape,
                    (
                        100 + column * 145 + generator.randint(-5, 5),
                        100 + row * 140 + generator.randint(-5, 5),
                    ),
                    size,
                    25.0 + generator.uniform(-2.0, 3.0),
                    generator.uniform(-30.0, 30.0),
                    generator.uniform(-7.0, 7.0),
                    generator.uniform(-7.0, 7.0),
                )
            )
        background, scene = make_rgbd_scene(
            solids,
            noise_std_mm=0.08,
            seed=seed + round_index,
            frame_id=f"benchmark-{round_index}",
        )
        pipeline = VisionPipeline3D(config=config, background_frame=background)
        # Measure steady-state service latency; camera and OpenCV kernels are
        # warmed once at startup before competition timing begins.
        pipeline.process(background)
        started = time.perf_counter()
        first = pipeline.process(scene)
        split = time.perf_counter()
        results = pipeline.process(scene)
        finished = time.perf_counter()
        timings.extend([(split - started) * 1000.0, (finished - split) * 1000.0])

        expected = sorted(f"{item.color_id}:{item.shape_id}" for item in solids)
        actual = sorted(item.class_key for item in results)
        detected += len(actual)
        if len(expected) == len(actual):
            correct += sum(left == right for left, right in zip(expected, actual))
        selected = [item for item in results if item.selected]
        selected_rounds += int(len(selected) == 1)
        unsafe_selections += sum(
            item.status.value != "PICKABLE" or item.pose_3d is None or item.grasp is None
            for item in selected
        )

    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)
    return BenchmarkReport3D(
        rounds=rounds,
        objects=rounds * 12,
        detected=detected,
        correct=correct,
        selected_rounds=selected_rounds,
        unsafe_selections=unsafe_selections,
        median_ms=float(np.median(timings)),
        p95_ms=float(ordered[p95_index]),
    )
