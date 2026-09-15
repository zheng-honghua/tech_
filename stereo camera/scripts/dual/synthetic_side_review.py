from __future__ import annotations

import argparse

from sorting_vision.shape_registry import load_shape_registry
from sorting_vision.synthetic_side import build_synthetic_side_contact_sheet


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a synthetic side-view smoke-test sheet")
    parser.add_argument("--shape-registry", default="config/shapes/competition-11.yaml")
    parser.add_argument("--output", default="output/review/dual-side-synthetic.png")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    target = build_synthetic_side_contact_sheet(load_shape_registry(args.shape_registry), args.output, seed=args.seed)
    print(target.resolve())
    print("SYNTHETIC_ONLY: do not use for calibration, probability thresholds, or accuracy claims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
