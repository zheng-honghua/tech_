"""Train from original labelled single-object captures, without legacy exemplars."""
import argparse
import json
from pathlib import Path
import cv2
from sorting_vision.geometry_rgbd_model import train_rgbd_geometry_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/rgbd-pilot')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    cv2.setNumThreads(1)
    report = train_rgbd_geometry_model(args.data_root, args.output)
    report['provenance'] = 'original single-object captures only; no mixed-scene labels or base model'
    Path(args.output).with_suffix('.report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
