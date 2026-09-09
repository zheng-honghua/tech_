"""Compare multi-02/03 using the existing, provisional appearance assignments.

These assignments are only evaluation bookkeeping for these reviewed batches.
They must never be used as a general geometry classifier or independent truth.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2

from prepare_reviewed_multi_02_03 import _appearance, _assign
from sorting_vision.rgbd_review import _review_tile, _write_contact_sheet


def evaluate(root: Path, output: Path, tag: str) -> dict:
    rows = []
    totals, correct = Counter(), Counter()
    excluded = []
    tiles = {}
    for file in sorted(root.glob('multi-*/frame-*/results-v2.json')):
        batch = file.parent.parent.name
        predictions = json.loads(file.read_text(encoding='utf-8'))
        try:
            assignments = _assign(batch, [_appearance(p, file.parent) for p in predictions])
        except ValueError as error:
            excluded.append({'frame': f'{batch}/{file.parent.name}', 'reason': str(error)})
            for p in predictions:
                crop = cv2.imread(str(file.parent / p['crop_image']))
                tiles.setdefault(batch, []).append(_review_tile(
                    crop, f'{file.parent.name} LABEL UNVERIFIED',
                    f"pred={p['shape_id']} {p['confidence']['shape']:.2f}", 320, 240,
                ))
            continue
        for appearance, truth in assignments:
            p = appearance['item']
            totals[truth] += 1
            correct[truth] += p['shape_id'] == truth
            feature = p['diagnostics']['shape_features']
            row = {'frame': f'{batch}/{file.parent.name}', 'object_id': p['object_id'],
                   'expected': truth, 'predicted': p['shape_id'],
                   'color': p['color_id'], 'confidence': p['confidence']['shape'],
                   'rejection': {k: v for k, v in feature.items() if 'rejected' in k},
                   'bbox': p['bbox_px']}
            rows.append(row)
            crop = cv2.imread(str(file.parent / p['crop_image']))
            title = f"{file.parent.name} {truth}"
            summary = f"pred={p['shape_id']} {p['confidence']['shape']:.2f}"
            tiles.setdefault(batch, []).append(_review_tile(crop, title, summary, 320, 240))
    for batch, images in tiles.items():
        _write_contact_sheet(images, output / f'{tag}-{batch}-objects.jpg', 4, 320, 240)
    report = {'provisional_reviewed_labels': True, 'same_batch_only': True,
              'excluded_frames': excluded, 'whole_dataset_accuracy_available': False,
              'correct': sum(correct.values()), 'total': sum(totals.values()),
              'recall': {k: {'correct': correct[k], 'total': v} for k, v in totals.items()},
              'rows': rows}
    (output / f'{tag}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True)
    parser.add_argument('--after', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for tag in ('before', 'after'):
        report = evaluate(Path(getattr(args, tag)), output, tag)
        print(tag, json.dumps({k: v for k, v in report.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
