"""Compare reviewed RGB-D replays against scene composition, not instance truth."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def evaluate(root: Path) -> dict:
    replay = json.loads((root / 'replay.json').read_text(encoding='utf-8'))
    rows, seen = [], set()
    duplicates = 0
    for frame in replay['frames']:
        source = Path(frame['source'])
        digest = hashlib.sha256((source / 'color.png').read_bytes()
                                + (source / 'depth.npy').read_bytes()).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        scene = json.loads((source / 'scene.json').read_text(encoding='utf-8'))
        expected = Counter({o['label_id']: o['count'] for o in scene['objects']})
        actual = Counter(frame['shapes'])
        rows.append({'frame': frame['frame'], 'frame_id': frame['frame_id'],
                     'pair_sha256': digest, 'expected': dict(expected), 'predicted': dict(actual),
                     'count_match': sum(expected.values()) == sum(actual.values()),
                     'composition_match': expected == actual,
                     'label_count_overlap': sum((expected & actual).values())})
    return {'evaluation': 'development replay; composition labels only, no per-instance accuracy',
            'exact_rgbd_duplicates_removed': duplicates, 'frames': len(rows),
            'count_match': sum(r['count_match'] for r in rows),
            'composition_match': sum(r['composition_match'] for r in rows),
            'label_count_overlap': sum(r['label_count_overlap'] for r in rows),
            'expected_objects': sum(sum(r['expected'].values()) for r in rows),
            'predicted_objects': sum(sum(r['predicted'].values()) for r in rows),
            'unknown': sum(r['predicted'].get('unknown', 0) for r in rows), 'rows': rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True)
    parser.add_argument('--after', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reports = {tag: evaluate(Path(getattr(args, tag))) for tag in ('before', 'after')}
    if [r['pair_sha256'] for r in reports['before']['rows']] != [r['pair_sha256'] for r in reports['after']['rows']]:
        raise ValueError('before/after replay sources differ')
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding='utf-8')
    for tag, report in reports.items():
        print(tag, {k: v for k, v in report.items() if k != 'rows'})


if __name__ == '__main__':
    main()
