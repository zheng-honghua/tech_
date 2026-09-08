"""Offline supervised depth-feature probe; never publishes a runtime model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(args.cache)
    rows = [r for r in json.loads((root / 'frames.json').read_text(encoding='utf-8')) if 'feature_index' in r]
    x = np.load(root / 'features.npz')['features'].astype(np.float64)
    labels = sorted(set(r['label_id'] for r in rows))
    y = np.array([labels.index(r['label_id']) for r in rows])
    train = np.array([r['batch_id'] == 'pilot-01' or (r['batch_id'] == 'pilot-03' and r['ordinal'] <= 20) for r in rows])
    val = np.array([r['batch_id'] == 'pilot-02' for r in rows])
    test = np.array([r['batch_id'] == 'pilot-03' and r['ordinal'] > 20 for r in rows])
    scale = x[train].std(axis=0)
    scale[scale < 1e-4] = 1
    z = (x - x[train].mean(axis=0)) / scale
    targets = np.eye(len(labels))[y[train]]
    report = []

    def score(name, scores):
        order = scores.argsort(axis=1)
        best = scores[np.arange(len(x)), order[:, -1]]
        margin = best - scores[np.arange(len(x)), order[:, -2]]
        for gate in (0., .55, .68):
            pred = np.where((best >= gate) & (margin >= (0. if not gate else .1)), order[:, -1], -1)
            report.append({'name': name, 'gate': gate, **{k: {'correct': int((pred[v] == y[v]).sum()), 'unknown': int((pred[v] == -1).sum()), 'total': int(v.sum())} for k, v in [('validation', val), ('temporal_holdout', test)]}})

    # These fixed candidate widths and regularizers are selected only on validation.
    for group, weights in [('all', np.ones(x.shape[1])), ('base', np.r_[np.ones(20), np.full(x.shape[1]-20, .25)])]:
        distance = np.mean(((z[:, None, :] - z[train][None, :, :]) * weights) ** 2, axis=2)
        for width in (.25, 1., 4.):
            kernel = np.exp(-distance / (2 * width))
            for regularizer in (.01, .1, 1.):
                coef = np.linalg.solve(kernel[train] + np.eye(train.sum()) * regularizer, targets)
                score(f'rbf_{group}_{width}_{regularizer}', kernel @ coef)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for r in sorted(report, key=lambda r: (r['validation']['correct'], -r['validation']['unknown']), reverse=True)[:12]:
        print(r)


if __name__ == '__main__':
    main()
