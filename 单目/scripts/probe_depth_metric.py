"""Evaluate group-weighted RGB-D distances with train-only normalization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sorting_vision.geometry_rgbd_model import DepthGeometryModel


def weighted_fit(x: np.ndarray, labels: list[str], weights: np.ndarray) -> DepthGeometryModel:
    return DepthGeometryModel.fit(x, labels, feature_weights=weights)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model-output', required=True, help='new candidate path; existing models are never overwritten')
    args = p.parse_args()
    if Path(args.model_output).exists():
        raise FileExistsError(args.model_output)
    root = Path(args.cache)
    rows = [r for r in json.loads((root / 'frames.json').read_text(encoding='utf-8')) if 'feature_index' in r]
    x = np.load(root / 'features.npz')['features']
    y = np.array([r['label_id'] for r in rows])
    train = np.array([r['batch_id'] == 'pilot-01' or (r['batch_id'] == 'pilot-03' and r['ordinal'] <= 20) for r in rows])
    val = np.array([r['batch_id'] == 'pilot-02' for r in rows])
    test = np.array([r['batch_id'] == 'pilot-03' and r['ordinal'] > 20 for r in rows])
    base = DepthGeometryModel.load('models/stable/rgbd/geometry-rgbd-multipose-v4.npz')
    bx, by = base.training_data()
    report, models = [], {}
    for source, tx, ty in [('aligned', x[train], y[train].tolist()), ('v4_plus_aligned', np.vstack((bx, x[train])), by + y[train].tolist())]:
        for silhouette in (.25, 1., 4.):
            for topology in (.25, 1., 4.):
                w = np.ones(x.shape[1])
                w[10:14] = silhouette
                w[20:] = topology
                name = f'{source}_sil{silhouette}_topo{topology}'
                model = weighted_fit(tx, ty, w)
                pred = np.array([model.predict_features(v)[0] for v in x])
                report.append({'name': name, **{k: {'correct': int((pred[m] == y[m]).sum()), 'unknown': int((pred[m] == 'unknown').sum()), 'wrong': int(((pred[m] != y[m]) & (pred[m] != 'unknown')).sum()), 'total': int(m.sum())} for k,m in [('validation', val), ('temporal_holdout', test)]}})
                models[name] = model
    ranked = sorted(report, key=lambda r: (r['validation']['correct'] - 2*r['validation']['wrong'], r['validation']['correct']), reverse=True)
    for r in ranked[:10]:
        print(r)
    best = ranked[0]['name']
    model_path = args.model_output
    models[best].save(model_path, {'candidate': best, 'status': 'experimental; not promoted', 'selection': 'validation correct minus twice wrong; no test-set selection', 'cache': str(root), 'legacy_base_leakage_warning': 'v4 may overlap old validation captures'})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'selected': best, 'model': model_path, 'candidates': report}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
