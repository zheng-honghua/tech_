"""Select depth-feature classifiers on a validation batch; report held-out frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sorting_vision.geometry_rgbd_model import DepthGeometryModel


def forest_fit(x, y, depth):
    cv2.setRNGSeed(20260905)
    forest = cv2.ml.RTrees_create()
    forest.setMaxDepth(depth)
    forest.setMinSampleCount(2)
    forest.setActiveVarCount(0)
    forest.setMaxCategories(32)
    forest.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 200, 0))
    forest.train(x.astype(np.float32), cv2.ml.ROW_SAMPLE, y.astype(np.int32))
    return forest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    args = p.parse_args()
    root = Path(args.cache)
    rows = [r for r in json.loads((root / 'frames.json').read_text(encoding='utf-8')) if 'feature_index' in r]
    x = np.load(root / 'features.npz')['features']
    labels = sorted(set(r['label_id'] for r in rows))
    y = np.array([labels.index(r['label_id']) for r in rows])
    train = np.array([r['batch_id'] == 'pilot-01' or (r['batch_id'] == 'pilot-03' and r['ordinal'] <= 20) for r in rows])
    val = np.array([r['batch_id'] == 'pilot-02' for r in rows])
    test = np.array([r['batch_id'] == 'pilot-03' and r['ordinal'] > 20 for r in rows])
    report = {'split': {'train': 'pilot-01 + pilot-03 ordinal 1..20 per class', 'validation': 'pilot-02', 'test': 'pilot-03 ordinal 21..30; same-session temporal holdout'}, 'sizes': {k: int(v.sum()) for k, v in [('train', train), ('validation', val), ('test', test)]}, 'candidates': []}
    baseline = np.array([labels.index(r['prediction_v4']) if r['prediction_v4'] in labels else -1 for r in rows])
    print('sizes', report['sizes'], flush=True)
    for name, pred in [('v4_runtime', baseline)]:
        report['candidates'].append({'name': name, **{k: {'correct': int((pred[v] == y[v]).sum()), 'total': int(v.sum()), 'unknown': int((pred[v] == -1).sum())} for k,v in [('validation',val),('test',test)]}})
    for k in (1, 3, 5):
        model = DepthGeometryModel.fit(x[train], [labels[i] for i in y[train]])
        model.neighbors = k
        pred = [model.predict_features(v)[0] for v in x]
        pred = np.array([labels.index(v) if v in labels else -1 for v in pred])
        report['candidates'].append({'name': f'aligned_knn_{k}', **{key: {'correct': int((pred[v] == y[v]).sum()), 'total': int(v.sum()), 'unknown': int((pred[v] == -1).sum())} for key,v in [('validation',val),('test',test)]}})
    base = DepthGeometryModel.load('models/stable/rgbd/geometry-rgbd-multipose-v4.npz')
    bx, by = base.training_data()
    model = DepthGeometryModel.fit(np.vstack((bx, x[train])), by + [labels[i] for i in y[train]])
    pred = [model.predict_features(v)[0] for v in x]
    pred = np.array([labels.index(v) if v in labels else -1 for v in pred])
    report['candidates'].append({'name': 'v4_plus_aligned_train', **{key: {'correct': int((pred[v] == y[v]).sum()), 'total': int(v.sum()), 'unknown': int((pred[v] == -1).sum())} for key,v in [('validation',val),('test',test)]}})
    model.save('models/experimental/rgbd/geometry-rgbd-runtime-aligned-candidate-next.npz', metadata={'split': report['split'], 'base': 'v4; legacy provenance, not independent validation', 'status': 'experimental strict ROI features; not deployed'})
    report['baseline_provenance_warning'] = 'v4 historical exemplars may overlap old batches; no independent holdout claim for v4 or its augmentation'
    report['forest_available'] = hasattr(cv2, 'ml')
    for depth in ((6, 12, 20) if hasattr(cv2, 'ml') else ()):
        model = forest_fit(x[train], y[train], depth)
        votes = model.getVotes(x.astype(np.float32), 0)
        probability = votes[1:].astype(float) / votes[1:].sum(axis=1, keepdims=True)
        order = np.argsort(probability, axis=1)
        top = probability[np.arange(len(x)), order[:, -1]]
        gap = top - probability[np.arange(len(x)), order[:, -2]]
        pred = votes[0][np.argmax(probability, axis=1)]
        for gate in (0.0, 0.45, 0.55):
            accepted = (top >= gate) & (gap >= (0 if gate == 0 else .1))
            selected = np.where(accepted, pred, -1)
            report['candidates'].append({'name': f'forest_{depth}_gate{gate}', **{key: {'correct': int((selected[v] == y[v]).sum()), 'total': int(v.sum()), 'unknown': int((selected[v] == -1).sum())} for key,v in [('validation',val),('test',test)]}})
    (root / 'learners.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
