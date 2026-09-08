"""Cache runtime-aligned depth features; offline only, no camera or motion."""
from __future__ import annotations

import argparse
from dataclasses import replace
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from sorting_vision.camera import load_rgbd_frame
from sorting_vision.config import load_config
from sorting_vision.geometry_rgbd_model import (
    DepthGeometryModel, extract_rgbd_geometry_features, detect_tray_roi_mask,
)
from sorting_vision.rgbd_dataset import load_rgbd_dataset_entries
from sorting_vision.rgbd import resize_rgbd_frame
from sorting_vision.pipeline3d import VisionPipeline3D
import sorting_vision.pipeline3d as pipeline_module


class FeatureRecorder:
    def __init__(self, model, fused_edges=False):
        self.model = model
        self.fused_edges = fused_edges
        self.features = []
        self.last_diagnostics = {}
        self.cached = None
        self.cursor = 0
        self.single = False

    def classify(self, points, rgb, depth, mask, intrinsics, origin):
        if self.cached is not None:
            if self.single:
                value = self.cached[self.cursor]
                self.cursor += 1
            else:
                self.features.append(None)
                return 'unknown', 0.0
        else:
            value = extract_rgbd_geometry_features(
                points, depth, mask, intrinsics, origin, rgb,
                include_fused_edges=self.fused_edges,
            )
        self.features.append(value)
        prediction = self.model.predict_features(value)
        return prediction[0], prediction[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--anchor-roi', action='store_true')
    parser.add_argument('--adaptive-reference-roi', action='store_true', help='use the current calibrated reference-area constraint')
    parser.add_argument('--reuse-features', action='store_true', help='recover an interrupted JSON export from a completed feature array')
    parser.add_argument('--fused-edges', action='store_true', help='append registered RGB/depth edge features')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=args.reuse_features)
    cv2.setNumThreads(1)
    # Feature audit deliberately omits suction search, which cannot affect shape.
    pipeline_module.find_suction_grasp = lambda *a, **kw: None
    cfg = load_config('config/d415-reviewed-20260905.yaml')
    model = DepthGeometryModel.load('models/stable/rgbd/geometry-rgbd-multipose-v4.npz')
    entries = load_rgbd_dataset_entries('data/rgbd-pilot')
    backgrounds = {}
    for e in entries:
        if e['label_id'] == 'empty_tray' and e['batch_id'] not in backgrounds:
            backgrounds[e['batch_id']] = load_rgbd_frame(e['absolute_sample_dir'])
    rows, vectors = [], []
    seen = set()
    cached = np.load(out / 'features.npz')['features'] if args.reuse_features else None
    cache_cursor = 0
    original_segment = pipeline_module.segment_depth_objects
    for batch, bg in backgrounds.items():
        recorder = FeatureRecorder(model, args.fused_edges)
        recorder.cached = cached
        recorder.cursor = cache_cursor
        def record_count(*a, **kw):
            objects, mask = original_segment(*a, **kw)
            recorder.single = len(objects) == 1
            return objects, mask
        pipeline_module.segment_depth_objects = record_count
        pipeline = VisionPipeline3D(cfg, background_frame=bg, shape_model=recorder)
        if not args.anchor_roi and not args.adaptive_reference_roi:
            pipeline.calibration = replace(pipeline.calibration, tray_roi_polygon=None)
        reference = detect_tray_roi_mask(resize_rgbd_frame(bg, cfg.rgbd.processing_scale).color_bgr)
        if args.anchor_roi:
            pipeline_module.detect_tray_roi_mask = lambda rgb: cv2.bitwise_and(detect_tray_roi_mask(rgb), reference)
        indices = Counter()
        for e in [v for v in entries if v['batch_id'] == batch and v['label_id'] != 'empty_tray']:
            indices[e['label_id']] += 1
            row = {k: e[k] for k in ('batch_id', 'label_id', 'sample_dir', 'frame_id')}
            row['ordinal'] = indices[e['label_id']]
            try:
                source = Path(e['absolute_sample_dir'])
                digest = hashlib.sha256((source / 'color.png').read_bytes() + (source / 'depth.npy').read_bytes()).hexdigest()
                row['pair_sha256'] = digest
                if digest in seen:
                    raise ValueError('duplicate_rgbd_pair')
                seen.add(digest)
                frame = load_rgbd_frame(source)
                recorder.features = []
                pipeline.reset_tracking()
                result = pipeline.process(frame)
                row['count'] = len(result)
                row['health'] = pipeline.health()
                if len(result) != 1 or len(recorder.features) != 1:
                    raise ValueError('not_single_runtime_instance')
                row['prediction_v4'] = result[0].shape_id
                row['feature_index'] = len(vectors)
                vectors.append(recorder.features[0])
            except (ValueError, OSError) as exc:
                row['error'] = str(exc)
            rows.append(row)
            if len(rows) % 25 == 0:
                print(f'{len(rows)} frames; {len(vectors)} single instances', flush=True)
        pipeline_module.detect_tray_roi_mask = detect_tray_roi_mask
        cache_cursor = recorder.cursor
    if cached is not None and cache_cursor != len(cached):
        raise ValueError('cached feature count does not match segmentation replay')
    np.savez_compressed(out / 'features.npz', features=np.asarray(vectors))
    (out / 'frames.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=lambda x: x.item()), encoding='utf-8')
    print(Counter(r.get('error', 'ok') for r in rows), flush=True)


if __name__ == '__main__':
    main()
