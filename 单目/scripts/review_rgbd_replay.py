"""Replay fixed RGB-D batches with explicit provenance for before/after review."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import cv2

from sorting_vision.camera import load_rgbd_frame
from sorting_vision.cli import _write_rgbd_results
from sorting_vision.config import load_config
from sorting_vision.geometry_rgbd_model import DepthGeometryModel
from sorting_vision.pipeline3d import VisionPipeline3D
from sorting_vision.rgbd_review import build_rgbd_review
from sorting_vision.rgbd_dataset import load_rgbd_dataset_entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--batch', action='append', required=True)
    parser.add_argument('--background-dir', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--legacy-roi', action='store_true', help='ablation only: disable saved workspace bounds')
    parser.add_argument('--single-samples', action='store_true', help='review ordinals 1,15,30 per class from a single-object manifest')
    args = parser.parse_args()
    target = Path(args.output_dir)
    if target.exists():
        raise FileExistsError(f'refusing to overwrite: {target}')
    cv2.setNumThreads(1)
    config = load_config(args.config)
    background = load_rgbd_frame(args.background_dir)
    model = DepthGeometryModel.load(args.model)
    pipeline = VisionPipeline3D(config=config, background_frame=background, shape_model=model)
    if args.legacy_roi:
        pipeline.calibration = replace(pipeline.calibration, tray_roi_polygon=None)
    target.mkdir(parents=True)
    pipeline.calibration.save(target / 'calibration.json')
    rows = []
    for batch in args.batch:
        if args.single_samples:
            entries = [e for e in load_rgbd_dataset_entries(args.data_root)
                       if e['batch_id'] == batch and e['label_id'] != 'empty_tray']
            counts = {}
            captures = []
            for entry in entries:
                label = entry['label_id']
                counts[label] = counts.get(label, 0) + 1
                if counts[label] in (1, 15, 30):
                    captures.append(Path(entry['absolute_sample_dir']))
        else:
            captures = [p.parent for p in sorted((Path(args.data_root) / batch).rglob('scene.json'))]
        for index, source in enumerate(captures, 1):
            frame = load_rgbd_frame(source)
            pipeline.reset_tracking()
            start = time.perf_counter()
            results = pipeline.process(frame)
            elapsed = (time.perf_counter() - start) * 1000
            output = target / batch / f'frame-{index:02d}'
            _write_rgbd_results(output, frame, pipeline, results)
            rows.append({'source': str(source.resolve()), 'frame': f'{batch}/frame-{index:02d}',
                         'frame_id': frame.frame_id, 'processing_ms': elapsed,
                         'shapes': [r.shape_id for r in results]})
            print(f'{batch}/{index:02d}: {len(results)} objects', flush=True)
    report = build_rgbd_review(target, target / 'review')
    manifest = {
        'evaluation': 'same-batch development replay; not independent holdout',
        'duplicate_removal': False,
        'legacy_roi': args.legacy_roi,
        'single_samples': args.single_samples,
        'model_sha256': hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        'background_dir': str(Path(args.background_dir).resolve()),
        'frame_count': len(rows), 'frames': rows,
        'candidate_count': report['candidate_count'],
        'unknown_shape_count': report['unknown_shape_count'],
    }
    (target / 'replay.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
