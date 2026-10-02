"""Archive the isolated v8 models, raw observation snapshot and frozen evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from experiment_color_geometry_v8 import ROOT, OUTPUT, V6, V7, write_json
from sorting_vision.visual_contract import LEGACY, V7 as COLOUR_RIDGE_CONTRACT
from sorting_vision.geometry_rgbd_model import DepthGeometryModel
from sorting_vision.side_geometry import SideGeometryModel
from sorting_vision.cross_view_topology import CrossViewTopologyModel
from sorting_vision.shape_registry import ShapeRegistry
from sorting_vision.color_v7 import CameraColorProfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    split = json.loads((OUTPUT / 'split-manifest.json').read_text(encoding='utf-8'))
    metrics = json.loads((OUTPUT / 'evaluation-metrics.json').read_text(encoding='utf-8'))
    review = json.loads((OUTPUT / 'review/application-visual-review.json').read_text(encoding='utf-8'))
    if split['geometry_model_contract'] != LEGACY or split['promotion_eligible'] or metrics['promotion_eligible']:
        raise ValueError('invalid v8 contract or promotion flag')
    if len(split['fit_ids']) != 348 or len(split['calibration_ids']) != 72 or len(split['regression_ids']) != 108:
        raise ValueError('frozen population changed')
    if set(split['fit_ids']) & set(split['calibration_ids']):
        raise ValueError('fit/calibration overlap')
    if metrics['application']['regression']['samples'] != 108 or metrics['application']['calibration']['samples'] != 72:
        raise ValueError('incomplete application evaluation')
    for path, signature in review['contact_hashes'].items():
        if sha(OUTPUT / path) != signature:
            raise ValueError('contact sheets changed after actual inspection: ' + path)
    registry = ShapeRegistry.load(ROOT / 'config/shapes/temporary-observed10.yaml')
    top = DepthGeometryModel.load(OUTPUT / 'top-rgbd.npz')
    side = SideGeometryModel.load(OUTPUT / 'side-lsd.npz', registry)
    joint = CrossViewTopologyModel.load(OUTPUT / 'joint-topology.npz', registry)
    if top.edge_parameters.get('feature_contract') != LEGACY or side.feature_contract != LEGACY or joint.feature_contract != LEGACY:
        raise ValueError('v7 semantic features must not enter legacy geometry models')
    for camera in ('primary', 'side'):
        CameraColorProfile.load(OUTPUT / f'{camera}-color-profile.json', camera)
        if sha(OUTPUT / f'{camera}-color-profile.json') != sha(V7 / f'{camera}-color-profile.json'):
            raise ValueError('reviewed fit-only v7 colour profile changed')
    previous_bundle = json.loads((V7 / 'bundle-manifest.json').read_text(encoding='utf-8'))
    capture_hashes = previous_bundle['capture_source_hashes']
    for identifier in split['training_candidates'] + split['regression_ids']:
        signature = hashlib.sha256(COLOUR_RIDGE_CONTRACT.encode())
        for name in ('metadata.json', 'primary-color.png', 'primary-depth.npy', 'side-color.png'):
            signature.update((ROOT / identifier / name).read_bytes())
        if signature.hexdigest() != capture_hashes[identifier]:
            raise ValueError('original capture changed since v7 review: ' + identifier)
    component = {'geometry_model': LEGACY, 'colour_and_ridges': COLOUR_RIDGE_CONTRACT,
                 'correspondence': 'legacy_mutual_unique', 'position': 'guarded_gauss_newton',
                 'ridge_tiers': 'recall__image_verified__geometry_and_image_supported'}
    component_fingerprint = 'colour_geometry_v8:' + hashlib.sha256(json.dumps(component, sort_keys=True).encode()).hexdigest()[:16]
    models = ['top-rgbd.npz', 'side-lsd.npz', 'joint-topology.npz', 'fusion-policy.json',
              'primary-color-profile.json', 'side-color-profile.json']
    scripts = ['experiment_color_geometry_v8.py', 'evaluate_color_geometry_v8.py', 'review_application_v8.py',
               'package_color_geometry_v8.py', 'experiment_color_ridge_v7.py', 'evaluate_color_ridge_v7.py',
               'replay_dual_application.py', 'train_dual_fusion_holdout.py']
    source_paths = [*sorted((ROOT / 'src').rglob('*.py')), *sorted((ROOT / 'config').rglob('*.yaml')),
        *sorted((ROOT / 'config').rglob('*.json')), *sorted((ROOT / 'config').rglob('*.png')),
        *[ROOT / 'scripts' / name for name in scripts], *sorted((ROOT / 'tests').glob('test_*.py')),
        ROOT / 'pyproject.toml', ROOT / 'AGENTS.md', ROOT / 'README.md', ROOT / '算法更新记录.md',
        ROOT / 'docs/v8双相机融合实验复现.md', ROOT / 'docs/reports/v6v7取长补短v8离线对照_20261001.md',
        ROOT / 'output/native-calibration-debug-20260918/calibration-native.json',
        V6 / 'bundle-manifest.json', V7 / 'bundle-manifest.json', V7 / 'split-manifest.json', V7 / 'evaluation-metrics.json',
        V7 / 'review/manual-annotations.json', V7 / 'review/visual-review.json',
        V7 / 'primary-color-profile.json', V7 / 'side-color-profile.json', V7 / 'coordinate-audit.json',
        ROOT / 'output/dual-joint-v4-holdout-20260917/holdout-fusion-records.jsonl',
        V6 / 'application-frozen108-review/application-records.jsonl',
        *[V7 / f'application-{partition}/application-records.jsonl' for partition in ('calibration', 'regression')]]
    # Only 48 small feature-cache entries are required for the controlled
    # spatial ablation; raw photographs remain external to the bundle.
    annotation = json.loads((V7 / 'review/manual-annotations.json').read_text(encoding='utf-8'))['records']
    for row in annotation:
        path = V7 / 'features' / (hashlib.sha256(row['sample'].encode()).hexdigest()[:20] + '.npz')
        if path.exists():
            source_paths.append(path)
    write_json(OUTPUT / 'coordinate-audit.json', {**json.loads((V7 / 'coordinate-audit.json').read_text(encoding='utf-8')),
        'version': 8, 'geometry_model_contract': LEGACY, 'component_fingerprint': component_fingerprint,
        'segmentation_domain': 'legacy uncorrected working primary / native side',
        'colour_domain': 'v7 corrected native RGB independently restored mask',
        'ridge_diagnostics_domain': 'original RGB; primary working crop converted to native endpoints',
        'corner_domain': 'legacy native contour turns and junctions; no subpixel displacement',
        'position_refinement': 'max-reprojection and depth guarded; candidate positions never authorize grasp'})
    bundle = {'version': 8, 'schema_version': 2, 'promotion_eligible': False,
        'component_contracts': component, 'component_fingerprint': component_fingerprint,
        'fit_ids': split['fit_ids'], 'calibration_ids': split['calibration_ids'], 'regression_ids': split['regression_ids'],
        'frozen_population_hash': split['frozen_population_hash'], 'calibration_not_refitted': True,
        'model_hashes': {name: sha(OUTPUT / name) for name in models},
        'observation_snapshot_hash': sha(OUTPUT / 'legacy-observations.npz'),
        'capture_source_hashes': capture_hashes, 'verified_capture_count': 528,
        'source_hashes': {path.relative_to(ROOT).as_posix(): sha(path) for path in source_paths},
        'application_review_hash': sha(OUTPUT / 'review/application-visual-review.json'),
        'actual_reviewed_application_pairs': len(review['observations']), 'source_marks_modified': False,
        'raw_captures_included': False, 'configured_calibration_background_included': True,
        'motion_executed': False, 'acceptance': metrics['acceptance'],
        'replication': 'extract snapshot into an isolated directory; use docs/v8双相机融合实验复现.md; supply unchanged original captures for replay'}
    write_json(OUTPUT / 'bundle-manifest.json', bundle)
    artifacts = [path for path in OUTPUT.rglob('*') if path.is_file() and path.suffix != '.zip' and path.name != 'archive-check.json']
    archive = OUTPUT / 'dual-color-geometry-v8-experiment.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in dict.fromkeys(source_paths + artifacts):
            zipped.write(path, 'snapshot/' + path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError('archive CRC verification failed')
    write_json(OUTPUT / 'archive-check.json', {'path': archive.name, 'bytes': archive.stat().st_size,
        'sha256': sha(archive), 'zip_crc_verified': True, 'promotion_eligible': False})
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'promotion_eligible': False}), flush=True)


if __name__ == '__main__':
    main()
