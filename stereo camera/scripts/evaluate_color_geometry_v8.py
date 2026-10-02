"""Evaluate frozen v8 replays and a controlled, unchanged-gate refinement ablation."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from experiment_color_geometry_v8 import ROOT, OUTPUT, V6, V7, relative_id, write_json
from evaluate_color_ridge_v7 import app_metrics, records_from, line_metrics, aggregate_line, cached
from experiment_color_ridge_v7 import bootstrap_config
from sorting_vision.dual_view import DualViewCalibration
from sorting_vision.sparse_stereo import reconstruct, SparseStereoLimits
from sorting_vision.visual_contract import LEGACY, visual_contract


def application_metrics(records):
    metrics = app_metrics(records)
    extra_timings = defaultdict(list)
    for row in records:
        for obj in row['results']:
            diag = obj['diagnostics']
            side = (diag.get('dual_view', {}).get('cross_view_topology') or {})
            for name, source in (('primary_color_classification_ms', diag), ('primary_ridge_diagnostics_ms', diag),
                                 ('side_color_classification_ms', side), ('side_ridge_diagnostics_ms', side)):
                if name in source:
                    extra_timings[name].append(source[name])
    metrics['stage_p95_ms'].update({name:float(np.percentile(values, 95)) for name, values in extra_timings.items()})
    metrics['geometric_candidate_corner_count'] = metrics.pop('confirmed_corner_count')
    objects = [max(row['results'], key=lambda item: item['diagnostics'].get('rgb_mask_pixels', 0))
               if row['results'] else None for row in records]
    correct = sum(item is not None and item['shape_id'] == row['true_label']
                  for row, item in zip(records, objects))
    confusion = defaultdict(Counter)
    for row, item in zip(records, objects):
        confusion[row['true_label']][item['shape_id'] if item else 'unknown'] += 1
    geometric = [len(item['diagnostics'].get('geometry_candidates_3d', [])) if item else 0 for item in objects]
    metrics.update(shape_correct=correct, shape_accuracy=correct / max(1, len(records)),
                   shape_unknown=sum(item is None or item['shape_id'] == 'unknown' for item in objects),
                   shape_wrong_known=sum(item is not None and item['shape_id'] not in {'unknown',row['true_label']}
                                         for row,item in zip(records,objects)),
                   shape_confusion={label:dict(counts) for label,counts in confusion.items()},
                   geometric_candidate_ridge_count=sum(geometric),
                   geometric_candidate_ridge_coverage=sum(count > 0 for count in geometric) / max(1, len(records)),
                   physical_correspondence_accuracy=None, measured_3d_accuracy_mm=None,
                   total_instances=sum(len(row['results']) for row in records),
                   multi_instance_frames=[relative_id(row['sample']) for row in records if len(row['results']) > 1])
    return metrics


def main():
    cv2.setNumThreads(2)
    split = json.loads((OUTPUT / 'split-manifest.json').read_text(encoding='utf-8'))
    datasets = {partition: records_from(OUTPUT / f'application-{partition}/application-records.jsonl')
                for partition in ('calibration', 'regression')}
    for partition, records in datasets.items():
        if [relative_id(row['sample']) for row in records] != split[partition + '_ids']:
            raise ValueError('frozen replay population/order changed: ' + partition)
    metrics = {partition: application_metrics(records) for partition, records in datasets.items()}
    historical = application_metrics(records_from(V6 / 'application-frozen108-review/application-records.jsonl'))
    historical['side_color'] = {'available': False, 'reason': 'v6 application had no independent side colour classifier'}
    previous = {partition: application_metrics(records_from(V7 / f'application-{partition}/application-records.jsonl'))
                for partition in datasets}
    annotations = json.loads((V7 / 'review/manual-annotations.json').read_text(encoding='utf-8'))['records']
    indexed = {relative_id(row['sample']): row for records in datasets.values() for row in records}
    previous_index = {relative_id(row['sample']): row for partition in datasets
                      for row in records_from(V7 / f'application-{partition}/application-records.jsonl')}
    line_rows = defaultdict(list)
    spatial = defaultdict(list)
    details = []
    calibration = DualViewCalibration.load(bootstrap_config().dual_view.calibration_path)
    for index, annotation in enumerate(annotations, 1):
        identifier, partition = annotation['sample'], annotation['partition']
        for name, record in (('v8', indexed[identifier]), ('v7', previous_index[identifier])):
            item = max(record['results'], key=lambda obj: obj['diagnostics'].get('rgb_mask_pixels', 0)) if record['results'] else None
            primary = item['diagnostics'] if item else {}
            side = (primary.get('dual_view', {}).get('cross_view_topology') or {})
            for camera, diag in (('primary', primary), ('side', side)):
                shape = (1080, 1920) if camera == 'primary' else (720, 1280)
                candidates = [line for line in diag.get('candidate_ridges_2d', []) if line.get('state') == 'CANDIDATE_2D']
                tiers = {'all_candidates': candidates}
                if name == 'v8':
                    tiers['image_verified'] = [line for line in candidates if line.get('image_verified', False)]
                for tier, lines in tiers.items():
                    line_rows[partition, camera, name, tier].append(line_metrics(
                        [line['endpoints_px'] for line in lines], annotation['cameras'][camera], shape))
        # This ablation uses the same v7-cached native masks/cloud, v6 corners,
        # mutual matching and SDK candidate for BOTH branches. It isolates 3D
        # numerical refinement; it is not an end-to-end v6-v8 comparison.
        data = cached(V7, identifier)
        if data is None:
            for mode in ('dlt', 'refined'):
                spatial[partition, mode].append({'nodes': 0, 'errors': [], 'failed': True, 'pairs': []})
            details.append({'sample': identifier, 'extraction_failed': True, 'denominator_retained': True})
            continue
        images = [cv2.imread(str(ROOT / identifier / f'{camera}-color.png')) for camera in ('primary', 'side')]
        graphs = {}
        with visual_contract(LEGACY):
            for mode, refine in (('dlt', False), ('refined', True)):
                graph = reconstruct(images[0], data['primary_mask'], images[1], data['side_mask'], data['points'],
                                    calibration, SparseStereoLimits(), refine_positions=refine)
                graphs[mode] = graph
                spatial[partition, mode].append({'nodes': len(graph['nodes']),
                    'errors': [max(node['primary_error_px'], node['side_error_px']) for node in graph['nodes']],
                    'pairs': [(node['primary_corner_id'], node['side_corner_id']) for node in graph['nodes']]})
        raw_pairs = {(node['primary_corner_id'], node['side_corner_id']) for node in graphs['dlt']['nodes']}
        refined_pairs = {(node['primary_corner_id'], node['side_corner_id']) for node in graphs['refined']['nodes']}
        details.append({'sample': identifier, 'baseline_pairs_preserved': raw_pairs <= refined_pairs,
                        'baseline_nodes': len(raw_pairs), 'refined_nodes': len(refined_pairs)})
        if index % 12 == 0:
            print(f'controlled refinement {index}/48', flush=True)
    spatial_summary = {}
    for (partition, mode), rows in spatial.items():
        errors = [value for row in rows for value in row['errors']]
        spatial_summary[f'{partition}/{mode}'] = {'denominator': 24, 'nodes': sum(row['nodes'] for row in rows),
            'coverage': sum(row['nodes'] > 0 for row in rows) / 24,
            'reprojection_p95_px': float(np.percentile(errors, 95)) if errors else None,
            'same_inputs_and_limits': True, 'physical_correspondence_accuracy': None}
    full, old = metrics['regression'], previous['regression']
    acceptance = {'shape_accuracy_improves_v7': full['shape_accuracy'] > old['shape_accuracy'],
        'colour_errors_do_not_increase_v7': all(full[camera + '_color']['wrong_acceptance_count'] <=
            old[camera + '_color']['wrong_acceptance_count'] for camera in ('primary', 'side')),
        'corner_coverage_preserved_v7': full['corner_coverage'] >= old['corner_coverage'],
        'controlled_baseline_pairs_preserved': all(row.get('baseline_pairs_preserved', True) for row in details),
        'no_illegal_depth_safety_upgrade': all(value['illegal_safety_upgrades'] == 0 for value in metrics.values()),
        'latency_improves_v7': full['latency_p95_ms'] < old['latency_p95_ms'], 'promotion_eligible': False}
    acceptance['image_verified_2d_ridge_precision_recall_improve_v7_coarse_annotations'] = all(
        aggregate_line(line_rows['regression',camera,'v8','image_verified'])[metric] >
        aggregate_line(line_rows['regression',camera,'v7','all_candidates'])[metric]
        for camera in ('primary','side') for metric in ('precision','recall'))
    write_json(OUTPUT / 'evaluation-metrics.json', {'version': 8, 'geometry_model_contract': LEGACY,
        'component_contract': split['component_contract'], 'application': metrics, 'previous_v7_application': previous,
        'historical_v6_application': historical, 'ridge_annotations': {'/'.join(key): aggregate_line(rows)
            for key, rows in line_rows.items()}, 'matching_refinement_ablation': spatial_summary, 'acceptance': acceptance,
        'limitations': ['NO_INDEPENDENT_3D_TRUTH', 'CODEX_COARSE_ANNOTATIONS_NOT_CERTIFIED_HUMAN_TRUTH',
                       'HIGH_RECALL_AND_IMAGE_VERIFIED_TIERS_HAVE_DIFFERENT_PRECISION_RECALL',
                       'HISTORICAL_V6_FIT_420_V8_FIT_348_NOT_A_CONTROLLED_MODEL_COMPARISON',
                       'SDK_NATIVE_CALIBRATION_CANDIDATE_STRICT_INVALID', 'SINGLE_OBJECT_CAPTURES',
                       'COARSE_CORRESPONDENCE_IDENTITIES_MOSTLY_AMBIGUOUS'], 'promotion_eligible': False})
    write_json(OUTPUT / 'evaluation-details.json', details)
    print(json.dumps({'application': metrics, 'acceptance': acceptance}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
