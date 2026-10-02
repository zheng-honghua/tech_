"""Save the actual v8 visual inspection and render frozen-record comparisons.

The findings below belong to this completed Codex development inspection.
Running the renderer does not certify a new person's inspection or 3D truth.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from experiment_color_geometry_v8 import ROOT, OUTPUT, V7, relative_id, write_json
from evaluate_color_ridge_v7 import records_from


def tile(image, bounds, lines, title, color):
    x, y, right, bottom = bounds
    patch = image[y:bottom, x:right].copy()
    for endpoints in lines:
        points = np.rint(np.asarray(endpoints).reshape(2, 2) - [x, y]).astype(int)
        cv2.line(patch, tuple(points[0]), tuple(points[1]), color, 1, cv2.LINE_AA)
    scale = min(250 / patch.shape[1], 250 / patch.shape[0])
    patch = cv2.resize(patch, None, fx=scale, fy=scale)
    result = np.full((280, 260, 3), 245, np.uint8)
    left, top = (260 - patch.shape[1]) // 2, 30 + (250 - patch.shape[0]) // 2
    result[top:top + patch.shape[0], left:left + patch.shape[1]] = patch
    cv2.putText(result, title, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, .4, (25, 25, 25), 1, cv2.LINE_AA)
    return result


def main():
    cv2.setNumThreads(2)
    review = OUTPUT / 'review'
    review.mkdir(exist_ok=True)
    datasets = {partition: records_from(OUTPUT / f'application-{partition}/application-records.jsonl')
                for partition in ('calibration', 'regression')}
    hashes, observations = {}, []
    for partition, records in datasets.items():
        expected = 72 if partition == 'calibration' else 108
        if len(records) != expected:
            raise ValueError('visual review requires complete replay populations')
        for index in range(1, expected // 6 + 1):
            path = OUTPUT / f'application-{partition}/application-{index:03}.jpg'
            hashes[path.relative_to(OUTPUT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        for record in records:
            objects = []
            for item in record['results']:
                diag = item['diagnostics']
                side = (diag.get('dual_view', {}).get('cross_view_topology') or {})
                candidates = diag.get('candidate_ridges_2d', [])
                objects.append({'object_id': item['object_id'], 'bbox_px': item['bbox_px'], 'status': item['status'],
                    'selected': item['selected'], 'rgb_mask_pixels': diag.get('rgb_mask_pixels'),
                    'depth_mask_pixels': diag.get('depth_mask_pixels'), 'object_valid_depth_ratio': diag.get('object_valid_depth_ratio'),
                    'primary_color': diag.get('primary_color'), 'side_color': side.get('color'),
                    'candidate_count': len(candidates), 'image_verified_count': sum(line.get('image_verified', False) for line in candidates),
                    'geometry_candidate_count': len(diag.get('geometry_candidates_3d', [])),
                    'confirmed_ridge_count': len(diag.get('confirmed_ridges_3d', []))})
            observations.append({'partition': partition, 'sample': relative_id(record['sample']),
                'primary_frame_id': record['primary_frame_id'], 'side_frame_id': record['side_frame_id'],
                'tray_roi_valid': record['health'].get('tray_roi_valid'), 'objects': objects})
    write_json(review / 'application-visual-review.json', {'reviewer': 'codex_visual', 'human_ground_truth': False,
        'source_marks_modified': False, 'method': 'actual view_image inspection of all calibration 001-012 and regression 001-018 RGB/depth/side contact sheets',
        'contact_hashes': hashes, 'observations': observations,
        'findings': [
            {'frames': ['1946', '1738', '2045', '10365', '10005', '9704'], 'finding': 'curved cones retain recall candidates including surface texture; candidates do not establish true straight ridges or a mesh; depth holes remain'},
            {'frames': ['973', '889', '4269', '4560'], 'finding': 'speckles/glints produce extra recall lines; image verification tier is separate; reject 4269 and do not claim all orange lines are ridges'},
            {'frames': ['14557'], 'finding': 'v7 red octahedron split absent in v8 frame; this is one observed recovery, not general split accuracy'},
            {'frames': ['6181', '8907'], 'finding': 'legacy mask path still splits one object in regression 6181 and calibration 8907; unresolved false instances retained'},
            {'frames': ['601', '7085', '6487'], 'finding': 'side object visible but reliable correspondence/shape evidence missing; ambiguity retained'},
            {'frames': ['922', '813', '3343', '2132'], 'finding': 'tray boundary/overhang views remain in population; side view does not repair unavailable primary depth'},
            {'frames': ['2609', '2199', '510', '647', '4560', '3778', '2005', '2620', '1885', '2457', '2571'],
             'finding': 'PICKABLE only with genuine primary depth; selected remains false; no hardware execution'},
            {'frames': ['4647', '4788', '8124', '8249'], 'finding': 'partial evidence only; geometric candidate points/lines are separated from image-and-geometry-supported ridge lines'}],
        'limitations': ['qualitative depth previews are not measured 3D truth', 'single-object data does not validate touching instances',
                       'review does not certify calibration or correspondence identity']})
    previous = {relative_id(row['sample']): row for row in records_from(V7 / 'application-regression/application-records.jsonl')}
    selected = [next(row for row in datasets['regression'] if int(row['primary_frame_id'].split('-')[-1]) == int(suffix))
                for suffix in ('1738', '973', '14557', '6181', '4560', '4647', '6487', '813')]
    rendered = []
    for record in selected:
        identifier = relative_id(record['sample'])
        current = max(record['results'], key=lambda item: item['diagnostics'].get('rgb_mask_pixels', 0))
        old = max(previous[identifier]['results'], key=lambda item: item['diagnostics'].get('rgb_mask_pixels', 0))
        tiles = []
        for camera in ('primary', 'side'):
            image = cv2.imread(str(ROOT / identifier / f'{camera}-color.png'))
            diag, old_diag = current['diagnostics'], old['diagnostics']
            if camera == 'side':
                diag = (diag.get('dual_view', {}).get('cross_view_topology') or {})
                old_diag = (old_diag.get('dual_view', {}).get('cross_view_topology') or {})
            candidates = [line for line in diag.get('candidate_ridges_2d', []) if line.get('state') == 'CANDIDATE_2D']
            prior = [line['endpoints_px'] for line in old_diag.get('candidate_ridges_2d', []) if line.get('state') == 'CANDIDATE_2D']
            points = [line['endpoints_px'] for line in candidates] + prior
            if camera == 'primary':
                x, y, w, h = current['bbox_px']
                points += [[[x, y], [x + w, y + h]]]
            if points:
                coords = np.asarray(points).reshape(-1, 2)
                x, y = np.floor(coords.min(axis=0) - 20).astype(int)
                right, bottom = np.ceil(coords.max(axis=0) + 20).astype(int)
                bounds = (max(0, x), max(0, y), min(image.shape[1], right), min(image.shape[0], bottom))
            else:
                bounds = (0, 0, image.shape[1], image.shape[0])
            verified = [line['endpoints_px'] for line in candidates if line.get('image_verified', False)]
            confirmed = [line[f'{camera}_uv'] for line in current['diagnostics'].get('confirmed_ridges_3d', [])]
            tiles.extend([tile(image, bounds, prior, camera + ' v7 candidates', (0, 150, 255)),
                          tile(image, bounds, [line['endpoints_px'] for line in candidates], camera + ' v8 recall candidates', (0, 150, 255)),
                          tile(image, bounds, verified, camera + ' v8 image verified', (0, 150, 255)),
                          tile(image, bounds, confirmed, camera + ' v8 supported 3D', (255, 255, 0))])
        body = np.hstack(tiles)
        header = np.full((30, body.shape[1], 3), 35, np.uint8)
        title = record['primary_frame_id'] + ' | ' + record['true_label'] + ' | v7=' + old['shape_id'] + ' v8=' + current['shape_id']
        cv2.putText(header, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1, cv2.LINE_AA)
        rendered.append(np.vstack([header, body]))
    for index in range(4):
        cv2.imwrite(str(review / f'focused-v7-v8-{index + 1:02}.jpg'), np.vstack(rendered[index * 2:index * 2 + 2]))
    print('saved actual review of 180 pairs and 8 frozen-record v7/v8 comparisons', flush=True)


if __name__ == '__main__':
    main()
