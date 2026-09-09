import importlib.util
from pathlib import Path

import pytest


def test_changed_composition_cannot_silently_enter_training():
    path = Path(__file__).parents[1] / 'scripts' / 'prepare_reviewed_multi_02_03.py'
    spec = importlib.util.spec_from_file_location('reviewed_assignments', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    objects = [
        {'item': {'frame_id': 'd415-000003388'}, 'hue': 110, 'area': 500},
        {'item': {'frame_id': 'd415-000003388'}, 'hue': 90, 'area': 200},
        {'item': {'frame_id': 'd415-000003388'}, 'hue': 90, 'area': 600},
    ]
    with pytest.raises(ValueError, match='instance_labels_need_review'):
        module._assign('multi-02', objects)


def test_explicit_labels_require_confirmation_and_matching_box():
    path = Path(__file__).parents[1] / 'scripts' / 'prepare_reviewed_multi_02_03.py'
    spec = importlib.util.spec_from_file_location('reviewed_assignments', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    objects = [{'item': {'frame_id': 'f', 'object_id': 'o', 'bbox_px': [1, 2, 30, 40]}}]
    annotation = {'frame_id': 'f', 'object_id': 'o', 'bbox_px': [1, 2, 30, 40],
                  'reviewed': False, 'label_id': 'pentagonal_prism'}
    with pytest.raises(ValueError, match='not confirmed'):
        module._confirmed_assignments(objects, [annotation])
    annotation['reviewed'] = True
    assert module._confirmed_assignments(objects, [annotation])[0][1] == 'pentagonal_prism'
    annotation['bbox_px'] = [9, 2, 30, 40]
    with pytest.raises(ValueError, match='bbox changed'):
        module._confirmed_assignments(objects, [annotation])
