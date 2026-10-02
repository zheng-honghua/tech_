"""Refit v6 semantic features on the isolated v7 split; attach v7 colour."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from experiment_color_ridge_v7 import ROOT, relative_id, write_json, normalized_model
from sorting_vision.geometry_rgbd_model import DepthGeometryModel, FUSED_FEATURE_NAMES
from sorting_vision.side_geometry import SideGeometryModel
from sorting_vision.cross_view_topology import CrossViewTopologyModel
from sorting_vision.shape_registry import ShapeRegistry
from sorting_vision.visual_contract import LEGACY, visual_contract
from train_dual_fusion_holdout import _select_policy

OUTPUT = ROOT/'output/dual-color-geometry-v8-20261001'
V6 = ROOT/'output/dual-color-first-v6-20260918'
V7 = ROOT/'output/dual-color-ridge-v7-20261001'


def observation_rows(identifiers, registry):
    snapshot=OUTPUT/'legacy-observations.npz'
    provenance_path=OUTPUT/'observation-provenance.json'
    if snapshot.exists():
        provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
        if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=provenance['snapshot_sha256']:
            raise ValueError('observation snapshot checksum mismatch')
        with np.load(snapshot,allow_pickle=False) as data:
            if str(data['contract'][0])!=LEGACY or data['identifiers'].tolist()!=identifiers:
                raise ValueError('observation snapshot contract/population mismatch')
            rows={key:{'sample':key,'label':str(data['labels'][index]),
                **{name:data[name][index].copy() for name in ('top','side','cross')},
                **{name:data[name].copy() for name in ('side_groups','cross_groups')},
                **{name:float(data[name][index]) for name in ('side_quality','topology_quality')}}
                for index,key in enumerate(identifiers)}
        if any(Path(key).parent.name!=row['label'] for key,row in rows.items()):
            raise ValueError('observation label mismatch')
        return rows,provenance['top_edge_parameters'],provenance['alignment_max_error']
    old_top=DepthGeometryModel.load(V6/'top-rgbd.npz')
    old_side=SideGeometryModel.load(V6/'side-lsd.npz',registry)
    old_cross=CrossViewTopologyModel.load(V6/'joint-topology.npz',registry)
    with np.load(V6/'top-feature-cache.npz',allow_pickle=False) as data:
        raw_top={relative_id(key):value for key,value in zip(data['sample_keys'],data['features'])}
        topology_quality={relative_id(key):float(value) for key,value in zip(data['sample_keys'],data['topology_quality'])}
    original, original_labels=old_top.training_data()
    alignment_error=float(np.max(np.abs(original-np.stack([raw_top[key] for key in identifiers]))))
    if alignment_error>1e-4 or any(Path(key).parent.name!=label for key,label in zip(identifiers,original_labels)):
        raise ValueError('historical training order cannot be proven')
    # Inverse normalization restores cached observation vectors, not forest
    # predictions. New means/scales and all estimators fit only the 348 IDs.
    raw_side=old_side.features*old_side.scale+old_side.mean
    raw_cross=old_cross.features*old_cross.scale+old_cross.mean
    if not np.array_equal(old_side.labels,old_cross.labels) or list(old_side.labels)!=original_labels:
        raise ValueError('side/cross observation label order mismatch')
    rows={}
    for index, identifier in enumerate(identifiers):
        directory=ROOT/identifier
        image_hash=hashlib.sha256((directory/'side-color.png').read_bytes()).hexdigest()
        signature=hashlib.sha256((directory.resolve().as_posix()+image_hash).encode())
        for name in ('primary-depth.npy','primary-color.png','metadata.json'):
            signature.update((directory/name).read_bytes())
        path=V6/'paired-cache'/(signature.hexdigest()+'.npz')
        if not path.exists():
            raise ValueError('historical observation source hash mismatch: '+identifier)
        with np.load(path,allow_pickle=False) as data:
            quality=float(data['side_quality'])
            if np.max(np.abs(data['features']-raw_top[identifier]))>1e-5:
                raise ValueError('paired/top raw feature mismatch')
        rows[identifier]={'sample':identifier,'label':original_labels[index],
            'top':raw_top[identifier],'side':raw_side[index],'cross':raw_cross[index],
            'side_groups':old_side.group_ids,'cross_groups':old_cross.group_ids,
            'side_quality':quality,'topology_quality':topology_quality[identifier]}
    np.savez_compressed(snapshot,contract=np.asarray([LEGACY]),identifiers=np.asarray(identifiers),
        labels=np.asarray(original_labels),side_groups=old_side.group_ids,cross_groups=old_cross.group_ids,
        **{name:np.stack([rows[key][name] for key in identifiers]) for name in
           ('top','side','cross','side_quality','topology_quality')})
    write_json(provenance_path,{'contract':LEGACY,'snapshot_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        'top_edge_parameters':old_top.edge_parameters,'alignment_max_error':alignment_error,
        'source_artifact_sha256':{name:hashlib.sha256((V6/name).read_bytes()).hexdigest() for name in
            ('top-rgbd.npz','side-lsd.npz','joint-topology.npz','top-feature-cache.npz','bundle-manifest.json')},
        'verified_paired_capture_signatures':420,'restoration':'inverse normalization of observation vectors',
        'old_forest_reused':False,'old_normalization_reused':False,'raw_images_modified':False})
    return rows,old_top.edge_parameters,alignment_error


def prepare_and_fit():
    cv2.setNumThreads(2)
    OUTPUT.mkdir(exist_ok=True)
    split=json.loads((V7/'split-manifest.json').read_text(encoding='utf-8'))
    baseline=json.loads((V6/'bundle-manifest.json').read_text(encoding='utf-8'))
    identifiers=[relative_id(value) for value in baseline['training_ids']]
    if identifiers!=split['training_candidates'] or set(split['fit_ids'])&set(split['calibration_ids']):
        raise ValueError('frozen population or isolated partitions changed')
    registry=ShapeRegistry.load(ROOT/'config/shapes/temporary-observed10.yaml')
    rows,edge_parameters,alignment_error=observation_rows(identifiers,registry)
    fit=[rows[key] for key in split['fit_ids']]
    calibration=[rows[key] for key in split['calibration_ids']]
    with visual_contract(LEGACY):
        top=DepthGeometryModel.fit(np.stack([row['top'] for row in fit]),[row['label'] for row in fit],feature_names=FUSED_FEATURE_NAMES)
        top.edge_parameters={**edge_parameters,'feature_contract':LEGACY}
        side=normalized_model(fit,'side','side_groups',SideGeometryModel,registry)
        cross=normalized_model(fit,'cross','cross_groups',CrossViewTopologyModel,registry)
        records=[]
        for row in calibration:
            label, confidence, reason=top.predict_features(row['top'])
            record={'sample':str(ROOT/row['sample']),'true_label':row['label'],
                'top_prediction':label,'top_confidence':confidence,'top_reason':reason,
                'top_class_scores':dict(top.last_class_scores),'side_quality':row['side_quality'],
                'topology_quality':row['topology_quality']}
            for prefix,model,key in (('classic_side',side,'side'),('side',cross,'cross')):
                prediction, probability,diag=model.predict_features(row[key])
                record.update({prefix+'_prediction':prediction,prefix+'_confidence':probability,
                    prefix+'_reason':diag['reason'],prefix+'_class_scores':dict(model.last_class_scores)})
            records.append(record)
        policy, policy_report=_select_policy(records,registry.registry_hash,strict=True)
    top.save(OUTPUT/'top-rgbd.npz',metadata={'fit_ids':split['fit_ids'],'calibration_ids':split['calibration_ids'],
        'calibration_not_refitted':True,'feature_source':'v6 cached observation vectors, not v6 forest predictions'})
    side.save(OUTPUT/'side-lsd.npz'); cross.save(OUTPUT/'joint-topology.npz'); policy.save(OUTPUT/'fusion-policy.json')
    for camera in ('primary','side'):
        shutil.copyfile(V7/(camera+'-color-profile.json'),OUTPUT/(camera+'-color-profile.json'))
    for partition,name in (('regression','holdout-fusion-records.jsonl'),('calibration','calibration-replay-inputs.jsonl')):
        (OUTPUT/name).write_text(''.join(json.dumps({'sample':str(ROOT/key),'true_label':Path(key).parent.name})+'\n'
            for key in split[partition+'_ids']),encoding='utf-8')
    split.update(version=8,geometry_model_contract=LEGACY,component_contract='v6_shape_mask__v7_independent_colour_ridges__v6_corners_v7_position_refinement',
        calibration_not_refitted=True,promotion_eligible=False)
    write_json(OUTPUT/'split-manifest.json',split)
    write_json(OUTPUT/'fit-report.json',{'fit_candidates':348,'fit_usable':len(fit),'calibration_candidates':72,
        'calibration_usable':len(calibration),'alignment_max_error':alignment_error,'policy_fit':policy_report,
        'calibration_not_refitted':True,'normalization_fit_only':True,
        'historical_observation_restoration':'side/cross inverse normalization; new estimators never inherit old forest or old normalization',
        'input_contract':LEGACY,'promotion_eligible':False})
    write_json(OUTPUT/'calibration-model-records.json',records)
    print(json.dumps({'fit':len(fit),'calibration':len(calibration),'alignment_error':alignment_error,
        'policy':policy_report['selected_thresholds'],'promotion_eligible':False}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.parse_args()
    prepare_and_fit()
