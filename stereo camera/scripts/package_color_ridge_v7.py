"""Create a verified isolated v7 source/model/review archive without captures."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from experiment_color_ridge_v7 import ROOT, DEFAULT_OUTPUT, write_json, metadata, bootstrap_config
from sorting_vision.visual_contract import V7, V7_PARAMETERS
from sorting_vision.geometry_rgbd_model import DepthGeometryModel
from sorting_vision.dual_view import DualViewCalibration
from sorting_vision.rgbd import CameraIntrinsics
from sorting_vision.sparse_stereo import project_primary


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / DEFAULT_OUTPUT
    split = json.loads((output/"split-manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output/"evaluation-metrics.json").read_text(encoding="utf-8"))
    if split["visual_contract"] != V7 or metrics["visual_contract"] != V7:
        raise ValueError("bundle contract mismatch")
    # Remove stale legacy parameter descriptions; vectors/predictions are
    # unchanged, and the original fit provenance is retained verbatim.
    path = output/"top-rgbd.npz"
    with np.load(path,allow_pickle=False) as data:
        arrays={key:data[key].copy() for key in data.files}
    arrays["edge_parameters_json"]=np.asarray(json.dumps({"input_contract":"rgb_silhouette_depth_owned_v2",
        "feature_contract":V7,"parameters":V7_PARAMETERS},ensure_ascii=False))
    np.savez_compressed(path,**arrays)
    DepthGeometryModel.load(path)
    calibration_path=ROOT/"output/native-calibration-debug-20260918/calibration-native.json"
    calibration=DualViewCalibration.load(calibration_path)
    failures=[]
    capture_hashes={}
    cache_hash_matches=0
    for identifier in split["training_candidates"]+split["regression_ids"]:
        info=metadata(identifier)
        native=CameraIntrinsics(**info["primary_intrinsics"])
        expected=calibration.primary_intrinsics
        if native!=expected:
            failures.append(identifier)
        signature=hashlib.sha256(V7.encode())
        for name in ("metadata.json","primary-color.png","primary-depth.npy","side-color.png"):
            signature.update((ROOT/identifier/name).read_bytes())
        capture_hashes[identifier]=signature.hexdigest()
        cache=output/"features"/(hashlib.sha256(identifier.encode()).hexdigest()[:20]+".npz")
        if cache.exists():
            with np.load(cache,allow_pickle=False) as data:
                if str(data["source_hash"][0])!=capture_hashes[identifier]:
                    raise ValueError("capture changed after feature extraction: "+identifier)
            cache_hash_matches+=1
    cloud=np.array([[0.,0.,500.],[50.,20.,600.],[-30.,-50.,700.]])
    pixels=project_primary(cloud,calibration)
    scale=bootstrap_config().rgbd.processing_scale
    intrinsics=calibration.primary_intrinsics
    working=np.c_[cloud[:,0]*intrinsics.fx*scale/cloud[:,2]+intrinsics.cx*scale,
                  cloud[:,1]*intrinsics.fy*scale/cloud[:,2]+intrinsics.cy*scale]
    coordinate_audit={"primary_native_size":[intrinsics.width,intrinsics.height],
        "side_native_size":[calibration.side_intrinsics.width,calibration.side_intrinsics.height],
        "processing_scale":scale,"native_intrinsics_match_count":528-len(failures),"mismatch_ids":failures,
        "rescale_round_trip_max_px":float(np.max(np.linalg.norm(working/scale-pixels,axis=1))),
        "primary_distortion":calibration.primary_distortion.tolist(),"side_distortion":calibration.side_distortion.tolist(),
        "calibration_hash":calibration.calibration_hash,"strict_valid":calibration.metrics.valid,
        "candidate_metrics":calibration.metrics.to_dict(calibration.quality_limits),
        "candidate_used":"existing SDK native matching candidate; no fitted per-frame projection shifts",
        "color_domain":"native primary RGB with nearest-restored working mask; side native RGB",
        "mask_coordinates":"working primary; restored to native before dual fusion",
        "crop_origin":"working local during extraction; native origin plus processing_scale in exported diagnostics",
        "triangulation":"native undistorted rays; joint reprojection residual in native distorted pixels",
        "thresholds_unchanged":split["thresholds"]}
    write_json(output/"coordinate-audit.json",coordinate_audit)
    models=["top-rgbd.npz","side-lsd.npz","joint-topology.npz","fusion-policy.json","primary-color-profile.json","side-color-profile.json"]
    source_paths=[*sorted((ROOT/"src").rglob("*.py")),*sorted((ROOT/"config").rglob("*.yaml")),
        *[ROOT/"scripts"/name for name in ("experiment_color_ridge_v7.py","review_annotations_v7.py","evaluate_color_ridge_v7.py",
          "package_color_ridge_v7.py","review_application_v7.py","replay_dual_application.py","train_dual_fusion_holdout.py")],
        *sorted((ROOT/"tests").glob("test_*.py")),ROOT/"pyproject.toml",ROOT/"AGENTS.md",ROOT/"README.md",ROOT/"算法更新记录.md",
        ROOT/"docs/v7颜色棱线实验复现.md",ROOT/"docs/reports/双相机颜色棱线v7离线对照_20261001.md",calibration_path,
        ROOT/"output/dual-color-first-v6-20260918/bundle-manifest.json",
        ROOT/"output/dual-joint-v4-holdout-20260917/holdout-fusion-records.jsonl",
        ROOT/"output/dual-color-first-v6-20260918/application-frozen108-review/application-records.jsonl"]
    review=json.loads((output/"review/visual-review.json").read_text(encoding="utf-8"))
    bundle={"version":7,"promotion_eligible":False,"visual_contract":V7,"parameters":V7_PARAMETERS,
        "schema_version":2,"frozen_population_hash":split["frozen_population_hash"],
        "fit_candidates":len(split["fit_ids"]),"calibration_candidates":len(split["calibration_ids"]),"regression_candidates":108,
        "fit_ids":split["fit_ids"],"calibration_ids":split["calibration_ids"],"regression_ids":split["regression_ids"],
        "model_hashes":{name:sha(output/name) for name in models},
        "capture_source_hashes":capture_hashes,"feature_source_hash_match_count":cache_hash_matches,
        "source_hashes":{path.relative_to(ROOT).as_posix():sha(path) for path in source_paths},
        "review_hash":sha(output/"review/visual-review.json"),"annotation_hash":sha(output/"review/manual-annotations.json"),
        "application_review_hash":sha(output/"review/application-visual-review.json"),
        "reviewed_raw_ids":review["reviewed_ids"],"source_marks_modified":False,
        "calibration_not_refitted":True,"calibration_hash":calibration.calibration_hash,
        "acceptance":metrics["acceptance"],"raw_captures_included":False,"motion_executed":False,
        "replication":"extract snapshot into an isolated directory, install editable package, supply unchanged data/dual/temporary; use docs/v7颜色棱线实验复现.md"}
    write_json(output/"bundle-manifest.json",bundle)
    artifacts=[path for path in output.rglob("*") if path.is_file() and "features" not in path.relative_to(output).parts
               and path.suffix not in {".zip"} and path.name != "archive-check.json"]
    archive=output/"dual-color-ridge-v7-experiment.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as zipped:
        for path in dict.fromkeys(source_paths+artifacts):
            zipped.write(path,"snapshot/"+path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("archive verification failed")
    write_json(output/"archive-check.json",{"path":archive.name,"sha256":sha(archive),"bytes":archive.stat().st_size,
        "zip_crc_verified":True,"promotion_eligible":False})
    print(json.dumps({"archive":str(archive),"bytes":archive.stat().st_size,"promotion_eligible":False}),flush=True)


if __name__=="__main__":
    main()
