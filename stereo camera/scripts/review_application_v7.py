"""Persist the completed application contact-sheet inspection and focused views.

The notes describe this development inspection, not certified human labels.
Run only after actually viewing both complete application contact-sheet sets.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from experiment_color_ridge_v7 import ROOT, DEFAULT_OUTPUT, write_json
from evaluate_color_ridge_v7 import cached, records_from
from sorting_vision.geometry_edges import extract_edge_topology
from sorting_vision.side_geometry import extract_side_features
from sorting_vision.visual_contract import LEGACY, visual_contract


def lines_tile(image, mask, lines, color, title):
    if mask is not None and np.any(mask):
        x, y, w, h = cv2.boundingRect(mask)
        x, y, right, bottom = max(0,x-20),max(0,y-20),min(image.shape[1],x+w+20),min(image.shape[0],y+h+20)
    else:
        x, y, right, bottom = 0, 0, image.shape[1], image.shape[0]
    region = image[y:bottom,x:right].copy()
    for endpoints in lines:
        points = np.rint(np.asarray(endpoints).reshape(2,2)-[x,y]).astype(int)
        cv2.line(region,tuple(points[0]),tuple(points[1]),color,1,cv2.LINE_AA)
    scale = min(250/region.shape[1],250/region.shape[0])
    region = cv2.resize(region,None,fx=scale,fy=scale)
    tile = np.full((280,260,3),245,np.uint8)
    left, top = (260-region.shape[1])//2,30+(250-region.shape[0])//2
    tile[top:top+region.shape[0],left:left+region.shape[1]] = region
    cv2.putText(tile,title,(5,19),cv2.FONT_HERSHEY_SIMPLEX,.42,(25,25,25),1,cv2.LINE_AA)
    return tile


def main():
    cv2.setNumThreads(2)
    output = ROOT/DEFAULT_OUTPUT
    datasets = {partition:records_from(output/f"application-{partition}/application-records.jsonl")
                for partition in ("calibration","regression")}
    hashes, observations = {}, []
    for partition, records in datasets.items():
        expected = 72 if partition=="calibration" else 108
        if len(records)!=expected:
            raise ValueError("inspection requires the complete frozen application replay")
        for index in range(1,expected//6+1):
            path = output/f"application-{partition}/application-{index:03}.jpg"
            hashes[path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in records:
            objects=[]
            for item in row["results"]:
                diag=item["diagnostics"]
                side=(diag.get("dual_view",{}).get("cross_view_topology") or {})
                objects.append({"object_id":item["object_id"],"bbox_px":item["bbox_px"],"status":item["status"],
                    "selected":item["selected"],"rgb_mask_pixels":diag.get("rgb_mask_pixels"),
                    "depth_mask_pixels":diag.get("depth_mask_pixels"),"object_valid_depth_ratio":diag.get("object_valid_depth_ratio"),
                    "primary_color":diag.get("primary_color"),"side_color":side.get("color"),
                    "candidate_count":len(diag.get("candidate_ridges_2d",[])),
                    "confirmed_ridge_count":len(diag.get("confirmed_ridges_3d",[]))})
            observations.append({"partition":partition,"sample":Path(row["sample"]).relative_to(ROOT).as_posix(),
                "primary_frame_id":row["primary_frame_id"],"side_frame_id":row["side_frame_id"],
                "tray_roi_valid":row["health"].get("tray_roi_valid"),"objects":objects})
    write_json(output/"review/application-visual-review.json",{
        "reviewer":"codex_visual","human_ground_truth":False,"source_marks_modified":False,
        "method":"actual view_image inspection of regression 001-018 and calibration 001-012, RGB/owned-depth/side overlays",
        "contact_hashes":hashes,"observations":observations,
        "findings":[
            {"frames":["1946","1738","2045","2637","10365","9704"],"finding":"cone contour visible; original depth holes preserved; no grasp upgrade from RGB"},
            {"frames":["973","889","1112","14605"],"finding":"surface speckles and glints remain uncertain, never treated as certified physical ridges"},
            {"frames":["4788","7883","4647","4688","8249"],"finding":"partial cyan geometry supported lines; not a complete object edge mesh"},
            {"frames":["4253","973","6487","813"],"finding":"confirmed 3D ridge field empty; 2D candidates and other graph edges are not confirmed 3D"},
            {"frames":["601","7085","6487","5439"],"finding":"visible side object does not establish unique correspondence; missing evidence retained"},
            {"frames":["14557"],"finding":"one red octahedron still produces two instances; unresolved split failure"},
            {"frames":["2609","510","647","4560","3778","2005","2620","1885","2457","2571"],
             "finding":"PICKABLE requires actual primary depth; all cold-start selections false; no mechanical execution"},
            {"frames":["922","813","10365","2435"],"finding":"tray-edge/overhang views and depth failures remain in the denominator"}],
        "limitations":["depth previews are qualitative; numerical owned-depth diagnostics retained",
            "not independent human approval or 3D ground truth","single-object captures cannot validate real touching objects"]})
    # Focused comparisons retain the same v7 mask for the legacy detector.
    # They are a ridge-stage ablation, not a historical end-to-end v6 replay.
    selected=[]
    for suffix in ("1738","4253","973","4560","14557","4647","6487","813"):
        selected.append(next(row for row in datasets["regression"] if int(row["primary_frame_id"].split("-")[-1])==int(suffix)))
    rows=[]
    for record in selected:
        identifier=Path(record["sample"]).relative_to(ROOT).as_posix()
        data=cached(output,identifier)
        item=max(record["results"],key=lambda value:value["diagnostics"].get("rgb_mask_pixels",0))
        diag=item["diagnostics"]
        side=(diag.get("dual_view",{}).get("cross_view_topology") or {})
        tiles=[]
        for camera in ("primary","side"):
            image=cv2.imread(str(ROOT/identifier/f"{camera}-color.png"))
            mask=data[f"{camera}_mask"] if data is not None else None
            legacy=[]
            if mask is not None:
                with visual_contract(LEGACY):
                    legacy=(extract_side_features(image,mask).segments_px if camera=="side" else
                            [line.points() for line in extract_edge_topology(image,mask).merged_lines])
            candidates=(diag if camera=="primary" else side).get("candidate_ridges_2d",[])
            straight=[line["endpoints_px"] for line in candidates if line.get("state")=="CANDIDATE_2D"]
            confirmed=[line[f"{camera}_uv"] for line in diag.get("confirmed_ridges_3d",[])]
            tiles.extend([lines_tile(image,mask,legacy,(0,150,255),f"{camera} legacy / same mask"),
                          lines_tile(image,mask,straight,(0,150,255),f"{camera} v7 2D candidates"),
                          lines_tile(image,mask,confirmed,(255,255,0),f"{camera} geometry supported 3D")])
        row=np.hstack(tiles)
        header=np.full((30,row.shape[1],3),35,np.uint8)
        cv2.putText(header,record["primary_frame_id"]+" | "+record["true_color"]+" "+record["true_label"]+" | "+item["status"],
                    (8,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
        rows.append(np.vstack([header,row]))
    for index in range(4):
        cv2.imwrite(str(output/f"review/focused-ridges-{index+1:02}.jpg"),np.vstack(rows[index*2:index*2+2]))
    print("saved independent application review for 180 pairs and 4 focused ridge comparisons",flush=True)


if __name__=="__main__":
    main()
