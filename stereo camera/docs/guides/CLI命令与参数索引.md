# CLI命令与参数索引

核对日期：2026-10-02。CLI部分由当前代码的`build_parser()`生成；仅构造参数解析器，没有连接相机、训练模型或执行控制。完整操作顺序见[使用手册](双相机工程完整使用手册.md)，模块职责见[模块说明](模块使用说明.md)。

## 1. 命令位置与帮助

所有示例先进入双相机工程根目录。CLI全局`--config`放在子命令前；独立脚本的`--config`跟在脚本名后。参数列表中的默认路径相对于当前工作目录。`--help`只查看帮助。

```powershell
Set-Location "C:\Users\d114\Desktop\tech_\stereo camera"
.\.venv\Scripts\python.exe -m sorting_vision.cli --help
.\.venv\Scripts\python.exe `
    -m sorting_vision.cli `
    --config config\dual\temporary.yaml camera-live `
    --help
.\.venv\Scripts\python.exe scripts\dual_rgbd_side_capture.py --help
```

## 2. CLI命令目录

| 命令 | 作用（程序帮助原文） |
|---|---|
| `detect` | detect objects in one tray image |
| `predict-image` | predict the geometry class of one RGB photograph |
| `predict-scene` | detect and classify multiple separated objects in one RGB image |
| `demo` | run the synthetic tray demonstration |
| `benchmark` | run repeatable 12-object synthetic scenes |
| `rgbd-demo` | run the synthetic RGB-D demonstration |
| `rgbd-benchmark` | run randomised 3-D solid benchmark scenes |
| `rgbd-detect` | detect a recorded RGB-D frame |
| `rgbd-calibrate` | fit the empty-tray plane and save RGB-D calibration |
| `dual-calibrate` | calibrate ChArUco intrinsics and T_side_from_primary |
| `dual-capture` | capture synchronized primary RGB-D and side RGB samples |
| `dual-multi-capture` | capture 5-8 object paired scenes for development/final acceptance |
| `dual-review` | build primary/side/depth contact sheets for human review |
| `dual-review-apply` | apply explicit per-sample human review JSONL decisions |
| `dual-evaluate` | apply independent-holdout dual-view promotion gates |
| `dual-temperature-fit` | fit top and side probability temperatures on a calibration split |
| `shape-registry-check` | validate the configured shape IDs, aliases and hash |
| `dual-side-train` | train and compare side-view grouped KNN and RTrees |
| `dual-fusion-fit` | calibrate and choose a safe Top-2 fusion policy |
| `dual-side-cnn-train` | train the experimental frozen MobileNetV3-Small side branch |
| `dual-side-cnn-export` | export the side CNN to ONNX/OpenVINO and an Orin TensorRT recipe |
| `dual-side-cnn-calibrate` | fit side CNN acceptance gates using reviewed real probability-calibration images |
| `dual-side-tensorrt-build` | build the exported FP16 engine on the target Orin Nano |
| `dual-side-backend-check` | compare OpenCV/OpenVINO/TensorRT class, acceptance and P95 latency |
| `rgbd-capture` | capture labelled RealSense RGB-D samples as self-contained folders |
| `rgbd-capture-assistant` | interactively capture and label a complete RealSense geometry batch |
| `rgbd-multi-capture` | batch-capture labelled multi-object RealSense test scenes |
| `rgbd-dataset-audit` | validate RGB-D sample bundles and depth quality |
| `rgbd-review` | build annotated/depth contact sheets for visual review |
| `rgbd-edge-audit` | audit RGB/depth-correlated internal edges by dataset batch |
| `geometry-rgbd-train` | train the metric point-cloud geometry baseline |
| `rgbd-face-audit` | visualise observed 3-D planes and face topology |
| `serve` | serve detection over newline JSON/TCP |
| `camera-live` | preview a UVC, RealSense, or dual camera source with motion interlock |
| `camera-record` | record a UVC or aligned RealSense capture session |
| `geometry-audit` | audit a folder-labelled RGB geometry dataset |
| `geometry-train` | train a lightweight RGB geometry model |
| `geometry-edge-audit` | export internal edge and topology diagnostics |
| `geometry-structure-audit` | export conservative noise-reduced contour lines and vertices |
| `geometry-scene-structure-audit` | extract noise-reduced structure for every separated object in scene images |
| `geometry-edge-compare` | compare legacy and edge-topology OpenCV models |
| `geometry-holdout-evaluate` | evaluate a trained OpenCV model on hash-distinct images |
| `geometry-evaluate` | leave-one-out evaluation of a geometry dataset |
| `geometry-cnn-train` | train a MobileNetV3-Small geometry classifier |
| `geometry-cnn-export` | export a CNN checkpoint to ONNX and OpenVINO |
| `geometry-benchmark` | measure geometry backend P50/P95 latency |
| `geometry-export` | export per-image geometry crops, masks and predictions |
| `calibrate` | save four-point calibration |

## `detect`

```text
usage: build_reference.py detect [-h] --image IMAGE [--background BACKGROUND]
                                 [--calibration CALIBRATION]
                                 [--output-dir OUTPUT_DIR] [--qrcode]
                                 [--shape-model SHAPE_MODEL]
                                 [--shape-backend {opencv,openvino}]
                                 [--shape-device SHAPE_DEVICE]

options:
  -h, --help            show this help message and exit
  --image IMAGE
  --background BACKGROUND
  --calibration CALIBRATION
  --output-dir OUTPUT_DIR
  --qrcode
  --shape-model SHAPE_MODEL
                        trained geometry RGB model (.npz)
  --shape-backend {opencv,openvino}
  --shape-device SHAPE_DEVICE
```

## `predict-image`

```text
usage: build_reference.py predict-image [-h] [--model MODEL]
                                        [--backend {auto,opencv,openvino}]
                                        [--device DEVICE]
                                        [--output-dir OUTPUT_DIR]
                                        [--json-only]
                                        image

positional arguments:
  image                 input JPG/PNG image path

options:
  -h, --help            show this help message and exit
  --model MODEL         NPZ model or OpenVINO model directory
  --backend {auto,opencv,openvino}
  --device DEVICE
  --output-dir OUTPUT_DIR
                        optional directory for result.json, annotated image,
                        crop and mask
  --json-only           print machine-readable JSON only
```

## `predict-scene`

```text
usage: build_reference.py predict-scene [-h] [--model MODEL]
                                        [--backend {auto,opencv,openvino}]
                                        [--device DEVICE]
                                        [--output-dir OUTPUT_DIR]
                                        [--json-only]
                                        image

positional arguments:
  image                 input scene JPG/PNG image path

options:
  -h, --help            show this help message and exit
  --model MODEL         geometry model path
  --backend {auto,opencv,openvino}
  --device DEVICE
  --output-dir OUTPUT_DIR
  --json-only
```

## `demo`

```text
usage: build_reference.py demo [-h] [--output-dir OUTPUT_DIR]

options:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
```

## `benchmark`

```text
usage: build_reference.py benchmark [-h] [--rounds ROUNDS] [--seed SEED]

options:
  -h, --help       show this help message and exit
  --rounds ROUNDS
  --seed SEED
```

## `rgbd-demo`

```text
usage: build_reference.py rgbd-demo [-h] [--output-dir OUTPUT_DIR]

options:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
```

## `rgbd-benchmark`

```text
usage: build_reference.py rgbd-benchmark [-h] [--rounds ROUNDS] [--seed SEED]

options:
  -h, --help       show this help message and exit
  --rounds ROUNDS
  --seed SEED
```

## `rgbd-detect`

```text
usage: build_reference.py rgbd-detect [-h] --frame-dir FRAME_DIR
                                      [--background-dir BACKGROUND_DIR]
                                      [--rgbd-calibration RGBD_CALIBRATION]
                                      [--rgbd-shape-model RGBD_SHAPE_MODEL]
                                      [--output-dir OUTPUT_DIR]

options:
  -h, --help            show this help message and exit
  --frame-dir FRAME_DIR
  --background-dir BACKGROUND_DIR
  --rgbd-calibration RGBD_CALIBRATION
  --rgbd-shape-model RGBD_SHAPE_MODEL
                        trained RGB-D geometry NPZ model
  --output-dir OUTPUT_DIR
```

## `rgbd-calibrate`

```text
usage: build_reference.py rgbd-calibrate [-h] --background-dir BACKGROUND_DIR
                                         [--camera-to-robot CAMERA_TO_ROBOT]
                                         [--output OUTPUT]

options:
  -h, --help            show this help message and exit
  --background-dir BACKGROUND_DIR
  --camera-to-robot CAMERA_TO_ROBOT
                        JSON file containing a 4x4 camera-to-robot transform;
                        identity by default
  --output OUTPUT
```

## `dual-calibrate`

```text
usage: build_reference.py dual-calibrate [-h] --primary-dir PRIMARY_DIR
                                         --side-dir SIDE_DIR
                                         --background-dir BACKGROUND_DIR
                                         --platform-id {temporary,competition}
                                         [--squares-x SQUARES_X]
                                         [--squares-y SQUARES_Y]
                                         [--square-length-mm SQUARE_LENGTH_MM]
                                         [--marker-length-mm MARKER_LENGTH_MM]
                                         [--dictionary DICTIONARY]
                                         --tray-pose-index TRAY_POSE_INDEX
                                         --output OUTPUT

options:
  -h, --help            show this help message and exit
  --primary-dir PRIMARY_DIR
  --side-dir SIDE_DIR
  --background-dir BACKGROUND_DIR
                        empty-tray RGB-D frame used to fit and bind the tray
                        depth plane
  --platform-id {temporary,competition}
  --squares-x SQUARES_X
  --squares-y SQUARES_Y
  --square-length-mm SQUARE_LENGTH_MM
  --marker-length-mm MARKER_LENGTH_MM
  --dictionary DICTIONARY
  --tray-pose-index TRAY_POSE_INDEX
                        zero-based paired image index where the board lies
                        flat in the tray
  --output OUTPUT
```

## `dual-capture`

```text
usage: build_reference.py dual-capture [-h] --dataset-root DATASET_ROOT
                                       --batch-id BATCH_ID
                                       --platform-id {temporary,competition}
                                       [--label LABEL]
                                       [--shape-registry SHAPE_REGISTRY]
                                       [--split {train,probability_calibration,final_holdout}]
                                       [--instance-id INSTANCE_ID]
                                       [--color-id COLOR_ID] [--count COUNT]
                                       [--camera-index CAMERA_INDEX]
                                       [--side-camera-index SIDE_CAMERA_INDEX]
                                       [--width WIDTH] [--height HEIGHT]
                                       [--fps FPS] [--depth-width DEPTH_WIDTH]
                                       [--depth-height DEPTH_HEIGHT]
                                       [--color-width COLOR_WIDTH]
                                       [--color-height COLOR_HEIGHT]
                                       [--side-width SIDE_WIDTH]
                                       [--side-height SIDE_HEIGHT]
                                       [--side-fps SIDE_FPS]
                                       [--dual-calibration DUAL_CALIBRATION]
                                       [--side-background-output SIDE_BACKGROUND_OUTPUT]
                                       [--headless]

options:
  -h, --help            show this help message and exit
  --dataset-root DATASET_ROOT
  --batch-id BATCH_ID
  --platform-id {temporary,competition}
  --label LABEL
  --shape-registry SHAPE_REGISTRY
  --split {train,probability_calibration,final_holdout}
  --instance-id INSTANCE_ID
  --color-id COLOR_ID
  --count COUNT
  --camera-index CAMERA_INDEX
  --side-camera-index SIDE_CAMERA_INDEX
  --width WIDTH
  --height HEIGHT
  --fps FPS
  --depth-width DEPTH_WIDTH
  --depth-height DEPTH_HEIGHT
  --color-width COLOR_WIDTH
  --color-height COLOR_HEIGHT
  --side-width SIDE_WIDTH
  --side-height SIDE_HEIGHT
  --side-fps SIDE_FPS
  --dual-calibration DUAL_CALIBRATION
  --side-background-output SIDE_BACKGROUND_OUTPUT
                        write the median side image of captured empty-tray
                        pairs
  --headless
```

## `dual-multi-capture`

```text
usage: build_reference.py dual-multi-capture [-h]
                                             [--dataset-root DATASET_ROOT]
                                             --batch-id BATCH_ID
                                             --platform-id {temporary,competition}
                                             --scene-split {development,final_acceptance}
                                             --composition COMPOSITION
                                             --occlusion {none,light,heavy}
                                             [--count COUNT]
                                             [--shape-registry SHAPE_REGISTRY]
                                             [--dual-calibration DUAL_CALIBRATION]
                                             [--camera-index CAMERA_INDEX]
                                             [--side-camera-index SIDE_CAMERA_INDEX]
                                             [--width WIDTH] [--height HEIGHT]
                                             [--fps FPS]
                                             [--depth-width DEPTH_WIDTH]
                                             [--depth-height DEPTH_HEIGHT]
                                             [--color-width COLOR_WIDTH]
                                             [--color-height COLOR_HEIGHT]
                                             [--side-width SIDE_WIDTH]
                                             [--side-height SIDE_HEIGHT]
                                             [--side-fps SIDE_FPS]
                                             [--headless]

options:
  -h, --help            show this help message and exit
  --dataset-root DATASET_ROOT
  --batch-id BATCH_ID
  --platform-id {temporary,competition}
  --scene-split {development,final_acceptance}
  --composition COMPOSITION
                        5-8 comma-separated shape IDs/names/aliases
  --occlusion {none,light,heavy}
  --count COUNT
  --shape-registry SHAPE_REGISTRY
  --dual-calibration DUAL_CALIBRATION
  --camera-index CAMERA_INDEX
  --side-camera-index SIDE_CAMERA_INDEX
  --width WIDTH
  --height HEIGHT
  --fps FPS
  --depth-width DEPTH_WIDTH
  --depth-height DEPTH_HEIGHT
  --color-width COLOR_WIDTH
  --color-height COLOR_HEIGHT
  --side-width SIDE_WIDTH
  --side-height SIDE_HEIGHT
  --side-fps SIDE_FPS
  --headless
```

## `dual-review`

```text
usage: build_reference.py dual-review [-h] --samples-root SAMPLES_ROOT
                                      --output-dir OUTPUT_DIR
                                      --platform-id {temporary,competition}
                                      --dual-calibration DUAL_CALIBRATION
                                      --side-background SIDE_BACKGROUND
                                      [--side-shape-model SIDE_SHAPE_MODEL]
                                      [--cross-view-shape-model CROSS_VIEW_SHAPE_MODEL]
                                      [--shape-registry SHAPE_REGISTRY]
                                      [--fusion-policy FUSION_POLICY]
                                      [--fusion-backend {auto,opencv,openvino,tensorrt}]
                                      [--deployment-target {host,n100,orin_nano}]
                                      [--rgbd-calibration RGBD_CALIBRATION]
                                      [--background-dir BACKGROUND_DIR]
                                      [--rgbd-shape-model RGBD_SHAPE_MODEL]
                                      [--columns COLUMNS]
                                      [--tile-width TILE_WIDTH]
                                      [--tile-height TILE_HEIGHT]

options:
  -h, --help            show this help message and exit
  --samples-root SAMPLES_ROOT
  --output-dir OUTPUT_DIR
  --platform-id {temporary,competition}
  --dual-calibration DUAL_CALIBRATION
  --side-background SIDE_BACKGROUND
  --side-shape-model SIDE_SHAPE_MODEL
                        trained side model; omit for a raw three-view contact
                        sheet before training
  --cross-view-shape-model CROSS_VIEW_SHAPE_MODEL
  --shape-registry SHAPE_REGISTRY
  --fusion-policy FUSION_POLICY
  --fusion-backend {auto,opencv,openvino,tensorrt}
  --deployment-target {host,n100,orin_nano}
  --rgbd-calibration RGBD_CALIBRATION
  --background-dir BACKGROUND_DIR
  --rgbd-shape-model RGBD_SHAPE_MODEL
  --columns COLUMNS
  --tile-width TILE_WIDTH
  --tile-height TILE_HEIGHT
```

## `dual-review-apply`

```text
usage: build_reference.py dual-review-apply [-h] --samples-root SAMPLES_ROOT
                                            --reviews REVIEWS
                                            [--shape-registry SHAPE_REGISTRY]

options:
  -h, --help            show this help message and exit
  --samples-root SAMPLES_ROOT
  --reviews REVIEWS
  --shape-registry SHAPE_REGISTRY
```

## `dual-evaluate`

```text
usage: build_reference.py dual-evaluate [-h] --records RECORDS
                                        [--output OUTPUT]

options:
  -h, --help         show this help message and exit
  --records RECORDS
  --output OUTPUT
```

## `dual-temperature-fit`

```text
usage: build_reference.py dual-temperature-fit [-h] --records RECORDS
                                               [--output OUTPUT]

options:
  -h, --help         show this help message and exit
  --records RECORDS
  --output OUTPUT
```

## `shape-registry-check`

```text
usage: build_reference.py shape-registry-check [-h] [--registry REGISTRY]
                                               [--alias ALIAS]

options:
  -h, --help           show this help message and exit
  --registry REGISTRY
  --alias ALIAS
```

## `dual-side-train`

```text
usage: build_reference.py dual-side-train [-h] --samples-root SAMPLES_ROOT
                                          --side-background SIDE_BACKGROUND
                                          [--shape-registry SHAPE_REGISTRY]
                                          [--output OUTPUT] [--report REPORT]
                                          [--allow-unreviewed]

options:
  -h, --help            show this help message and exit
  --samples-root SAMPLES_ROOT
  --side-background SIDE_BACKGROUND
  --shape-registry SHAPE_REGISTRY
  --output OUTPUT
  --report REPORT
  --allow-unreviewed
```

## `dual-fusion-fit`

```text
usage: build_reference.py dual-fusion-fit [-h] --records RECORDS
                                          [--shape-registry SHAPE_REGISTRY]
                                          [--output OUTPUT] [--report REPORT]

options:
  -h, --help            show this help message and exit
  --records RECORDS
  --shape-registry SHAPE_REGISTRY
  --output OUTPUT
  --report REPORT
```

## `dual-side-cnn-train`

```text
usage: build_reference.py dual-side-cnn-train [-h] --samples-root SAMPLES_ROOT
                                              --side-background SIDE_BACKGROUND
                                              [--shape-registry SHAPE_REGISTRY]
                                              [--output OUTPUT]
                                              [--epochs EPOCHS] [--seed SEED]
                                              [--synthetic-pretrain-per-class SYNTHETIC_PRETRAIN_PER_CLASS]
                                              [--allow-unreviewed]
                                              [--report REPORT]

options:
  -h, --help            show this help message and exit
  --samples-root SAMPLES_ROOT
  --side-background SIDE_BACKGROUND
  --shape-registry SHAPE_REGISTRY
  --output OUTPUT
  --epochs EPOCHS
  --seed SEED
  --synthetic-pretrain-per-class SYNTHETIC_PRETRAIN_PER_CLASS
  --allow-unreviewed
  --report REPORT
```

## `dual-side-cnn-export`

```text
usage: build_reference.py dual-side-cnn-export [-h] --checkpoint CHECKPOINT
                                               --output-dir OUTPUT_DIR
                                               [--target-backend {openvino,tensorrt,both}]

options:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
  --output-dir OUTPUT_DIR
  --target-backend {openvino,tensorrt,both}
```

## `dual-side-cnn-calibrate`

```text
usage: build_reference.py dual-side-cnn-calibrate [-h] --model-dir MODEL_DIR
                                                  --samples-root SAMPLES_ROOT
                                                  --side-background SIDE_BACKGROUND
                                                  [--shape-registry SHAPE_REGISTRY]
                                                  [--backend {opencv,openvino}]
                                                  [--device DEVICE]
                                                  [--report REPORT]

options:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
  --samples-root SAMPLES_ROOT
  --side-background SIDE_BACKGROUND
  --shape-registry SHAPE_REGISTRY
  --backend {opencv,openvino}
  --device DEVICE
  --report REPORT
```

## `dual-side-tensorrt-build`

```text
usage: build_reference.py dual-side-tensorrt-build [-h] --model-dir MODEL_DIR
                                                   [--trtexec TRTEXEC]

options:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
  --trtexec TRTEXEC
```

## `dual-side-backend-check`

```text
usage: build_reference.py dual-side-backend-check [-h] --model-dir MODEL_DIR
                                                  --samples-root SAMPLES_ROOT
                                                  --side-background SIDE_BACKGROUND
                                                  [--shape-registry SHAPE_REGISTRY]
                                                  [--backend {opencv,openvino,tensorrt}]
                                                  [--device DEVICE]
                                                  [--output OUTPUT]

options:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
  --samples-root SAMPLES_ROOT
  --side-background SIDE_BACKGROUND
  --shape-registry SHAPE_REGISTRY
  --backend {opencv,openvino,tensorrt}
                        repeat to choose backends; defaults to all three
  --device DEVICE
  --output OUTPUT
```

## `rgbd-capture`

```text
usage: build_reference.py rgbd-capture [-h] --dataset-root DATASET_ROOT
                                       --batch-id BATCH_ID --label LABEL
                                       [--color-width COLOR_WIDTH]
                                       [--color-height COLOR_HEIGHT]
                                       [--depth-width DEPTH_WIDTH]
                                       [--depth-height DEPTH_HEIGHT]
                                       [--fps FPS]
                                       [--discard-frames DISCARD_FRAMES]
                                       [--count COUNT] [--headless]

options:
  -h, --help            show this help message and exit
  --dataset-root DATASET_ROOT
  --batch-id BATCH_ID
  --label LABEL
  --color-width COLOR_WIDTH
  --color-height COLOR_HEIGHT
  --depth-width DEPTH_WIDTH
  --depth-height DEPTH_HEIGHT
  --fps FPS
  --discard-frames DISCARD_FRAMES
  --count COUNT         stop after N captures; zero means manual capture until
                        q
  --headless
```

## `rgbd-capture-assistant`

```text
usage: build_reference.py rgbd-capture-assistant [-h]
                                                 --dataset-root DATASET_ROOT
                                                 --batch-id BATCH_ID
                                                 [--target-per-label TARGET_PER_LABEL]
                                                 [--start-label START_LABEL]
                                                 [--color-width COLOR_WIDTH]
                                                 [--color-height COLOR_HEIGHT]
                                                 [--depth-width DEPTH_WIDTH]
                                                 [--depth-height DEPTH_HEIGHT]
                                                 [--fps FPS]
                                                 [--discard-frames DISCARD_FRAMES]
                                                 [--stable-frames STABLE_FRAMES]
                                                 [--motion-threshold MOTION_THRESHOLD]
                                                 [--min-valid-depth-ratio MIN_VALID_DEPTH_RATIO]
                                                 [--max-sync-delta-ms MAX_SYNC_DELTA_MS]
                                                 [--auto-advance]

options:
  -h, --help            show this help message and exit
  --dataset-root DATASET_ROOT
  --batch-id BATCH_ID
  --target-per-label TARGET_PER_LABEL
  --start-label START_LABEL
  --color-width COLOR_WIDTH
  --color-height COLOR_HEIGHT
  --depth-width DEPTH_WIDTH
  --depth-height DEPTH_HEIGHT
  --fps FPS
  --discard-frames DISCARD_FRAMES
  --stable-frames STABLE_FRAMES
  --motion-threshold MOTION_THRESHOLD
  --min-valid-depth-ratio MIN_VALID_DEPTH_RATIO
  --max-sync-delta-ms MAX_SYNC_DELTA_MS
  --auto-advance
```

## `rgbd-multi-capture`

```text
usage: build_reference.py rgbd-multi-capture [-h]
                                             [--dataset-root DATASET_ROOT]
                                             --batch-id BATCH_ID
                                             --composition COMPOSITION
                                             [--scene-index SCENE_INDEX]
                                             [--captures-per-scene CAPTURES_PER_SCENE]
                                             [--interval-ms INTERVAL_MS]
                                             [--color-width COLOR_WIDTH]
                                             [--color-height COLOR_HEIGHT]
                                             [--depth-width DEPTH_WIDTH]
                                             [--depth-height DEPTH_HEIGHT]
                                             [--fps FPS]
                                             [--discard-frames DISCARD_FRAMES]
                                             [--stable-frames STABLE_FRAMES]
                                             [--motion-threshold MOTION_THRESHOLD]
                                             [--min-valid-depth-ratio MIN_VALID_DEPTH_RATIO]
                                             [--max-sync-delta-ms MAX_SYNC_DELTA_MS]
                                             [--auto-start] [--headless]
                                             [--max-captures MAX_CAPTURES]

options:
  -h, --help            show this help message and exit
  --dataset-root DATASET_ROOT
  --batch-id BATCH_ID
  --composition COMPOSITION
                        scene object counts, e.g. "三棱柱:2,四棱锥:1,圆锥:1"
  --scene-index SCENE_INDEX
                        scene number; zero resumes the latest incomplete scene
  --captures-per-scene CAPTURES_PER_SCENE
  --interval-ms INTERVAL_MS
  --color-width COLOR_WIDTH
  --color-height COLOR_HEIGHT
  --depth-width DEPTH_WIDTH
  --depth-height DEPTH_HEIGHT
  --fps FPS
  --discard-frames DISCARD_FRAMES
  --stable-frames STABLE_FRAMES
  --motion-threshold MOTION_THRESHOLD
  --min-valid-depth-ratio MIN_VALID_DEPTH_RATIO
  --max-sync-delta-ms MAX_SYNC_DELTA_MS
  --auto-start
  --headless
  --max-captures MAX_CAPTURES
```

## `rgbd-dataset-audit`

```text
usage: build_reference.py rgbd-dataset-audit [-h] --data-root DATA_ROOT
                                             [--output-report OUTPUT_REPORT]

options:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
  --output-report OUTPUT_REPORT
```

## `rgbd-review`

```text
usage: build_reference.py rgbd-review [-h] --results-root RESULTS_ROOT
                                      --output-dir OUTPUT_DIR
                                      [--columns COLUMNS]
                                      [--tile-width TILE_WIDTH]
                                      [--tile-height TILE_HEIGHT]

options:
  -h, --help            show this help message and exit
  --results-root RESULTS_ROOT
  --output-dir OUTPUT_DIR
  --columns COLUMNS
  --tile-width TILE_WIDTH
  --tile-height TILE_HEIGHT
```

## `rgbd-edge-audit`

```text
usage: build_reference.py rgbd-edge-audit [-h] --data-root DATA_ROOT
                                          --output-dir OUTPUT_DIR
                                          [--batch-id BATCH_ID]
                                          [--limit-per-class LIMIT_PER_CLASS]

options:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
  --output-dir OUTPUT_DIR
  --batch-id BATCH_ID
  --limit-per-class LIMIT_PER_CLASS
```

## `geometry-rgbd-train`

```text
usage: build_reference.py geometry-rgbd-train [-h] --data-root DATA_ROOT
                                              --output OUTPUT
                                              [--output-report OUTPUT_REPORT]
                                              [--base-model BASE_MODEL]
                                              [--batch-id BATCH_ID]
                                              [--strict-single-object]
                                              [--fused-edges]
                                              [--baseline-output BASELINE_OUTPUT]

options:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
  --output OUTPUT
  --output-report OUTPUT_REPORT
  --base-model BASE_MODEL
                        append new samples to exemplars from an existing RGB-D
                        v3 model
  --batch-id BATCH_ID   train only this capture batch; repeat to select
                        multiple batches
  --strict-single-object
                        reject labelled captures when RGB-D segmentation finds
                        more than one object
  --fused-edges         train the v4 model with RGB/depth-correlated internal-
                        edge features
  --baseline-output BASELINE_OUTPUT
                        with --fused-edges, save a same-sample legacy-feature
                        baseline model
```

## `rgbd-face-audit`

```text
usage: build_reference.py rgbd-face-audit [-h] --frame-dir FRAME_DIR
                                          --background-dir BACKGROUND_DIR
                                          [--output-dir OUTPUT_DIR]

options:
  -h, --help            show this help message and exit
  --frame-dir FRAME_DIR
  --background-dir BACKGROUND_DIR
  --output-dir OUTPUT_DIR
```

## `serve`

```text
usage: build_reference.py serve [-h] [--source {file,uvc,realsense,dual}]
                                [--frame-dir FRAME_DIR]
                                [--background-dir BACKGROUND_DIR]
                                [--rgbd-calibration RGBD_CALIBRATION]
                                [--background BACKGROUND]
                                [--calibration CALIBRATION]
                                [--camera-index CAMERA_INDEX] [--width WIDTH]
                                [--height HEIGHT] [--fps FPS]
                                [--depth-width DEPTH_WIDTH]
                                [--depth-height DEPTH_HEIGHT]
                                [--color-width COLOR_WIDTH]
                                [--color-height COLOR_HEIGHT]
                                [--shape-model SHAPE_MODEL]
                                [--rgbd-shape-model RGBD_SHAPE_MODEL]
                                [--shape-backend {opencv,openvino}]
                                [--shape-device SHAPE_DEVICE]
                                [--side-camera-index SIDE_CAMERA_INDEX]
                                [--side-width SIDE_WIDTH]
                                [--side-height SIDE_HEIGHT]
                                [--side-fps SIDE_FPS]
                                [--dual-calibration DUAL_CALIBRATION]
                                [--side-background SIDE_BACKGROUND]
                                [--side-shape-model SIDE_SHAPE_MODEL]
                                [--cross-view-shape-model CROSS_VIEW_SHAPE_MODEL]
                                [--fusion-policy FUSION_POLICY]
                                [--fusion-backend {auto,opencv,openvino,tensorrt}]
                                [--platform-id {temporary,competition}]
                                [--host HOST] [--port PORT]

options:
  -h, --help            show this help message and exit
  --source {file,uvc,realsense,dual}
  --frame-dir FRAME_DIR
  --background-dir BACKGROUND_DIR
  --rgbd-calibration RGBD_CALIBRATION
  --background BACKGROUND
  --calibration CALIBRATION
  --camera-index CAMERA_INDEX
  --width WIDTH
  --height HEIGHT
  --fps FPS
  --depth-width DEPTH_WIDTH
  --depth-height DEPTH_HEIGHT
  --color-width COLOR_WIDTH
  --color-height COLOR_HEIGHT
  --shape-model SHAPE_MODEL
                        trained geometry RGB model (.npz)
  --rgbd-shape-model RGBD_SHAPE_MODEL
                        trained RGB-D geometry NPZ model
  --shape-backend {opencv,openvino}
  --shape-device SHAPE_DEVICE
  --side-camera-index SIDE_CAMERA_INDEX
  --side-width SIDE_WIDTH
  --side-height SIDE_HEIGHT
  --side-fps SIDE_FPS
  --dual-calibration DUAL_CALIBRATION
  --side-background SIDE_BACKGROUND
  --side-shape-model SIDE_SHAPE_MODEL
  --cross-view-shape-model CROSS_VIEW_SHAPE_MODEL
  --fusion-policy FUSION_POLICY
  --fusion-backend {auto,opencv,openvino,tensorrt}
  --platform-id {temporary,competition}
  --host HOST
  --port PORT
```

## `camera-live`

```text
usage: build_reference.py camera-live [-h] --source {uvc,realsense,dual}
                                      [--camera-index CAMERA_INDEX]
                                      [--width WIDTH] [--height HEIGHT]
                                      [--fps FPS] [--depth-width DEPTH_WIDTH]
                                      [--depth-height DEPTH_HEIGHT]
                                      [--color-width COLOR_WIDTH]
                                      [--color-height COLOR_HEIGHT]
                                      [--shape-model SHAPE_MODEL]
                                      [--rgbd-shape-model RGBD_SHAPE_MODEL]
                                      [--shape-backend {opencv,openvino}]
                                      [--shape-device SHAPE_DEVICE]
                                      [--background BACKGROUND]
                                      [--calibration CALIBRATION]
                                      [--background-dir BACKGROUND_DIR]
                                      [--rgbd-calibration RGBD_CALIBRATION]
                                      [--side-camera-index SIDE_CAMERA_INDEX]
                                      [--side-width SIDE_WIDTH]
                                      [--side-height SIDE_HEIGHT]
                                      [--side-fps SIDE_FPS]
                                      [--dual-calibration DUAL_CALIBRATION]
                                      [--side-background SIDE_BACKGROUND]
                                      [--side-shape-model SIDE_SHAPE_MODEL]
                                      [--cross-view-shape-model CROSS_VIEW_SHAPE_MODEL]
                                      [--fusion-policy FUSION_POLICY]
                                      [--fusion-backend {auto,opencv,openvino,tensorrt}]
                                      [--platform-id {temporary,competition}]
                                      [--headless] [--max-frames MAX_FRAMES]

options:
  -h, --help            show this help message and exit
  --source {uvc,realsense,dual}
  --camera-index CAMERA_INDEX
  --width WIDTH
  --height HEIGHT
  --fps FPS
  --depth-width DEPTH_WIDTH
  --depth-height DEPTH_HEIGHT
  --color-width COLOR_WIDTH
  --color-height COLOR_HEIGHT
  --shape-model SHAPE_MODEL
                        trained geometry RGB model (.npz)
  --rgbd-shape-model RGBD_SHAPE_MODEL
                        trained RGB-D geometry NPZ model
  --shape-backend {opencv,openvino}
  --shape-device SHAPE_DEVICE
  --background BACKGROUND
  --calibration CALIBRATION
  --background-dir BACKGROUND_DIR
  --rgbd-calibration RGBD_CALIBRATION
  --side-camera-index SIDE_CAMERA_INDEX
  --side-width SIDE_WIDTH
  --side-height SIDE_HEIGHT
  --side-fps SIDE_FPS
  --dual-calibration DUAL_CALIBRATION
  --side-background SIDE_BACKGROUND
  --side-shape-model SIDE_SHAPE_MODEL
  --cross-view-shape-model CROSS_VIEW_SHAPE_MODEL
  --fusion-policy FUSION_POLICY
  --fusion-backend {auto,opencv,openvino,tensorrt}
  --platform-id {temporary,competition}
  --headless
  --max-frames MAX_FRAMES
```

## `camera-record`

```text
usage: build_reference.py camera-record [-h] --source {uvc,realsense}
                                        [--camera-index CAMERA_INDEX]
                                        [--width WIDTH] [--height HEIGHT]
                                        [--fps FPS] --session SESSION
                                        [--label LABEL] [--headless]
                                        [--max-frames MAX_FRAMES]

options:
  -h, --help            show this help message and exit
  --source {uvc,realsense}
  --camera-index CAMERA_INDEX
  --width WIDTH
  --height HEIGHT
  --fps FPS
  --session SESSION
  --label LABEL
  --headless
  --max-frames MAX_FRAMES
```

## `geometry-audit`

```text
usage: build_reference.py geometry-audit [-h] --data-root DATA_ROOT
                                         [--output-report OUTPUT_REPORT]

options:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
  --output-report OUTPUT_REPORT
```

## `geometry-train`

```text
usage: build_reference.py geometry-train [-h] --data-root DATA_ROOT
                                         [--additional-data-root ADDITIONAL_DATA_ROOT]
                                         --output OUTPUT
                                         [--feature-set {legacy,edge-topology,structure-topology}]
                                         [--output-report OUTPUT_REPORT]

options:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
  --additional-data-root ADDITIONAL_DATA_ROOT
                        additional labelled batch; exact duplicate images are
                        skipped
  --output OUTPUT
  --feature-set {legacy,edge-topology,structure-topology}
  --output-report OUTPUT_REPORT
```

## `geometry-edge-audit`

```text
usage: build_reference.py geometry-edge-audit [-h] --data-root DATA_ROOT
                                              --output-dir OUTPUT_DIR
                                              [--output-report OUTPUT_REPORT]

options:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
  --output-dir OUTPUT_DIR
  --output-report OUTPUT_REPORT
```

## `geometry-structure-audit`

```text
usage: build_reference.py geometry-structure-audit [-h] --data-root DATA_ROOT
                                                   --output-dir OUTPUT_DIR
                                                   [--output-report OUTPUT_REPORT]

options:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
  --output-dir OUTPUT_DIR
  --output-report OUTPUT_REPORT
```

## `geometry-scene-structure-audit`

```text
usage: build_reference.py geometry-scene-structure-audit [-h]
                                                         --data-root DATA_ROOT
                                                         --output-dir OUTPUT_DIR
                                                         [--output-report OUTPUT_REPORT]

options:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
  --output-dir OUTPUT_DIR
  --output-report OUTPUT_REPORT
```

## `geometry-edge-compare`

```text
usage: build_reference.py geometry-edge-compare [-h] --data-root DATA_ROOT
                                                --legacy-model LEGACY_MODEL
                                                --edge-model EDGE_MODEL
                                                [--output-report OUTPUT_REPORT]

options:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
  --legacy-model LEGACY_MODEL
  --edge-model EDGE_MODEL
  --output-report OUTPUT_REPORT
```

## `geometry-holdout-evaluate`

```text
usage: build_reference.py geometry-holdout-evaluate [-h]
                                                    --training-data-root TRAINING_DATA_ROOT
                                                    --test-data-root TEST_DATA_ROOT
                                                    --model MODEL
                                                    [--output-report OUTPUT_REPORT]

options:
  -h, --help            show this help message and exit
  --training-data-root TRAINING_DATA_ROOT
  --test-data-root TEST_DATA_ROOT
  --model MODEL
  --output-report OUTPUT_REPORT
```

## `geometry-evaluate`

```text
usage: build_reference.py geometry-evaluate [-h] --data-root DATA_ROOT
                                            --model MODEL
                                            [--backend {opencv,openvino}]
                                            [--device DEVICE]
                                            [--output-report OUTPUT_REPORT]

options:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
  --model MODEL
  --backend {opencv,openvino}
  --device DEVICE
  --output-report OUTPUT_REPORT
```

## `geometry-cnn-train`

```text
usage: build_reference.py geometry-cnn-train [-h] --data-root DATA_ROOT
                                             [--additional-data-root ADDITIONAL_DATA_ROOT]
                                             --output OUTPUT [--epochs EPOCHS]
                                             [--seed SEED] [--no-pretrained]
                                             [--resume RESUME]
                                             [--fine-tune-backbone]
                                             [--skip-cross-validation]
                                             [--output-report OUTPUT_REPORT]

options:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
  --additional-data-root ADDITIONAL_DATA_ROOT
                        additional labelled batch; exact duplicate images are
                        skipped
  --output OUTPUT
  --epochs EPOCHS
  --seed SEED
  --no-pretrained
  --resume RESUME       continue from a compatible CNN checkpoint
  --fine-tune-backbone  unfreeze the last MobileNet feature blocks at a lower
                        learning rate
  --skip-cross-validation
  --output-report OUTPUT_REPORT
```

## `geometry-cnn-export`

```text
usage: build_reference.py geometry-cnn-export [-h] --checkpoint CHECKPOINT
                                              --output-dir OUTPUT_DIR
                                              [--precision {fp32,fp16,int8}]
                                              [--data-root DATA_ROOT]
                                              [--output-report OUTPUT_REPORT]

options:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
  --output-dir OUTPUT_DIR
  --precision {fp32,fp16,int8}
  --data-root DATA_ROOT
                        required calibration data for INT8
  --output-report OUTPUT_REPORT
```

## `geometry-benchmark`

```text
usage: build_reference.py geometry-benchmark [-h] --data-root DATA_ROOT
                                             --backend {opencv,openvino}
                                             --model MODEL
                                             [--batch-size {1,12}]
                                             [--warmup WARMUP]
                                             [--iterations ITERATIONS]
                                             [--device DEVICE]
                                             [--output-report OUTPUT_REPORT]

options:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
  --backend {opencv,openvino}
  --model MODEL
  --batch-size {1,12}
  --warmup WARMUP
  --iterations ITERATIONS
  --device DEVICE
  --output-report OUTPUT_REPORT
```

## `geometry-export`

```text
usage: build_reference.py geometry-export [-h] --data-root DATA_ROOT
                                          --model MODEL
                                          --output-dir OUTPUT_DIR
                                          [--backend {opencv,openvino}]
                                          [--device DEVICE]

options:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
  --model MODEL
  --output-dir OUTPUT_DIR
  --backend {opencv,openvino}
  --device DEVICE
```

## `calibrate`

```text
usage: build_reference.py calibrate [-h] --point POINT [--output OUTPUT]

options:
  -h, --help       show this help message and exit
  --point POINT    corner x,y in TL,TR,BR,BL order; repeat four times
  --output OUTPUT
```

## 3. 独立脚本目录

下面按源码列出独立脚本及参数。无参数脚本通常绑定历史数据或固定实验目录，执行前先读文件，不能作为任意新数据的训练命令。参数以各脚本`--help`和源码为准。v7/v8的标注、审查记录与打包脚本绑定既有开发实验，不代替新照片的人工审查。

### [`scripts/audit_color_first_retrain.py`](../../scripts/audit_color_first_retrain.py)

Compare a completed retrain on frozen IDs; never fit or tune a model.

参数：`--run-dir`、`--baseline-dir`、`--baseline-application`、`--frozen-records`

### [`scripts/audit_dual_depth_quality.py`](../../scripts/audit_dual_depth_quality.py)

Audit saved aligned depth frames without loading the dataset into memory.

参数：`--dataset-root`、`--output-dir`、`--worst-count`

### [`scripts/calibrate_reviewed_colors.py`](../../scripts/calibrate_reviewed_colors.py)

Build a local D415 colour profile from explicitly reviewed blue/cyan crops.

参数：`--results-root`、`--base-config`、`--output`

### [`scripts/check_sparse_platform.py`](../../scripts/check_sparse_platform.py)

Offline geometric preflight. A pass is NOT competition/robot acceptance.

参数：`--config`、`--output`

### [`scripts/compare_dual_regression.py`](../../scripts/compare_dual_regression.py)

Compare two fixed-holdout dual runs and build visual review overviews.

参数：`--before`、`--after`、`--output-dir`

### [`scripts/compare_reviewed_rgbd.py`](../../scripts/compare_reviewed_rgbd.py)

Compare multi-02/03 using the existing, provisional appearance assignments.

参数：`--before`、`--after`、`--output-dir`

### [`scripts/compare_rgb_overhang.py`](../../scripts/compare_rgb_overhang.py)

Compare frozen application boxes; generate visual evidence, never train.

参数：`--before`、`--after`、`--output-dir`

### [`scripts/compare_rgbd_composition.py`](../../scripts/compare_rgbd_composition.py)

Compare reviewed RGB-D replays against scene composition, not instance truth.

参数：`--before`、`--after`、`--output`

### [`scripts/compare_rgbd_learners.py`](../../scripts/compare_rgbd_learners.py)

Select depth-feature classifiers on a validation batch; report held-out frames.

参数：`--cache`

### [`scripts/compare_side_color_components.py`](../../scripts/compare_side_color_components.py)

Audit frozen side masks and render RGB/depth/before/after evidence.

参数：`--before`、`--after`、`--output-dir`

### [`scripts/compare_top_holdout_models.py`](../../scripts/compare_top_holdout_models.py)

Compare frozen and augmented RGB-D models on one cached dual holdout.

参数：`--cache`、`--records`、`--model`、`--output`

### [`scripts/debug_native_calibration.py`](../../scripts/debug_native_calibration.py)

Read factory rays and solve a separate, provenance-labelled extrinsic file.

参数：`--session`、`--output-dir`

### [`scripts/dual/synthetic_side_review.py`](../../scripts/dual/synthetic_side_review.py)

离线开发脚本；运行前查看源码。 

参数：`--shape-registry`、`--output`、`--seed`

### [`scripts/dual_apriltag_calibrate.py`](../../scripts/dual_apriltag_calibrate.py)

Live three-AprilTag stereo calibration for top RGB-D plus side RGB.

参数：`--config`、`--platform-id`、`--output`、`--primary-intrinsics`、`--side-intrinsics`、`--generate-tags-dir`、`--session-dir`、`--replay-session`、`--side-camera-index`、`--color-width`、`--color-height`、`--depth-width`、`--depth-height`、`--fps`、`--side-width`、`--side-height`、`--side-fps`、`--tag-size-mm`、`--fixed-tag-a`、`--fixed-tag-b`、`--free-tag`、`--tag-ids`、`--fixed-tag-inset-mm`、`--tray-width-mm`、`--tray-height-mm`、`--dictionary`、`--required-poses`、`--discard-frames`、`--stable-frames`、`--max-corner-motion-px`、`--min-sharpness`、`--min-reference-depth-ratio`、`--detection-width`

### [`scripts/dual_rgbd_side_capture.py`](../../scripts/dual_rgbd_side_capture.py)

Interactive paired capture: top RealSense RGB-D plus side UVC RGB.

参数：`--config`、`--dataset-root`、`--batch-id`、`--platform-id`、`--side-camera-index`、`--color-width`、`--color-height`、`--depth-width`、`--depth-height`、`--fps`、`--side-width`、`--side-height`、`--side-fps`、`--start-label`、`--shape-registry`、`--split`、`--instance-id`、`--color-id`、`--target-per-label`、`--dual-calibration`、`--discard-frames`、`--stable-frames`、`--motion-threshold`、`--min-valid-depth-ratio`、`--max-rgbd-sync-ms`、`--max-pair-delta-ms`、`--min-side-blur`、`--auto-advance`

### [`scripts/evaluate_color_geometry_v8.py`](../../scripts/evaluate_color_geometry_v8.py)

Evaluate frozen v8 replays and a controlled, unchanged-gate refinement ablation.

参数：未静态发现argparse参数；先检查源码中的固定输入和输出。

### [`scripts/evaluate_color_ridge_v7.py`](../../scripts/evaluate_color_ridge_v7.py)

Frozen offline metrics and controlled stage ablations for the v7 experiment.

参数：未静态发现argparse参数；先检查源码中的固定输入和输出。

### [`scripts/evaluate_rgbd_edge_fusion.py`](../../scripts/evaluate_rgbd_edge_fusion.py)

Compare legacy and fused-edge KNN models on identical batch-held-out samples.

参数：`--cache`、`--report`、`--candidate-output`、`--baseline-output`

### [`scripts/experiment_color_geometry_v8.py`](../../scripts/experiment_color_geometry_v8.py)

Refit v6 semantic features on the isolated v7 split; attach v7 colour.

参数：未静态发现argparse参数；先检查源码中的固定输入和输出。

### [`scripts/experiment_color_ridge_v7.py`](../../scripts/experiment_color_ridge_v7.py)

Reproducible isolated v7 development run; never promotes or edits captures.

参数：`phase`、`--output-dir`

### [`scripts/export_pending_labels.py`](../../scripts/export_pending_labels.py)

Export ambiguous frame instances for explicit human label confirmation.

参数：`--results-root`、`--output-dir`

### [`scripts/export_primary_sdk_rays.py`](../../scripts/export_primary_sdk_rays.py)

Read a chosen RealSense factory colour profile without starting streaming.

参数：`--serial`、`--width`、`--height`、`--fps`、`--output`

### [`scripts/investigate_rgbd_accuracy.py`](../../scripts/investigate_rgbd_accuracy.py)

Cache runtime-aligned depth features; offline only, no camera or motion.

参数：`--output`、`--anchor-roi`、`--adaptive-reference-roi`、`--reuse-features`、`--fused-edges`

### [`scripts/package_color_geometry_v8.py`](../../scripts/package_color_geometry_v8.py)

Archive the isolated v8 models, raw observation snapshot and frozen evidence.

参数：未静态发现argparse参数；先检查源码中的固定输入和输出。

### [`scripts/package_color_ridge_v7.py`](../../scripts/package_color_ridge_v7.py)

Create a verified isolated v7 source/model/review archive without captures.

参数：未静态发现argparse参数；先检查源码中的固定输入和输出。

### [`scripts/prepare_reviewed_multi_02_03.py`](../../scripts/prepare_reviewed_multi_02_03.py)

Build reviewed single-object RGB-D samples from multi-02 and multi-03.

参数：`--source-root`、`--detect-root`、`--background-dir`、`--annotations`、`--output-root`、`--batch`、`--train-through`

### [`scripts/probe_connected_dual.py`](../../scripts/probe_connected_dual.py)

Bounded live acquisition check; no classifier or robot commands.

参数：`--side-camera-index`、`--output-dir`

### [`scripts/probe_depth_classification.py`](../../scripts/probe_depth_classification.py)

Offline supervised depth-feature probe; never publishes a runtime model.

参数：`--cache`、`--output`

### [`scripts/probe_depth_metric.py`](../../scripts/probe_depth_metric.py)

Evaluate group-weighted RGB-D distances with train-only normalization.

参数：`--cache`、`--output`、`--model-output`

### [`scripts/replay_dual_application.py`](../../scripts/replay_dual_application.py)

Replay saved pairs through the real application pipeline, without motion.

参数：`--run-dir`、`--config`、`--output-dir`、`--dual-calibration`、`--show-top-edges`、`--show-side-geometry`、`--show-side-edges`、`--show-sparse-geometry`、`--input-records`、`--shape-registry`、`--opencv-threads`

### [`scripts/review_annotations_v7.py`](../../scripts/review_annotations_v7.py)

Persist the independent visual review and manually placed thumbnail annotations.

参数：未静态发现argparse参数；先检查源码中的固定输入和输出。

### [`scripts/review_application_v7.py`](../../scripts/review_application_v7.py)

Persist the completed application contact-sheet inspection and focused views.

参数：未静态发现argparse参数；先检查源码中的固定输入和输出。

### [`scripts/review_application_v8.py`](../../scripts/review_application_v8.py)

Save the actual v8 visual inspection and render frozen-record comparisons.

参数：未静态发现argparse参数；先检查源码中的固定输入和输出。

### [`scripts/review_rgbd_replay.py`](../../scripts/review_rgbd_replay.py)

Replay fixed RGB-D batches with explicit provenance for before/after review.

参数：`--data-root`、`--batch`、`--background-dir`、`--model`、`--config`、`--output-dir`、`--legacy-roi`、`--single-samples`

### [`scripts/rgb_intrinsics_calibrate.py`](../../scripts/rgb_intrinsics_calibrate.py)

Calibrate one camera from its RGB stream using a rigid checkerboard.

参数：`--platform-id`、`--source`、`--camera-id`、`--camera-index`、`--realsense-serial`、`--video-file`、`--video-sample-interval-ms`、`--video-max-adjacent-difference`、`--width`、`--height`、`--fps`、`--backend`、`--fourcc`、`--autofocus`、`--focus`、`--auto-exposure`、`--exposure`、`--auto-white-balance`、`--white-balance`、`--corners-x`、`--corners-y`、`--square-size-mm`、`--required-frames`、`--detection-width`、`--discard-frames`、`--auto-capture`、`--auto-solve`、`--stable-frames`、`--max-corner-motion-px`、`--min-sharpness`、`--headless`、`--auto-interval-ms`、`--session-dir`、`--replay-session`、`--output`、`--generate-board-dir`

### [`scripts/summarize_dual_retrain.py`](../../scripts/summarize_dual_retrain.py)

Summarize an already frozen regression; never selects model parameters.

参数：`--run-dir`、`--baseline-dir`

### [`scripts/summarize_sparse_stereo.py`](../../scripts/summarize_sparse_stereo.py)

Audit a completed sparse reconstruction replay, without selecting thresholds.

参数：`--records`、`--baseline-records`、`--config`、`--calibration`、`--output-dir`

### [`scripts/train_clean_rgbd.py`](../../scripts/train_clean_rgbd.py)

Train from original labelled single-object captures, without legacy exemplars.

参数：`--data-root`、`--output`

### [`scripts/train_dual_fusion_holdout.py`](../../scripts/train_dual_fusion_holdout.py)

Train and replay the actual top RGB-D + side RGB Top-2 fusion chain.

参数：`--samples-root`、`--side-background`、`--config`、`--shape-registry`、`--base-top-model`、`--output-dir`、`--top-model-output`、`--side-model-output`、`--policy-output`、`--cross-model-output`、`--dual-calibration`、`--holdout-per-group`、`--calibration-per-group`、`--seed`、`--full-rgb-instance`、`--fixed-holdout-records`、`--resume-features`

### [`scripts/train_dual_group_holdout.py`](../../scripts/train_dual_group_holdout.py)

Train a provisional side-view model with fixed random holdout per batch/label.

参数：`--samples-root`、`--side-background`、`--shape-registry`、`--output-dir`、`--model-output`、`--holdout-per-group`、`--seed`
