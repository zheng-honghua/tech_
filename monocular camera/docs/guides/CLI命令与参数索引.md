# CLI命令与参数索引

核对日期：2026-10-02。CLI部分由当前代码的`build_parser()`生成；仅构造参数解析器，没有连接相机、训练模型或执行控制。完整操作顺序见[使用手册](单目工程完整使用手册.md)，模块职责见[模块说明](模块使用说明.md)。

## 1. 命令位置与帮助

所有示例先进入双相机工程根目录。CLI全局`--config`放在子命令前；独立脚本的`--config`跟在脚本名后。参数列表中的默认路径相对于当前工作目录。`--help`只查看帮助。

```powershell
Set-Location "C:\Users\d114\Desktop\tech_\monocular camera"
.\.venv\Scripts\python.exe -m sorting_vision.cli --help
.\.venv\Scripts\python.exe `
    -m sorting_vision.cli `
    --config config\d435if-initial.yaml camera-live `
    --help
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-train --help
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
| `rgbd-capture` | capture labelled RealSense RGB-D samples as self-contained folders |
| `rgbd-capture-assistant` | interactively capture and label a complete RealSense geometry batch |
| `rgbd-multi-capture` | batch-capture labelled multi-object RealSense test scenes |
| `rgbd-dataset-audit` | validate RGB-D sample bundles and depth quality |
| `rgbd-review` | build annotated/depth contact sheets for visual review |
| `rgbd-edge-audit` | audit RGB/depth-correlated internal edges by dataset batch |
| `geometry-rgbd-train` | train the metric point-cloud geometry baseline |
| `rgbd-face-audit` | visualise observed 3-D planes and face topology |
| `serve` | serve detection over newline JSON/TCP |
| `camera-live` | preview a UVC or RealSense camera with motion interlock |
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
usage: build_reference.py serve [-h] [--source {file,uvc,realsense}]
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
                                [--shape-device SHAPE_DEVICE] [--host HOST]
                                [--port PORT]

options:
  -h, --help            show this help message and exit
  --source {file,uvc,realsense}
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
  --host HOST
  --port PORT
```

## `camera-live`

```text
usage: build_reference.py camera-live [-h] --source {uvc,realsense}
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
                                      [--headless] [--max-frames MAX_FRAMES]

options:
  -h, --help            show this help message and exit
  --source {uvc,realsense}
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

下面按源码列出独立脚本及参数。无参数脚本通常绑定历史数据或固定实验目录，执行前先读文件，不能作为任意新数据的训练命令。参数以各脚本`--help`和源码为准。本单目工程不包含双机或v7/v8命令。

### [`scripts/calibrate_reviewed_colors.py`](../../scripts/calibrate_reviewed_colors.py)

Build a local D415 colour profile from explicitly reviewed blue/cyan crops.

参数：`--results-root`、`--base-config`、`--output`

### [`scripts/compare_reviewed_rgbd.py`](../../scripts/compare_reviewed_rgbd.py)

Compare multi-02/03 using the existing, provisional appearance assignments.

参数：`--before`、`--after`、`--output-dir`

### [`scripts/compare_rgbd_composition.py`](../../scripts/compare_rgbd_composition.py)

Compare reviewed RGB-D replays against scene composition, not instance truth.

参数：`--before`、`--after`、`--output`

### [`scripts/compare_rgbd_learners.py`](../../scripts/compare_rgbd_learners.py)

Select depth-feature classifiers on a validation batch; report held-out frames.

参数：`--cache`

### [`scripts/evaluate_rgbd_edge_fusion.py`](../../scripts/evaluate_rgbd_edge_fusion.py)

Compare legacy and fused-edge KNN models on identical batch-held-out samples.

参数：`--cache`、`--report`、`--candidate-output`、`--baseline-output`

### [`scripts/export_pending_labels.py`](../../scripts/export_pending_labels.py)

Export ambiguous frame instances for explicit human label confirmation.

参数：`--results-root`、`--output-dir`

### [`scripts/investigate_rgbd_accuracy.py`](../../scripts/investigate_rgbd_accuracy.py)

Cache runtime-aligned depth features; offline only, no camera or motion.

参数：`--output`、`--anchor-roi`、`--adaptive-reference-roi`、`--reuse-features`、`--fused-edges`

### [`scripts/prepare_reviewed_multi_02_03.py`](../../scripts/prepare_reviewed_multi_02_03.py)

Build reviewed single-object RGB-D samples from multi-02 and multi-03.

参数：`--source-root`、`--detect-root`、`--background-dir`、`--annotations`、`--output-root`、`--batch`、`--train-through`

### [`scripts/probe_depth_classification.py`](../../scripts/probe_depth_classification.py)

Offline supervised depth-feature probe; never publishes a runtime model.

参数：`--cache`、`--output`

### [`scripts/probe_depth_metric.py`](../../scripts/probe_depth_metric.py)

Evaluate group-weighted RGB-D distances with train-only normalization.

参数：`--cache`、`--output`、`--model-output`

### [`scripts/review_rgbd_replay.py`](../../scripts/review_rgbd_replay.py)

Replay fixed RGB-D batches with explicit provenance for before/after review.

参数：`--data-root`、`--batch`、`--background-dir`、`--model`、`--config`、`--output-dir`、`--legacy-roi`、`--single-samples`

### [`scripts/train_clean_rgbd.py`](../../scripts/train_clean_rgbd.py)

Train from original labelled single-object captures, without legacy exemplars.

参数：`--data-root`、`--output`
