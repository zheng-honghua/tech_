# 智能分拣 RGB-D 视觉模块

面向智能分拣赛项的固定俯视 RGB-D 三维视觉系统。系统识别任意可见姿态的立体物块，并向吸盘控制端输出机器人坐标系下的三维位置、表面法向、接近方向、四元数和抓取质量。

当前可使用普通USB摄像头进行预览、二维临时识别和数据采集；计划使用的Intel RealSense D415通过可选适配器接入。没有深度时系统始终禁止下发抓取。真实比赛准确率必须在相机和样品到位后重新验收。

原来的单目二维`VisionPipeline`仍然保留用于兼容和显示，但它只能处理平面投影，不能作为任意三维姿态的安全抓取依据。新项目应使用`VisionPipeline3D`。

## 快速开始

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli rgbd-demo --output-dir output/rgbd-demo
.\.venv\Scripts\python.exe -m sorting_vision.cli rgbd-benchmark --rounds 30
.\.venv\Scripts\python.exe -m pytest
```

RGB-D演示会生成彩色帧、深度帧、标定文件、标注图、货物裁剪图和`results-v2.json`。合成基准生成30轮、每轮12件的随机位置、旋转、倾斜和深度噪声场景。合成结果只能用于软件回归，不能代替实物验收。

## 普通摄像头开发

USB/UVC摄像头实时预览（默认1280×720、30 FPS；设备索引按Windows当前枚举结果指定）：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli camera-live --source uvc --camera-index 0
```

窗口中按`s`模拟工具头开始运动，按`r`模拟停止，按`q`退出。运动期间只保留原始预览，不运行识别，也不显示旧目标。RGB模式的结果统一为`DEPTH_REQUIRED`，`pose_3d`和`grasp`为空且`selected=false`。

采集带清单的数据：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli camera-record `
  --source uvc --session data/session-001 --label red-block
```

无显示器运行时增加`--headless`；自动测试时可用`--max-frames 100`限制帧数。

## RGB几何例图与单件分类

若只想给一张图片直接预测几何类别，请使用`predict-image`。完整中文说明见[单图预测使用说明.md](单图预测使用说明.md)。

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli predict-image `
  "图片路径.jpg" --output-dir "output\single-image-demo"
```

文件夹名作为类别标签，当前支持三棱柱、三棱锥、四棱锥、五棱柱、六棱柱、六棱锥和正八面体。先审计图片：

一张图片中有多个彼此分开的物块时，使用`predict-scene`。它会逐个保存裁剪、掩膜和棱线拓扑诊断；完整说明见[多物块场景预测使用说明.md](多物块场景预测使用说明.md)。

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli predict-scene `
  "多个物块.jpg" --output-dir "output\multi-object"
```

当前默认使用`models/geometry-rgb-morph-color.npz`。它不使用霍夫变换：先以开运算去除细碎纹理，再用闭运算连接短小断点，同时在Lab空间划分大色块，将稳定的色面边界作为棱线辅助证据。外轮廓、可见面顶点和旧几何特征仍参与分类。场景分割会排除低亮度、低饱和度或细长松散的线缆杂物；超出画面的物块会保留为候选，但固定拒识为`object_out_of_frame`。

当前RGB场景接口只保证处理背景清晰、彼此留有间隔的彩色物块。接触或重叠物块可能被合并为一个候选，应改用分水岭/实例分割和D415深度后再进行抓取验证。

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-audit `
  --data-root "几何测试_1"
```

系统提供两个可独立选择、但使用相同预测接口的几何后端：

- `opencv`：HOG、Hu矩、轮廓、边缘方向和明暗面特征，模型保存为NPZ。
- `openvino`：MobileNetV3-Small CNN，训练输入为192×192，部署使用ONNX/OpenVINO。

训练OpenCV轻量RGB基线并执行留一评测：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-train `
  --data-root "几何测试_1" --output models/geometry-rgb.npz

.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-evaluate `
  --backend opencv --data-root "几何测试_1" `
  --model models/geometry-rgb.npz `
  --output-report output/geometry-evaluation.json
```

模型可接入单图或USB预览：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli detect `
  --image sample.jpg --shape-backend opencv `
  --shape-model models/geometry-rgb.npz

.\.venv\Scripts\python.exe -m sorting_vision.cli camera-live `
  --source uvc --camera-index 0 --shape-backend opencv `
  --shape-model models/geometry-rgb.npz
```

当前34张图来自同一批次且每类仅3–6张，评测报告固定标记`same_batch_only=true`，不能作为比赛准确率验收。模型证据不足时返回`unknown`；即使识别出类别，RGB模式仍保持`DEPTH_REQUIRED`和`selected=false`。

将全部例图、标准化裁剪、掩膜、标注图和预测结果整理到新目录：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-export `
  --data-root "几何测试_1" --model models/geometry-rgb.npz `
  --output-dir "几何测试_1_计算结果"
```

为避免误覆盖人工检查结果，目标目录已存在且非空时命令会拒绝执行。

### OpenCV棱线拓扑实验模型

棱线版仍使用阈值分割取得物块掩膜，但分类特征主要来自物块内部棱线、交点、平行/汇聚关系及可见面。它与旧NPZ模型并存，不会覆盖旧模型：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-edge-audit `
  --data-root "几何测试_1" --output-dir "几何测试_1_棱线分析"

.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-train `
  --feature-set edge-topology --data-root "几何测试_1" `
  --output models/geometry-rgb-edges.npz

.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-edge-compare `
  --data-root "几何测试_1" `
  --legacy-model models/geometry-rgb.npz `
  --edge-model models/geometry-rgb-edges.npz `
  --output-report output/geometry-edge-comparison.json
```

实时切换只需要替换模型路径，后端名称仍为`opencv`：

```powershell
# 旧特征模型
.\.venv\Scripts\python.exe -m sorting_vision.cli camera-live `
  --source uvc --camera-index 0 --shape-backend opencv `
  --shape-model models/geometry-rgb.npz

# 棱线拓扑模型
.\.venv\Scripts\python.exe -m sorting_vision.cli camera-live `
  --source uvc --camera-index 0 --shape-backend opencv `
  --shape-model models/geometry-rgb-edges.npz
```

v2模型按棱线拓扑55%、外轮廓20%、HOG与方向20%、亮度5%进行分组距离计算。棱线不足、拓扑矛盾或类别间隔不足会拒识为`unknown`。当前图集来自同一批次，因此新旧对照只能用于开发；棱线版在独立批次证明可靠前保持实验状态。

v3进一步排除了沿物块外轮廓延伸的伪棱，并采用更严格的安全拒识门限。测试目录中与训练集哈希相同的图片会自动排除：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-holdout-evaluate `
  --training-data-root "几何测试_1" --test-data-root "几何测试_2" `
  --model models/geometry-rgb-edges-v3.npz `
  --output-report "几何测试_2_优化结果/strict-holdout.json"
```

需要让实时模型学习两个拍摄批次时，使用附加数据目录。训练器按SHA-256去重：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-train `
  --feature-set edge-topology --data-root "几何测试_1" `
  --additional-data-root "几何测试_2" `
  --output models/geometry-rgb-edges-faces.npz
```

扩展模型可用于当前摄像头试验，但测试2已经参与训练，不能再用于泛化验收；应另拍测试3。旧v1/v2模型仍可加载。

### 形态学与色块辅助模型（当前默认）

当前v4模型由测试1和测试2共52张去重图片训练，命令如下：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-train `
  --feature-set edge-topology --data-root "几何测试_1" `
  --additional-data-root "几何测试_2" `
  --output models/geometry-rgb-morph-color.npz
```

`geometry-edge-audit`和`predict-scene`的逐物块目录会额外生成`color-blocks.png`，用于核对色块分割是否对应真实可见面。色块只是辅助，弱色差但梯度清楚的真实棱仍可保留；细小印刷纹理和小色斑会通过面积过滤及开运算移除。旧的`geometry-rgb-edges-faces.npz`仍可用`--model`手动选择。

测试图片统一采用`类别ID_batch批次_序号`命名，例如`triangular_pyramid_batch02_003.jpg`；父文件夹仍是唯一真实标签。测试1与测试2中SHA-256相同的11张图片会自动排除，不能重复计入独立评测。

根据测试1训练、测试2去重留出的结果，v4增加了逐类别安全门限：六棱柱需要更大的类别间隔，六棱锥同时检查类别距离。在当前18张严格留出图片上，正确接受2张、错误接受由2张降为0；其余返回`unknown`。这项改进优先降低误分拣，不代表总体识别率已达到比赛要求。

### CNN训练和OpenVINO导出

训练依赖与部署依赖相互隔离。开发电脑安装训练环境：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[cnn-train]"
```

训练MobileNetV3-Small。默认执行确定性的分层三折同批次评测，然后使用全部图片训练最终模型：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-cnn-train `
  --data-root "几何测试_1" --output models/geometry-cnn.pt `
  --output-report output/geometry-cnn-training.json
```

导出ONNX并使用当前图集进行INT8校准。量化后训练集回放准确率下降超过2个百分点时，部署元数据会自动选择FP16模型：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-cnn-export `
  --checkpoint models/geometry-cnn.pt `
  --output-dir models/geometry-cnn-openvino `
  --precision int8 --data-root "几何测试_1"
```

评测并接入摄像头：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-evaluate `
  --backend openvino --model models/geometry-cnn-openvino `
  --data-root "几何测试_1"

.\.venv\Scripts\python.exe -m sorting_vision.cli camera-live `
  --source uvc --camera-index 0 --shape-backend openvino `
  --shape-model models/geometry-cnn-openvino --shape-device CPU
```

CNN最大概率低于0.65或前两类概率差小于0.12时返回`unknown`。空画面、多主体和主体出界在进入模型前直接拒绝。OpenCV与CNN不会自动投票，代码中仅保留未启用的组合接口。

### Ubuntu N100部署和测速

N100只安装运行依赖，不安装PyTorch：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[cnn]"
```

默认使用OpenVINO CPU后端。分别测试单件和12件批处理的P50/P95：

```bash
python -m sorting_vision.cli geometry-benchmark \
  --backend openvino --model models/geometry-cnn-openvino \
  --data-root "几何测试_1" --batch-size 1

python -m sorting_vision.cli geometry-benchmark \
  --backend openvino --model models/geometry-cnn-openvino \
  --data-root "几何测试_1" --batch-size 12
```

每项默认预热20次并测试200次。目标为单件P95不超过30 ms、12件批量P95不超过150 ms；实际N100结果才是最终结论。

## RGB-D数据与标定

一帧回放数据使用以下目录结构：

```text
frame/
  color.png
  depth.npy
  metadata.json
```

`depth.npy`保存相机原始深度值，`metadata.json`中的`depth_scale_to_mm`负责转换为毫米。彩色图和深度图必须已经对齐。

从空托盘帧拟合托盘平面并生成标定：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli rgbd-calibrate `
  --background-dir data/empty-tray-frame `
  --camera-to-robot camera-to-robot.json `
  --output rgbd-calibration.json
```

`camera-to-robot.json`是4×4齐次变换矩阵。省略时使用单位矩阵，适合算法测试但不能直接控制真实机械机构。

检测录制帧：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli rgbd-detect `
  --frame-dir data/scene-frame `
  --rgbd-calibration rgbd-calibration.json `
  --output-dir output/real
```

也可以用`--background-dir`代替现成标定，程序会自动拟合托盘平面并暂时使用单位外参。

## 三维处理流程

1. 根据内参把深度像素反投影为相机坐标系点云。
2. 以空托盘深度拟合托盘平面，根据离平面高度提取物块。
3. 通过三维高度、RGB边界和分水岭拆分接触物块。
4. 从有效可见表面识别颜色；使用点云尺寸、主轴、平面误差、曲率和轮廓辅助特征识别常见立体几何体。
5. 在物块表面搜索满足吸盘直径、边缘余量、平面度、有效深度率和法向倾角要求的抓取点。
6. 将抓取点和方向变换到机器人坐标系，连续两帧稳定后只选择一个目标。

默认吸盘直径15 mm、最大表面倾角35°，全部安全阈值位于`config/default.yaml`。

## 结果协议v2

控制端只执行`selected=true`且`status=PICKABLE`的结果。核心字段：

```json
{
  "schema_version": 2,
  "class_key": "yellow:cuboid",
  "pose_3d": {
    "position_mm": {"x": 12.3, "y": 41.2, "z": 672.1},
    "quaternion_xyzw": {"x": 0, "y": 0, "z": 0, "w": 1},
    "surface_normal": {"x": 0, "y": 0, "z": -1},
    "approach_vector": {"x": 0, "y": 0, "z": 1}
  },
  "grasp": {
    "cup_diameter_mm": 15,
    "flatness_rmse_mm": 0.2,
    "edge_clearance_mm": 13.5,
    "valid_depth_ratio": 1,
    "score": 0.96
  },
  "status": "PICKABLE",
  "selected": true
}
```

`center_mm`和`angle_deg`仅为旧界面兼容字段。真实控制必须使用`pose_3d`。

状态包括：

- `PICKABLE`：类别和抓取面均通过安全门限。
- `UNCERTAIN`：颜色或立体类别置信度不足。
- `OCCLUDED`：靠近边界、接触或抓取空间不足。
- `DEPTH_INVALID`：物块深度缺失或全局深度/托盘平面异常。
- `NO_GRASP_SURFACE`：没有满足吸盘要求的表面。
- `DEPTH_REQUIRED`：普通RGB开发结果，只能显示，不能抓取。

## JSON/TCP控制

启动本地服务：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli serve `
  --frame-dir data/scene-frame `
  --rgbd-calibration rgbd-calibration.json
```

服务使用一行一个JSON消息，支持：

```json
{"type":"detect","request_id":"1"}
{"type":"motion_start","request_id":"2"}
{"type":"motion_stop","request_id":"3"}
{"type":"ack_pick","request_id":"4"}
{"type":"health","request_id":"5"}
```

`motion_start`会立即清除旧目标；控制端必须收到其确认响应后才能开始运动。运动中相机仍取流，但`detect`只返回`BUSY_MOVING`和空结果。`motion_stop`后默认丢弃8帧、至少等待300 ms并要求连续3帧画面稳定，超时则返回`MOTION_UNSTABLE`。机械端完成抓取后还必须发送`ack_pick`。

## 接入D415和模型

安装可选D415依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[realsense]"
```

D415实时模式要求已有空托盘帧或RGB-D标定：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli camera-live `
  --source realsense --rgbd-calibration rgbd-calibration.json
```

- `RealSenseD415Source`以640×480、30 FPS请求彩色和深度流，将深度对齐到彩色图并读取设备深度比例、内参和两路时间戳。
- 实现`sorting_vision.camera.RGBDSource.read()`仍可接入其他厂商SDK，不需要修改三维流水线。
- 实现`sorting_vision.classification3d.ShapeModel3D.classify()`即可接入RGB-D或点云神经网络。
- 几何分类器是无样品情况下的基线；样品到位后，应以多姿态真实数据训练模型，二维轮廓只能作为辅助校验。
- 每种立体类别至少采集200个独立姿态，并覆盖侧躺、不同面朝上、倾斜、接触、遮挡、黑色表面、反光和深度孔洞；数据按采集批次划分训练、验证和测试集。

实物验收目标为组合类别准确率不低于99%、抓取位置误差P95不超过3 mm、法向误差P95不超过5°，并完成30轮完整托盘零误分拣测试。

## 旧二维兼容命令

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli demo --output-dir output/demo
.\.venv\Scripts\python.exe -m sorting_vision.cli benchmark --rounds 30
```

这些命令只验证单目二维旧接口，不代表立体识别能力。
