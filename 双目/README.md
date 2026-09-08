# 双目工作空间（开发基线）

> 当前目录由既有单相机工程复制而来，用于下一步双摄像头开发。现有源码、配置和模型仍按单相机链路工作；尚未实现双相机同步、内外参标定、跨视角配准、遮挡互补、识别融合或抓取坐标变换。完成实机验证前，第二摄像头结果不得驱动机械运动。

# 智能分拣 RGB-D 视觉模块

面向智能分拣赛项的固定俯视 RGB-D 三维视觉系统。项目可完成托盘工作区定位、RGB-D 实例分割、颜色与立体形状识别、可见面拓扑分析、吸盘抓取位姿估计、目标选择以及 JSON/TCP 安全输出。

当前硬件为 Intel RealSense D435if，使用通用 RealSense SDK 适配器；普通 USB/UVC 摄像头仍可用于二维预览、数据采集和算法开发。没有有效深度时，系统始终输出不可执行状态。原来的单目二维 `VisionPipeline` 仅用于兼容和显示，新项目的执行链路应使用 `VisionPipeline3D`。

> **安全边界：**只有整帧 `health.ok=true`，并且对象同时满足 `status=PICKABLE`、`selected=true` 时，控制端才可继续使用 `pose_3d`。RGB-only、实验模型、离线诊断和绿色显示框都不能单独授权机械动作。

## 当前推荐版本

| 类型 | 当前文件 | 用途与限制 |
| --- | --- | --- |
| D435if 初始配置 | [`config/d435if-initial.yaml`](config/d435if-initial.yaml) | 当前相机入口；1280×720 RGB/深度、30 FPS。阈值尚待 D435if 实拍复审。 |
| RGB-D 兼容基线 | [`models/stable/rgbd/geometry-rgbd-multipose-v4.npz`](models/stable/rgbd/geometry-rgbd-multipose-v4.npz) | 原 D415 数据训练的多姿态 KNN，只用于 D435if 迁移对照，重新验证前不视为 D435if 稳定模型。 |
| RGB 开发模型 | [`models/stable/rgb/geometry-rgb-morph-color.npz`](models/stable/rgb/geometry-rgb-morph-color.npz) | 单目图片和 UVC 开发；结果固定不可抓取。 |
| 融合棱线候选 | [`models/experimental/rgbd/`](models/experimental/rgbd/) | 阶段23候选未达到推广门槛，不得用于实际分拣。 |

真实比赛准确率必须在最终相机、光照、托盘和实物样品到位后重新验收。模型分级规则见[模型说明](models/README.md)，历次算法变化、实测效果和已知限制见[算法更新记录](算法更新记录.md)。

## 阅读导航

- [快速开始](#快速开始)：安装依赖并运行不需要相机的合成自检。
- [按任务选择入口](#按任务选择入口)：单图、现有 RGB-D、D435if 采集、训练、复审和服务端分别从哪里开始。
- [项目结构与模块导航](#项目结构与模块导航)：目录职责和源码分层入口。
- [普通摄像头开发](#普通摄像头开发)：UVC 预览与 RGB-only 安全限制。
- [RGB 几何例图与单件分类](#rgb几何例图与单件分类)：OpenCV、结构线和 CNN 实验。
- [RGB-D 数据与标定](#rgb-d数据与标定)：帧格式、空托盘平面和录制帧检测。
- [三维处理流程](#三维处理流程)：RGB-D 主链路。
- [结果协议 v2](#结果协议v2)与[JSON/TCP 控制](#jsontcp控制)：控制端字段和运动互锁。
- [RGB-D 棱线融合实验](#rgb-d-棱线融合实验)：融合候选的训练、审查和未推广结论。
- [完整模块使用手册](docs/guides/模块使用说明.md)：每个 Python 模块、CLI 命令、配置、脚本及对应测试的详细说明。
- [RGB-D 输出判读说明](docs/guides/RGB-D结果颜色与字段说明.md)：框色、`tri` 等缩写、置信度、`selected` 和联系表。
- [视觉审查流程](docs/guides/视觉审查流程.md)：生成联系表后的人工检查清单。
- [D435if 迁移与重新标定](docs/guides/D435if迁移与重新标定.md)：换相机后的流配置、空托盘、模型重验和推广条件。

## 快速开始

以下命令都必须在包含 `pyproject.toml` 的仓库根目录执行。首次使用创建虚拟环境并安装开发依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

.\.venv\Scripts\python.exe -m sorting_vision.cli rgbd-demo --output-dir "output\rgbd-demo"
.\.venv\Scripts\python.exe -m sorting_vision.cli rgbd-benchmark --rounds 30
.\.venv\Scripts\python.exe -m pytest -q
```

后续示例保留 `.\.venv\Scripts\python.exe` 的完整写法，因此无需激活虚拟环境。若仓库被复制或移动后旧 `.venv` 无法启动，应删除并在新路径重新创建虚拟环境，不要把 `.venv` 上传到 GitHub。PowerShell 换行符是行末反引号 `` ` ``；反引号后不能再有空格。

RGB-D演示会生成彩色帧、深度帧、标定文件、标注图、货物裁剪图和`results-v2.json`。合成基准生成30轮、每轮12件的随机位置、旋转、倾斜和深度噪声场景。合成结果只能用于软件回归，不能代替实物验收。

## 按任务选择入口

| 目标 | 推荐入口 | 继续阅读 |
| --- | --- | --- |
| 验证代码能否运行 | `rgbd-demo`、`rgbd-benchmark`、`pytest` | [快速开始](#快速开始) |
| 用一张 RGB 图片识别单个物块 | `predict-image` | [单图预测使用说明](docs/guides/单图预测使用说明.md) |
| 用一张 RGB 图片识别多个分离物块 | `predict-scene` | [多物块场景预测使用说明](docs/guides/多物块场景预测使用说明.md) |
| 用已有 RGB-D 帧测试稳定 v4 | `rgbd-detect` | [RGB-D 数据采集与训练](docs/guides/RGBD数据采集与训练.md) |
| 连续采集单物体训练集 | `rgbd-capture-assistant` | [D435if 数据集拍摄助手](docs/guides/D435if数据集拍摄助手.md) |
| 采集多物体整盘测试集 | `rgbd-multi-capture` | [D435if 多物体批量测试拍摄](docs/guides/D435if多物体批量测试拍摄.md) |
| 检查数据包是否完整 | `rgbd-dataset-audit` | [模块手册：数据与审查](docs/guides/模块使用说明.md#7-数据集采集与视觉审查) |
| 生成 RGB/深度联系表并人工复审 | `rgbd-review` | [视觉审查流程](docs/guides/视觉审查流程.md) |
| 训练多姿态 RGB-D KNN 候选 | `geometry-rgbd-train` | [RGB-D 数据采集与训练](docs/guides/RGBD数据采集与训练.md) |
| 检查可见面或融合棱线 | `rgbd-face-audit`、`rgbd-edge-audit` | [输出颜色与字段说明](docs/guides/RGB-D结果颜色与字段说明.md) |
| 接入机械控制端 | `serve` | [JSON/TCP 控制](#jsontcp控制) |

## 项目结构与模块导航

| 目录/文件 | 作用 | 使用规则 |
| --- | --- | --- |
| [`src/sorting_vision/`](src/sorting_vision/) | 应用代码 | 模块分层、公共 API 和调用示例见[完整模块使用手册](docs/guides/模块使用说明.md)。 |
| [`config/`](config/) | D435if 初始配置、默认参数与 D415 历史复现配置 | 新环境先复制配置生成候选，不要把未验证参数写成“已审”。 |
| [`models/stable/`](models/stable/) | 当前推荐模型 | 稳定表示当前证据下推荐，仍不绕过安全门。 |
| [`models/experimental/`](models/experimental/) | 未推广候选 | 只能离线实验和对照。 |
| [`models/archive/`](models/archive/) | 早期兼容模型 | 仅用于复现，不作为新运行默认值。 |
| [`data/`](data/) | 本机 RGB-D 原始采集 | 数据量大且可能含现场信息；上传前按数据发布策略筛选。 |
| [`fixtures/rgb/`](fixtures/rgb/) | 小型 RGB 测试图集 | 用于开发回归，不等同独立测试集。 |
| [`tests/`](tests/) | Pytest 自动化测试 | 修改对应模块时运行聚焦测试，合入前运行全量测试。 |
| [`scripts/`](scripts/) | 数据集特定的离线调查脚本 | 多数不是稳定公共接口；运行前阅读脚本说明和数据口径。 |
| [`docs/guides/`](docs/guides/) | 使用教程 | 面向安装、采集、训练、预测和视觉审查。 |
| [`docs/reports/`](docs/reports/) | 阶段实测报告 | 保留真实结果、失败候选和已知限制。 |
| [`artifacts/evaluations/`](artifacts/evaluations/) | 关键评估证据 | 用于复核阶段21–23结论。 |
| [`output/`](output/) | 每次运行生成的结果 | 临时目录；为不同测试使用独立子目录，避免覆盖。 |

源码按职责分为六层：

1. 入口与协议：[`cli.py`](src/sorting_vision/cli.py)、[`server.py`](src/sorting_vision/server.py)、[`types.py`](src/sorting_vision/types.py)。
2. 相机、标定与配置：[`camera.py`](src/sorting_vision/camera.py)、[`rgbd.py`](src/sorting_vision/rgbd.py)、[`calibration.py`](src/sorting_vision/calibration.py)、[`config.py`](src/sorting_vision/config.py)。
3. RGB 开发链路：[`pipeline.py`](src/sorting_vision/pipeline.py)、[`segmentation.py`](src/sorting_vision/segmentation.py)、[`classification.py`](src/sorting_vision/classification.py)。
4. RGB-D 执行链路：[`pipeline3d.py`](src/sorting_vision/pipeline3d.py)、[`geometry3d.py`](src/sorting_vision/geometry3d.py)、[`classification3d.py`](src/sorting_vision/classification3d.py)、[`grasp3d.py`](src/sorting_vision/grasp3d.py)。
5. 几何模型与棱线拓扑：[`geometry_rgb.py`](src/sorting_vision/geometry_rgb.py)、[`geometry_rgbd_model.py`](src/sorting_vision/geometry_rgbd_model.py)、[`geometry_edges.py`](src/sorting_vision/geometry_edges.py)、[`face_topology3d.py`](src/sorting_vision/face_topology3d.py)。
6. 数据、审查与测试：[`rgbd_dataset.py`](src/sorting_vision/rgbd_dataset.py)、[`rgbd_review.py`](src/sorting_vision/rgbd_review.py)、[`tests/`](tests/)。

每个模块的输入输出、主要类/函数、对应 CLI 和测试文件见[模块使用说明](docs/guides/模块使用说明.md)。

## 普通摄像头开发

D435if 的成组采集、深度几何训练与离线检测步骤见 [RGB-D 数据采集与训练](docs/guides/RGBD数据采集与训练.md)，换机后的必做工作见 [D435if 迁移与重新标定](docs/guides/D435if迁移与重新标定.md)。连续拍摄整个分类批次推荐使用 [D435if 数据集拍摄助手](docs/guides/D435if数据集拍摄助手.md)，多物块整盘场景的批量采集见 [D435if 多物体批量测试拍摄](docs/guides/D435if多物体批量测试拍摄.md)。每次拍摄会将彩色图、原始深度、预览和内参元数据放进同一个样本文件夹，避免 RGB 与深度错配。

RGB-D 几何模型已加入局部平面拓扑：从物块内部深度拟合可见面，计算法向、二面角、邻接和三面汇聚关系，再与整体点云尺寸及 RGB 边界证据共同分类。可用 `rgbd-face-audit` 检查每个物块实际提取到的平面。

实时 RGB-D 识别已启用白色托盘 ROI 安全门：只处理托盘底面和内沿的物块，托盘外画面不参与分割或抓取。算法、健康字段和实拍回归结果见[托盘ROI识别升级说明.md](docs/reports/托盘ROI识别升级说明.md)。

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

若只想给一张图片直接预测几何类别，请使用`predict-image`。完整中文说明见[单图预测使用说明.md](docs/guides/单图预测使用说明.md)。

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli predict-image `
  "图片路径.jpg" --output-dir "output\single-image-demo"
```

训练数据以父文件夹名作为类别标签，当前支持三棱柱、三棱锥、四棱锥、五棱柱、五棱锥、六棱柱、六棱锥、正八面体和圆锥。训练前先审计图片是否可读、标签是否有效以及是否存在重复样本：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-audit `
  --data-root "fixtures/rgb/batch-01"
```

一张图片中有多个彼此分开的物块时，使用`predict-scene`。它会逐个保存裁剪、掩膜和棱线拓扑诊断；完整说明见[多物块场景预测使用说明.md](docs/guides/多物块场景预测使用说明.md)。

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli predict-scene `
  "多个物块.jpg" --output-dir "output\multi-object"
```

当前默认使用`models/stable/rgb/geometry-rgb-morph-color.npz`。它不使用霍夫变换：先以开运算去除细碎纹理，再用闭运算连接短小断点，同时在Lab空间划分大色块，将稳定的色面边界作为棱线辅助证据。外轮廓、可见面顶点和旧几何特征仍参与分类。场景分割会排除低亮度、低饱和度或细长松散的线缆杂物；超出画面的物块会保留为候选，但固定拒识为`object_out_of_frame`。

当前RGB场景接口只保证处理背景清晰、彼此留有间隔的彩色物块。接触或重叠物块可能被合并为一个候选，应改用分水岭/实例分割和 D435if 深度后再进行抓取验证。

系统提供两个可独立选择、但使用相同预测接口的几何后端：

- `opencv`：HOG、Hu矩、轮廓、边缘方向和明暗面特征，模型保存为NPZ。
- `openvino`：MobileNetV3-Small CNN，训练输入为192×192，部署使用ONNX/OpenVINO。

训练OpenCV轻量RGB基线并执行留一评测：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-train `
  --data-root "fixtures/rgb/batch-01" --output models/experimental/rgb/geometry-rgb-next.npz

.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-evaluate `
  --backend opencv --data-root "fixtures/rgb/batch-01" `
  --model models/archive/rgb/geometry-rgb.npz `
  --output-report output/geometry-evaluation.json
```

模型可接入单图或USB预览：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli detect `
  --image sample.jpg --shape-backend opencv `
  --shape-model models/archive/rgb/geometry-rgb.npz

.\.venv\Scripts\python.exe -m sorting_vision.cli camera-live `
  --source uvc --camera-index 0 --shape-backend opencv `
  --shape-model models/archive/rgb/geometry-rgb.npz
```

当前34张图来自同一批次且每类仅3–6张，评测报告固定标记`same_batch_only=true`，不能作为比赛准确率验收。模型证据不足时返回`unknown`；即使识别出类别，RGB模式仍保持`DEPTH_REQUIRED`和`selected=false`。

将全部例图、标准化裁剪、掩膜、标注图和预测结果整理到新目录：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-export `
  --data-root "fixtures/rgb/batch-01" --model models/archive/rgb/geometry-rgb.npz `
  --output-dir "output/rgb-batch-01-evaluation"
```

为避免误覆盖人工检查结果，目标目录已存在且非空时命令会拒绝执行。

### OpenCV棱线拓扑实验模型

棱线版仍使用阈值分割取得物块掩膜，但分类特征主要来自物块内部棱线、交点、平行/汇聚关系及可见面。它与旧NPZ模型并存，不会覆盖旧模型：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-edge-audit `
  --data-root "fixtures/rgb/batch-01" --output-dir "output/rgb-batch-01-edge-audit"

.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-train `
  --feature-set edge-topology --data-root "fixtures/rgb/batch-01" `
  --output models/experimental/rgb/geometry-rgb-edges-next.npz

.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-edge-compare `
  --data-root "fixtures/rgb/batch-01" `
  --legacy-model models/archive/rgb/geometry-rgb.npz `
  --edge-model models/archive/rgb/geometry-rgb-edges.npz `
  --output-report output/geometry-edge-comparison.json
```

实时切换只需要替换模型路径，后端名称仍为`opencv`：

```powershell
# 旧特征模型
.\.venv\Scripts\python.exe -m sorting_vision.cli camera-live `
  --source uvc --camera-index 0 --shape-backend opencv `
  --shape-model models/archive/rgb/geometry-rgb.npz

# 棱线拓扑模型
.\.venv\Scripts\python.exe -m sorting_vision.cli camera-live `
  --source uvc --camera-index 0 --shape-backend opencv `
  --shape-model models/archive/rgb/geometry-rgb-edges.npz
```

v2模型按棱线拓扑55%、外轮廓20%、HOG与方向20%、亮度5%进行分组距离计算。棱线不足、拓扑矛盾或类别间隔不足会拒识为`unknown`。当前图集来自同一批次，因此新旧对照只能用于开发；棱线版在独立批次证明可靠前保持实验状态。

v3进一步排除了沿物块外轮廓延伸的伪棱，并采用更严格的安全拒识门限。测试目录中与训练集哈希相同的图片会自动排除：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-holdout-evaluate `
  --training-data-root "fixtures/rgb/batch-01" --test-data-root "fixtures/rgb/batch-02" `
  --model models/archive/rgb/geometry-rgb-edges-v3.npz `
  --output-report "output/rgb-batch-02-strict-holdout.json"
```

需要让实时模型学习两个拍摄批次时，使用附加数据目录。训练器按SHA-256去重：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-train `
  --feature-set edge-topology --data-root "fixtures/rgb/batch-01" `
  --additional-data-root "fixtures/rgb/batch-02" `
  --output models/experimental/rgb/geometry-rgb-edges-faces-next.npz
```

扩展模型可用于当前摄像头试验，但测试2已经参与训练，不能再用于泛化验收；应另拍测试3。旧v1/v2模型仍可加载。

### 形态学与色块辅助模型（当前默认）

当前v4模型由测试1、测试2和测试3共120张去重图片训练，并新增了五棱锥与圆锥类别：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-train `
  --feature-set edge-topology --data-root "fixtures/rgb/batch-01" `
  --additional-data-root "fixtures/rgb/batch-02" `
  --additional-data-root "fixtures/rgb/batch-03" `
  --output models/experimental/rgb/geometry-rgb-morph-color-next.npz
```

共131个图片文件，训练器跳过了11个SHA-256完全重复样本。测试3的68张图已全部通过读取与主体预处理，但由于已参与最终训练，其留一法结果仍属于`same_batch_only=true`，不能当作比赛泛化准确率。

`geometry-edge-audit`和`predict-scene`的逐物块目录会额外生成`color-blocks.png`，用于核对色块分割是否对应真实可见面。色块只是辅助，弱色差但梯度清楚的真实棱仍可保留；细小印刷纹理和小色斑会通过面积过滤及开运算移除。旧的`geometry-rgb-edges-faces.npz`仍可用`--model`手动选择。

### 去噪结构线与顶点（CNN输入候选）

`geometry-structure-audit`会将外轮廓拟合为稳定多边形，对内部棱线进行断线共线合并、候选交点重建、端点吸附、重复线删除和最终图连通性复核。它不使用霍夫变换；长期贴着外轮廓的线和悬空端点会被删除，高顶点数的圆滑轮廓不会将连续明暗渐变强制解释为多面体棱。

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-structure-audit `
  --data-root "fixtures/rgb/batch-03" `
  --output-dir "output/structure-audit-v2/test3"
```

对无标签的多物块场景，先拆分互不接触的物块，再逐物块导出结构：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-scene-structure-audit `
  --data-root "fixtures/rgb/mixed-scenes" `
  --output-dir "output/structure-final/mixed-scenes"
```

每张图生成`clean_mask.png`、`raw_edge_map.png`、`clean_line_map.png`、`vertex_heatmap.png`、`structure_overlay.png`和`structure.json`。叠加图中绿线是拟合外轮廓，黄线是通过复核的内部棱，蓝/红/粉点分别是轮廓顶点、内部交点和边界附着点。`总览/<类别>.jpg`将同类所有图排成联系表，用于逐张人工复核。`clean_line_map.png`和`vertex_heatmap.png`也可作为后续双分支CNN的输入。

结构特征已经接入一个独立的实验分类模型，不覆盖当前推荐模型：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-train `
  --feature-set structure-topology `
  --data-root "fixtures/rgb/batch-01" `
  --additional-data-root "fixtures/rgb/batch-02" `
  --additional-data-root "fixtures/rgb/batch-03" `
  --output models/experimental/rgb/geometry-rgb-structure-next.npz

.\.venv\Scripts\python.exe -m sorting_vision.cli predict-scene `
  "fixtures/rgb/mixed-scenes/WIN_20260826_09_02_14_Pro.jpg" `
  --model models/experimental/rgb/geometry-rgb-structure.npz `
  --output-dir output/structure-recognition/scene
```

该模型在测试3的同批次留一评测中准确率为22.1%，当前只用于比较和继续研究，不能用于实际分拣。训练图片回放准确率会因样本参与建模而虚高，不作为成功率。

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
  --data-root "fixtures/rgb/batch-01" `
  --additional-data-root "fixtures/rgb/batch-02" `
  --additional-data-root "fixtures/rgb/batch-03" `
  --output models/experimental/cnn/geometry-cnn-next.pt `
  --output-report output/geometry-cnn-training.json
```

如果训练需要分段进行，可使用`--resume models/experimental/cnn/geometry-cnn-next.pt`继续训练。程序会核对类别顺序和源图哈希，数据不一致时拒绝续训。
默认先冻结ImageNet特征骨干训练分类头；分类头稳定后可组合`--resume`与`--fine-tune-backbone`，使用较小学习率微调最后三个特征块。

多个数据目录会按SHA-256自动去重。默认三折评测仍是图片级分层，不是批次独立验证；要测量真实泛化能力，必须保留一个完全不参与训练的新批次。

Windows下若项目路径包含中文，PyTorch的DLL自动搜索可能失败。训练入口会在导入Torch前预加载wheel自带DLL，因此应始终通过`python -m sorting_vision.cli geometry-cnn-train`启动，不要用临时脚本直接`import torch`判断项目是否可训练。

CNN分类器的训练单位始终是“一个完整物块裁剪”。整盘或复合图像不能直接使用整张图的文件夹标签训练分类器；正式采集复合图时，应标注每个物块的边界框或实例掩膜，再由数据工具裁成单物块样本。复合图像主要用于训练/验证多目标分割、接触物块拆分和整盘漏检率。相机内参、托盘平面和机械外参属于几何标定，不属于CNN训练标签。

导出ONNX并使用当前图集进行INT8校准。量化后训练集回放准确率下降超过2个百分点时，部署元数据会自动选择FP16模型：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-cnn-export `
  --checkpoint models/experimental/cnn/geometry-cnn.pt `
  --output-dir models/experimental/cnn/openvino-next `
  --precision int8 --data-root "fixtures/rgb/batch-01"
```

评测并接入摄像头：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-evaluate `
  --backend openvino --model models/experimental/cnn/openvino `
  --data-root "fixtures/rgb/batch-01"

.\.venv\Scripts\python.exe -m sorting_vision.cli camera-live `
  --source uvc --camera-index 0 --shape-backend openvino `
  --shape-model models/experimental/cnn/openvino --shape-device CPU
```

CNN最大概率低于0.65或前两类概率差小于0.12时返回`unknown`。空画面、多主体和主体出界在进入模型前直接拒绝。OpenCV与CNN不会自动投票，代码中仅保留未启用的组合接口。

当前实验模型使用119张有效去重图片、9个类别，经18轮冻结骨干训练和4轮末端特征块微调。一张多主体图被跳过。训练回放原始Top-1为55.0%，安全门限下接受17/120且零错误；这不是泛化准确率。用测试1+2训练、测试3已知类别严格留出时，原始Top-1为42.3%，安全门限下只正确接受1/52且错误接受1/52，因此当前CNN仍只能用于开发预览，不得用于机械执行。

已导出`models/experimental/cnn/openvino`的FP32模型，PyTorch与OpenVINO在119张有效图上Top-1完全一致。当前开发电脑测得单件P95约9.7 ms、12件批量P95约79.9 ms；N100性能仍需实机验证。CNN仍属于实验预览模型。

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
  --backend openvino --model models/experimental/cnn/openvino \
  --data-root "fixtures/rgb/batch-01" --batch-size 1

python -m sorting_vision.cli geometry-benchmark \
  --backend openvino --model models/experimental/cnn/openvino \
  --data-root "fixtures/rgb/batch-01" --batch-size 12
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
.\.venv\Scripts\python.exe -m sorting_vision.cli `
  --config config/d435if-initial.yaml rgbd-detect `
  --frame-dir data/scene-frame `
  --rgbd-calibration rgbd-calibration.json `
  --rgbd-shape-model models/stable/rgbd/geometry-rgbd-multipose-v4.npz `
  --output-dir output/real
```

也可以用`--background-dir`代替现成标定，程序会自动拟合托盘平面并暂时使用单位外参。省略 `--rgbd-shape-model` 时使用规则几何基线，不会自动加载稳定 v4，因此正式比较必须显式传入模型路径。

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

检测框颜色、`tri/pent/hex` 等缩写、两行置信度、`selected`、联系表顶部字段和棱线审查图的完整判读方法见[RGB-D结果颜色与字段说明](docs/guides/RGB-D结果颜色与字段说明.md)。特别注意：框色表示安全状态，不是分类真值颜色；绿色框也不能单独证明类别正确或授权机械动作。

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

## 接入 D435if 和模型

安装可选 RealSense 依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[realsense]"
```

D435if 实时模式要求使用当前相机重新拍摄的空托盘帧或重新生成的 RGB-D 标定：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli `
  --config config/d435if-initial.yaml camera-live `
  --source realsense --rgbd-calibration rgbd-calibration.json `
  --rgbd-shape-model models/stable/rgbd/geometry-rgbd-multipose-v4.npz
```

- `RealSenseSource` 按 D435if 配置请求 1280×720、30 FPS 彩色和深度流，将深度对齐到彩色图并读取设备深度比例、内参和两路时间戳；`RealSenseD415Source` 仅作为旧代码兼容类保留。
- 实现`sorting_vision.camera.RGBDSource.read()`仍可接入其他厂商SDK，不需要修改三维流水线。
- 实现`sorting_vision.classification3d.ShapeModel3D.classify()`即可接入RGB-D或点云神经网络。
- 几何分类器是无样品情况下的基线；样品到位后，应以多姿态真实数据训练模型，二维轮廓只能作为辅助校验。
- D435if 的更宽深度视场和不同成像分布会改变 ROI、深度边缘和模型特征；必须重新拍空托盘、单物体多姿态和多物体批次。原 D415 v4 只能作为迁移基线。
- 每种立体类别至少采集200个独立姿态，并覆盖侧躺、不同面朝上、倾斜、接触、遮挡、黑色表面、反光和深度孔洞；数据按采集批次划分训练、验证和测试集。

实物验收目标为组合类别准确率不低于99%、抓取位置误差P95不超过3 mm、法向误差P95不超过5°，并完成30轮完整托盘零误分拣测试。

## 旧二维兼容命令

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli demo --output-dir output/demo
.\.venv\Scripts\python.exe -m sorting_vision.cli benchmark --rounds 30
```

这些命令只验证单目二维旧接口，不代表立体识别能力。

## RGB-D 棱线融合实验

RGB-D v4 融合实验会将弱 RGB 梯度线与深度平面边界逐条核对，并追加棱数量、长度、RGB/深度互相支持率及交点关系等特征。旧 v1–v3 NPZ 继续使用原特征路径，不会因为升级代码而增加融合计算。

先按批次导出可视化审查结果：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli `
  --config config/d415-reviewed-20260905.yaml `
  rgbd-edge-audit --data-root data/rgbd-pilot `
  --batch-id pilot-01 --batch-id pilot-02 --batch-id pilot-03 `
  --limit-per-class 10 --output-dir output/rgbd-edge-audit
```

训练融合候选时可同时保存完全相同样本的旧特征基线：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli `
  --config config/d415-reviewed-20260905.yaml geometry-rgbd-train `
  --data-root data/rgbd-pilot --strict-single-object --fused-edges `
  --output models/experimental/rgbd/geometry-rgbd-fused-candidate-next.npz `
  --baseline-output models/experimental/rgbd/geometry-rgbd-fused-baseline-next.npz
```

2026-09-08 的候选未通过跨批次和多物体推广门槛，因此实时运行仍推荐 `models/stable/rgbd/geometry-rgbd-multipose-v4.npz`，不要使用 `models/experimental/` 下的模型执行分拣。实测与视觉审查见[RGB-D棱线融合试验报告](docs/reports/RGB-D棱线融合试验报告_20260908.md)。模型分级、归档规则和当前工作区结构见[工作空间说明](WORKSPACE.md)。
