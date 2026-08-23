# 智能分拣 RGB-D 视觉模块

面向智能分拣赛项的固定俯视 RGB-D 三维视觉系统。系统识别任意可见姿态的立体物块，并向吸盘控制端输出机器人坐标系下的三维位置、表面法向、接近方向、四元数和抓取质量。

当前没有真实相机、样品或CAD，因此仓库包含厂商无关的RGB-D框架、常见立体几何基线、文件回放、JSON/TCP服务和可重复合成验证。真实比赛准确率必须在相机和样品到位后重新验收。

原来的单目二维`VisionPipeline`仍然保留用于兼容和显示，但它只能处理平面投影，不能作为任意三维姿态的安全抓取依据。新项目应使用`VisionPipeline3D`。

## 快速开始

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli rgbd-demo --output-dir output/rgbd-demo
.\.venv\Scripts\python.exe -m sorting_vision.cli rgbd-benchmark --rounds 30
.\.venv\Scripts\python.exe -m pytest
```

RGB-D演示会生成彩色帧、深度帧、标定文件、标注图、货物裁剪图和`results-v2.json`。合成基准生成30轮、每轮12件的随机位置、旋转、倾斜和深度噪声场景。合成结果只能用于软件回归，不能代替实物验收。

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
{"type":"ack_pick","request_id":"2"}
{"type":"health","request_id":"3"}
```

机械端完成抓取后必须发送`ack_pick`，视觉端会清除旧目标并要求新的连续帧确认。外参无效、全局有效深度不足或托盘平面偏移超限时，健康状态为故障并禁止选择目标。

## 接入真实相机和模型

- 实现`sorting_vision.camera.RGBDSource.read()`即可接入任意厂商SDK，不需要修改三维流水线。
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
