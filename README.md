# 智能分拣视觉模块

这是面向智能分拣赛项的独立视觉算法基线。它从固定俯视相机图像中输出货物的颜色、形状、抓取点、平面角度、置信度和标准化裁剪图，不包含机械控制、储物盒映射或显示界面。

当前版本在没有真实样品和训练权重的情况下即可运行：使用背景差分、分水岭、Lab颜色分类和轮廓几何分类。`InstanceSegmenter`与`HybridShapeClassifier`提供训练模型接入点，之后可以无缝替换为轻量实例分割和形状分类模型。

## 快速开始

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli demo --output-dir output/demo
.\.venv\Scripts\python.exe -m sorting_vision.cli benchmark --rounds 30
.\.venv\Scripts\python.exe -m pytest
```

演示输出包括`scene.png`、`annotated.png`、每个货物的裁剪图和`results.json`。
合成基准会生成30轮、每轮12件的随机旋转/位置场景，并报告识别准确率、唯一目标选择和单帧耗时。它只验证软件回归，不能代替真实样品验收。

检测真实图片：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli detect `
  --image data/frame.png `
  --background data/empty-tray.png `
  --calibration calibration.json `
  --output-dir output/real
```

建议始终拍摄一张相同曝光、相同位置的空托盘图。没有`--background`时，程序会使用图像边缘的中值颜色估计托盘背景，稳定性较低。

## 四点标定

按左上、右上、右下、左下顺序提供托盘内边界角点：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli calibrate `
  --point 120,80 --point 1710,95 --point 1690,970 --point 135,955 `
  --output calibration.json
```

输出坐标原点为矫正后托盘左下角，X轴向右，Y轴向上。托盘实际尺寸和矫正分辨率在`config/default.yaml`中设置。

## 控制端约定

- 控制端只执行`selected=true`且`status=PICKABLE`的结果。
- 同一目标必须在连续两帧中类别一致、位置差不超过3 mm才会被选中。
- 机械端完成取件后调用`pipeline.reset_tracking()`，然后重新采图；不得复用旧坐标。
- `class_key`采用`颜色ID:形状ID`格式，储物盒映射由控制端维护。
- 圆形、正方形和正六边形等旋转对称物体的`angle_deg`为`null`。

## 模型扩展

- 实例分割：实现`sorting_vision.segmentation.InstanceSegmenter.segment(image)`并返回二值掩膜列表，然后注入`VisionPipeline(instance_model=...)`。
- 形状模型：传入接收`(BGR裁剪图, 二值掩膜)`并返回`(shape_id, confidence)`的可调用对象。
- 二维码：命令行添加`--qrcode`即可启用OpenCV二维码扩展。
- OCR和缺陷：使用`CallableExtension("ocr", analyzer)`或`CallableExtension("defect", analyzer)`注入，主输出协议保持不变。

## 样品到位后的工作

每种颜色和形状至少采集200张独立物体图，并采集不少于100组完整托盘场景。数据需覆盖旋转、接触、遮挡、阴影、反光和不同色温，并按采集批次划分训练、验证、测试集。实际色卡应更新`config/default.yaml`中的颜色原型。
