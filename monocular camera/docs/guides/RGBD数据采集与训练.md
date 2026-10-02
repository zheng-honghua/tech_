# RGB-D单相机数据采集、平面标定与训练

更新：2026-10-02。此页处理一台RealSense的color.png／depth.npy／metadata.json格式。本工程不读取侧相机进行融合。

## 1. 帧格式与单位

```text
sample/
  color.png
  depth.npy
  metadata.json
  depth-preview.png  （数据集采集时的预览）
```

RGB与深度已对齐。depth.npy是原始深度数值，metadata中的depth_scale_to_mm转换为毫米；RGBDFrame.depth_mm提供转换后的数组。无效零值不能参与点云或平面拟合。彩色预览不是测量数据。

内参、尺寸、深度比例、设备与时间戳必须和该帧一致，不借用另一个相机的metadata。

## 2. 环境与单批采集

```powershell
Set-Location "C:\Users\d114\Desktop\tech_\monocular camera"
.\.venv\Scripts\python.exe `
    -m sorting_vision.cli `
    --config config\d435if-initial.yaml rgbd-capture `
    --dataset-root data\rgbd-new `
    --batch-id pilot-20261002 `
    --label empty_tray `
    --count 5
```

当前配置决定流参数；主动改档位时显式传color-width／height、depth-width／height和fps，并重核标定。正常手动窗口按空格保存，Q退出；headless自动保存需正数count并先确认画面。

每个物块批次先拍空盘，单物块按真实label采集。连续逐类操作见[拍摄助手](D435if数据集拍摄助手.md)。

## 3. 目录与清单

```text
data/rgbd-new/
  manifest.jsonl
  pilot-20261002/
    empty_tray/<sample-id>/
    cone/<sample-id>/
```

manifest和metadata记录batch、label及质量来源。仅复制color.png到一个文件夹不能得到RGB-D训练样本。不要删照片留下清单，拍错先记录并审查。

## 4. 数据审计与看图

```powershell
.\.venv\Scripts\python.exe `
    -m sorting_vision.cli rgbd-dataset-audit `
    --data-root data\rgbd-new `
    --output-report output\rgbd-audit-20261002.json
.\.venv\Scripts\python.exe `
    -m sorting_vision.cli `
    --config config\d435if-initial.yaml rgbd-review `
    --results-root output\rgbd-replay-20261002 `
    --output-dir output\rgbd-review-20261002
```

审计检查文件、尺寸、标签和深度；rgbd-review读取已检测的结果目录，不直接读取原始数据。先按第6节或[视觉审查教程](视觉审查流程.md)逐样本回放到output/rgbd-replay-20261002，再生成联系表。联系表必须实际查看ROI、孔洞、反光、数量和分割。统计通过不代替逐图审查。详见[视觉审查](视觉审查流程.md)。

## 5. 空盘平面标定

background-dir指向一组完整空盘帧，不是包含多个样本的批次目录。将下面路径换为第2节实际打印的样本：
```powershell
.\.venv\Scripts\python.exe `
    -m sorting_vision.cli `
    --config config\d435if-initial.yaml rgbd-calibrate `
    --background-dir data\rgbd-new\pilot-20261002\empty_tray\ACTUAL_SAMPLE_ID `
    --output output\rgbd-calibration-20261002.json
```

输出保存主内参、料盘平面和相机到机器人变换。默认机械变换为单位矩阵，只用于开发；真实抓取需--camera-to-robot实测4×4变换JSON及实际坐标核验。不能把平面拟合成功称为机械外参通过。

换盘位置、工作距离、相机、档位后重新确认。空间ROI应覆盖真正料盘，不包含画面外第二个白盘。

## 6. 离线检测

```powershell
.\.venv\Scripts\python.exe `
    -m sorting_vision.cli `
    --config config\d435if-initial.yaml rgbd-detect `
    --frame-dir data\rgbd-new\pilot-20261002\cone\ACTUAL_SAMPLE_ID `
    --rgbd-calibration output\rgbd-calibration-20261002.json `
    --output-dir output\rgbd-detect-20261002
```

也可用--background-dir代替rgbd-calibration。shape模型可显式--rgbd-shape-model。看results-v2.json、标注和health，缺深度／无抓取面分别处理，不只看形状名。

## 7. 训练候选与同样本对照

只把明确训练数据放入data/rgbd-fit，或用--batch-id选择训练批次。通用训练器不按双机split自动隔离；不要传混有校准／测试的总目录。
```powershell
.\.venv\Scripts\python.exe `
    -m sorting_vision.cli `
    --config config\d435if-initial.yaml geometry-rgbd-train `
    --data-root data\rgbd-fit `
    --strict-single-object `
    --fused-edges `
    --baseline-output models\experimental\rgbd-baseline-20261002.npz `
    --output models\experimental\rgbd-candidate-20261002.npz `
    --output-report output\rgbd-training-20261002.json
```

严格单物块会拒绝误拆／多目标样本，失败仍计入开发统计。base-model追加旧观测会改变训练来源，必须披露；不默认混入旧D415资料。LOO训练报告不能替代独立实体／批次测试。

## 8. 可见面与内部棱线诊断

```powershell
.\.venv\Scripts\python.exe `
    -m sorting_vision.cli rgbd-edge-audit `
    --data-root data\rgbd-fit `
    --output-dir output\rgbd-edge-20261002 `
    --limit-per-class 10
.\.venv\Scripts\python.exe `
    -m sorting_vision.cli rgbd-face-audit `
    --frame-dir data\rgbd-new\pilot-20261002\cone\ACTUAL_SAMPLE_ID `
    --background-dir data\rgbd-new\pilot-20261002\empty_tray\ACTUAL_SAMPLE_ID `
    --output-dir output\rgbd-face-20261002
```

局部平面、深度边界、RGB线和图支持用于验证部分可见结构。深度孔洞不等于棱；OBB不等于实物网格；圆柱／圆锥轮廓不自动成为直棱。

## 9. 采集质量与正式测试

物块工作距离、照明、红外纹理、反光、图深对齐和同步先确认，再调分类。新实体／批次保留独立校准和测试，失败样本不排除分母。提高原生深度分辨率后需重新核配置、标定和新数据，不把插值锐化当测量精度。

旧v4是D415训练基线，D435if需要新实拍复验，见[换机指南](D435if迁移与重新标定.md)。更多参数见[CLI索引](CLI命令与参数索引.md)。
