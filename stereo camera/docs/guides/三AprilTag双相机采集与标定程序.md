# 三 AprilTag 双相机采集与标定程序

本说明对应两个独立程序：`scripts/dual_rgbd_side_capture.py` 同步采集俯视 D435if 的 RGB/深度和侧置 UVC 的 RGB；`scripts/dual_apriltag_calibrate.py` 使用两个对角固定 AprilTag 和一个自由移动 AprilTag 标定两相机及料盘坐标系。两者只修改双目工作空间，不改变单目程序。

## 0. 三种分辨率怎么改

三种图像分别有独立的宽高入口：

| 图像 | 临时命令参数 | `config/dual/temporary.yaml` 长期配置 |
|---|---|---|
| 俯视 RGB | `--color-width`、`--color-height` | `camera.realsense_color_width`、`realsense_color_height` |
| 俯视原始深度流 | `--depth-width`、`--depth-height` | `camera.realsense_depth_width`、`realsense_depth_height` |
| 侧视 RGB | `--side-width`、`--side-height` | `dual_view.side_width`、`side_height` |

俯视 RGB 和深度共用 `--fps`／`camera.realsense_fps`；侧视帧率使用 `--side-fps`／`dual_view.side_fps`。命令参数只覆盖本次运行，不会改写 YAML。比如临时使用当前已验证的三种输入尺寸：

```powershell
.\.venv\Scripts\python.exe scripts\dual_apriltag_calibrate.py `
  --config config\dual\temporary.yaml --platform-id temporary `
  --color-width 640 --color-height 480 `
  --depth-width 640 --depth-height 480 --fps 15 `
  --side-width 1280 --side-height 720 --side-fps 30 `
  --side-camera-index 0 --tag-size-mm 30 `
  --fixed-tag-a 0 --fixed-tag-b 1 --free-tag 2 `
  --fixed-tag-inset-mm 20 --required-poses 20 `
  --detection-width 640 `
  --session-dir data\apriltag-calibration `
  --output config\dual\temporary\calibration.json
```

D435 深度会对齐到俯视 RGB，所以最终显示和保存的对齐深度数组宽高跟俯视 RGB 相同；`--depth-width/--depth-height` 控制的是对齐前的传感器深度流采样。分辨率组合必须是 RealSense Viewer 中真实存在的流配置，否则 SDK 会报告 `Couldn't resolve requests`。当前侧相机驱动只公布 `1280×720@30` 的 YUY2/NV12 档位，因此虽然程序提供侧视覆盖参数，这台相机现阶段应保持 `1280×720@30 NV12`。

任意一种分辨率改变后，旧内参、外参、空盘参考和标定文件都不能继续使用，必须重新完成三 AprilTag 标定和空盘采集。

## 1. 三个标签怎么放

默认使用 `DICT_APRILTAG_36h11`：固定标签为 ID 0、ID 1，自由标签为 ID 2。

- ID 0 的中心固定在料盘坐标原点角附近，中心距两条内边均为 `--fixed-tag-inset-mm`。
- ID 1 固定在对角，中心距另外两条内边使用同一距离。
- 两个固定标签必须平贴在同一料盘平面，且印刷图案的上边朝同一个方向；程序以它们建立 `+X/+Y`，不能把其中一个旋转 90°或 180°。
- ID 2 不固定。采集每一帧后把它移动到另一位置，并改变旋转和倾角；两台相机必须同时看见完整四角。
- `--tag-size-mm` 指标签黑色正方形外边的实测宽度，不包括四周白色静区。打印后用卡尺测量，不要按 PNG 整张图的宽度填写。

可先离线生成三个标签和摆放预览，不会打开相机：

```powershell
.\.venv\Scripts\python.exe scripts\dual_apriltag_calibrate.py `
  --platform-id temporary --tag-size-mm 30 `
  --generate-tags-dir output\apriltag-print
```

`three-apriltag-layout-preview.png` 只解释方向，不按比例打印。三个独立 PNG 带白色静区；打印时应调整纸面缩放，直到黑色方框实测宽度等于命令中的毫米值。

## 2. 双相机同步拍摄程序

先拍空盘，再按数字键拍各类物块：

```powershell
.\.venv\Scripts\python.exe scripts\dual_rgbd_side_capture.py `
  --config config\dual\temporary.yaml `
  --dataset-root data\dual --batch-id temporary-01 `
  --platform-id temporary --side-camera-index 0 `
  --start-label empty_tray --target-per-label 10
```

系统只有两台相机：俯视 RGB-D 主相机和侧置 RGB 辅助相机。窗口把俯视相机的 RGB、俯视相机的深度以及侧相机的 RGB 分成三个画面显示，但这不是三相机系统。按键：`0–9` 选标签，`Space` 合格保存，`F` 强制保存并在元数据标为质量覆盖，`N/P` 前后切类，`A` 跳到下一未完成类别，`Q` 退出。普通物块样本默认要求本批次先有至少一份 `empty_tray`。

质量门检查两路静止、俯视有效深度比例、D435if RGB/深度时间差、两相机配对时间差和侧图清晰度。每个样本目录保存：

- `primary-color.png`：俯视 RGB；
- `primary-depth.npy`：与俯视 RGB 对齐的原始深度；
- `side-color.png`：侧视 RGB；
- `metadata.json`：内参、双路帧 ID、设备/主机时间戳、配对误差、平台、标定哈希、标签和质量结果。

根目录的 `dual-manifest.jsonl` 用于断点续拍计数。`F` 仅供诊断或保留难例，质量覆盖样本不能直接进入正式训练/验收，必须人工复审。

## 3. 实时三标签标定程序

先确认两机刚性固定，曝光、白平衡和焦距模式在整次标定中不切换，空料盘处于机械定位挡块中。临时配置的侧相机保持自动曝光/自动白平衡和固定焦距；示例中的 30 mm 必须换成标签黑框实测值：

```powershell
.\.venv\Scripts\python.exe scripts\dual_apriltag_calibrate.py `
  --config config\dual\temporary.yaml --platform-id temporary `
  --side-camera-index 0 --tag-size-mm 30 `
  --fixed-tag-a 0 --fixed-tag-b 1 --free-tag 2 `
  --fixed-tag-inset-mm 20 --required-poses 20 `
  --session-dir data\apriltag-calibration `
  --output config\dual\temporary\calibration.json
```

操作顺序：

1. 两个固定标签按对角位置和相同方向贴好，清空盘面；确认两机都识别出 ID 0、1，按 `R` 保存固定参考和空盘深度。
2. 手持或固定 ID 2，使两机都看到完整标签，按 `Space` 保存。每次明显改变位置、画面尺度、旋转或倾角，覆盖画面中心、四角、近处和远处；过于相似的姿态会被拒绝。
3. 至少保存 20 对有效姿态后按 `C` 求解并保存。`Q` 可退出但不会生成标定结果。

程序分别求俯视/侧视内参与畸变，再求 `T_side_from_primary`；固定对角标签建立 `T_tray_from_primary`，并只在投影出的料盘多边形内使用空盘深度拟合平面。原始标定图、深度、双路时间戳和姿态签名保存在 session 目录，便于复核。

只有两机重投影 RMS 均不超过 0.8 px、共视极线误差 P95 不超过 3 px、标签尺度/固定对角位置误差不超过 1% 时，标定文件才为 `valid=true`。无效结果仍保存用于诊断，但融合运行会返回 `CALIBRATION_INVALID`。

## 4. 现场注意事项

- 两个固定标签最好放在料盘有效工作区外侧或可在标定后移除；若会遮挡物块，标定完成后移除，但相机和料盘绝不能再移动。
- 标签发生翘曲、污损、反光或黑框尺寸不一致时重打；普通纸应贴在平整硬板上。
- 临时平台与比赛平台必须使用不同目录和不同标定文件；更换支架、相机分辨率、焦距、料盘位置或光照后重新标定、重拍空盘。
- 三标签标定只建立视觉几何关系，不替代 RGB-D 到机器人坐标的机械外参标定。
- 侧视仍只提供形状证据，不能生成抓取坐标，也不能把任何深度或遮挡拒识升级为可抓取。

## 5. 黑屏或预览卡顿

DirectShow `IAMStreamConfig` 原始能力表实测：设备 `0` 是侧置 `USB Camera`，设备 `1` 是 RealSense Depth，设备 `2` 是 RealSense RGB。侧相机驱动只声明 `1280×720@30 YUY2` 和 `1280×720@30 NV12`，没有声明MJPEG；OpenCV请求MJPG时虽然DirectShow返回成功，读回格式仍是YUY2，MSMF则直接拒绝。当前配置使用已确认生效且数据量更低的 `DirectShow + NV12`。不要把索引 `2` 当侧相机，否则会让OpenCV与librealsense同时抢RealSense RGB并造成超时。

侧相机强制手动曝光 `-6` 会黑屏或读帧失败，所以临时配置使用自动曝光与自动白平衡。若重新插拔后编号变化，应重新读取 DirectShow 设备名称，不能只按画面分辨率猜索引。

截图证明 RealSense Viewer 的 RGB+深度与 Windows“相机”可以同时运行，之后的独立进程基线也测得主相机约 15 FPS、侧相机约 30 FPS，因此硬件和当前 USB 连接不是根因。真正问题是 OpenCV/DirectShow 与 librealsense 在同一 Python 进程内争用，以及串行执行 AprilTag 检测时没有持续排空 RealSense 帧流。

当前程序的稳定方案是：侧置 UVC 相机在独立子进程中以 `DirectShow + NV12` 连续采集，通过有界 JPEG 队列传回最新帧；RealSense 的创建、等待帧和关闭全部由同一个专用线程负责；双视角配对线程等待侧相机跨过主帧时间后再选最近帧。2026-09-09 在正式脚本相同启动顺序和每对都执行双路 AprilTag 检测的条件下，连续 120 对通过，处理速度 15.06 FPS，配对误差 P95 为 14.61 ms、最大 39.24 ms，120/120 均满足 50 ms 门槛。

这台侧相机的驱动没有公布 MJPEG 档位，所以 GStreamer 不能凭空增加相机端 MJPEG；当前不需要安装或切换 GStreamer。程序内部 JPEG 只用于跨进程传帧，不会改变相机的 `NV12` 采集格式。

实时AprilTag检测支持缩放后检测、再回到原图做亚像素角点细化。若其他平台使用更高分辨率且预览卡顿，可加：

```powershell
--detection-width 720
```

不要低于 320 像素；标签在缩小图中太小时会降低检出率。启动脚本前仍需关闭 RealSense Viewer、Windows“相机”、微信视频等占用摄像头的软件。当前已验证配置为俯视 RGB/深度 `640×480@15`、侧视 `1280×720@30 NV12`；不要自行改回更高的 RealSense 分辨率或帧率，修改后必须重新做持续取帧测试和完整标定。
