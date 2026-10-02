# 两路RGB内参、三AprilTag外参与采集程序

更新：2026-10-02。内参、双机／料盘外参、主深度平面和机器人外参分别完成，不能互相替代。现场尺寸必须实测。环境见[排错篇](环境与相机排错.md)。

## 1. 文件职责与顺序

| 步骤 | 输入 | 输出 |
|---|---|---|
| 主／侧RGB内参 | 棋盘格RGB图／视频 | 两套intrinsics、畸变、质量 |
| 双机／料盘关系 | 共视AprilTag＋空盘深度 | 双机变换、料盘系、哈希和质量 |
| 主深度平面／机械外参 | 单RGB-D空盘＋实测机械变换 | RGBDCalibration |
| 侧背景 | 空盘同步侧图 | 中位背景 |

SDK对齐深度投影与主RGB标定还须核对；缩放／裁剪端点必须恢复原生像素再使用原生内参。

## 2. 准备环境

```powershell
Set-Location "C:\Users\d114\Desktop\tech_\stereo camera"
.\.venv\Scripts\python.exe scripts\rgb_intrinsics_calibrate.py --help
.\.venv\Scripts\python.exe scripts\dual_apriltag_calibrate.py --help
```

新候选保存output/calibration-20261002，避免覆盖旧数据绑定的标定。求解通过后在自己的候选配置引用新路径。

## 3. 棋盘生成与尺寸

```powershell
.\.venv\Scripts\python.exe scripts\rgb_intrinsics_calibrate.py `
    --generate-board-dir output\calibration-20261002\checkerboard `
    --corners-x 10 `
    --corners-y 7 `
    --square-size-mm 20
```

10×7为70个内角点，纸面11×8格。100%／实际大小打印，禁用适合页面，贴刚性平板。实测格边长，下文20 mm仅在实测一致时用。全部角点应完整清晰可见。

## 4. 主RGB内参

```powershell
.\.venv\Scripts\python.exe scripts\rgb_intrinsics_calibrate.py `
    --platform-id temporary `
    --source realsense `
    --camera-id primary `
    --width 1920 `
    --height 1080 `
    --fps 15 `
    --corners-x 10 `
    --corners-y 7 `
    --square-size-mm 20 `
    --required-frames 25 `
    --session-dir output\calibration-20261002\intrinsics-session `
    --output output\calibration-20261002\primary-intrinsics.json
```

仅启用RealSense彩色流，不采深度。棋盘覆盖画面中心／四角、多倾角与尺度；等READY后按空格记录，保存至少25个有效多样姿态后按C求解。Q只退出，不求解。完整拍法、自动模式与照片复算见[棋盘格内参教程](棋盘格内参标定完整教程.md)。

## 5. 侧RGB内参

先结束前一个程序，再运行；索引改为真实侧相机：
```powershell
.\.venv\Scripts\python.exe scripts\rgb_intrinsics_calibrate.py `
    --platform-id temporary `
    --source uvc `
    --camera-id side `
    --camera-index 1 `
    --width 1280 `
    --height 720 `
    --fps 30 `
    --backend DSHOW `
    --fourcc NV12 `
    --corners-x 10 `
    --corners-y 7 `
    --square-size-mm 20 `
    --required-frames 25 `
    --session-dir output\calibration-20261002\intrinsics-session `
    --output output\calibration-20261002\side-intrinsics.json
```

检查valid、拒绝原因、RMS及会话原图。原生尺寸、焦距、裁切和设备与采集一致。UVC标定支持--autofocus on/off、--focus等控制参数，须与正式采集的侧相机设置一致，并核对启动打印的control_status；参数请求不代表驱动执行成功。板翘曲、角点缺失或姿态覆盖少时先补拍。

## 6. 已有视频模式

```powershell
.\.venv\Scripts\python.exe scripts\rgb_intrinsics_calibrate.py `
    --source video `
    --platform-id temporary `
    --camera-id primary `
    --video-file data\calibration\temporary\intrinsics\videos\primary.mp4 `
    --width 1920 `
    --height 1080 `
    --corners-x 10 `
    --corners-y 7 `
    --square-size-mm 20 `
    --required-frames 25 `
    --headless `
    --auto-capture `
    --output output\calibration-20261002\primary-video-intrinsics.json
```

视频必须保留原尺寸、停稳姿态和相同焦距。程序检查相邻帧变化和姿态多样性；有效帧不足则补视频。侧视频改camera-id、video-file和1280×720。自动存储规则见脚本--help；显式路径更便于复核。

## 7. 三标签生成与摆放

```powershell
.\.venv\Scripts\python.exe scripts\dual_apriltag_calibrate.py `
    --platform-id temporary `
    --tag-size-mm 30 `
    --generate-tags-dir output\calibration-20261002\apriltags
```

运行上面的生成命令后，终端先询问三个ID。按“固定A、固定B、自由码”的顺序输入三个不同整数，空格或逗号分隔，输入q取消；有效后才生成图片。生成时不再自动沿用45／17／50。重复或超出所选字典范围会提示重新输入。

也可在命令里一次指定三个ID。你这次选择固定A=34、固定B=35、自由码=14，命令如下；以后改这三个数字即可：

```powershell
.\.venv\Scripts\python.exe scripts\dual_apriltag_calibrate.py `
    --platform-id temporary `
    --tag-ids 34 35 14 `
    --tag-size-mm 30 `
    --generate-tags-dir output\calibration-20261002\custom-apriltags-34-35-14
```

旧的--fixed-tag-a、--fixed-tag-b、--free-tag参数也保留，生成模式需同时指定全部三个，不能和--tag-ids混用。生成图片不打开摄像头，不读取相机标定文件；非交互终端必须显式指定三个ID。输出的three-apriltag-printing.json和终端结果记录ID与角色，程序还打印后续标定应使用的--tag-ids参数。

tag-size是黑色编码方块的实测边长，不含白边；30为示例。刚性平贴，固定标签在料盘对角且同向，自由标签变化位置／高度／倾角。

同向指编码角0→角1的方向一致，不是把两张不同ID的黑色图案看成相同外形。旧20260912会话两固定码方向相差约179°，新版已明确拒绝，见[审查与方向箭头](../reports/标定程序优化与多行命令核验_20261002.md)。新摆放先对照生成的布局，固定标签必须同向／共面；联合固定角点投影P95也须≤3 px。

实测料盘宽高、固定标签中心距边界的内缩距离。示例数值不代表你现场。先在PowerShell填写：
```powershell
$tagSizeMm = $null
$trayWidthMm = $null
$trayHeightMm = $null
$fixedInsetMm = $null
$fixedTagA = 34
$fixedTagB = 35
$freeTag = 14
```

将前四个null改成真实毫米数值，后三个已经填写你这次的34／35／14；以后换码时同步修改这三个ID。尺寸留空会报参数错误，防止误用猜测值。只做标定且不指定任何ID时，仍保留读取YAML旧ID的兼容行为；使用自定义码时请明确传入同一组--tag-ids。

## 8. 外参采集与求解

```powershell
.\.venv\Scripts\python.exe scripts\dual_apriltag_calibrate.py `
    --config config\dual\temporary.yaml `
    --platform-id temporary `
    --side-camera-index 1 `
    --tag-ids $fixedTagA $fixedTagB $freeTag `
    --primary-intrinsics output\calibration-20261002\primary-intrinsics.json `
    --side-intrinsics output\calibration-20261002\side-intrinsics.json `
    --tag-size-mm $tagSizeMm `
    --tray-width-mm $trayWidthMm `
    --tray-height-mm $trayHeightMm `
    --fixed-tag-inset-mm $fixedInsetMm `
    --required-poses 20 `
    --session-dir output\calibration-20261002\stereo-session `
    --output output\calibration-20261002\calibration.json
```

1. 清空盘面，两台都看到固定标签，R保存固定参考和空盘深度。
2. 固定标签不动，自由标签在两路都清晰看到时放稳。
3. 空格记录姿态，覆盖位置、倾角和高度，至少20个有效多样姿态。
4. 查看接受／拒绝消息，不重复同姿态刷计数。
5. 按C求解，检查原始会话和输出质量；Q只退出，不求解。

采集默认要求连续3帧角点最大移动≤1.5 px、目标区域拉普拉斯方差≥50、角点距离图像边界≥8 px。两路同步沿用配置门槛，主RGB与深度间隔≤20 ms。R参考帧中两固定标签包围矩形的有效深度至少85%；这不等于料盘平面已经拟合通过。阈值是采集筛选条件，清晰度分数随分辨率和曝光变化；先改善照明、停稳、焦距和共视，再根据实拍核对阈值。

必须先R保存空盘参考，再按空格采自由标签。若已经采了姿态后重新按R，程序会启动新子会话并清零姿态计数，旧照片保留。相机或固定标签在会话中移动后必须重拍参考和全部自由姿态。默认20是最低数量；建议先采25–30个，保持两路均覆盖位置、尺度、倾角和高度。失败时可继续补拍再按C，不用放宽比赛档门槛。

两套内参须valid、尺寸一致。同步、共视、空盘平面、标签尺寸与安装错误会影响结果。

## 9. 会话回放

```powershell
.\.venv\Scripts\python.exe scripts\dual_apriltag_calibrate.py `
    --config config\dual\temporary.yaml `
    --platform-id temporary `
    --tag-ids $fixedTagA $fixedTagB $freeTag `
    --primary-intrinsics output\calibration-20261002\primary-intrinsics.json `
    --side-intrinsics output\calibration-20261002\side-intrinsics.json `
    --tag-size-mm $tagSizeMm `
    --tray-width-mm $trayWidthMm `
    --tray-height-mm $trayHeightMm `
    --fixed-tag-inset-mm $fixedInsetMm `
    --session-dir output\calibration-20261002\stereo-session `
    --replay-session `
    --output output\calibration-20261002\calibration-replay.json
```

回放不打开相机，仅重算已有观测，不能补出缺失视角或修复移动后新几何。使用启动时打印的实际会话目录；目录已有照片时，新的实时运行会在run-*子目录保存。新会话记录标签／料盘尺寸与两套内参哈希，回放不允许悄悄换参数；历史会话缺少这些字段时会提示人工核对，不宣称已自动验证。

## 10. 质量判断

| 标定条件 | 严格比赛档 | 当前临时开发档 |
|---|---:|---:|
| 主／侧内参RMS | 各≤0.8 px | 各≤0.8 px |
| 联合投影P95 | ≤3 px | ≤3.5 px |
| 尺度误差 | ≤1% | ≤3.5% |

检查valid、quality_profile、quality_limits和重投影原图。当前三标签joint_projection_p95_px取“原始像素重投影P95”和“去畸变像素极线P95”的较大值，详细数值及逐姿态残差在board.projection_diagnostics中。旧三标签文件该字段仅来自原始像素极线距离，不能把两次不同口径的数值直接当精度提高百分比。内参固定，三标签不重新拟合内参；固定标签同向检查与八角点联合料盘位姿也参与求解。临时通过只能称开发通过。稀疏运行另要求极线3 px、重投影2 px、射线角2°、深度一致性5 mm，不随临时档放宽。

历史SDK候选严格预检未通过，当前安装要重新验证。运行外参后还要检查空盘投影、盘边及实物对应，不靠逐帧平移框掩盖误差。

## 11. 深度平面、机械变换、背景

主空盘用rgbd-calibrate，格式见[RGB-D教程](RGBD数据采集与训练.md)。camera-to-robot默认单位矩阵是软件开发默认，不是实际机器人标定。机械变换需按真实坐标验证。

重新采侧中位背景，将新文件写入候选配置。新数据绑定新哈希；旧照片使用原文件，不修改历史哈希使校验通过。

## 12. 保存与排错

保留打印资产、实测尺寸、两套内参、固定参考、自由姿态RGB／深度、时间戳与质量文件。按设备→尺寸／焦距→检测→共视→平面→投影／尺度定位。

每次求解保存calibration-summary.json和publish-status.json。质量失败不会写入--output指定的标定文件；原有输出保持不动。通过时写入输出，已有文件先备份为*.backup-*。诊断文件不是正式标定输出，请检查published及退出码：0表示通过并发布，2表示质量失败，1表示退出未完成或输入／求解失败。默认配置引用路径不自动改变。采集会话中的实际图像尺寸也会与内参核对，尺寸不匹配立即拒绝。

备用ChArUco入口及零基tray-pose-index见[验收篇](双视角采集标定与验收.md)。采物块使用[三画面教程](双视角形状与颜色数据采集完整教程.md)。
