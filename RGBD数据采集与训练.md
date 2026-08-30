# D415 深度数据采集与训练

## 1. 安装与连接

关闭 RealSense Viewer，连接 D415，然后安装可选依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[realsense]"
```

采集程序默认使用彩色 `1280×720@30`、深度 `640×360@30`，并把深度对齐到彩色图。若设备不支持这组流，可把彩色也改成 `640×480`。

## 2. 采集文件包

每个拍摄批次必须先拍真正的空托盘：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli rgbd-capture `
  --dataset-root "data\rgbd-geometry" --batch-id batch-01 --label 空托盘
```

窗口内按空格保存一组，按 `q` 退出。再放入单个物块，修改标签继续拍摄：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli rgbd-capture `
  --dataset-root "data\rgbd-geometry" --batch-id batch-01 --label 三棱柱
```

每按一次空格会创建一个独立文件夹：

```text
data/rgbd-geometry/batch-01/triangular_prism/<sample-id>/
  color.png              # 彩色图
  depth.npy              # 原始深度值（训练使用）
  depth-preview.png      # 仅供人查看
  metadata.json          # 内参、比例、时间戳、标签和流配置
```

根目录的 `manifest.jsonl` 是总清单。不要编辑 `depth.npy`，也不要用伪彩色预览图训练。

## 3. 拍摄要求

- 每张只放一个完整物块，手离开画面后再拍；每次改变朝向、位置或稳定面。
- 每类建议至少 3 个独立批次，每批 30–60 张；另留一个完整批次只做测试。
- 每批重新拍空托盘，保持相机高度、焦距、曝光和分辨率与比赛一致。
- 黑色或反光物块出现大面积深度空洞时应调整距离、光照或曝光，不能靠插值伪造抓取面。

## 4. 审计和训练

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli rgbd-dataset-audit `
  --data-root "data\rgbd-geometry" --output-report "output\rgbd-audit.json"

.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-rgbd-train `
  --data-root "data\rgbd-geometry" --output "models\geometry-rgbd.npz" `
  --output-report "output\rgbd-training.json"
```

第一版模型使用点云主轴尺寸、表面平面性、厚度/深度分布和外轮廓共同判断。它必须由真实 D415 数据训练后才能使用，训练回放结果不等于泛化准确率。

## 5. 离线检测

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli rgbd-detect `
  --frame-dir "data\rgbd-geometry\batch-01\triangular_prism\<sample-id>" `
  --background-dir "data\rgbd-geometry\batch-01\empty_tray\<sample-id>" `
  --rgbd-shape-model "models\geometry-rgbd.npz" `
  --output-dir "output\rgbd-detect"
```

只有深度健康、吸附面有效且结果为 `PICKABLE`、`selected=true` 时，控制端才可使用抓取位姿。
