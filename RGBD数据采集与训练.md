# D415 深度数据采集与训练

多姿态KNN模型格式、独立批次对比和当前实测结果见[RGBD多姿态识别升级说明.md](RGBD多姿态识别升级说明.md)。

## 1. 安装与连接

以下命令均在仓库根目录执行。先创建并激活虚拟环境，再关闭RealSense Viewer、连接D415并安装可选依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,realsense]"
```

如果虚拟环境已经安装，只需执行激活命令。若PowerShell禁止激活，可改用`.\.venv\Scripts\python.exe -m pip ...`和`.\.venv\Scripts\python.exe -m sorting_vision.cli ...`。

采集程序默认使用彩色 `1280×720@30`、深度 `640×360@30`，并把深度对齐到彩色图。若设备不支持这组流，可把彩色也改成 `640×480`。

## 2. 采集文件包

需要连续完成空托盘和全部类别时，推荐使用[D415数据集拍摄助手.md](D415数据集拍摄助手.md)中的`rgbd-capture-assistant`；它支持按数字切换类别、计数恢复、画面稳定和深度质量检查。下面的`rgbd-capture`适合只采集一个指定类别。

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

`batch-id`只能包含英文字母、数字、下划线和连字符；`label`必须是程序已注册的中文类别或英文ID。当前支持空托盘、三棱柱、三棱锥、四棱锥、五棱柱、五棱锥、六棱柱、六棱锥、正八面体和圆锥；权威映射位于`src/sorting_vision/geometry_rgb.py`的`GEOMETRY_LABELS`中。

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
  --data-root "data\rgbd-geometry" --output "models\geometry-rgbd-multipose-v3.npz" `
  --output-report "output\rgbd-training.json"
```

当前 v3 模型使用点云主轴尺寸、表面平面性、厚度/深度分布和外轮廓共同判断，并保存每类的多姿态样本特征。预测时取最接近的 3 个同类姿态计算距离，不再把侧躺、倾斜和正放样本压缩为单一类别质心。它必须由真实 D415 数据训练后才能使用，训练回放结果不等于泛化准确率。

可只使用指定拍摄批次训练，以便把另一批完整保留为独立测试集：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-rgbd-train `
  --data-root "data\rgbd-pilot" --batch-id pilot-01 `
  --output "models\geometry-rgbd-pilot01-knn.npz" `
  --output-report "output\rgbd-pilot01-knn-training.json"
```

`--batch-id`可重复填写以选择多个批次；不填写时使用数据根目录下的全部批次。正式准确率必须用未参与训练的完整批次计算，不能引用训练回放准确率。

当前 v2 模型还会在物块内部以 RANSAC 提取可见平面，计算面法向、面积、拟合误差、面邻接、二面角和三面汇聚关系。RGB 梯度与色块边界只作为平面边界的辅助证据；程序不使用霍夫变换，也不会把推断出的隐藏面当作吸盘表面。

拍完一组空托盘和物块后，可以先导出平面诊断图：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli rgbd-face-audit `
  --frame-dir "data\rgbd-geometry\batch-01\triangular_prism\<物块样本ID>" `
  --background-dir "data\rgbd-geometry\batch-01\empty_tray\<空托盘样本ID>" `
  --output-dir "output\rgbd-face-audit"
```

输出的 `annotated-faces.png` 用不同颜色标出实际观测平面，`face-topology.json` 保存法向、邻接、二面角、三面汇聚数和证据质量。若一个应当清晰可见的平面被切成很多碎片，应先调整 D415 距离、曝光和光照，再修改阈值。

## 5. 离线检测

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli rgbd-detect `
  --frame-dir "data\rgbd-geometry\batch-01\triangular_prism\<sample-id>" `
  --background-dir "data\rgbd-geometry\batch-01\empty_tray\<sample-id>" `
  --rgbd-shape-model "models\geometry-rgbd-multipose-v3.npz" `
  --output-dir "output\rgbd-detect"
```

只有深度健康、吸附面有效且结果为 `PICKABLE`、`selected=true` 时，控制端才可使用抓取位姿。
