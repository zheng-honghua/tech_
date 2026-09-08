# D415 深度数据采集与训练

多姿态KNN模型格式、独立批次对比和当前实测结果见[RGBD多姿态识别升级说明.md](../reports/RGBD多姿态识别升级说明.md)。

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
  --data-root "data\rgbd-geometry" --output "models\experimental\rgbd\geometry-rgbd-multipose-next.npz" `
  --output-report "output\rgbd-training.json"
```

当前 v3 模型使用点云主轴尺寸、表面平面性、厚度/深度分布和外轮廓共同判断，并保存每类的多姿态样本特征。预测时取最接近的 3 个同类姿态计算距离，不再把侧躺、倾斜和正放样本压缩为单一类别质心。它必须由真实 D415 数据训练后才能使用，训练回放结果不等于泛化准确率。

可只使用指定拍摄批次训练，以便把另一批完整保留为独立测试集：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-rgbd-train `
  --data-root "data\rgbd-pilot" --batch-id pilot-01 `
  --output "models\experimental\rgbd\geometry-rgbd-pilot01-knn-next.npz" `
  --output-report "output\rgbd-pilot01-knn-training.json"
```

若单件采集时可能有手、相邻物块或托盘碎片进入 ROI，增加
`--strict-single-object`。它会拒绝 RGB-D 分割出零个或多个实例的标注帧，
而不是按最大连通域继续训练；被拒绝的相对路径会写进训练报告。该选项默认
关闭，以保持已有模型的复现行为。严格模式仍需要在训练后以未参与训练的批次
回放和人工查看 RGB/深度联系表，不能仅凭训练报告部署模型。

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli `
  --config "config\d415-reviewed-20260905.yaml" geometry-rgbd-train `
  --data-root "data\rgbd-pilot" --batch-id pilot-03 --strict-single-object `
  --output "models\experimental\rgbd\geometry-rgbd-pilot03-strict-candidate-next.npz" `
  --output-report "models\experimental\rgbd\geometry-rgbd-pilot03-strict-candidate-next.report.json"
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
  --rgbd-shape-model "models\stable\rgbd\geometry-rgbd-multipose-v4.npz" `
  --output-dir "output\rgbd-detect"
```

只有深度健康、吸附面有效且结果为 `PICKABLE`、`selected=true` 时，控制端才可使用抓取位姿。

输出图片的框色、`tri/pent/hex` 等缩写、置信度、状态、十字抓取点和 `results-v2.json` 字段见[RGB-D结果颜色与字段说明](RGB-D结果颜色与字段说明.md)。框色是安全状态色，不是分类真值。

## 6. 2026-09-05：使用新数据的ROI优化版本

当前仍推荐 `models/stable/rgbd/geometry-rgbd-multipose-v4.npz` 和 `config/d415-reviewed-20260905.yaml`。程序已增加空托盘参考约束：正常托盘平移继续跟随当前画面，检测区域异常扩大到参考面积1.5倍以上才限制到参考范围；深度仍用于分割、三维形状及吸附面检查。

启动时传入**当前相机位置下拍摄的空托盘** `--background-dir`，新建标定会含 `tray_roi_polygon`。若只传入旧 `--rgbd-calibration` 而文件没有该字段，程序兼容旧动态ROI行为，不会凭空获得新约束。不要混用其他拍摄位置的空托盘，也不要把离线单位变换标定用于机器人动作。

最终新多物块回放数量匹配由13/31升至26/31，形状组成完全匹配由10/31升至12/31；二者均不是独立逐实例分类准确率。新训练候选尚未稳定优于v4，未替换。复现命令、实际查看的RGB/深度图及剩余问题见[深度识别优化报告](../reports/深度识别优化报告_20260905.md)。

2026-09-06继续增加了可选深度特征加权训练。加权候选在旧场景临时已审实例中为50/55，但新场景整帧组成由12/31降为10/31，故仍不替换v4。公开训练接口支持 `DepthGeometryModel.fit(features, labels, feature_weights=weights)`，权重必须为与特征数一致的正有限数组；模型保持v3格式。实验切分、复现脚本和视觉审查见[深度加权分类试验报告](../reports/深度加权分类试验报告_20260906.md)。

## 7. 2026-09-08：RGB-D 棱线融合候选

`rgbd-edge-audit` 可将 RGB 弱边缘、深度面边界、融合保留线和拒绝线分别导出。绿色为 RGB 与深度面直接对应，橙色为通过两侧局部平面复核的弱证据；外轮廓、深度孔洞、同一平面阴影线和圆滑轮廓会被过滤。建议先审图再训练：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli `
  --config config/d415-reviewed-20260905.yaml `
  rgbd-edge-audit --data-root data/rgbd-pilot `
  --batch-id pilot-01 --batch-id pilot-02 --batch-id pilot-03 `
  --output-dir output/rgbd-edge-audit
```

融合模型使用 `rgbd_geometry_v4_fused_edges` 格式；旧模型格式兼容。当前实验在403个可用单实例的嵌套留批评估中低于同样本旧特征基线，多物体组成回放也下降，因此候选未推广，当前推荐组合仍是 `models/stable/rgbd/geometry-rgbd-multipose-v4.npz` 与 `config/d415-reviewed-20260905.yaml`。不要因棱线候选输出了类别而绕过 `DEPTH_INVALID`、`OCCLUDED`、`NO_GRASP_SURFACE` 或运动互锁。
