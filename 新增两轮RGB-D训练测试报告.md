# 新增两轮 RGB-D 训练测试报告

## 数据与人工审查

本次处理 `data/rgbd-multi-scenes/multi-02` 和 `multi-03`，各 10 帧。托盘 ROI、RGB/深度对齐和物块可见性正常。`multi-03/frame-06` 将一个三棱锥过分割成两个候选，因此不加入训练，但保留在测试中。

多物体采集只保存了场景组成，没有逐物块标签。训练前按稳定颜色族、相对面积和可见几何进行逐物块对应，并检查了分组总览：

- `multi-02`：蓝色四棱锥；小青色三棱锥；大青色五棱柱。
- `multi-03`：大蓝色六棱柱；小蓝色五棱锥；小青色三棱锥；大青色三棱柱。

生成的数据位于 `data/rgbd-reviewed-multi-02-03`，共 66 个单物块样本。每个 `metadata.json` 都记录原始帧、边界框、分配规则和 `reviewed=true`。复现脚本为 `scripts/prepare_reviewed_multi_02_03.py`。

## 训练与结果

旧模型 `models/geometry-rgbd-multipose-v3.npz` 的 223 个样本被保留，新模型追加 66 个样本：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-rgbd-train `
  --data-root data/rgbd-reviewed-multi-02-03 `
  --base-model models/geometry-rgbd-multipose-v3.npz `
  --output models/geometry-rgbd-multipose-v4.npz
```

同一批数据回放中，`multi-02` 从 56.7% 提升到 80.0%，`multi-03` 从 80.0% 提升到 92.5%。为减少训练回放偏差，另用第 1–8 帧训练、第 9–10 帧测试：旧模型为 10/14（71.4%），追加训练模型为 12/14（85.7%）；场景完全正确为 2/4。

以上仍是同批次留出结果，不代表比赛现场泛化准确率。下一轮应更换光照、距离和摆放批次，并使用新的 `batch-id`。

## 当前使用

实时 D415 使用最终模型：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli camera-live `
  --source realsense `
  --color-width 1920 --color-height 1080 `
  --depth-width 640 --depth-height 480 --fps 30 `
  --rgbd-shape-model models/geometry-rgbd-multipose-v4.npz
```

诊断结果位于 `output/multi-02-03-test`、`output/multi-02-03-v4` 和 `output/multi-02-03-holdout`。RGB-D 抓取安全门限没有改变。
