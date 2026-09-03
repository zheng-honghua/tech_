# RGB-D 多姿态识别 v3

## 本次改动

RGB-D 几何分类器由“每类单一质心”升级为多姿态 KNN。模型保存训练样本的标准化三维特征，预测时分别计算各类别最接近的 3 个姿态距离，再进行距离和类别间隔拒识。这样可分别保留正放、侧躺和倾斜物块的特征。旧版 `rgbd_geometry_v1`、`rgbd_geometry_v2_face_topology` 模型仍可加载。

训练支持 `--batch-id`。可先用一个批次训练、另一个批次测试：

```powershell
.\.venv\Scripts\python.exe -m sorting_vision.cli geometry-rgbd-train `
  --data-root "data\rgbd-pilot" --batch-id pilot-01 `
  --output "models\geometry-rgbd-pilot01-knn.npz" `
  --output-report "output\rgbd-pilot01-knn-training.json"
```

不填写 `--batch-id` 时会合并全部批次。本次最终候选模型是 `models/geometry-rgbd-multipose-v3.npz`。

## 当前实测

训练 `pilot-01`、独立测试 `pilot-02`（135张物块图）：

| 模型 | 整体正确率 | 明确分类率 | 明确分类准确率 |
| --- | ---: | ---: | ---: |
| v2 单质心 | 37.8% | 51.1% | 73.9% |
| v3 多姿态KNN | 59.3% | 75.6% | 78.4% |

合并两个批次后，模型在 `pilot-02` 训练回放为97.0%，但该数字不能表示新场景性能。正式验收需要再拍一个完全不参与训练的 `pilot-03`。

## 可视化审查

审查图位于 `output/rgbd-pilot02-knn-holdout-review/`。绿色为托盘区域，青色为RGB候选区域，紫色为最终深度主体，橙色为检测框。审查确认大部分主体选择正确，但少量画面存在第二个白色托盘导致ROI异常，锥体斜面也有深度轮廓缺失。

## 当前限制

- 59.3%仍未达到比赛要求，最终模型不得直接控制机械执行。
- 六棱锥和五棱锥跨批次识别最弱，需要补充稳定面、侧躺和倾斜姿态。
- 下一批应固定相机、托盘和曝光，并移走画面外的第二个白色托盘。
- RGB-D结果仍必须同时满足 `PICKABLE` 和 `selected=true` 才能下发。
