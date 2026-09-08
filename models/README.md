# 模型分级

## Stable

| 用途 | 路径 | 状态 |
| --- | --- | --- |
| RGB-D 多姿态识别 | `stable/rgbd/geometry-rgbd-multipose-v4.npz` | 当前效果最好的 RGB-D 推荐模型 |
| RGB 单目开发识别 | `stable/rgb/geometry-rgb-morph-color.npz` | RGB 开发默认；没有深度，禁止驱动机器人 |

## Experimental

- `experimental/rgbd/`：严格单实例、运行时对齐、特征加权和融合棱线等未推广候选。
- `experimental/rgb/geometry-rgb-structure.npz`：点线结构研究版本。
- `experimental/cnn/`：PyTorch 与 OpenVINO 开发预览模型；现有跨批次证据不足，不属于机器人稳定模型。

## Archive

- `archive/rgb/`：RGB 早期轮廓、棱线和色面模型，用于历史复现及兼容测试。
- `archive/rgbd/`：RGB-D v1–v3、早期 pilot 与 holdout 模型。
- `archive/cnn/`：已被当前 CNN 开发版本取代的训练候选。
- `archive/redundant/`：被更严格评估版本取代、仅为可恢复留存的重复候选。

实验模型升级为稳定模型前，必须更新独立批次验证、视觉审查、延迟结果、README 推荐路径和 `算法更新记录.md`，不能只移动文件。

