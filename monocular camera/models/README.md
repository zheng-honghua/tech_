# 模型分级、输入与加载

更新：2026-10-02。模型路径相对于当前工程运行目录；加载前核对格式、类别顺序、输入语义和来源。

## 原有基线

| 文件 | 用途与边界 |
|---|---|
| stable/rgb/geometry-rgb-morph-color.npz | RGB开发；缺真实深度，固定不可执行 |
| stable/rgbd/geometry-rgbd-multipose-v4.npz | D415多姿态基线；D435if新环境先复验 |

## 候选与历史

experimental保存未推广候选，archive保存兼容／复现版本。具体文件以本机目录为准。RGB普通轮廓、edge-topology、structure-topology和主RGB-D特征语义不同；旧模型不静默消费改变后的掩膜／棱线语义。

本工程没有v6/v7/v8侧视模型，不能从双机复制模型后直接加载。

## 使用和训练

predict-image的--model是RGB模型；rgbd-detect的--rgbd-shape-model是匹配主RGB-D模型。配置不保证自动加载主模型，按入口显式核对。完整操作见[模型教程](../docs/guides/模型训练与后端.md)与[配置](../docs/guides/配置与模型版本.md)。

## 推广所需资料

独立实体／批次测试、实际看图、错误接受／拒识覆盖、深度与抓取状态、现场标定／机械坐标、目标机端到端P95及互锁。保留训练／校准／测试ID、原始哈希、模型／策略／配置与指纹。不能只改目录名或只看训练准确率升级stable。
