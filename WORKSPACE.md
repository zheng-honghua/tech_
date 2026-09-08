# 工作空间说明

本仓库以 `main` 为唯一开发主线。2026-09-08 盘点时只有本地 `main` 和远端跟踪分支 `origin/main`，没有需要合并的其他分支。

## 当前入口

- RGB-D 稳定模型：`models/stable/rgbd/geometry-rgbd-multipose-v4.npz`
- RGB-D 推荐配置：`config/d415-reviewed-20260905.yaml`
- RGB 开发模型：`models/stable/rgb/geometry-rgb-morph-color.npz`
- 代码：`src/sorting_vision/`
- 自动化测试：`tests/`
- 离线调查脚本：`scripts/`
- 使用说明：`docs/guides/`
- 历史试验报告：`docs/reports/`
- 保留的评估证据：`artifacts/evaluations/`
- RGB 测试图集：`fixtures/rgb/`
- 原始 RGB-D 采集：`data/`，不随意删除或改写

## 目录规则

- `models/stable/`：已经被当前文档推荐的版本。只有这里的 RGB-D 模型可作为运行候选，仍须通过深度、同步、抓取面和运动互锁。
- `models/experimental/`：可复现实验，但没有通过推广门槛，不得用于实际分拣。
- `models/archive/`：为兼容测试和历史复现保留的早期版本，不再推荐。
- `output/`：临时生成目录，可随时清空；程序运行会重新产生内容。
- `archive/`：本机可恢复归档，已加入 `.gitignore`，不作为当前代码或模型来源。

2026-09-08 整理时，历史 `output` 中的关键阶段21–23 JSON、联系表和特征缓存已迁入 `artifacts/evaluations/`。其余生成结果位于本机 `archive/generated-output-20260908/`，确认不再需要后可以单独永久删除。

