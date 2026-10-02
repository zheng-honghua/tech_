# 视觉工作区Git与GitHub使用指南

更新：2026-10-02。本机Git根目录为tech_，包含两个独立工程。提交前看清路径与本次修改，原始照片和模型产物按项目数据管理规则单独保存。

## 1. 查看当前状态

```powershell
Set-Location "C:\Users\d114\Desktop\tech_"
git status --short
git branch --show-current
git remote -v
git diff --stat
```

当前分支和远端以实时输出为准，不沿用某日只有main的盘点。M是修改，??是未跟踪，D是删除；目录缺失不代表删除属于你的本次工作。

## 2. 查看本次文档变更

```powershell
git diff -- "stereo camera/README.md" "stereo camera/docs" "monocular camera/README.md" "monocular camera/docs"
git diff `
    --check -- "stereo camera/README.md" "stereo camera/docs" "monocular camera/README.md" "monocular camera/docs"
```

git diff默认不显示未跟踪文件正文；还要实际打开新文件。检查模型、原照片、配置与历史实验记录是否意外被覆盖。

## 3. 分支与提交

需要独立分支时可用codex/前缀或团队约定名称。先确认工作区已有修改如何保留，再创建／切换，不自动丢弃。
```powershell
git switch -c codex/documentation-refresh
git add -- "stereo camera/README.md" "stereo camera/docs/README.md" "stereo camera/docs/guides"
git diff --cached --stat
git diff --cached --check
git commit -m "docs: rewrite vision operation guides"
```

示例仅暂存这些路径；按本次范围增加单目文档和算法记录，避免git add .把无关删除或大产物一并纳入。提交前确认更新记录有文档维护说明，算法变化为无。

## 4. 推送与PR

配置正确远端、提交检查完成后：
```powershell
git push -u origin codex/documentation-refresh
```

推送需要网络和相应账号权限。PR说明写具体改变、命令核验和实机未测范围；不要把文档校验写成新相机已验收。

## 5. 原始数据与模型备份

备份RGB、原深度、metadata、清单、标定、模型／策略、特征契约、冻结ID、审查和报告。虚拟环境、临时裁剪和个人现场敏感资料通常不提交。大型实验ZIP保留独立校验和与存储来源。

output里有已审实验包和复现快照，不能一概清空。删除前确认哪些是唯一证据，先备份。

## 6. 冲突和恢复

先git status和diff，确认冲突位置再编辑、核验、暂存。不要照抄git reset --hard、git clean或全目录删除去“修环境”。旧记录或原数据被改时先恢复可验证来源，并记录发生了什么。
