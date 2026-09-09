# 单目 / 双目视觉工作区

本仓库现按相机方案拆分为两个并列工作空间：

- `monocular camera/`：拆分前的完整工程，包含当前代码、模型、本地数据、历史产物、虚拟环境及未提交修改。
- `stereo camera/`：双视角工程，包含源码、配置、模型、脚本、测试、夹具和项目文档。

进入对应目录后再执行安装、测试和命令。注意：两个项目中保留的 `.venv` 启动器均包含重命名前的绝对路径，移动后不能直接使用；继续开发前需要在各自项目目录中重建虚拟环境。

例如，为双目工作空间建立独立环境：

```powershell
Set-Location ".\stereo camera"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

> `stereo camera/` 已实现双相机采集、标定与融合实验链路，但仍未完成比赛平台独立验收；第二摄像头的结果不能单独用于机械运动。
