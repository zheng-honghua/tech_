# 单目 / 双目视觉工作区

本仓库现按相机方案拆分为两个并列工作空间：

- `monocular camera/`：拆分前的完整工程，包含当前代码、模型、本地数据、历史产物、虚拟环境及未提交修改。
- `stereo camera/`：双视角工程，包含源码、配置、模型、脚本、测试、夹具和项目文档。

进入对应目录后再执行安装、测试和命令。两个项目的包名都叫 `sorting-vision`，因此必须使用各自目录中的 `.venv`，不能混用。工程更名后至少要在对应环境中重新执行一次可编辑安装；如果环境本身无法启动，再删除并重建该项目的 `.venv`。

## 进入 stereo 双目虚拟环境

```powershell
Set-Location "C:\Users\d114\Desktop\tech_\stereo camera"
.\.venv\Scripts\Activate.ps1
```

激活成功后，PowerShell 提示符前会出现 `(.venv)`。首次使用或工程改名后重新安装双目依赖：

```powershell
python -m pip install -e ".[dev,realsense,dual,cnn,cnn-train]"
python -m sorting_vision.cli --help
python -m pytest -q --basetemp .pytest-tmp
```

退出虚拟环境：

```powershell
deactivate
```

如果 PowerShell 禁止运行 `Activate.ps1`，只为当前窗口临时允许脚本后再激活：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

也可以不激活环境，直接指定双目 Python：

```powershell
& "C:\Users\d114\Desktop\tech_\stereo camera\.venv\Scripts\python.exe" `
  -m sorting_vision.cli --help
```

## 进入 monocular 单目虚拟环境

```powershell
Set-Location "C:\Users\d114\Desktop\tech_\monocular camera"
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,realsense,cnn,cnn-train]"
python -m sorting_vision.cli --help
python -m pytest -q --basetemp .pytest-tmp
```

单目安装完成时应看到 `Successfully installed sorting-vision-0.11.0`。切换到另一个工程前先运行 `deactivate`，再进入另一个目录并激活它自己的 `.venv`。

> `stereo camera/` 已实现双相机采集、标定与融合实验链路，但仍未完成比赛平台独立验收；第二摄像头的结果不能单独用于机械运动。

两套环境的已安装版本和实测结果见[环境安装与测试记录](环境安装与测试记录_20260909.md)。
