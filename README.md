# 视觉工程使用入口

更新：2026-10-02。本工作区有两个独立工程，都叫`sorting_vision`，因此应在各自目录运行各自的Python。当前双相机已经接好并能启动；下一步先采小批空盘和物块，检查保存结果。

| 你要做的事 | 使用工程 | 教程入口 |
|---|---|---|
| 主相机RGB-D＋侧相机RGB同步采集、标定、v6/v7/v8实验 | `stereo camera` | [双相机完整使用手册](stereo%20camera/docs/guides/双相机工程完整使用手册.md) |
| 一台RealSense的RGB-D处理，或一台普通相机的RGB开发 | `monocular camera` | [单目完整使用手册](monocular%20camera/docs/guides/单目工程完整使用手册.md) |
| 查所有双相机模块和参数 | `stereo camera` | [模块说明](stereo%20camera/docs/guides/模块使用说明.md)、[命令索引](stereo%20camera/docs/guides/CLI命令与参数索引.md) |
| 查所有单目模块和参数 | `monocular camera` | [模块说明](monocular%20camera/docs/guides/模块使用说明.md)、[命令索引](monocular%20camera/docs/guides/CLI命令与参数索引.md) |

## 先确认运行目录

双相机：

```powershell
Set-Location "C:\Users\d114\Desktop\tech_\stereo camera"
.\.venv\Scripts\python.exe -m sorting_vision.cli --help
```

单目：

```powershell
Set-Location "C:\Users\d114\Desktop\tech_\monocular camera"
.\.venv\Scripts\python.exe -m sorting_vision.cli --help
```

`(.venv)`不会改变工作目录，也不能证明使用了哪个环境。在`tech_`目录直接运行`scripts\dual_rgbd_side_capture.py`会找不到脚本。教程中的`PS ...>`和`>>`是终端提示符，复制命令时不要复制它们。

两个工程的操作文档分别见[双相机文档目录](stereo%20camera/docs/README.md)和[单目文档目录](monocular%20camera/docs/README.md)。历史报告记录当时实验条件与结果；当前操作流程以各工程使用手册为入口。

长命令使用PowerShell反引号分行。续行符必须是行末最后一个字符，后面不能有空格或注释；一次复制完整代码块，最后一行不加续行符。相机标定见[棋盘格内参](stereo%20camera/docs/guides/棋盘格内参标定完整教程.md)和[三AprilTag外参](stereo%20camera/docs/guides/三AprilTag双相机采集与标定程序.md)。
