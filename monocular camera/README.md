# 单目RGB／RGB-D视觉工程

更新：2026-10-02。固定俯视RealSense主链及普通RGB开发；本工程不包含侧视融合。

## 快速开始

```powershell
Set-Location "C:\Users\d114\Desktop\tech_\monocular camera"
.\.venv\Scripts\python.exe `
    -c "import sys, sorting_vision; print(sys.executable); print(sorting_vision.__file__)"
.\.venv\Scripts\python.exe -m sorting_vision.cli --help
```

无需反复激活；先确认本工程目录和自己的Python。新用户从[单目完整手册](docs/guides/单目工程完整使用手册.md)按顺序操作；设备、索引和PowerShell问题见[环境排错](docs/guides/环境与相机排错.md)。

## 按任务选择入口

| 任务 | 教程 |
|---|---|
| 环境、采集、标定、训练、运行全流程 | [单目完整手册](docs/guides/单目工程完整使用手册.md) |
| 单RGB图片／多分离物块 | [单图](docs/guides/单图预测使用说明.md)／[场景](docs/guides/多物块场景预测使用说明.md) |
| 主RGB-D采集／平面／训练 | [RGB-D教程](docs/guides/RGBD数据采集与训练.md) |
| 逐类单主机／多物块采集 | [助手](docs/guides/D435if数据集拍摄助手.md)／[场景采集](docs/guides/D435if多物体批量测试拍摄.md) |
| 实际看图、联系表、拒绝记录 | [视觉审查](docs/guides/视觉审查流程.md) |
| 配置、模型版本及CNN后端 | [配置](docs/guides/配置与模型版本.md)／[模型](docs/guides/模型训练与后端.md) |
| 全部模块、命令、脚本和测试入口 | [模块](docs/guides/模块使用说明.md)／[参数](docs/guides/CLI命令与参数索引.md) |
| 换设备／平台、服务／机械互锁 | [迁移](docs/guides/D435if迁移与重新标定.md)／[服务](docs/guides/服务接口与部署.md) |

双视角功能使用独立[双相机工程](../stereo%20camera/README.md)。

## 当前模型与配置

config/d435if-initial.yaml为D435if初始迁移配置，主RGB／深度请求1280×720、30 FPS。models/stable/rgbd/geometry-rgbd-multipose-v4.npz来自旧D415；models/stable/rgb/geometry-rgb-morph-color.npz用于RGB开发。

stable表示原环境验证记录，旧D415 v4是D435if迁移基线。实验配置与模型独立保存；仅移动文件或成功加载不构成推广。详见[模型分级](models/README.md)。

## 项目结构与模块导航

源码src/sorting_vision，参数config，模型models，原始数据data，自动测试tests，离线调查scripts，教程docs/guides，历史依据docs/reports。artifacts/evaluations保留旧阶段的报告、联系表和小型特征缓存，详见[证据说明](artifacts/README.md)。

Git根目录在父目录tech_，两个工程有同名包和独立环境。数据和固定实验资料不能任意清空；规则见[WORKSPACE](WORKSPACE.md)，小图集用途见[fixtures](fixtures/README.md)。

## 普通摄像头开发

UVC使用camera-live --source uvc，索引由实际画面确认。RGB-only输出DEPTH_REQUIRED、pose／grasp为空、selected=false。真实三维定位使用完整主RGB-D数据包，不从单RGB补造深度。

## RGB-D数据与标定

color.png／depth.npy／metadata.json为单RGB-D格式，原深度通过比例转毫米。主空盘平面、主相机机械变换分别核对，默认单位矩阵不等于实测机器人标定。完整步骤见[RGB-D教程](docs/guides/RGBD数据采集与训练.md)。

## 结果协议v2

保持schema_version=2。health、PICKABLE、selected、位姿与抓取面及外部互锁共同判断。分类名、绿色框、部分可见线图不单独授权。字段／状态见[输出说明](docs/guides/RGB-D结果颜色与字段说明.md)。

## JSON/TCP控制

逐行UTF-8 JSON，请求health、detect、motion_start、motion_stop、ack_pick。运动请求为状态通知，服务不执行机器人运动。完整服务与客户端例子见[接口篇](docs/guides/服务接口与部署.md)。

## 开发验证与记录

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src tests
```

根据实际修改跑相关检查。视觉／模型改动还需实际看图，记录版本、失败分母、覆盖和计时边界。文档维护仅核对命令与链接，不代表现场验收。算法历程见[更新记录](算法更新记录.md)，所有教程见[文档目录](docs/README.md)。
