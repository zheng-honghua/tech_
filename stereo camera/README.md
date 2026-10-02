# 双相机RGB-D视觉工程

更新：2026-10-02。主RealSense采RGB与真实深度，侧UVC提供独立RGB；当前两台相机接好后可从小批试拍继续。

## 快速开始

```powershell
Set-Location "C:\Users\d114\Desktop\tech_\stereo camera"
.\.venv\Scripts\python.exe `
    -c "import sys, sorting_vision; print(sys.executable); print(sorting_vision.__file__)"
.\.venv\Scripts\python.exe -m sorting_vision.cli --help
```

无需反复激活；先确认本工程目录和自己的Python。新用户从[双相机完整手册](docs/guides/双相机工程完整使用手册.md)按顺序操作；设备、索引和PowerShell问题见[环境排错](docs/guides/环境与相机排错.md)。

## 按任务选择入口

| 任务 | 教程 |
|---|---|
| 环境、采集、标定、训练、运行全流程 | [双相机完整手册](docs/guides/双相机工程完整使用手册.md) |
| 单RGB图片／多分离物块 | [单图](docs/guides/单图预测使用说明.md)／[场景](docs/guides/多物块场景预测使用说明.md) |
| 主RGB-D采集／平面／训练 | [RGB-D教程](docs/guides/RGBD数据采集与训练.md) |
| 逐类单主机／多物块采集 | [助手](docs/guides/D435if数据集拍摄助手.md)／[场景采集](docs/guides/D435if多物体批量测试拍摄.md) |
| 实际看图、联系表、拒绝记录 | [视觉审查](docs/guides/视觉审查流程.md) |
| 配置、模型版本及CNN后端 | [配置](docs/guides/配置与模型版本.md)／[模型](docs/guides/模型训练与后端.md) |
| 全部模块、命令、脚本和测试入口 | [模块](docs/guides/模块使用说明.md)／[参数](docs/guides/CLI命令与参数索引.md) |
| 换设备／平台、服务／机械互锁 | [迁移](docs/guides/D435if迁移与重新标定.md)／[服务](docs/guides/服务接口与部署.md) |

双机专用：[采集教程](docs/guides/双视角形状与颜色数据采集完整教程.md)、[内参／AprilTag](docs/guides/三AprilTag双相机采集与标定程序.md)、[融合](docs/guides/双视角几何识别与安全融合.md)、[验收](docs/guides/双视角采集标定与验收.md)、[稀疏重建](docs/guides/更换平台与双视角稀疏三维重建复现.md)、[v7](docs/v7颜色棱线实验复现.md)、[v8](docs/v8双相机融合实验复现.md)。

## 当前模型与配置

基础临时配置config/dual/temporary.yaml；v6完整颜色物块、v7新颜色棱线语义、v8独立组合分别使用其命名配置。旧v4主RGB-D基线和RGB开发模型仍保留。v6/v7/v8均为离线开发，promotion_eligible=false；v8形状71.30%未超过历史v6 74.07%，训练口径不同，见[报告](docs/reports/v6v7取长补短v8离线对照_20261001.md)。

stable表示原环境验证记录，旧D415 v4是D435if迁移基线。实验配置与模型独立保存；仅移动文件或成功加载不构成推广。详见[模型分级](models/README.md)。

## 项目结构与模块导航

源码src/sorting_vision，参数config，模型models，原始数据data，自动测试tests，离线调查scripts，教程docs/guides，历史依据docs/reports。当前v7/v8冻结包和观测快照在output内，它们是复现来源。

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
