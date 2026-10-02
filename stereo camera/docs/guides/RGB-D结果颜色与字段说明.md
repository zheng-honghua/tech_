# RGB-D结果、状态颜色与字段判读

更新：2026-10-02。形状预测、空间结构和可执行抓取分别判断；图像有绿色框不等于已选目标。

## 1. 先看哪几个字段

按health→status→selected→pose_3d／grasp→分类及诊断顺序检查。schema_version=2是协议格式版本，不是模型v2。color_id／shape_id是标签，class_key组合它们，display_name供显示；未知标签保留unknown。

| 字段 | 含义 |
|---|---|
| frame_id、object_id | 本帧和对象身份，不能当作跨帧永久实体ID |
| color_id／color_name | 最终颜色标签 |
| shape_id／shape_name | 最终形状标签，可能被拒识 |
| confidence | 分割、颜色、形状、位姿、抓取及组合分 |
| bbox_px | 图像框，坐标尺度看对应诊断 |
| center_mm、angle_deg | 几何摘要，不能代替完整抓取位姿 |
| pose_3d | 位置、四元数、法向与接近向量 |
| grasp | 杯径、平面误差、边缘余量、有效深度与评分 |
| status | 单对象安全状态 |
| selected | 当前目标是否被流水线选择 |
| diagnostics | 掩膜、深度、模型拒识和结构证据 |

位置单位毫米，图像误差单位像素，时间单位看字段后缀。pose_3d所处坐标由主RGB-D标定中的camera_to_robot决定；默认单位矩阵保留相机系，不能当成真实机器人系。

## 2. 状态框色

pipeline3d的显示色使用BGR，以下是人眼看到的颜色：

| 状态 | 大致框色 | 解释 |
|---|---|---|
| PICKABLE | 绿色 | 真实深度／分类／抓取条件通过，还需selected与health |
| UNCERTAIN | 黄／橙黄 | 标签／置信度或其他接受条件不足 |
| OCCLUDED | 红色 | 可见性／重叠不足 |
| DEPTH_INVALID | 紫／品红 | 物块或整帧深度／平面／同步无效 |
| NO_GRASP_SURFACE | 橙色 | 无合格平整吸附面或边缘余量 |
| DEPTH_REQUIRED | RGB开发状态 | 缺真实深度，pose／grasp为空，selected=false |

单张depth-preview的伪彩不是上述状态色；颜色渐变仅编码深度范围。没有物块的空结果也需检查health和漏检，不能当作零错误。

## 3. health与selected

整帧健康失败会令结果不可执行并清除选择。health中看ok、reason及深度／平面／同步诊断；服务还给run_state、can_compute、can_pick和camera_error。

PICKABLE和selected是两个条件：静态离线冷启动常因稳定帧不足而selected=false；实时连续帧还受运动互锁。检测到颜色／形状不能绕过这些条件。

## 4. 置信度与统计口径

confidence.shape是该模型输出，confidence.combined是多项分数的几何组合，不等于在新现场验证过的成功率。

报告分别记录：
- 原始最高分类分是否正确。
- 接受后正确／错误及接受覆盖。
- 全部帧含unknown的正确率／宏召回。
- 深度安全状态、可抓数量、selected与非法升级。
- 端到端／处理耗时及实际计时边界。

49/49接受正确不代表108/108正确。漏检、提取失败和unknown保留分母；不能只统计有效特征子集。

## 5. 轮廓、面、线、点的证据

完整RGB轮廓用于形状；真实深度掩膜用于空间／抓取归属。颜色明暗面不直接等于多个物块，分面和实例分离分别判断。

RGB线可能来自纹理、反光和阴影；局部深度孔洞不是棱。部分可见面图不是完整物体网格，OBB是参考包围体。圆锥／圆柱曲线的多边形近似不能自动生成真实直棱或角点。

双机diagnostics.dual_view含fusion_state、side_reason、完整概率、投影／对应与质量。v7/v8另有两路颜色、candidate_ridges_2d、image_verified、correspondences、spatial_quality和confirmed_ridges_3d；v8 geometry_candidates_3d是几何候选，仍需双路图像支持才能成为确认线。重投影残差是自洽程度，像素扰动敏感性不是实测定位精度。

## 6. 形状缩写与拒识

常见缩写tri／quad／pent／hex表示相应棱柱／多边形族，pyramid／pyr为棱锥，oct为八面体；完整真值用shape_id及注册表，不据缩写重新建类别。

distance_rejected表示特征离训练分布过远；margin_rejected表示候选分数难分；conflict／association／calibration原因属于不同阶段。先看具体原因再补数据或改模块，降低概率门不能修坏深度。

## 7. 输出文件的查看顺序

原图／原深度→annotated与掩膜／裁剪→results-v2.json→health及diagnostics→联系表和报告。联系表有重建候选时记录对应图层，避免参考盒误作真实棱。

更多操作见[视觉审查](视觉审查流程.md)、[RGB-D教程](RGBD数据采集与训练.md)和[服务接口](服务接口与部署.md)。
