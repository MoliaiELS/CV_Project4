# 旧项目阅读与复用审计

本审计基于 GitHub 实际代码与子模块记录，不把 README 的功能介绍当作已经运行成功的证据。审计对象为 [`Zijun-Mo/vslam`](https://github.com/Zijun-Mo/vslam/tree/8d419149b51473ccc61478fea417f5351ade9c6c)，固定提交 `8d419149b51473ccc61478fea417f5351ade9c6c`（提交时间 2025-12-01，提交信息“增强VGGT ID持久化和双阶段BA”）。本次未批量克隆旧仓库，也未改动其代码。

## 给未接触旧项目的组员

旧项目解决的是“相机一边移动，系统一边估计位置并维护地图”：读入视频，筛选关键帧，把一小段关键帧交给 VGGT，预测相机、深度和点跟踪，再交给改造后的 ORB-SLAM3 管理关键帧、地图点与优化。ROS2 负责节点间的数据传递。

可以把它理解为两层：VGGT 是提供相机与几何初值的模型；ORB-SLAM3 是让持续输入的视频保持一致地图的工程框架。旧项目展示的彩色点云只是三维点集合。Project 4 的 3DGS 则还要从这些初值出发，通过图像重建损失训练高斯的形状、颜色、透明度，最终渲染未参与训练的新视角。因此“旧项目能输出点云”并不意味着 Part 1 已完成。

本次 Part 1 只需要三种初始化接同一套 3DGS，并公平评估。采用离线图像目录比重建完整 ROS2 系统更直接：`train RGB → COLMAP / VGGT / VGGT-ω → 相机和点云 → 同一 3DGS runner → test renders 与指标`。

## 代码和子模块地图

| 路径 | 实际作用 | 与本次工作的关系 |
| --- | --- | --- |
| `src/vggt_ros/vggt_ros/vggt_node.py` | 加载 `facebook/VGGT-1B`；518 预处理；滑窗推理；解码相机、深度和轨迹；发布 ROS 消息 | 模型调用和坐标处理可参考；整节点不宜直接作为离线 Part 1 入口 |
| `src/vggt_ros/vggt_ros/geometry_utils.py` | 根据深度、内参、w2c 外参，把二维轨迹反投影到世界坐标 | 可借鉴反投影公式；本次应独立验证坐标、有效深度、有限值与形状 |
| `src/vggt_ros/vggt_ros/keyframe_selector.py` | Lucas–Kanade 光流视差筛帧，维护滑窗 | 适用于视频流；Part 1 dense 必须使用全部允许训练帧，不能悄悄用此函数替换正式输入清单 |
| `src/orb_slam3_vggt_frontend/src/vggt_frontend_node.cpp` | 将 VGGT 结果接入修改后的 ORB-SLAM3 | 主要属于旧 SLAM 路线；不需要成为本次 Part 1 的依赖 |
| `src/orb_slam3_lib/orb_slam3/src/Tracking.cc`、`Optimizer.cc` | VGGT 地图点、跨窗 ID、关键帧与 BA 的定制改动 | 仅作后续对比或历史材料；不能当作统一 3DGS 训练器 |
| `src/orb_slam3_lib/orb_slam3/src/System.cc` | 包含 TUM、EuRoC、KITTI 等轨迹导出及 Atlas 存储 | 轨迹文件格式可参考；没有替代本项目独立 ATE 评测器 |
| `src/vslam_bringup/config/vslam_params.yaml` | CUDA、8 帧滑窗、视差阈值、模型名、旧机器路径 | 参数不能原样移植；存在 `/home/jun/...`、`/docs/myroom.mp4` 等机器专属路径 |
| `src/vslam_msgs/msg/VggtOutput.msg` | 相机、二维/三维轨迹、有效性、颜色等 ROS 消息 | 说明旧系统中间量；本次用磁盘上的统一场景格式交接 |
| `vggt/` | 官方 VGGT Git 子模块 | 本次实际可复用的主要软件资产 |
| `src/video_reader/` | 视频输入 Git 子模块 | 不属于 Part 1 的必需依赖 |

`.gitmodules` 确认 `vggt/` 来自 `https://github.com/facebookresearch/vggt.git`，锁定提交为 `44b3afbd1869d8bde4894dd8ea1e293112dd5eba`。`src/video_reader/` 来自 `https://github.com/PnX-HKUSTGZ/video_reader.git`，锁定 `0db85a3a296165a5ebbcafd332bc67da035c9b75`。普通 clone 不会自动下载子模块源码。

## 可以直接借用什么，还缺什么

最可靠的复用路径是固定上述官方 VGGT 提交，调用其模型、图像预处理、相机解码和 COLMAP 导出辅助函数。该提交的 [`demo_colmap.py`](https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/demo_colmap.py) 支持 no-BA 和 BA，读 `SCENE_DIR/images/`，输出 `SCENE_DIR/sparse/` 的 `cameras.bin / images.bin / points3D.bin` 以及 `points.ply`。它还处理回到原图尺度的相机参数。实际调用仍须验证 pycolmap API 版本、图像顺序和路径，不能只依赖 README 的示例命令。

旧仓库 README 所说的 gsplat 接入，是官方 VGGT README 给出的下一步训练建议。核查的旧仓库树没有独立 3DGS 训练器、可复现实验矩阵、标准 PSNR/SSIM/LPIPS 评测或完整 `vslam_evals`。在锁定的官方 VGGT 提交下，也没有名为 `demo_gsplat.py` 的文件。3DGS 训练、save/resume/render、split 清单、ATE 对齐、结果汇总和三种初始化的比较必须在本项目补齐。

VGGT-ω 是 Part 1 新增的第三路初始化，旧仓库没有此实现，需按它自己的官方 reproduction 指南和权重要求配置，不能给 VGGT 改名代替。

## 不应直接继承的实现假设

- `vggt_node.py:219` 无条件调用 `torch.cuda.get_device_capability()`，即使前面已经退回 CPU 也会失败；后续 autocast 同样写死 CUDA。独立离线入口需要真实处理设备分支。
- 节点把输入帧反序后推理，再把预测反序。此约定服务于旧滑窗；新入口应固定显式 frame ID 和相机对应关系，并验证投影，不能机械复制反序。
- `geometry_utils.py` 使用 `depth > 0.1` 过滤点；单目几何存在尺度自由度，这个常数不是通用的“0.1 米”保证。
- 光流筛帧与 8 帧滑窗会改变可见数据，不能据此把子序列命名为正式 dense/full。
- `INTERFACE.md` 的示例消息定义落后于实际 `.msg` 文件；集成时以代码和输出形状验收为准。
- VGGT 官方资源表是 H100 上 **aggregator** 的 Flash Attention 3 测量，不能把其中 8 帧约 3.23GB、0.11s 直接当成含权重、所有预测头、预处理、导出和 3DGS 的完整端到端预算；本项目资源估计与实测应分开记录。

## 环境与运行边界

旧系统预期 Ubuntu 22.04、ROS2 Humble、Python 3.10、PyTorch 2.3.1 / torchvision 0.18.1，以及 C++ 的 OpenCV、Eigen、Pangolin、g2o、DBoW2、Sophus 和 colcon。`requirements.txt` 混合 ROS、演示、LightGlue、可选 pycolmap/pyceres 等依赖，不适合作为本项目最小环境直接安装。

本项目只复用离线 VGGT 时可绕开 ROS2、Pangolin 和 ORB-SLAM3 C++ 构建；3DGS 的 CUDA 扩展与 NVIDIA GPU 仍需要单独准备。旧仓库的历史结果、7-Scenes 图表或性能数字不属于本次三个指定场景的新实验，不能填入 Part 1 结果表。

上游顶层未声明统一许可证；旧 README 分别提到 ORB-SLAM3 和 VGGT 的许可。若未来复制旧 C++ 或 Python 代码，应保存来源与对应上游许可；本项目以调用官方依赖和记录版本的方式复用，不将历史团队成果署名为本次原创。

## 本地数据查找记录

早期只读检查课程文件时未找到场景包。之后用户提供完整数据包，已核实全部三个指定场景并完成标准化，详见 [完整数据审计](local_data_audit.md)。

课程 PDF 的 Baidu/Google 链接仍作为备用来源，详见 [数据下载记录](data_sources.md)。当前输入以用户上传归档的hash和公开固定split为准；旧项目数据与历史成绩不参与本次比较。

## 两人分工的可复用接口

用户负责 Part 1：固定输入清单、三种初始化、冻结相机的统一 3DGS、指标与比较。熟悉旧项目的组员负责 Part 2：复用同一 runner，从各自允许的 sparse RGB 重新估计几何。Part 1 的 VGGT dense 结果只有在输入、版本、过滤、seed、训练预算和相机设置完全一致时，才能作为 Part 2 full 对照复用。

交接应包含原始 frame ID、train/test 列表、处理后内参、相机方向、点云、模型版本、完整配置与 run ID、checkpoint、原始 test render 和运行日志。只交一个点云或截图不足以完成复用。
