# 来源与 AI 使用记录

## 用户授权的本次范围

2026-11-14：先完成 Project 4 Part 1；优先实现代码，本地资源不足则给出估算，之后由用户同步/拉取到服务器运行。两人团队，一人熟悉旧仓库，用户需要能够独立理解和使用它。之后补充：数据尚未下载，要求检查手册中的数据入口。

随后用户提供目录中的 `project4_3d.zip`，要求检查是否为所需场景资源。已核验完整包、实际读取并标准化三场景，公开固定 train/test 清单，继续原 Part1 代码任务。数据已到位，GPU实验仍未运行。

上传文档作为课程要求和执行方案的证据。后续用户明确要求清理目录并推送 `MoliaiELS/CV_Project4`，因此该次GitHub发布获得直接授权。没有代交Canvas、发送组员消息或申请模型授权。

## 课程与方案

- `Project 4_new__Generative_Sparse_View_3D_Reconstruction.pdf`，Fall 2026，Part 1 见第 3–4 页，mandatory 数据与指标见第 5–6 页。
- `Project4_Plan_and_Checkpoint.pdf`，v2.0，2026-11-13；本次采用 Part 1 所列统一接口与九项比较，7k 是建议预算，尚待 pilot 冻结。

## 实现依据（固定版本）

- [旧项目](https://github.com/Zijun-Mo/vslam/tree/8d419149b51473ccc61478fea417f5351ade9c6c)：方法来源与复用审计，无本次实验数据。
- [VGGT](https://github.com/facebookresearch/vggt/tree/a288dd0f14786c93483e45524328726ab7b1b4ce)：模型、官方预处理和相机 decoder。
- [VGGT-Ω](https://github.com/facebookresearch/vggt-omega/tree/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6)、[reproduction 指南](https://github.com/facebookresearch/vggt-omega/blob/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6/reproduction.md)：1B-416 权重、输入约定与资源参考。
- [gsplat 1.5.3](https://github.com/nerfstudio-project/gsplat/tree/937e29912570c372bed6747a5c9bf85fed877bae)、[训练示例](https://github.com/nerfstudio-project/gsplat/blob/937e29912570c372bed6747a5c9bf85fed877bae/examples/simple_trainer.py)：rasterization、参数、学习率与 DefaultStrategy API；本项目独立数据/训练/恢复接口。
- [COLMAP](https://github.com/colmap/colmap/tree/3.11.1)：PINHOLE RGB-only SfM，SIFT 与 mapper（包含 BA）。
- [3DGS 论文与项目](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)、[LPIPS](https://github.com/richzhang/PerceptualSimilarity)：方法和评价引用。正式论文需补齐 BibTeX 和课程阅读清单引用。

外部源码、模型、课程数据保留各自许可。本仓库不包含旧团队的报告图片或用旧结果冒充新结果。

## 本轮使用 AI 的工作记录

AI 协助：阅读文档、静态审计旧仓库、查验官方 API、编写数据与相机转换、初始化、训练、评价、编排、测试、说明文档；通过并行独立检查发现并修复缓存/指纹及上游 opacity reset 问题。

核心提示主题（供课程 prompt list 整理）：

1. 根据两份 PDF 区分课程要求、组内建议和本次用户授权，先实现 Part 1。
2. 审计旧 ROS2/VGGT/ORB-SLAM3 仓库，说明可复用资产与缺失 3DGS 流程。
3. 实现三种 RGB-only 初始化，准确恢复模型预处理到 canonical K，拒绝 test 泄漏。
4. 实现统一 3DGS、完整 save/resume、train-only Sim3 test rendering 和一致指标。
5. 独立检查实验覆盖、失败证据、阶段缓存、输入/权重/输出指纹，并编写回归测试。
6. 查找课程数据链接；无法在合理时间下载时保存真实访问状态及服务器命令。

这些是主题摘要，不冒称逐字完整对话导出。最终提交前应从聊天记录补充教师要求的原始 prompt list、人工审核和实际运行记录。未实测 CUDA、模型精度或课程场景成绩。

## Part 3

- 继续完成课程项目 Part 3，目标是可提交。先读课程要求，检查 Part 1 和 Part 2，合并 GitHub、本地和课程服务器的版本。
- 实验优先用课程服务器，暂停使用 IDAC。训练启动后持续检查，遵守共享资源规则，保留其他组的作业。
- 根据课程要求实现两轮生成与重建、主动视角规划、几何一致性门控与计算量匹配的对照。用固定测试集的原始 3DGS 渲染核验指标，依据实际运行记录写图表和报告。

生成组件使用官方 Difix-ref 推理模型和代码。模型、依赖和固定版本列于 [Part 3 复现文档](part3.md)。
