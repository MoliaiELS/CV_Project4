# Part 2 正式实验

三个指定场景各完成一组独立的 Full/Sparse 实验。各分支使用自己的训练 RGB 估计相机和点云后，固定相机训练标准 3DGS 7,000 步，随机种子为 42。Waymo 的原序列采样步长为 10，DL3DV 和 RealEstate10K 为 30；各场景的两条分支保留一致的测试集。paired ATE 在同一组 sparse 训练相机上拟合和比较。

| 场景 | Full / Sparse 训练帧 | 测试帧 | Full / Sparse PSNR（dB） |
| --- | ---: | ---: | ---: |
| Waymo-405841 | 174 / 20 | 25 | 17.689 / 16.726 |
| Dl3dv-2 | 267 / 11 | 39 | 18.609 / 15.479 |
| Re10k-1 | 244 / 10 | 35 | 11.364 / 11.234 |

- [最终指标](../results/part2_formal_7000/comparison.csv)包含 PSNR、SSIM、LPIPS、paired ATE、训练时间和 checkpoint 校验值。
- [收敛数值](../results/part2_supplement_v1/convergence.csv)和[三指标收敛图](../results/part2_supplement_v1/figures/convergence_all_metrics.png)覆盖 30 个检查点和 990 条逐帧评价。
- [轨迹对照](../results/part2_supplement_v1/figures/paired_trajectories.png)、[相机诊断](../results/part2_supplement_v1/camera_diagnostics.csv)和[投影覆盖率](../results/part2_supplement_v1/coverage.csv)提供几何分析。投影覆盖率衡量渲染视角上的 Gaussian alpha 支持。
- [各阶段耗时](../results/part2_supplement_v1/runtime.csv)与[耗时图](../results/part2_supplement_v1/figures/runtime_breakdown.png)记录正式 Full/Sparse 对照。[课程报告正文](../report/part2.tex)包含结果与局限。

Part 1 的 9 个 checkpoint 已在课程服务器导入核验。其中三个对应 VGGT Full；重渲染得到的 PSNR 与原归档一致。此结果用于跨服务器复现验收。Part 2 正式表使用原先独立训练的一组 Full/Sparse 对照。

## 复现与归档

[版本与原始归档 SHA-256](../results/part2_supplement_v1/course_merge_provenance.json)记录两个历史训练版本。Part 2 的训练绑定 `bd2e3ec` 的 `p4/` 文件内容及原始 manifest、初始化和权重。重渲染、逐帧诊断和补充收敛脚本应在对应的历史工作树中使用，并准备脚本所引用的原始 run 和数据目录。Part 1 replay 使用 `bde9244` 对应的原始包。合并工作树包含供检查的源码和汇总结果。

[summarize_part2_three_scenes.py](../scripts/summarize_part2_three_scenes.py)用于汇总正式结果。补充分析脚本包括 [固定检查点评价](../scripts/part2_convergence.py)、[相机与覆盖诊断](../scripts/part2_diagnostics.py)、[初始化速度实验](../scripts/part2_fast_initialize.py)和 [Part 1 checkpoint 重渲染](../scripts/part1_checkpoint_replay.py)。这些脚本复用原实验目录、配置和 GPU 环境。[环境准备和 pilot 记录](part2_hpc.md)

课程服务器上的 `output/handoff/part2-final/` 保存两个团队原始 ZIP、源码快照、含课程图像的失败案例和期中展示提纲。原始训练 checkpoint 仍可从各案例的历史运行目录核对。
