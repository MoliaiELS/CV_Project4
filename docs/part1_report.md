# Part 1 报告材料填写结构

状态：代码与 CPU 测试就绪；真实三场景实验 pending。此文件是待实验填充的结构，不是最终课程报告。

## 问题与方法

在 Waymo-405841、Dl3dv-2、Re10k-1 的相同训练 RGB 上，分别运行 COLMAP、VGGT 和 VGGT-Ω reproduction，得到相机与点云。所有分支接入同一 vanilla 3DGS runner，固定相机，使用相同下游分辨率、损失、预算、seed 和测试清单。COLMAP 包含必需的 BA，神经分支不进行 BA。

7k 步是阶段比较预算，应在 pilot 后冻结。VGGT 使用官方方形 518 预处理，Ω reproduction 使用 balanced 416；裁剪、resize、padding 的变换已反映到下游 K。记录各方法的输入数、点云数、模型和权重 hash、显存与耗时。

## 实验应填入的证据

1. `results/part1/coverage.csv` 中的 9 条实际记录；completed / failed / pending 分开。
2. 三场景 test PSNR、SSIM、LPIPS；初始化 ATE RMSE 及 reference 来源、单位、Sim3 对齐说明。
3. 初始化、训练、渲染与评价成本；COLMAP mapper 时间包括 BA，不冒称已有单独 BA 计时。
4. 初始/最终 Gaussian 数；CPU/GPU 配置；固定迭代点的收敛曲线。
5. 相同 held-out 相机的三方法 raw Gaussian renders 对比，解释可见差异及失败案例。

没有真实实验结果前不填写结论。可以讨论的实际证据包括：某方法在同预算下的收益、注册失败、轨迹退化、漂浮点、细节与空洞；不能仅凭图像模糊断言位姿错误。

## 两页 checkpoint slides 的内容

- 方法页：三初始化 → 统一相机/点云 → 同一 3DGS；附三场景、输入数、预算和评价边界。
- 结果页：覆盖表 + 一张最有解释力的同视角图/收敛曲线；只写有测量支持的一条结论和实际局限。若尚未运行，直接显示 pending，不用模板数值替代。

最终整份课程报告仍需以官方 CVPR LaTeX 模板包含全部三个 Part；当前只开展 Part 1，不能据此声称 Project 4 已全部完成。
