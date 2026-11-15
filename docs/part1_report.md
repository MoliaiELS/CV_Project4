# Part 1 正式实验结果

三个课程场景各完成 COLMAP、VGGT 和 VGGT-Ω 初始化后的 3DGS 训练，共九项正式实验。三种初始化在各场景使用同一训练集和同一留出集，GS 优化统一进行 7,000 步，随机种子为 42。课程提供的 Waymo 标定用于统一去畸变。初始化从训练 RGB 估计几何与相机，3DGS 固定各自估计的相机。测试时使用训练相机拟合的 Sim(3)，将留出视角从参考坐标系转换到估计坐标系后渲染。

| 场景 | 训练帧 | 留出帧 |
| --- | ---: | ---: |
| Waymo-405841 | 174 | 25 |
| Dl3dv-2 | 267 | 39 |
| Re10k-1 | 244 | 35 |

## 数据和图表

- [九项最终指标、位姿和耗时](../results/part1-final/coverage.csv)
- [45 组留出集指标](../results/part1-final/convergence.csv)
- [三场景 PSNR 收敛图](../results/part1-final/convergence.png)
- [版本、划分哈希和计时范围](../results/part1-final/provenance.json)
- [可并入课程论文的 LaTeX 正文](../report/part1.tex)

## 实测结果

VGGT-Ω 在三个场景的初始化耗时最短，为 36–42 秒。VGGT 为 75–122 秒，COLMAP 为 390–806 秒。COLMAP 的计时包括 CPU SIFT 特征提取、穷举匹配与 mapper 中的束调整。端到端耗时取成功正式尝试的总墙钟时间，包含初始化、训练、渲染与指标评估。

画质最优方法随场景和指标变化。Dl3dv-2 的 COLMAP 获得三项最优指标，PSNR 为 23.61 dB，SSIM 为 0.771，LPIPS 为 0.077。Re10k-1 的 VGGT-Ω 获得三项最优指标，分别为 18.33 dB、0.640 和 0.068。Waymo-405841 的 VGGT 获得最高 PSNR 和 SSIM，分别为 17.74 dB 和 0.655，COLMAP 的 LPIPS 最低，为 0.440。

训练相机中心的 ATE RMSE 以参考轨迹单位计。COLMAP 在 Waymo-405841 和 Dl3dv-2 的 ATE 最低，VGGT-Ω 在 Re10k-1 最低。ATE 的排序与留出图像质量的排序存在差异，例如 Waymo-405841 的 COLMAP ATE 低于 VGGT，而后者的 PSNR 更高。

留出集指标在第 500、1,000、2,000、4,000 和 7,000 步计算。部分曲线在中途达到最高值，例如 Dl3dv-2 的 COLMAP 在第 2,000 步为 24.00 dB，在第 7,000 步为 23.61 dB。最后一个检查点是预先固定的画质比较点。

## 正式运行配置

| 场景 | COLMAP PINHOLE 内参 | CPU 线程 | 穷举匹配块 |
| --- | --- | ---: | ---: |
| Waymo-405841 | 场景共享 | 8 | 200 |
| Dl3dv-2 | 每图独立 | 4 | 50 |
| Re10k-1 | 场景共享 | 8 | 50 |

第一次 COLMAP 初始化在 Waymo-405841 和 Re10k-1 分别注册 107/174 帧和 66/244 帧。最终正式运行完成了全量注册。Waymo-405841 的共享内参方案采用 200 的匹配块，以覆盖全部 174 张训练图像并避开分块匹配崩溃。耗时比较对应表中的实际运行配置。

VGGT 使用 518 输入分辨率和 CPU 中间特征卸载。VGGT-Ω 使用 416 分辨率的 reproduction 模型版本。位姿、画质和运行时间的差异对应整套初始化配置，包括输入处理、模型版本和实现方式。正式结果对应 [历史运行源码](https://github.com/MoliaiELS/CV_Project4/tree/bde9244f692cc497355a307826ce1ca25f39cd99)。提交版复现步骤见[项目 README](../README.md)。
