# Part 1 计算资源与服务器运行建议

## 当前机器可以完成的工作

2026-11-14 本机检查到 **Intel Iris Xe 集成显卡、约 16 GB 内存（系统报告 15.73 GiB）**，未发现 NVIDIA CUDA GPU。可以在本机完成资料与依赖核查、数据与 split 审计、相机转换、NumPy 几何测试、结果汇总与服务器代码准备。当前统一 3DGS 后端依赖 gsplat 的 CUDA 渲染和反向传播，完整实验需要 NVIDIA 服务器。

本地 `.venv-cpu` 是接口检查环境，使用 Python 3.14；它不是正式 GPU 运行环境，也不应直接复制到 Linux 服务器。没有把 CPU 测试通过写成 GPU 训练、save/resume/render 或九项实验已运行成功。

## 服务器配置

| 资源 | 建议 | 说明 |
| --- | --- | --- |
| 操作系统 | Ubuntu 22.04 或 24.04 | 现有服务器只要 CUDA 编译链兼容也可使用 |
| Python | 3.10 或 3.11 | 为 PyTorch、gsplat 和 COLMAP 相关依赖准备独立环境 |
| GPU 软件 | PyTorch 2.6.0、CUDA 12.4 构建、gsplat 1.5.3 | NVIDIA 驱动需支持 CUDA 12.4；需 `nvcc`/C++ 编译器构建 gsplat CUDA 扩展，单装 PyTorch wheel 不等于完整编译链 |
| GPU 显存 | **24 GB 用于首个 pilot**；dense 全量建议可提供 **48 GB 或 80 GB 单卡** | 已固定训练帧数为 Waymo174、DL3DV267、Re10k244；24 GB 不能视为联合几何推理全量成功保证 |
| 主机内存 | **32 GB 起步，64 GB 推荐** | COLMAP 匹配、图像/几何导出及数据处理也使用主机内存 |
| CPU | 8 核起步，16 核较方便 | 影响 COLMAP、数据处理和评测；保持各方法运行条件记录完整 |
| 可用磁盘 | 先预留 **至少 80 GB** | 含数据、约两套 1B 权重、环境、日志、checkpoint 与 renders；若高斯接近上限并保留全部 checkpoint，需扩到 160–250 GB 或按统一规则归档 |

模型推理、COLMAP 和 3DGS 可以按实验顺序串行运行，不要求同时把两套基础模型和训练器驻留显存；模型初始化结束后应释放进程。建议先租/使用一张可用卡完成校准，再决定是否需要多卡并行。单个实验的代码路线按单卡设计，不假定已经实现跨卡分片。

## 显存预估的依据与边界

[VGGT 官方固定版本 README](https://github.com/facebookresearch/vggt/blob/44b3afbd1869d8bde4894dd8ea1e293112dd5eba/README.md) 的 H100 测量对象是 aggregator，使用 Flash Attention 3；例如 50/100/200 帧记录为 11.41/21.15/40.63 GB。该表不是本项目完整端到端显存保证，不能据此声称“100 帧必定能放进 24 GB”。

[VGGT-ω 官方 README](https://github.com/facebookresearch/vggt-omega#runtime-and-gpu-memory) 给出 A100 上 **1B-512** 的端到端峰值：50/100/200/300/500 帧分别为 9.66/13.37/20.82/28.26/43.15 GB，测试输入经 balanced 预处理得到 624×416。它包含模型权重和前向激活。本方案使用的是 **1B-416-Reproduction**，并按 [reproduction 指南](https://github.com/facebookresearch/vggt-omega/blob/main/reproduction.md) 设置 `image_resolution=416`；前述表不能标成该 416 权重的实测，也不能简单按像素数比例承诺显存。

因此先用真实数据测量各路初始化峰值。若全量 dense OOM，应先扩大显存或采用明确、统一且有记录的可行实现，再重新冻结比较协议。少取帧只能作为标明 pilot 的接口测试，不能把它作为 dense/full 的正式替代品。初始化和 3DGS 的显存峰值应分别报告；3DGS 还受下游分辨率、最终高斯数和 densification 调度影响。

磁盘也存在高斯数相关的上界：SH degree 3 时每个高斯约 59 个 float 参数，仅参数与 Adam 两个状态就约 708 字节/高斯，300 万高斯的一个完整 checkpoint 可超过 2 GB；九次实验的多份 checkpoint 加起来可能超过 80 GB。80 GB 是启动预留，不能当作所有最坏情况的总上限。

## 时间预算：先校准，再安排九项正式实验

**当前没有真实场景 GPU 耗时测量。** 在中等图像尺寸、适中高斯数量且单步约 0.2–1.0 秒的条件下，九次 7k 训练本身约为 3.5–17.5 GPU 小时；给初始化和 checkpoint 渲染留出余量，可先按 **5–20 GPU 小时**安排窗口。这是条件化的排期量级，不是基准成绩，也不包括网络下载、环境编译及可能较长的 COLMAP CPU 时间。大图、大量 dense 帧、密集高斯或较慢 GPU 都可能超出该范围。

正式估计使用：

```text
每个条件耗时 = 初始化与导出时间
             + 7000 × 平均训练单步时间
             + 各 checkpoint 的 test 渲染与指标时间

九项串行总耗时 = 三个场景 × 三种初始化的上述耗时之和
GPU 小时 = 各 GPU 阶段秒数之和 / 3600
```

建议第一个 VGGT 条件先跑 `--pilot-steps 100`，记录初始化/导出时间、CUDA 峰值、分辨率、初始高斯数和训练 step 速度。前 100 步可验证编译与基本接口，但尚未覆盖 densification；再运行覆盖首次细化的短程 pilot，并估计高斯增长后速度，随后统一冻结 7k、保存步数、seed、分辨率和资源上限。训练器日志同时记录累计训练时间与高斯数量，可据此更新排期。`--pilot-steps` 保留原 7k 调度，输出到独立 pilot 根目录。

保存、恢复、渲染至少做一次实际 GPU 验收，并核对恢复后步数、optimizer、scheduler、随机状态与细化状态；本机静态检查不足以替代此项。

## 时间比较的公平性

所有方法记录“加载权重 / SfM、前向推理、过滤与导出、训练、渲染与评测”的边界和硬件。COLMAP 的初始化包含它所需的 SfM 与 BA；不能把 COLMAP BA 从时间里删掉后与基础模型比较。VGGT 主协议为 feed-forward no-BA，额外 BA 若尝试必须另列。

COLMAP 使用 CPU SIFT 还是 CUDA SIFT 必须在日志注明；若当前安装只支持 CPU，可以正常报告该配置下的结果，并把硬件差异作为耗时分析的限制，不把它解释成纯算法差异。三路下游 3DGS 使用相同训练设备、相机冻结、图像分辨率、损失、seed、步数和评测协议。

## 当前缺项与下一步

1. **数据已到位并标准化。** 用户上传的 `project4_3d.zip` 已通过完整 CRC，三场景784张正式RGB全部读取并验证；旧 `.partial` 不用于实验。
2. 包内没有显式官方 split，已公开每8帧测试1帧的本组固定列表。标准化后 train/test 为174/25、267/39、244/35，相机暂标 reference。同步 `data/prepared` 可直接复用相同输入。
3. 将本项目源代码、固定配置和依赖版本同步到 NVIDIA 服务器；准备 VGGT 与 VGGT-ω 权重，核对各权重的来源、访问状态与 hash。
4. 运行环境检查和首个 pilot，记录资源实测；在本机缺少 CUDA 的情况下，完整初始化/训练/渲染结果仍标 pending。
5. 跑完三个场景的全部九个条件后，生成真实覆盖表、训练曲线、同视角对比和分析，再写最终 Part 1 结论。

记录各阶段的实际运行状态，并保留失败日志和原因。
