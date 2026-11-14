# Project 4 · Part 1

本仓库实现 AIAA3201 Part 1：**COLMAP / VGGT / VGGT-Ω → 同一套 vanilla 3DGS → 同一测试集评价**。三个指定场景共 9 项比较。当前交付是代码、CPU 验证和服务器运行流程；**尚无真实场景训练成绩**。实际状态见 [coverage.md](results/part1/coverage.md)，不能用合成测试或旧项目数字补齐。

如果你不了解旧项目，先读 [旧仓库说明与复用审计](docs/reuse_audit.md)。旧仓库是 VGGT + ORB-SLAM3 / ROS2 的 SLAM 原型，并非现成的 3DGS 项目。本实现复用其 VGGT 技术路线，使用固定官方模型源码；不需要安装旧 ROS 环境。

代码仓库：[MoliaiELS/CV_Project4](https://github.com/MoliaiELS/CV_Project4)。服务器可直接获取：

```bash
git clone git@github.com:MoliaiELS/CV_Project4.git
cd CV_Project4
```

仓库包含源码、固定 split、测试和审计说明。正式数据保存在本地 `data/`，原始包位于 `data/downloads/project4_3d.zip`，打包交付位于 `output/`；这些文件、虚拟环境和模型权重不进入 Git。新服务器需另行同步 `data/prepared/` 或原始数据包。

## 已实现的流程

1. 导入课程 `rgb/ + cameras.json`、COLMAP txt/bin、NeRF/nerfstudio transforms 或纯 RGB；明确 train/test，统一下游图像尺寸，保存输入 SHA256。
2. 只把真实 train RGB 交给初始化器；输出统一 `scene.npz`，含 OpenCV `c2w`、`K`、点云、RGB 和原图对应关系。
3. 使用 gsplat 1.5.3 的标准 SH、L1+SSIM、增密/剪枝训练；固定相机，关闭 appearance optimization。默认 7,000 步、seed 42、黑色背景，下游图像长边不超过 960。三方法配置一致。
4. 保存完整 Gaussian、optimizer、scheduler、增密、采样和随机状态，支持断点续训。只通过训练相机拟合一个 Sim3，将测试 reference 相机映射至估计坐标系，再渲染原始 Gaussian 场景。
5. 计算 test PSNR / SSIM / LPIPS、训练轨迹 ATE RMSE、耗时与显存；输出覆盖表、收敛曲线、同视角比较图。缺权重、注册失败、退化对齐、缺 LPIPS 都明确保留状态。

参考相机只在数据标准化和评价侧使用，初始化与训练不加载 `reference.npz`。若相机来自原数据 SfM，报告为 reference；仅确认其为真实 GT 后才使用 `--reference-kind ground_truth`。ATE 经 Sim3 对齐，单位是否为米取决于 reference，不能因此声称恢复绝对尺度。

## 1. 课程数据已在本机验收

用户上传的 **project4_3d.zip** 已归档到 `data/downloads/`，包含全部三个指定场景，完整 CRC 校验通过，已安全解压到 `data/raw/project4_3d/` 并完成标准化。源 ZIP 为 907,794,590 bytes，SHA256 `2e2a2658d6735761b5dd3c090c5d35a4a2cca04088cde6ae0cb231f90214c415`。详细事实见 [完整数据审计](docs/local_data_audit.md)。

在空的服务器工作区复制同一压缩包后，可重现解包（本机已解包，不需重复覆盖）：

```bash
python scripts/inspect_archive.py data/downloads/project4_3d.zip
python scripts/inspect_archive.py data/downloads/project4_3d.zip --extract-to data/raw
python scripts/prepare_course_data.py
```

也可直接同步本机 `data/prepared/`，保留本次预处理结果、manifest 和 reference。不要复制 Windows 的 `.venv-cpu` 到 Linux。重新准备时应记录图像库版本并让两位组员共用同一份 prepared 输入；文件指纹会识别不同预处理输出。

手册的 [Google Drive](https://drive.google.com/drive/folders/1euG7pnbFowljVWoNLcbCmil81IVsIEfM?usp=drive_link) 和 [百度网盘](https://pan.baidu.com/s/1Sa18zCeYiYA2gWAllo11dg?pwd=p3bm)（提取码 `p3bm`）保留为备用，详见 [数据来源记录](docs/data_sources.md)。该 Drive 归档大小与上传包不同，不直接当作逐字节相同的数据版本。本机旧 `.partial` 不用于实验。

## 2. Linux / NVIDIA 服务器环境

推荐 Python 3.10/3.11、CUDA toolkit 12.4（含 `nvcc`）、支持 CUDA 的 NVIDIA 驱动、GCC/G++ 和 Git。先建独立环境：

```bash
conda create -n project4 python=3.10 -y
conda activate project4
bash scripts/setup_server.sh
```

脚本固定 PyTorch 2.6.0 + torchvision 0.21.0 + cu124、NumPy 1.26.4、gsplat 1.5.3 commit `937e299...`。VGGT 原 requirements 指向更老 torch，Ω 要求 torch≥2.6，因此这里显式选择统一环境，**仍须通过服务器 pilot 验证**。不要在此环境继续执行两份上游 requirements 覆盖这些版本。若服务器 CUDA/toolchain 不同，应在新环境统一调整并记录 `pip freeze`，避免混用不同后端。

另装 **COLMAP 3.11.1**（优先）并确认 `colmap -h` 能运行。Ubuntu 的包版本可能不同；初始化器记录版本和完整命令，如参数不兼容会保留日志报错。CPU SIFT 是本协议的默认设置，时间表会说明该设置，不能将跨硬件差异直接归因于算法。

模型源码由 setup 脚本固定：

| 模型 | 源码 commit | 权重 |
| --- | --- | --- |
| VGGT | `a288dd0f14786c93483e45524328726ab7b1b4ce` | `facebook/VGGT-1B` 的 `model.pt` |
| VGGT-Ω | `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6` | `facebook/VGGT-Omega` 的 `vggt_omega_1b_416_reproduce.pt` |

```bash
python scripts/download_weights.py
python -m p4.doctor --output results/server_doctor.json
python scripts/gpu_smoke.py --output runs/gpu_smoke
```

若官方模型页要求账户授权，先在模型页完成你自己的访问流程，再通过 `HF_TOKEN` 使用已有凭证；脚本不会申请、代替接受条款或打印 token。Ω 使用 reproduction checkpoint 和 `image_resolution=416`，不替换为原 512 权重。LPIPS 首次使用还会下载 AlexNet 特征权重；离线服务器需提前缓存。

`gpu_smoke.py` 用合成场景真实执行训练、save/resume、渲染、增密和 opacity reset，并检查续训数值一致性。这是服务器验收，**不是课程实验**。本机无 CUDA，尚未运行该脚本。

## 3. 已固定的数据协议

完整上传包没有显式官方 train/test。按执行方案允许的固定划分，采用自然排序后索引 `0,8,16,...` 为测试，其余为训练；**这是本组自定 split**。源图像列表已保存到 [configs/splits](configs/splits/README.md)。实际标准化并校验的结果为：

| 场景 | 真实来源 | Train / Test | 下游尺寸 |
| --- | --- | ---: | --- |
| Waymo-405841 | `405841/FRONT/rgb` + `calib` + `gt` | 174 / 25 | 960×640 |
| Dl3dv-2 | `DL3DV-2/rgb` + `cameras.json` | 267 / 39 | 256×256 |
| Re10k-1 | `Re10k-1/images` + `cameras.json` | 244 / 35 | 256×256 |

```bash
python scripts/prepare_course_data.py --raw data/raw/project4_3d --output data/prepared
```

该脚本检查已有 prepared 是否符合固定列表与图像 SHA，校验通过就复用；新服务器则从原包按这些列表生成。正式 RGB 共784张，排除12张 DL3DV notebook 备份和 Re10k 视频。Waymo 按真实标定去畸变后 resize，读取 `gt` 的 OpenCV c2w，不重复应用车辆外参，不用 `gt_old` 或深度初始化。DL3DV/Re10k 读取已核实的 xyzw c2w 与归一化 K。

所有源相机统一保守标为 `reference`，相机表的存在不等于已证明米制 GT。训练使用三种方法各自估计的相机和点云，源相机保存在独立评价文件。另支持 COLMAP txt/bin、NeRF/nerfstudio transforms 和纯 RGB 的通用数据接口，见 `python -m p4.data prepare --help`。外部数据的源 SfM 点云不会复制成初始化结果。

每个 prepared 目录包含：

```text
images/         # 三种方法共用的规范 RGB
manifest.json   # 原帧ID、划分、尺寸、图像SHA、预处理与split hash
reference.npz   # 仅评价的相机（如果源数据提供）
```

## 4. 先 pilot，再冻结并跑九项实验

配置见 [configs/part1.json](configs/part1.json)。默认 7k 是执行方案建议的阶段预算，不是课程指定门槛。必须在正式比较前根据首场景试跑统一冻结预算，不能给单个方法私自减少训练帧或步数。

```bash
python -m p4.experiment run --dry-run
python -m p4.experiment run --scenes Re10k-1 --methods vggt --pilot-steps 100
python -m p4.experiment run
python -m p4.experiment summarize
python -m p4.visualize
```

pilot 写入独立的 `runs/part1_pilot/`，不会算入正式完成数量。神经初始化默认对所有 train 帧联合推理；OOM 会报出帧数和预处理尺寸，**不会自动截短 full 或拼接未经验证的窗口**。完整 dense 若装不下，先换大显存 GPU，或另外实现、记录并验证重叠窗口的 Sim3 融合协议。

每个 run 记录 `status.json`、冻结配置、初始化日志与点云、训练日志与 checkpoint、500/1k/2k/4k/7k 的测试渲染与指标、版本和文件指纹。COLMAP 未全注册时该主条件标记 failed。LPIPS 缺失时不会完成该条件。修改协议、源码、权重或数据后请换 `output_root`；校验不匹配不会静默复用。

意外中断可手动恢复到相同预算：

```bash
python -m p4.train train --dataset data/prepared/Re10k-1 --scene runs/part1/Re10k-1/vggt/init/scene.npz --output runs/part1/Re10k-1/vggt/train --config runs/part1/Re10k-1/vggt/frozen_config.json --resume runs/part1/Re10k-1/vggt/train/checkpoints/step_004000.pt
```

矩阵命令会复用校验通过的已完成阶段；未完成训练默认归档该次尝试并重新训练。手动 resume 完成后，可单独执行 dry-run 中列出的 render / evaluate 命令获取结果。正式比较使用最终固定步数；中间测试曲线仅分析收敛，不据此选择最佳 checkpoint。

## 5. 输出、复现与后续交接

- `results/part1/coverage.csv|json|md`：真实完成覆盖、质量、ATE、耗时、Gaussian 数与显存；缺项保持 pending。
- `results/part1/*_convergence.png`：固定迭代点的 test PSNR 曲线，仅使用通过校验的条件。
- `python -m p4.visualize`：同测试视角的 GT / 三方法渲染比较图；缺项明确标识。
- [资源预估](docs/resources.md)、[验证记录](docs/validation.md)、[报告填充结构](docs/part1_report.md)、[来源与 AI 使用记录](docs/sources_and_prompts.md)。

```bash
python -m pytest -q
```

当前 Part 1 的 VGGT dense checkpoint 以后可以交给 Part 2 成为 full 条件，但对方仍须核对 manifest/split、过滤、权重、seed、预算和相机设置。sparse 分支必须重新估计几何。这里不声称已经实现 Part 2 或 Part 3，也不将旧团队的 ATE 作为本次结果。

上游 gsplat 1.5.3 的 opacity reset 有布尔优先级错误，本训练器明确补上同一 reset 调用；三种初始化均使用该修正，记录在 provenance 中。论文分析应描述所运行的完整系统，不把所有差异归因于模型大小。
