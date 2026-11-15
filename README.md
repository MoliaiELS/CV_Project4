# Project 4 · Part 1 + Part 2

AIAA 3201 Project 4 的 Part 1 比较 COLMAP、VGGT 和 VGGT-Ω 的相机与点云初始化。三个课程场景的 **9 项正式比较已完成**。每项使用同一套 3D Gaussian Splatting 训练器和同场景固定的测试视角。

- [最终指标](results/part1-final/coverage.csv)包含 PSNR、SSIM、LPIPS、对齐后的 ATE 和各阶段耗时。
- [收敛曲线](results/part1-final/convergence.png)比较五个固定迭代点的测试视角质量。
- [Part 1 报告与复现记录](docs/part1_report.md)、[CVPR LaTeX 正文与编译方法](report/README.md)。
- [Part 2 最终结果与复现说明](docs/part2_results.md)记录三个场景的 **6 项 Full/Sparse 正式实验**。[报告正文](report/part2.tex)与[课程服务器交接说明](docs/integration_handoff.md)。

如果你不了解旧项目，先读 [旧仓库说明与复用审计](docs/reuse_audit.md)。旧仓库是 VGGT + ORB-SLAM3 / ROS2 的 SLAM 原型，并非现成的 3DGS 项目。本实现复用其 VGGT 技术路线，使用固定官方模型源码；不需要安装旧 ROS 环境。

```bash
git clone https://github.com/MoliaiELS/CV_Project4.git
cd CV_Project4
```

仓库包含源码、固定划分、测试与汇总结果。运行时需要另外准备课程数据和模型权重。原始数据、环境、逐帧渲染和模型 checkpoint 存放在本地或服务器上。

## 已实现的流程

1. 导入课程 `rgb/ + cameras.json`、COLMAP txt/bin、NeRF/nerfstudio transforms 或纯 RGB；明确 train/test，统一下游图像尺寸，保存输入 SHA256。
2. 只把真实 train RGB 交给初始化器；输出统一 `scene.npz`，含 OpenCV `c2w`、`K`、点云、RGB 和原图对应关系。
3. 使用 gsplat 1.5.3 的标准 SH、L1+SSIM、增密/剪枝训练；固定相机，关闭 appearance optimization。默认 7,000 步、seed 42、黑色背景，下游图像长边不超过 960。三方法配置一致。
4. 保存完整 Gaussian、optimizer、scheduler、增密、采样和随机状态，支持断点续训。只通过训练相机拟合一个 Sim3，将测试 reference 相机映射至估计坐标系，再渲染原始 Gaussian 场景。
5. 计算 test PSNR / SSIM / LPIPS、训练轨迹 ATE RMSE、耗时与显存；输出覆盖表、收敛曲线、同视角比较图。缺权重、注册失败、退化对齐、缺 LPIPS 都明确保留状态。

参考相机只在数据标准化和评价侧使用，初始化与训练不加载 `reference.npz`。若相机来自原数据 SfM，报告为 reference；仅确认其为真实 GT 后才使用 `--reference-kind ground_truth`。ATE 经 Sim3 对齐，单位是否为米取决于 reference，不能因此声称恢复绝对尺度。

## 1. 准备课程数据

将 `--raw` 指向直接包含 `405841/`、`DL3DV-2/` 和 `Re10k-1/` 的目录。本机原始场景直接位于 `data/`，可在安装依赖后运行：

```bash
python scripts/prepare_course_data.py --raw data --output data/prepared
```

课程服务器可将 `--raw` 设为 `/data/disk1/shared/aiaa3201/project4_3d`，脚本读取这个共享目录，并将标准化数据写入自己的工作区。也可以从课程压缩包准备数据：

```bash
python scripts/inspect_archive.py data/downloads/project4_3d.zip
python scripts/inspect_archive.py data/downloads/project4_3d.zip --extract-to data/raw
python scripts/prepare_course_data.py
```

归档校验与原始数据结构见[数据审计](docs/local_data_audit.md)。复现既有 checkpoint 时，使用原运行的 `data/prepared/` 与环境，保留 manifest、参考相机和图像指纹。

手册的 [Google Drive](https://drive.google.com/drive/folders/1euG7pnbFowljVWoNLcbCmil81IVsIEfM?usp=drive_link) 和 [百度网盘](https://pan.baidu.com/s/1Sa18zCeYiYA2gWAllo11dg?pwd=p3bm)（提取码 `p3bm`）保留为备用，详见 [数据来源记录](docs/data_sources.md)。该 Drive 归档大小与上传包不同，不直接当作逐字节相同的数据版本。本机旧 `.partial` 不用于实验。

## 2. Linux / NVIDIA 服务器环境

推荐 Python 3.10/3.11、CUDA toolkit 12.4（含 `nvcc`）、支持 CUDA 的 NVIDIA 驱动、GCC/G++ 和 Git。先建独立环境：

```bash
conda create -n project4 python=3.10 -y
conda activate project4
bash scripts/setup_server.sh
```

脚本固定 PyTorch 2.6.0 + torchvision 0.21.0 + cu124、NumPy 1.26.4、gsplat 1.5.3 commit `937e299...`。VGGT 原 requirements 指向更老 torch，Ω 要求 torch≥2.6。IADC 正式实验使用了上述统一环境。若服务器 CUDA/toolchain 不同，可在独立环境安装匹配版本并记录 `pip freeze`。

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

Hugging Face 如需账户授权，可通过 `HF_TOKEN` 使用已有凭证。IADC 的 Ω 实验使用 [ModelScope 的同一 reproduction checkpoint](https://modelscope.cn/models/facebook/VGGT-Omega)。把下载的文件放到下列位置并用 `sha256sum` 核对。

| 文件 | SHA-256 |
| --- | --- |
| `weights/vggt/model.pt` | `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0` |
| `weights/vggt-omega/vggt_omega_1b_416_reproduce.pt` | `e900833330eb0e570e10543e6108d0dd704fa7c8dbc32738a645987875298896` |

采用 [VGGT-Ω 官方复现设置](https://github.com/facebookresearch/vggt-omega/blob/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6/reproduction.md)的 416 档。LPIPS 首次使用会下载 AlexNet 特征权重，离线服务器需提前缓存。

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

## 4. 复现九项 Part 1 实验

[正式复现配置](configs/part1_reproduce.json)固定 7,000 步、seed 42、相同划分和训练设置。Waymo 与 Re10k 采用共享 PINHOLE 内参、8 个 COLMAP 工作线程，DL3DV 采用逐图内参、4 个工作线程。各场景运行 CPU SIFT 和穷举匹配，Waymo 的匹配块大小为 200，另两场景为 50。VGGT 对全部训练视角联合推理，并将预测头使用的特征卸载到 CPU。

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENBLAS_NUM_THREADS=1
python -m p4.experiment run --config configs/part1_reproduce.json --dry-run
python -m p4.experiment run --config configs/part1_reproduce.json
python -m p4.experiment summarize --config configs/part1_reproduce.json --output results/part1_reproduce
python -m p4.visualize --config configs/part1_reproduce.json --output results/part1_reproduce
```

训练和评价文件位于 `runs/part1_reproduce/`。[探索配置](configs/part1.json)可用于试跑。首次部署可以用 `python -m p4.experiment run --scenes Re10k-1 --methods vggt --pilot-steps 100`，输出位于 `runs/part1_pilot/`。显存不足时初始化器会给出帧数和尺寸，并保留失败状态。

每个 run 记录 `status.json`、冻结配置、初始化日志与点云、训练日志与 checkpoint、500/1k/2k/4k/7k 的测试渲染与指标、版本和文件指纹。COLMAP 未全注册时该主条件标记 failed。LPIPS 缺失时不会完成该条件。修改协议、源码、权重或数据后请换 `output_root`；校验不匹配不会静默复用。

意外中断可手动恢复到相同预算：

```bash
python -m p4.train train --dataset data/prepared/Re10k-1 --scene runs/part1_reproduce/Re10k-1/vggt/init/scene.npz --output runs/part1_reproduce/Re10k-1/vggt/train --config runs/part1_reproduce/Re10k-1/vggt/frozen_config.json --resume runs/part1_reproduce/Re10k-1/vggt/train/checkpoints/step_004000.pt
```

矩阵命令会复用校验通过的已完成阶段；未完成训练默认归档该次尝试并重新训练。手动 resume 完成后，可单独执行 dry-run 中列出的 render / evaluate 命令获取结果。正式比较使用最终固定步数；中间测试曲线仅分析收敛，不据此选择最佳 checkpoint。

## 5. 输出与后续交接

已发布的 [9 项指标与 45 个评价点](results/part1-final/)来自 IADC 完成的正式运行。生成命令分别保存覆盖表、各场景测试曲线与同视角的参考图和各方法渲染。

- [资源预估](docs/resources.md)、[验证记录](docs/validation.md)、[来源与 AI 使用记录](docs/sources_and_prompts.md)。

```bash
python -m pytest -q
```

Part 1 的 VGGT dense checkpoint 可以用于 Part 2 full 对照，复用条件包括相同的 manifest/split、过滤、权重、seed、预算和相机设置。sparse 分支从稀疏图像重新估计几何。已归档的 Part 2 结果使用课程服务器上独立训练的 full 对照。

上游 gsplat 1.5.3 的 opacity reset 有布尔优先级错误，本训练器明确补上同一 reset 调用；三种初始化均使用该修正，记录在 provenance 中。论文分析应描述所运行的完整系统，不把所有差异归因于模型大小。
