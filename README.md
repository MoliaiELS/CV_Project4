# Project 4 · Generative Sparse-View 3D Reconstruction

AIAA 3201 课程项目。我们用同一套 3D Gaussian Splatting 后端比较稠密几何初始化，测量未知位姿的全视角与稀疏视角重建，并验证两轮生成修复与场景更新。实验覆盖 Waymo-405841、Dl3dv-2 和 Re10k-1，另在 Mip-NeRF 360 Bonsai 上加测。

- [最终 CVPR 报告](report/Project4_Report.pdf) · [报告源码与编译方法](report/README.md)
- Part 1 · [稠密初始化指标](results/part1-final/coverage.csv) · [收敛曲线](results/part1-final/convergence.png)
- Part 2 · [Full/Sparse 对照](results/part2_formal_7000/comparison.csv) · [留出视角和覆盖率](results/part2_supplement_v1/figures/failure_cases.png)
- Part 3 · [两轮闭环指标](results/part3/comparison.csv) · [同视角渲染](results/part3/figures/heldout_comparison.png) · [运行与复现实验](docs/part3.md)
- 补充实验 · [稀疏重建与回灌对照](results/part3_mse30/) · [长预算与联合位姿对照](results/postpre_optimization/) · [复现说明](docs/postpre_optimization.md)

## 环境与权重

在装有 NVIDIA 驱动、CUDA Toolkit 12.4（含 `nvcc`）和 C++ 编译器的 Linux 环境运行。使用 Python 3.10 或 3.11，并安装 [COLMAP 3.11.1](https://github.com/colmap/colmap/releases/tag/3.11.1)。以下命令在仓库根目录执行。

```bash
git clone --branch submission --single-branch https://github.com/MoliaiELS/CV_Project4.git
cd CV_Project4
conda create -n project4 python=3.10 -y
conda activate project4
bash scripts/setup_server.sh
python scripts/download_weights.py
```

安装脚本固定 PyTorch 2.6.0 与 torchvision 0.21.0（CUDA 12.4）以及 gsplat 1.5.3，并检出 VGGT 与 VGGT-Ω 的固定源码版本。预训练权重由官方发布页提供。若需要授权，先在发布页开通访问，再在本地环境中设置 `HF_TOKEN`。首次计算 LPIPS 时需要下载或预先缓存 AlexNet 权重。

| 模型 | 官方发布 | 仓库相对权重路径 |
| --- | --- | --- |
| VGGT | [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B) | `weights/vggt/model.pt` |
| VGGT-Ω | [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) | `weights/vggt-omega/vggt_omega_1b_416_reproduce.pt` |
| Difix-ref | [nvidia/difix_ref](https://huggingface.co/nvidia/difix_ref/tree/f88399d769a3648f04c5b74f41661bfa25361607) | `weights/difix_ref/`，按 Part 3 步骤下载并校验 |

核对正式运行使用的 VGGT 权重文件。

```bash
sha256sum -c <<'SHA256'
d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0  weights/vggt/model.pt
e900833330eb0e570e10543e6108d0dd704fa7c8dbc32738a645987875298896  weights/vggt-omega/vggt_omega_1b_416_reproduce.pt
SHA256
```

## 数据与 Part 1

从课程渠道获取课程数据包，将包含 `405841/`、`DL3DV-2/` 和 `Re10k-1/` 的目录作为 `--raw`。下例假定解压位置为 `data/raw/project4_3d`，使用仓库中的[固定划分](configs/splits/README.md)。

```bash
python -m scripts.prepare_course_data --raw data/raw/project4_3d --output data/prepared
python -m p4.experiment run --config configs/part1_reproduce.json
python -m p4.experiment summarize --config configs/part1_reproduce.json --output runs/part1_summary
python -m p4.visualize --config configs/part1_reproduce.json --output runs/part1_summary
```

Part 1 配置运行 COLMAP、VGGT 和 VGGT-Ω 三种初始化与同预算的 7,000 步 3DGS。可给运行命令加 `--dry-run` 检查参数，或加 `--scenes Re10k-1 --methods vggt` 先运行一个条件。

## Part 2 · 从训练 RGB 估计位姿

下例独立训练 Re10k-1 的 Full 和 Sparse 两个条件。Dl3dv-2 也用步长 30，Waymo-405841 用步长 10。三个场景使用偏移 1，在原序列上采样，并沿用各自的留出集。

```bash
scene=Re10k-1
stride=30
dense="data/prepared/$scene"
sparse="data/part2_sparse/$scene"
run="runs/part2_fresh/$scene"
python -m p4.sparse --dataset "$dense" --output "$sparse" --stride "$stride" --offset 1

for condition in full sparse; do
  dataset="$dense"
  if [ "$condition" = sparse ]; then dataset="$sparse"; fi
  case_dir="$run/$condition"
  python -m p4.initialize --method vggt --neural-backend offload \
    --repo third_party/vggt --weights weights/vggt/model.pt \
    --dataset "$dataset" --output "$case_dir/init"
  python -m p4.train train --dataset "$dataset" --scene "$case_dir/init/scene.npz" \
    --config configs/part1.json --output "$case_dir/train"
  python -m p4.train render --dataset "$dataset" --scene "$case_dir/init/scene.npz" \
    --checkpoint "$case_dir/train/checkpoints/step_007000.pt" --output "$case_dir/renders"
  python -m p4.evaluate renders --dataset "$dataset" --render-dir "$case_dir/renders" \
    --output "$case_dir/metrics.json"
done
```

初始化和训练使用相应条件的训练 RGB。留出评估采用同一批参考相机，经训练相机拟合的 Sim(3) 对齐，计算原始渲染的 PSNR、SSIM 和 LPIPS。正式 Full/Sparse 的配对 ATE 在共同稀疏训练相机上计算。

## Part 3 · 从新的稀疏检查点运行两轮闭环

在同一个终端沿用上节的 `$sparse` 和 `$run`。先安装 Difix-ref 的独立依赖并下载固定版本权重。下面的 60 步 × 两轮适合验证端到端流程。它从上节完成增密的 7,000 步稀疏模型出发，比较纯真实图继续训练和加入生成图的门控融合。

```bash
python -m pip install --no-deps --target .venv-difix -r requirements-part3.txt
export PYTHONPATH="$PWD/.venv-difix${PYTHONPATH:+:$PYTHONPATH}"
model_dir=weights/difix_ref
python -m scripts.part3_restore --prepare-assets --model-dir "$model_dir"

python - "$sparse" "$run/sparse" <<'PY'
from pathlib import Path
import sys
from p4.common import sha256_file, write_json

dataset, sparse_run = (Path(arg).resolve() for arg in sys.argv[1:])
checkpoint = sparse_run / "train/checkpoints/step_007000.pt"
write_json(sparse_run / "m0.json", {
    "dataset": str(dataset),
    "initializer": str(sparse_run / "init/scene.npz"),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": sha256_file(checkpoint),
})
PY

python -m scripts.part3_closed_loop --supplemental "$run/sparse/m0.json" \
  --model-dir "$model_dir" --arms real_only gated --views 2 --steps 60 \
  --keep-checkpoints gated --output "runs/part3_fresh/$scene"
```

输出目录包含 `comparison.csv`、初始模型及两轮更新的原始留出渲染、指标和生成记录。正式原始闭环的每轮预算是 3,000 步、四个目标视角，可在新输出目录使用 `--views 4 --steps 3000 --arms real_only raw_replay ungated gated`。Bonsai、增强初始化和长预算实验的配置见 [Part 3 实验说明](docs/part3.md)与[补充复现指南](docs/postpre_optimization.md)。
