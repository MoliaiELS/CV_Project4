# Part 3 闭环生成重建

从 Part 2 的稀疏 3DGS 起点开始，每轮在训练相机之间选择目标视角，用 Difix-ref 修复原始渲染，再将伪观测加入 3DGS 训练。第二轮根据各分支第一轮的 3DGS 重新生成。原闭环的三场景和 Bonsai 结果，以及两组改进起点的结果，分别保存在[原闭环](../results/part3/README.md)和[稀疏重建改进](../results/part3_mse30/README.md)。

## 环境和输入

在仓库根目录运行。按[项目 README](../README.md)准备 Linux CUDA 环境、课程数据和 VGGT。完整评分使用 LPIPS AlexNet，离线机器需要在 `TORCH_HOME` 下准备 AlexNet 缓存。

```bash
assets=${PART2_ASSETS:-$PWD}
runs=${PART3_RUNS:-$PWD/runs}
model=${P4_DIFIX_MODEL_DIR:-$HOME/weights/difix_ref}
difix_deps=${P4_DIFIX_DEPS:-$HOME/.cache/cvproject4-difix-deps}
vggt_repo=${VGGT_REPO:-$assets/third_party/vggt}
vggt_weights=${VGGT_WEIGHTS:-$assets/weights/vggt/model.pt}

python -m pip install --no-deps --target "$difix_deps" -r requirements-part3.txt
export PYTHONPATH="$difix_deps${PYTHONPATH:+:$PYTHONPATH}"
python -m scripts.part3_restore --prepare-assets --model-dir "$model"
python -m scripts.part3_restore --verify-assets --model-dir "$model"
```

权重和定制流水线由入口逐文件核验 SHA-256，约需 5.2 GB 存储。`--prepare-assets` 可通过 `--endpoint` 使用可访问的 Hugging Face 镜像。 [Difix3D+ 论文](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Difix3D_Improving_3D_Reconstructions_with_Single-Step_Diffusion_Models_CVPR_2025_paper.html) · [固定源码](https://github.com/nv-tlabs/Difix3D/tree/c76edc595586e16732c91ddee82f3a6d83a8a9cc) · [固定权重和许可](https://huggingface.co/nvidia/difix_ref/tree/f88399d769a3648f04c5b74f41661bfa25361607)

`--assets-root` 重放需要归档的 `data/part2_sparse/{scene}/`，包含 `manifest.json`、`reference.npz` 和图像，以及 `runs/part2_formal/{scene}/full_sparse_7000_v1/sparse/` 下的 `init/scene.npz` 和 `train/checkpoints/step_007000.pt`。加载器核验这些文件的原始哈希。新工作区重新生成的数据清单可能因来源路径而产生不同的字节哈希。使用新生成的数据时，按 README 的 Part 2 流程独立训练稀疏 M0，再通过 `--supplemental` 提供自己的数据、初始化和检查点收据。

以下函数为已训练的模型生成收据，假定其初始化和训练输出分别位于同一目录的 `init/` 和 `train/`。

```bash
write_m0_receipt() {
  python - "$1" "$2" "${3:-7000}" <<'PY'
import sys
from pathlib import Path
from p4.common import sha256_file, write_json

dataset, case_root = (Path(item).resolve() for item in sys.argv[1:3])
step = int(sys.argv[3])
initializer = case_root / "init/scene.npz"
checkpoint = case_root / "train/checkpoints" / f"step_{step:06d}.pt"
write_json(case_root / "m0.json", {
    "dataset": str(dataset),
    "initializer": str(initializer),
    "initializer_sha256": sha256_file(initializer),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": sha256_file(checkpoint),
})
PY
}
```

## 原闭环与加测场景

先验收一个场景，再在新目录运行课内矩阵。正式实验每轮训练 3,000 步、使用四个新视角、种子 42 和伪图权重 0.2。`--scene` 可重复指定，`--arms` 可用于补跑分支。

```bash
python -m scripts.part3_closed_loop --assets-root "$assets" --model-dir "$model" \
  --scene Re10k-1 --steps 60 --views 2 --output "$runs/part3_pilot" --keep-checkpoints none

formal="$runs/part3_formal"
python -m scripts.part3_closed_loop --assets-root "$assets" --model-dir "$model" \
  --output "$formal" --keep-checkpoints gated
python -m scripts.part3_closed_loop --assets-root "$assets" --model-dir "$model" \
  --scene Re10k-1 --arms fixed_gated --output "$formal" --keep-checkpoints gated
```

| 分支 | 使用的额外观测 |
| --- | --- |
| `real_only` | 同样的真实训练帧序列与同样的更新预算 |
| `raw_replay` | 当前模型在选定视角的原始渲染 |
| `ungated` | Difix-ref 修复图，像素统一权重 |
| `gated` | Difix-ref 修复图，由训练图的几何和颜色一致性加权 |
| `fixed_gated` | 固定选择训练相机中点，沿用同样的修复与加权 |

Bonsai 使用来自 Mip-NeRF 360 的 12 张训练图和 37 张留出图。脚本对 292 张源图自然排序，每第 8 帧留出测试，从其余帧等距选取训练图，再读取所需的 49 张图和发布相机。下游长边限制为 512 像素，发布相机用于评价。 [Mip-NeRF 360 项目](https://jonbarron.info/mipnerf360/) · [固定 Bonsai 输入](../results/part3/inputs/Bonsai/source_receipt.json)

```bash
bonsai_data="$PWD/data/prepared/Bonsai"
bonsai="$runs/part3_bonsai"
python -m scripts.prepare_part3_bonsai --output "$bonsai_data"
python -m p4.initialize --method vggt --dataset "$bonsai_data" --repo "$vggt_repo" \
  --weights "$vggt_weights" --output "$bonsai/init"
python -m p4.train train --dataset "$bonsai_data" --scene "$bonsai/init/scene.npz" \
  --config results/part3_mse30/train_config.json --output "$bonsai/train"
write_m0_receipt "$bonsai_data" "$bonsai"
python -m scripts.part3_closed_loop --supplemental "$bonsai/m0.json" --model-dir "$model" \
  --output "$formal" --keep-checkpoints gated
python -m scripts.summarize_part3 --run-root "$formal" --assets-root "$assets" \
  --supplemental "$bonsai/m0.json" --output "$runs/part3_summary"
```

只汇总课内场景时，去掉汇总命令中的 `--supplemental`。加 `--videos-output "$runs/part3_videos"` 可导出逐场景和合并 MP4，需要 FFmpeg 或 `imageio-ffmpeg`。中断后用相同参数续跑，控制器核验训练协议、父检查点和伪观测。`--keep-checkpoints all` 保留两轮检查点，`gated` 保留门控分支的最终检查点。

## 历史渐进闭环基线

渐进规划第一轮在相邻训练相机之间的 20% 和 80% 处生成候选，第二轮进入 1/3、1/2 和 2/3。每个区间最多选一个有重投影支持的低覆盖目标。DL3DV 用相邻训练图的深度正向投影真实 RGB，经 z-buffer 和双侧颜色深度一致性检查筛选。`warp_only` 使用这些投影像素，`warp_gated` 另外用两次一致的 Difix 修复补洞。Re10k 的 `progressive_gated` 直接用双参考修复和重投影门控。每轮使用八个目标视角和 3,000 步更新。 [历史基线](../results/part3_progressive/README.md) · [FWD](https://arxiv.org/abs/2206.08355) · [BRPO](https://arxiv.org/html/2602.21535v1)

```bash
python -m scripts.part3_progressive --assets-root "$assets" --model-dir "$model" \
  --scene Dl3dv-2 --arms warp_only warp_gated --views 8 --steps 3000 \
  --synthetic-weight 0.4 --output "$runs/part3_warp_3000_weight04" --keep-checkpoints gated
python -m scripts.part3_progressive --assets-root "$assets" --model-dir "$model" \
  --scene Re10k-1 --arms progressive_gated --views 8 --steps 3000 \
  --synthetic-weight 1.0 --output "$runs/part3_progressive_re10k_weight1" --keep-checkpoints gated
```

## 从跨视角一致的种子点重建 DL3DV

用原有的 11 张训练图和 VGGT 相机三角化经验证的图像匹配，再从 VGGT 深度选取通过相邻视角深度与颜色检查的点。新种子限制为 100,000 个。下面的深度解码从原稀疏 VGGT 初始化同目录的 `scene.json` 和 `offload_cache/layer_{04,11,17,23}.npy` 读取模型特征。若要重新生成缓存，按 Part 2 的 VGGT `--neural-backend offload` 流程初始化。

```bash
dataset="$assets/data/part2_sparse/Dl3dv-2"
sparse=${DL3DV_VGGT_CASE:-$assets/runs/part2_formal/Dl3dv-2/full_sparse_7000_v1/sparse}
run="$runs/part3_mse30/Dl3dv-2"
python -m scripts.extract_vggt_depth_cache --dataset "$dataset" --initializer "$sparse/init/scene.npz" \
  --repo "$vggt_repo" --weights "$vggt_weights" --output "$run/frozen_depth"
python -m scripts.init_pycolmap_sparse --dataset "$dataset" --output "$run/pycolmap/init" --matches-only
```

匹配需要 PyCOLMAP 3.11.1。多线程匹配的几何验证可能有小幅变化。精确重放已验证的种子时，直接将 `DL3DV_MATCHES_DB`、`DL3DV_DEPTH_MAPS` 指向归档的 `database.db`、`depth_maps.npz`，它们的哈希在[种子收据](../results/part3_mse30/seed_receipt.json)中。新训练和闭环请使用空输出目录。

```bash
case_root="$run/vggt_matched_seeds"
python -m scripts.seed_sparse_colmap_depth --dataset "$dataset" --camera-initializer "$sparse/init/scene.npz" \
  --matches-database "${DL3DV_MATCHES_DB:-$run/pycolmap/init/database.db}" \
  --depth-maps "${DL3DV_DEPTH_MAPS:-$run/frozen_depth/depth_maps.npz}" --output "$case_root/init"
python -m p4.train train --dataset "$dataset" --scene "$case_root/init/scene.npz" \
  --config results/part3_mse30/train_config.json --output "$case_root/train"
write_m0_receipt "$dataset" "$case_root"
python -m scripts.part3_progressive --supplemental "$case_root/m0.json" --model-dir "$model" \
  --arms warp_gated real_only warp_only --views 8 --steps 3000 --synthetic-weight 0.4 \
  --keep-checkpoints gated --output "$case_root/closed_loop"
python -m scripts.compare_sparse_quality --dataset "$dataset" \
  --baseline "${DL3DV_BASELINE_METRICS:-$runs/part3_warp_3000_weight04/Dl3dv-2/warp_gated/round2/metrics.json}" \
  --candidate "$case_root/closed_loop/Dl3dv-2/warp_gated/round2/metrics.json" \
  --output "$case_root/improvement.json"
```

使用归档基线时，令 `DL3DV_BASELINE_METRICS` 指向包含逐帧数据的原始 `metrics.json`。比较器核对留出帧和顺序，分别计算逐帧 MSE 几何均值和按像素加权的汇总 MSE。

## Re10k 的 DA3-Large 初始化

从固定划分的 10 张训练图联合估计相机和深度。固定 DA3 的射线位姿、504 像素处理分辨率和 100,000 个初始点，再训练 7,000 步。准备官方源码和 DA3 环境，并将发布的 `config.json` 和 `model.safetensors` 放在同一个权重目录。 [固定 DA3 源码](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4) · [固定 DA3-Large-1.1 权重](https://huggingface.co/depth-anything/DA3-LARGE-1.1/tree/0e109ae307c5982f319a67cf6f9f99ccdc0ec97c)

```bash
dataset="$assets/data/part2_sparse/Re10k-1"
da3_repo=${DA3_REPO:-$PWD/third_party/Depth-Anything-3}
da3_large_weights=${DA3_LARGE_WEIGHTS:-$HOME/weights/DA3-LARGE-1.1}
case_root="$runs/part3_mse30/Re10k-1/da3_large"
python -m scripts.init_da3_sparse --dataset "$dataset" --repo "$da3_repo" \
  --model-dir "$da3_large_weights" --model-release da3-large-1.1 --output "$case_root/init"
python -m p4.train train --dataset "$dataset" --scene "$case_root/init/scene.npz" \
  --config results/part3_mse30/train_config.json --output "$case_root/train"
write_m0_receipt "$dataset" "$case_root"
python -m scripts.part3_progressive --supplemental "$case_root/m0.json" --model-dir "$model" \
  --arms warp_gated real_only --views 8 --steps 3000 --synthetic-weight 0.4 \
  --keep-checkpoints gated --output "$case_root/closed_loop"
python -m scripts.compare_sparse_quality --dataset "$dataset" \
  --baseline "${RE10K_BASELINE_METRICS:-$runs/part3_progressive_re10k_weight1/Re10k-1/progressive_gated/round2/metrics.json}" \
  --candidate "$case_root/closed_loop/Re10k-1/warp_gated/round2/metrics.json" \
  --output "$case_root/improvement.json"
```

[两场景优化结果](../results/part3_mse30/README.md) · [累计 40k 的配对闭环](postpre_optimization.md#part-3-同伪图的融合对照)

## 评价协议

所有分支使用相同的留出帧、原始 3DGS 渲染和 PSNR、SSIM、LPIPS 实现。以训练相机拟合的全局 Sim(3) 映射参考位姿。目标规划、修复参考和训练均来自稀疏训练输入。Difix-ref 的预训练包含 DL3DV-10K，课程 DL3DV 场景与公开预训练样本是否完全独立尚未核实。
