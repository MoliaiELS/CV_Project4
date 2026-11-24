# Part 3 闭环生成重建

以 Part 2 的稀疏 VGGT + 3DGS 为初始模型。每轮从当前模型渲染训练相机之间的新视角，以一个真实稀疏训练图像为参考调用 Difix-ref，再用生成图像和真实图像更新同一个 3DGS。第二轮从各分支自己的第一轮 checkpoint 重新选视角、渲染和生成。

## 环境和权重

Linux 课程环境的 CUDA/PyTorch/gsplat 与 Part 2 相同。先按 [服务器说明](part2_handoff.md)准备 Part 2 资产和 AlexNet LPIPS 缓存。在自己的目录安装扩散依赖叠加层并下载权重。

```bash
python -m pip install --no-deps --target "$HOME/.cache/cvproject4-difix-deps" -r requirements-part3.txt
export PYTHONPATH="$HOME/.cache/cvproject4-difix-deps${PYTHONPATH:+:$PYTHONPATH}"
python -m scripts.part3_restore --prepare-assets --model-dir "$HOME/weights/difix_ref"
python -m scripts.part3_restore --verify-assets --model-dir "$HOME/weights/difix_ref"
```

权重、定制流水线和组件由入口逐文件核验 SHA-256，约需 5.2 GB 可用存储。课程服务器使用可访问的 HTTPS 镜像时，可为 `--prepare-assets` 增加 `--endpoint https://hf-mirror.com`。镜像文件使用同一组官方哈希校验。[Difix3D+ 论文](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Difix3D_Improving_3D_Reconstructions_with_Single-Step_Diffusion_Models_CVPR_2025_paper.html) · [固定源码](https://github.com/nv-tlabs/Difix3D/tree/c76edc595586e16732c91ddee82f3a6d83a8a9cc) · [固定权重和许可](https://huggingface.co/nvidia/difix_ref/tree/f88399d769a3648f04c5b74f41661bfa25361607)

## 运行

在仓库根目录，用课程 RTX A6000 环境的 Python 执行。写入自己有权限和存储空间的运行目录。先跑一项小规模验收，再用新的输出目录执行正式矩阵。

```bash
assets=/data/disk2/home/zguo400/CV_Project4
model="$HOME/weights/difix_ref"
python -m scripts.part3_closed_loop --assets-root "$assets" --model-dir "$model" \
  --scene Re10k-1 --steps 60 --views 2 --output runs/part3_pilot --keep-checkpoints none
python -m scripts.part3_closed_loop --assets-root "$assets" --model-dir "$model" \
  --output runs/part3_formal --keep-checkpoints gated
python -m scripts.summarize_part3 --run-root runs/part3_formal --assets-root "$assets" \
  --output output/part3_summary
```

套件默认按场景串行运行四个分支。可以用重复的 `--scene` 缩小场景范围，`--arms` 缩小分支范围，后续再用同一协议和输出目录补齐。中断时用相同参数继续，控制器校验代码、训练协议、父 checkpoint 和伪观测指纹。正式运行每轮 3,000 次更新、四个新视角，固定随机种子 42 和伪图像权重 0.2。使用 `--keep-checkpoints all` 可保留全部中间和最终训练状态，`gated` 保留门控分支的最终 checkpoint。阶段指标、生成图像、置信度、训练日志和哈希都写入运行目录。

| 分支 | 使用的额外观测 |
| --- | --- |
| `real_only` | 同样的真实训练帧序列与同样的更新预算 |
| `raw_replay` | 当前模型在选定视角的原始渲染 |
| `ungated` | Difix-ref 修复图，像素统一权重 |
| `gated` | Difix-ref 修复图，由训练视图的几何和颜色一致性加权 |
| `fixed_gated`（可选） | 等间隔选择训练相机中点，并使用相同的修复与一致性权重 |

## 加测场景

```bash
python -m scripts.prepare_part3_bonsai --output data/prepared/Bonsai
```

Bonsai 取自 Mip-NeRF 360。脚本从固定版本的输入数据镜像获取 49 张必要图像和发布的相机文件。292 个文件名自然排序后，每第 8 帧留出测试，共 37 帧。从其余帧等距抽取 12 帧作为稀疏训练输入，长边 512 像素。Bonsai 单独用这些训练 RGB 做 VGGT 初始化与 7,000 步 3DGS 训练，发布的 COLMAP 相机仅用于评价。将自建模型的 `dataset`、`initializer`、`checkpoint` 和 `checkpoint_sha256` 写入一个 JSON，再用 `--supplemental` 传给闭环及汇总脚本。[Mip-NeRF 360 项目](https://jonbarron.info/mipnerf360/)

## 评价边界

主动规划在训练相机之间插值平移、朝向和内参，用低分辨率投影 alpha 找到缺少支持的视角。门控使用当前 3DGS 的期望深度，将目标像素重投影到相邻真实训练视图，检查参考视图遮挡和颜色一致性。无可靠重投影的像素保留较小权重。留出相机和图像只在评价中读取，按 Part 2 的 train-only Sim(3) 协议渲染并计算 PSNR、SSIM、LPIPS。测试视角的报告预览用 3DGS 原始渲染。

Difix-ref 预训练资料包含 DL3DV-10K。课程 DL3DV 场景是否与公开预训练样本完全独立尚未核实。项目内的参考图、训练和视角规划均遵守固定划分。
