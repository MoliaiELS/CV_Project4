# 汇报后实验复现

三场景沿用原课程协议的固定测试视角、分辨率、原始 RGB 评价与训练相机 Sim(3) 对齐。[实验记录](../results/postpre_optimization/README.md)

实验优先回答稀疏视角的相机中心误差与留出渲染质量为何不同。30,000 步和 5,000 步每轮是本次长预算对照的设定。先按[项目说明](../README.md)准备数据和环境。命令中的 `$assets`、`$output` 和 `$scratch` 分别由运行者设置为已有实验资产、输出和临时空间的目录。

## Part 1 稠密收敛对照

使用原 Part 1 交接中的三种初始化和完全相同的训练相机顺序。DL3DV 以 30,000 步为统一终点，从第一步开始用相同的学习率日程。第 7,000、15,000、30,000 步在训练结束后统一评分。原版 7,000 步测试结果属于另一条学习率日程。

```bash
python -m scripts.part1_long_budget --datasets-root "$PART1_DATASETS" --archive-root "$PART1_ARCHIVE" \
  --output "$output/part1_dl3dv_long" --scratch "$scratch/part1" --scenes Dl3dv-2 --steps 30000
```

将 `$PART1_DATASETS` 设为按原 Part 1 清单准备的图像目录，将 `$PART1_ARCHIVE` 设为用户提供的解包归档目录。归档中含有 `datasets/<scene>/` 的原始清单和参考相机，以及 `cases/<scene>/<method>/` 的初始化与训练收据。给同一条命令加 `--check-inputs` 可独立核验数据和初始化而无需 CUDA。默认把逐帧指标、训练时间和三个代表帧的渲染保存到正式输出，成功后释放当前训练的临时检查点。可在启动时加 `--keep-final` 以保留最终检查点。

## Part 2 位姿和高斯联合优化

全视角从 `data/prepared` 取图，稀疏视角从 `data/part2_sparse` 取图。基线诊断在相同的原始稀疏训练帧上计算 ATE、相对位姿、焦距误差和保留观测重投影。

```bash
python -m scripts.part2_pose_diagnostics --assets-root "$assets" --output "$output/part2_pose_baseline.json"
```

每个 fixed/joint 配对从同一个已估计相机和 1,000 步高斯检查点分叉。Joint 在第 1,001–6,000 次更新期间允许光度损失对训练相机外参反传。首台训练相机固定，其他相机使用旋转和平移的弱先验。两组固定内参、训练图采样顺序、增密日程和 7,000 步最终预算。最后 1,000 步冻结所有相机。

```bash
scene=Dl3dv-2
kind=sparse
dataset="$assets/data/part2_sparse/$scene"
initializer="$assets/runs/part2_formal/$scene/full_sparse_7000_v1/$kind/init/scene.npz"
case_dir="$scratch/part2/$scene/$kind"
python -m p4.train train --dataset "$dataset" --scene "$initializer" \
  --config configs/part1.json --output "$case_dir/warmup" --stop-after 1000
warmup="$case_dir/warmup/checkpoints/step_001000.pt"
python -m p4.train train --dataset "$dataset" --scene "$initializer" \
  --config configs/part1.json --output "$case_dir/fixed" --resume "$warmup"
python -m scripts.train_joint_poses train --dataset "$dataset" --initializer "$initializer" \
  --warmup "$warmup" --output "$case_dir/joint"
python -m scripts.train_joint_poses export --dataset "$dataset" --initializer "$initializer" \
  --checkpoint "$case_dir/joint/checkpoints/step_007000.pt" --output "$case_dir/refined"
```

课程运行把保存列表设为 1,000、4,000 和 7,000，便于中断恢复。全视角对照将 `kind` 改为 `full`，并将 `dataset` 改为 `"$assets/data/prepared/$scene"`。评价 fixed 时用原始初始化与固定分支的 7,000 步检查点。评价 joint 时用 `refined/scene.npz` 与 `refined/checkpoint.pt`。两者都用 `p4.train render` 和 `p4.evaluate renders`。留出相机来自同一参考记录，按各自训练相机拟合一个全局 Sim(3)，不做逐图优化。导出文件的高斯参数与归一化坐标继续绑定，可用于 Part 3 固定相机回灌。

为检验对齐敏感性，用固定组的训练相机计算同一个 Sim(3) 并复评联合模型。该操作不改检查点或重新拟合单张留出相机。

```bash
python -m scripts.evaluate_fixed_alignment --dataset "$dataset" --initializer "$initializer" \
  --refined "$case_dir/refined" --output "$output/part2_shared_alignment/$scene/$kind"
```

## Part 3 同伪图的融合对照

DL3DV 长预算起点从已发布的一致性种子重新训练 30,000 步，保持原 11 张真实图与 6,000 步增密截止。把 `TrainConfig` 的 `steps` 改为 30,000，`save_steps` 改为 7,000、15,000、30,000，再按[原 M0 收据方法](part3.md#从跨视角一致的种子点重建-dl3dv)记录第 30,000 步检查点及其 SHA-256。

```bash
python -m scripts.part3_paired_warp fit --m0 "$PART3_M0" --model-dir "$DIFIX_MODEL_DIR" \
  --output "$output/part3_dl3dv_long" --steps 5000 --views 8 --synthetic-weight 0.4
python -m scripts.part3_paired_warp evaluate --m0 "$PART3_M0" --output "$output/part3_dl3dv_long"
```

将 `$PART3_M0` 设为该起点的收据文件，将 `$DIFIX_MODEL_DIR` 设为已准备的 Difix-ref 权重目录。闭环三条分支各增加 10,000 步，最后达到 40,000 步。仅真实组、未加权组和置信度加权组的真实图采样状态相同。未加权组和加权组首轮共享同一组像素，次轮保留共同目标相机并从各自第一轮重新生成。训练完成后才对固定留出集执行原始渲染和评分。该对照固定规划以检验融合，历史闭环的逐分支主动规划结果另见原 Part 3 表。
