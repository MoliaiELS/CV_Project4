# Part 2 优化结果与 HPC 交接

本次交付包含三个场景的 Full / Sparse 7000 步正式结果、五个迭代阶段的补充评估、初始化优化入口和复现脚本。所有实验在课程 HPC 的 RTX A6000 上完成，未使用代理。原六组正式结果、checkpoint 和训练源码指纹保持不变。Part 3 尚未执行。

## 从哪里开始

- [补充分析与七张图](../results/part2_supplement_v1/report.md)
- [六组正式结果](../results/part2_formal_7000/comparison.csv)与[逐帧指标](../results/part2_formal_7000/per_frame_metrics.csv)
- [收敛指标](../results/part2_supplement_v1/convergence.csv)：500 / 1000 / 2000 / 4000 / 7000 步，共 30 组、990 次逐帧评估。
- [初始化测试](../results/part2_supplement_v1/fast_initialization_benchmark.json)与[Part 1 checkpoint 重放](../results/part2_supplement_v1/part1_checkpoint_replay.json)
- [HPC 资产清单](../results/part2_supplement_v1/hpc_artifacts.json)：逐文件大小、SHA-256、相对路径和环境版本。
- [Part 2 LaTeX 正文](../report/part2.tex)及[整合说明](../report/part2_integration.md)；正文片段尚未编译成最终论文。
- [同学 A 的两页展示提纲](../results/part2_supplement_v1/midterm_part2_notes.md)

Git 只包含源码、文档、JSON/CSV、PNG/SVG 和三个不足 4 KB 的诊断轨迹 NPZ。训练 checkpoint、VGGT/LPIPS 权重、课程图像、逐帧渲染、环境以及 ZIP 交付包均留在 HPC；没有使用 Git LFS 上传这些文件。

## 服务器位置

资产根目录为：

    /data/disk2/home/zguo400/CV_Project4

以下路径相对此根目录。清单中的每条 relative_path 都按这个根目录解析。

| 内容 | 路径 |
| --- | --- |
| Part 2 六组正式运行 | runs/part2_formal/{scene}/full_sparse_7000_v1/{full,sparse}/ |
| 7000 步 checkpoint | 上述目录下 train/checkpoints/step_007000.pt |
| 中间 checkpoint | 同目录 step_000500.pt、step_001000.pt、step_002000.pt、step_004000.pt |
| 每组原始初始化 | 上述目录下 init/scene.npz 和 init/scene.json |
| Full / Sparse 数据 | data/prepared/{scene}/、data/part2_sparse/{scene}/ |
| Part 1 九个导入 checkpoint | runs/part1_import_901c5ae/checkpoints/{scene}/{colmap,vggt,vggt-omega}/step_007000.pt |
| Part 1 原始 manifest、初始化与日志 | results/hpc_setup/part1_handoff_901c5ae/archive/ |
| 原始/补充渲染与缓存 | runs/part2_formal/、runs/part2_supplement_v1/ |
| VGGT 权重 | weights/vggt/model.pt |
| LPIPS 特征权重缓存 | weights/torch_cache/hub/checkpoints/ |
| 原正式结果包 | results/part2_formal_7000/summary_bundle.zip |
| 原补充结果包 | results/part2_supplement_v1/part2_enhanced_bundle.zip |

scene 为 Re10k-1、Waymo-405841、Dl3dv-2，注意大小写。共保留 Part 2 的 30 个 checkpoint 和 Part 1 的 9 个 checkpoint。不要修改共享输入；继续实验时使用自己的工作目录和新的输出路径。

环境为 /data/disk2/home/zguo400/envs/project4，Python 3.10.22、PyTorch 2.6.0+cu124、gsplat 1.5.3、NumPy 1.26.4；精确版本见资产清单。原始课程数据仍在 /data/disk1/shared/aiaa3201/project4_3d。

服务器目录按 students 组提供读取/目录遍历权限，同学 B 的 yguo704 账号属于该组。其他账号请先运行下面的 verify 检查实际访问权限。仓库不包含 SSH 密码或 GitHub 凭据。

## 为什么不能直接用当前 main 加载旧 checkpoint

checkpoint 校验训练包源码及原始 manifest。对应版本为：

| 用途 | Git commit | p4 包 SHA-256 |
| --- | --- | --- |
| Part 2 原正式训练与补充评估 | bd2e3ecae209df5de315fcddb258accf15a57592 | c8b6722a8b18deaecd6766279d8e69617e762ec9cb9add52300ba171424bf51c |
| Part 1 原正式训练 | bde9244f692cc497355a307826ce1ca25f39cd99 | fba9b9a1b9d91d72f1d4455e0b56f4b95b2d09c05aa81b1a16e06e77fdce66bff |
| CPU head-feature 初始化实现 | 901c5aedfe4627e7af93814e8f37e847237adb7b | 单文件快照与指纹记录在 adapter_provenance.json |

本次发布基于最新 main，保留 Part 1 的更新。下面的 prepare 会在新目录恢复匹配 Part 2 的旧 p4 包，并叠加本次脚本、报告和轻量结果；不会切换或改写共享实验工作区。不要关闭 checkpoint 指纹校验。Part 1 重放脚本另行恢复其历史 p4 包及原始 manifest。

## 同学在课程 HPC 上接手

用自己的 HPC 账号登录，在自己的目录中执行。仓库分支为 member-a/part2-supplement；PR 合并后可改用 main。

~~~bash
unset http_proxy https_proxy all_proxy ftp_proxy wss_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY WSS_PROXY
export NO_PROXY='*' no_proxy='*'
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1

git -c http.proxy= clone --branch member-a/part2-supplement \
  https://github.com/MoliaiELS/CV_Project4.git CV_Project4-part2-release
cd CV_Project4-part2-release
export PATH=/data/disk2/home/zguo400/envs/project4/bin:$PATH

python -m scripts.part2_handoff verify
# 完整 SHA-256 校验会读取约十几 GB 的资产；建议首次复制后运行一次：
python -m scripts.part2_handoff verify --sha256

python -m scripts.part2_handoff prepare \
  --destination "$HOME/CV_Project4-part2-repro"
cd "$HOME/CV_Project4-part2-repro"
source ./part2_direct.env
~~~

prepare 的目标必须不存在且位于发布仓库/共享资产目录之外。它只在本地克隆已有 Git 对象；大型输入使用链接，少量预览/初始化缓存复制到自己的目录。临时目录、Matplotlib 缓存、CUDA 扩展编译缓存和后续输出都归当前使用者，避免写入同学 A 的缓存目录。若迁移了资产根目录，可传入 --assets-root /new/location；需要同时更新环境路径与相应数据来源。

## 查看结果、重新生成图表

在上述新工作区内：

~~~bash
python -m scripts.part2_handoff verify
python -m scripts.build_part2_supplement
~~~

这一步使用随仓库交付的数值记录和 HPC 上的预览，不重新训练，也不重新运行 VGGT。输出位于 results/part2_supplement_v1/，包括新的 ZIP 汇总包。original_bundle_manifest.json 是此前 HPC 原始交付包的校验记录，不是当前 Git 文件列表；发布文件另有 release_manifest.json。

## 重新验证 checkpoint 和初始化

仍在上述新工作区内，按顺序运行，避免同时占用共享 GPU：

~~~bash
python -m scripts.part2_convergence
python -m scripts.part2_diagnostics
python -m scripts.part2_fast_initialize --benchmark
python -m scripts.part1_checkpoint_replay
python -m scripts.build_part2_supplement
~~~

- convergence 复用已有 7000 步指标，从原 checkpoint 重新评估中间点；diagnostics 重放最终 RGB/alpha 并核对原 RGB。
- --benchmark 默认复用 prepare 复制来的已完成初始化，重新核对相机/点云并读取原计时；它不会把缓存校验耗时当作新推理耗时。
- 如需真正重新测试初始化时间，请使用下面的新输出目录。不要删除/覆盖共享历史结果；新增条件另行记录。
- Part 1 replay 使用原始训练源码、manifest、输入图像和导入 checkpoint。三个 VGGT Full 的 PSNR 已完全复现归档值；SSIM/LPIPS 仅有浮点级差异。
- 部分历史 JSON 保留原始绝对路径作为 provenance；迁移服务器时需重新生成诊断记录，不能把旧路径当作新运行证据。

~~~bash
python -m scripts.part2_fast_initialize \
  --dataset data/prepared/Re10k-1 \
  --output runs/my_new_initialization/Re10k-1
~~~

新入口复用固定版本的 CPU head-feature 卸载，实现与原 CPU/disk 路线一致的相机和初始点云。它等待至少 30,000 MiB 空闲显存及相应 CPU 内存；资源不足时明确报错。需要更低显存时保留原 CPU/disk 路线，不要减少 Full 输入帧数来绕过内存需求。

如需从头训练新的对照，先查看 python -m scripts.part2_formal --help，使用新的 --output（例如 runs/my_new_formal/Re10k-1）；原 shared runs/part2_formal 下的条件保持只读。新训练必须沿用固定划分、seed、预算及评价协议，或者作为独立消融明确记录变化。不要直接在共享输入目录上执行 resume；历史 checkpoint 继续训练还需核对 optimizer 恢复兼容性。

## 可用于报告的结论及边界

原初始化约 926 / 661 / 836 秒，新路线约 228 / 141 / 230 秒（Re10k / Waymo / Dl3dv），观察到约 4.05 / 4.70 / 3.64 倍耗时比；新峰值显存为 19.70 / 15.77 / 20.99 GiB。这是单次新测试与历史记录的比较，未做重复负载控制。没有因此宣称最终渲染分数提升。

Full 在三个场景的平均 PSNR/SSIM 均高于 Sparse；Waymo Sparse 的 LPIPS 更低。较低的相机中心 ATE 不代表更好的朝向、内参或视图质量。Re10k 约 20% 的焦距偏差也存在于同学 B 的独立初始化中，未用参考内参替换模型输出。最新交接已核对 784 张 PNG，图像指纹和划分一致，参考相机数值一致。

7000 步终点预先固定，中间测试集峰值仅用于分析收敛，不用来挑选模型。Alpha 覆盖率是投影支持的代理量，不是真实三维完整度。仅一个随机种子、三个场景，不做总体显著性结论。
