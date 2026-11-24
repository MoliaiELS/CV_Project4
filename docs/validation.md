# 验证记录 · 2026-11-14

当前状态：**Part 1 实现与 CPU 验证完成，真实场景 GPU 实验尚未运行，正式完成数 0/9。**

## 已执行并通过

本机 Windows 11、Python 3.14.6，使用独立 `.venv-cpu`。它验证 CPU 数据/几何/编排部分，不是正式服务器依赖组合。

```text
.venv-cpu/Scripts/python.exe -m pytest -q
43 passed（补充实际 Waymo / Re10k 适配测试后）
```

覆盖：

- COLMAP txt/bin 相机读取、NeRF OpenGL → OpenCV、课程 rgb/cameras.json 的 xyzw c2w 和归一化 K。
- 非平凡旋转、非零平移、非方形图片 resize 后的投影与内参；场景归一化不改变像素投影。
- 明确 train/test、拒绝交叉/缺失划分、训练输入指纹、拒绝将不完整课程数据当 full。
- VGGT/Ω 裁剪/填充/resize 到 canonical K 的恢复；置信度与无效深度过滤、确定性点采样和颜色对应。
- 训练相机拟合 Sim3、测试不进入拟合、退化轨迹报错、已知 PSNR/SSIM、严格测试渲染清单。
- 精确九项矩阵、CLI 参数解析、pending/failed 不伪造数值、不完整 LPIPS 重试、输入/权重变化校验。
- 产物删除/篡改时拒绝缓存；上游重跑使下游失效；跨 Windows/Linux 的解释器路径和分隔符不会误拒相同 receipt。

另外通过 Python `compileall`、Git Bash `bash -n scripts/setup_server.sh`、所有主 CLI help / dry-run、无数据时汇总和可视化的 pending 路径。`gpu_smoke.py` 的合成相机/图像 fixture 已在 CPU 上验证可加载、test 相机数为2、训练对齐 ATE < 1e-10；**未执行其 CUDA 段**。

同视角 panel 和轨迹图曾使用临时合成文件进行生成与目视布局核验，篡改 render 后也能拒用；没有把临时图保存为课程成果。

## 实际数据与文档检查

已核对课程要求、数据说明以及官方模型和训练后端的源码。固定版本和权重来源见 [项目 README](../README.md)。

早期网络下载只到17,714,945字节后中止；之后用户提供了完整 `project4_3d.zip`。全包 CRC 通过，已解包、读取并标准化三个场景的全部784张正式RGB，核对完整相机、图像hash、SO(3)和训练轨迹非退化条件。Waymo实际去畸变后变为960×640，另两个场景保留256×256。包无显式官方split，已公开本组每8帧测试1帧的列表。实际 train/test：174/25、267/39、244/35。详见 `docs/local_data_audit.md` 和 `results/data_audit/preparation.json`。

新增实际格式测试覆盖 Waymo per-frame calib / gt、非零畸变在 resize 前移除、gt_old 与车辆外参不被重复应用、缺失图像/标定拒绝、Re10k images/cameras 组合。实际 prepared 全部图像已重算SHA验证，所有 reference 相机与图像一一对应。源图像预览已目视检查；没有将 source RGB 预览当成重建成果。

## 仍需 NVIDIA 服务器验收

1. `scripts/setup_server.sh` 的安装、两套模型在固定 torch2.6.0 环境的实际前向、COLMAP3.11.1 参数与注册情况。
2. 正式 checkpoint 的 save/resume 数值一致性、增密/opacity reset、test render 和 LPIPS 权重可用性。入口 `python scripts/gpu_smoke.py`。
3. 使用已经准备的固定数据先跑真实 pilot，测量初始化/训练显存与耗时，再冻结统一预算。继续在报告中明确 reference 来源和单位，不能仅凭文件名声称绝对尺度。
4. 三初始化 × 三场景的九项正式实验、测试指标、曲线、同视角图片与结论。

本机没有 torch/COLMAP/CUDA/模型权重；未宣称 CUDA 训练已验证。单元测试中的昂贵子进程采用假产物，只用于检查编排、缓存与状态逻辑，不是神经模型或渲染器成绩。

上游 gsplat1.5.3 的 opacity-reset 运算符问题已在统一 trainer 明确修正并写入 provenance；最终论文需描述这一实现选择。
