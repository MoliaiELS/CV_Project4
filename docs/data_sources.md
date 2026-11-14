# 数据下载入口与待验收项

**当前状态更新：用户已在项目目录提供完整 `project4_3d.zip`，CRC 通过，三场景已解包和标准化。** 当前输入以该包 hash 为准，详见 [完整本地数据审计](local_data_audit.md) 和 `results/data_audit/preparation.json`。下文保留早先下载入口及受限尝试的历史记录；不再需要依赖未完成的网络下载。

以下入口直接来自课程要求 PDF 第 5 页的链接注释。早期检查时已读取公开 Google Drive 页面并核对归档名称和大小；随后用户提供完整包，最终格式与划分结论以 [本地数据审计](local_data_audit.md) 为准。

## 手动下载

推荐下载课程打包数据，暂时不需要自行下载完整 Waymo 或 DL3DV 数据集。

- [Google Drive 课程文件夹](https://drive.google.com/drive/folders/1euG7pnbFowljVWoNLcbCmil81IVsIEfM?usp=drive_link)：选择 **Project4.zip**，不要选同目录的 Project3.zip。
- [Project4.zip 文件页面](https://drive.google.com/file/d/1yZxTOkIo-FkSsL7TpOC1Zpe0N8Cau_rE/view)：已观察到文件 ID 为 `1yZxTOkIo-FkSsL7TpOC1Zpe0N8Cau_rE`，MIME 为 `application/x-zip-compressed`，大小 **907,755,770 字节**（页面显示 865.7 MB，即约 865.7 MiB）。这不是预估值。
- [百度网盘备用入口](https://pan.baidu.com/s/1Sa18zCeYiYA2gWAllo11dg?pwd=p3bm)：提取码 **p3bm**。该链接来自 PDF；本次未能通过文本浏览工具进一步核对百度目录内容。

下载后可以先放在 `data/downloads/Project4.zip`，再解压到 `data/raw/`；如果原压缩包自带 Project4 顶层文件夹，保留它，先检查文件树，不要为了适配代码立即重命名或删除元数据。

## 服务器下载示例

Linux 上可用 `gdown` 对已观察到的文件 ID 下载；本机曾遇到网络错误，服务器可重试。若下载失败，可使用网页下载，或直接同步已准备的数据。

```bash
python -m pip install gdown
mkdir -p data/downloads data/raw
python -m gdown 1yZxTOkIo-FkSsL7TpOC1Zpe0N8Cau_rE -O data/downloads/Project4.zip
python -m zipfile -l data/downloads/Project4.zip
python -m zipfile -e data/downloads/Project4.zip data/raw
```

下载完成后应核对文件名、字节数及 ZIP 完整性。文件大小只能发现部分传输错误；此处没有声称已获取官方 checksum。解压前先看 `zipfile -l` 输出，避免覆盖已有数据。

## 数据范围与拿到文件后的核查

课程指定三个 mandatory 场景：`Waymo-405841`（室外）、`Dl3dv-2`（室外）、`Re10k-1`（室内）。本次 Part 1 对三个场景分别运行 COLMAP、VGGT、VGGT-ω，共 9 项；dense 表示全部允许训练帧，不包括保留测试图像。Part 2 后续的 1/10 或 1/30 稀疏率不应套用到 Part 1。

解包后依次确认：

1. 三个场景的实际目录、图像文件名、原始 frame ID 和分辨率。
2. 是否提供官方 train/test 列表与采样编号规则；若没有，建立固定并公开的划分，再统一用于所有初始化。
3. 是否提供相机内外参、COLMAP 模型、`transforms*.json` 或其他相机记录，及其 c2w/w2c、坐标轴和单位说明。
4. 相机记录是真实 GT、经 SfM 得到的 reference，还是其他来源。不能把 reference 默认标成 GT。
5. 若数据含畸变相机，需要统一图像去畸变和内参，不可把畸变参数丢弃后直接训练。
6. 测试 RGB 不进入初始化或训练；只使用训练相机对应关系确定全局对齐，再用于测试渲染相机。

上述数据检查已完成：三个场景784张正式RGB已标准化，包内无显式官方split，本组固定列表已公开。真实重建实验仍pending，ATE单位须随reference来源说明。合成测试不代替真实场景实验。

## 早期 ZIP 前缀检查记录

在尚未获得完整上传包时，先从不完整网络下载的 local file headers 核实了 **`DL3DV-2/`**，其相机表有306条、首图256×256。该阶段不能判断完整split或另外两场景。后续完整包检查已补齐这些信息，详见完整审计。

相机记录含 `image_name`、`cam_quat`、`cam_trans` 和归一化的 `fx/fy/cx/cy`。已通过 [S3PO-GS 固定版本 dl3dvParser](https://github.com/3DAgentWorld/S3PO-GS/blob/d3a5c0926e5c5bceac362e5d0646e60575cbcd0a/utils/dataset.py) 核实：四元数为 **xyzw**，旋转和平移构成 **OpenCV c2w**；归一化水平内参乘图像宽度，垂直内参乘高度。本项目 adapter 保留源坐标，reference 单独存放，不输入初始化或训练。相关约定与来源记录在 manifest 中。

`p4.data` 已支持全部三个实际场景格式，并校验相机与RGB一一对应；完整包标准化已执行。数据相机只用于评价，不替代三种方法各自的估计结果。

## 访问与检查记录

Google Drive connector 可按文件 ID 返回 Project4.zip 名称、类型与精确大小，但文件夹 children 搜索返回空；改为读取公开文件夹 HTML 后确认归档 ID，随后再次用 connector 元数据核对该归档。

实际下载尝试遇到以下环境限制，不能把失败尝试当作数据已到位：Drive 原始 fetch 返回 HTTP 413，指出文件 907755770 字节超过该工具的 268435456 字节上限；curl 访问 `drive.usercontent.google.com` 连接超时；gdown 访问 `drive.google.com` 返回 SSL `UNEXPECTED_EOF_WHILE_READING`。手动浏览器下载或服务器网络可能正常，可使用上面命令重试。这些错误不能证明分享链接失效。

早期 PowerShell 下载短时速度约0.14MB/s，故中止并标记为partial；该未完成文件已在目录整理时清理。随后改用用户上传的完整包，旧下载不参与实验。这一网络观测不代表服务器下载速度。

早期搜索课程文件时尚无场景数据；后续用户上传的完整包已成为当前正式输入。

相关要求来源：

- 课程 PDF：`Project 4_new__Generative_Sparse_View_3D_Reconstruction.pdf`，第3、5页。
- 执行方案：`Project4_Plan_and_Checkpoint.pdf`，第2、4页。
