# 用户上传数据包的完整审计

本记录以 2026-11-14 用户实际提供、现归档为 `data/downloads/project4_3d.zip` 的数据包为准。采用 Python `zipfile` 读取完整 central directory、对全部文件执行 CRC 校验、读取所有正式 RGB 图像的尺寸头并解析相机文件；审计阶段没有解压全部数据，也没有运行初始化或训练。

## 归档身份与完整性

| 项目 | 实际结果 |
| --- | --- |
| 本地归档 | `project4_3d.zip` |
| 字节数 | 907,794,590 |
| SHA-256 | `2e2a2658d6735761b5dd3c090c5d35a4a2cca04088cde6ae0cb231f90214c415` |
| ZIP entries | 1,612（包含目录 entries） |
| 文件未压缩字节数总和 | 907,551,842 |
| 完整 CRC 检查 | 通过，`ZipFile.testzip()` 返回 `None` |
| 顶层目录 | `project4_3d/` |

课程 Google Drive 中先前观察到的 `Project4.zip` 是 907,755,770 字节，与本地包相差 38,820 字节，且顶层目录不同。不能因为来源相关就假定两个归档逐字节相同。本次实验来源应记录用户实际上传包的上述 hash；之前网络产生的 `Project4.zip.partial` 不再承担数据输入用途。

## 三个正式场景

| 课程场景名 | 包内实际目录 | 正式 RGB 数 | 原始尺寸（宽×高） | 图像编号 |
| --- | --- | ---: | --- | --- |
| Waymo-405841 | `project4_3d/405841/FRONT/rgb/` | 199 | 1920×1280，全部一致 | `000000.png`–`000198.png` |
| Dl3dv-2 | `project4_3d/DL3DV-2/rgb/` | 306 | 256×256，全部一致 | `frame_00001.png`–`frame_00306.png` |
| Re10k-1 | `project4_3d/Re10k-1/images/` | 279 | 256×256，全部一致 | `00000.png`–`00278.png` |

大小写需要按实际路径处理：`DL3DV-2` 不是 `Dl3dv-2`，Waymo 在包中以数字 `405841/FRONT` 表示，而 Re10k 的图片目录是 `images`，不是 `rgb`。项目可以使用统一的课程场景名，但必须在 manifest 中保留实际来源路径和原始 frame ID。

DL3DV 的 `rgb/.ipynb_checkpoints/` 还包含 **12 张备份 PNG**，它们不在正式相机表中，应排除；否则按递归 PNG 个数会误计为 318 张。另有 `.ipynb_checkpoints/intrinsics-checkpoint.json`，属于笔记本备份。Re10k 的 `images/` 中还包含 `video.mp4`，不应作为图片输入。正式 RGB 合计 **784 张**。

## 相机、标定与参考轨迹

### DL3DV-2

`cameras.json` 是 306 条记录的列表，`image_name` 无重复，与排除隐藏备份后的 306 张 RGB 完全一一对应。字段为：

```text
cam_id, cam_quat, cam_trans, cx, cy, fx, fy, image_name
```

`intrinsics.json` 提供相同的归一化内参：`fx=0.7970605911654538`、`fy=0.7972243051546298`、`cx=cy=0.5`。所有相机记录的四元数、平移均为有限值；四元数范数最大偏差约 `6.67e-8`；各帧内参相同。

相机四元数/平移的 c2w 与轴向语义不是单凭字段名就能证明的。本项目采用的 S3PO 数据格式约定及其固定源码依据见 [数据来源说明](data_sources.md)：xyzw 四元数、OpenCV c2w，归一化水平/垂直内参分别乘实际图像宽/高。此相机表作为评测 reference，不能在无位姿初始化时直接拿来充当估计结果。

### Re10k-1

`cameras.json` 有 279 条记录，`image_name` 无重复，与 279 张正式 RGB 完全一一对应。字段在 DL3DV 同类字段基础上增加 `timestamp`。`intrinsics.json` 中 `fx=0.8572737166601563`、`fy=0.857483078`、`cx=cy=0.5`；各帧内参相同。四元数、平移均为有限值，四元数范数最大偏差约 `5.80e-8`。

该场景具备相机 reference，不应因其使用 `images/` 目录而走“只有 RGB、无相机”的普通目录分支。仅从归档本身不能证明此轨迹具有米制绝对尺度，因此 ATE 单位需按实际 reference 来源声明。

### Waymo-405841

`405841/FRONT/` 下有五个并列目录，各有 199 个同编号文件：

```text
rgb/       RGB PNG
calib/     标定 TXT
gt/        4×4 位姿 TXT
gt_old/    4×4 位姿 TXT
depth/     深度 PNG
```

全部 199 份标定文本相同。首份标定包含 `Tr_velo_to_cam`（3×4）、`fx/fy/cx/cy` 和 `k1/k2/p1/p2/k3`；其中 `fx=fy=2066.697564417299`，`cx=950.5512774150723`，`cy=641.1870541472169`。径向与切向畸变含非零项（例如 `k1≈0.03626`、`k2≈-0.35323`），不能丢弃畸变后把原图当作已去畸变针孔图像。

`gt/` 的全部位姿形状为 4×4，值有限，旋转正交误差最大约 `3.33e-16`，旋转行列式约 1。`gt/000000.txt` 的平移为零。对全部 199 帧，`gt_old/` 与 `gt/` 的旋转逐元素相同，平移之差恰好是同一常量：

```text
old_translation - gt_translation =
[-2113.6153150005457, 9287.32107309636, -86.413]
```

因此两目录在数值上仅有固定世界原点平移的差异；它们不是两条独立测量的轨迹。该数值关系本身不证明相机朝向、c2w/w2c 或米制单位语义；这些语义仍应与读取该课程格式的原始数据代码核对并写入 adapter provenance。深度、`gt/`、`gt_old/` 不应输入三种 RGB 几何初始化。统一图像去畸变和评测相机校准需要明确记录处理方式。

## 与早先下载前缀中的 DL3DV 元数据比较

本次对完整上传包与早期网络下载前缀提取的两份临时 JSON 分别比较，结果如下（临时文件已在整理时清理，比较记录保留）：

| 文件 | 上传包内字节数 | 早期前缀提取字节数 | 字节相同 | JSON 对象相同 |
| --- | ---: | ---: | --- | --- |
| DL3DV `cameras.json` | 146,171 | 125,670 | 否 | **是，完整 306 条记录相等** |
| DL3DV `intrinsics.json` | 92 | 89 | 否 | **是** |

完整包的 DL3DV `cameras.json` SHA-256 为 `863287f0a96d5fa18d7587ebdca7b5a739af85c5330567cee5681f58e33cfa7f`；早期文件为 `bcddc1f5709a7d5c9592e87e7c75ad23eeeace44273c4cfbe460d7cc95d05284`。内参文件相应为 `9a71a50d80d49b889ffc5cd9a11b03ec1e45f640bbbbb4ad9e10e45412c7a1c8` 与 `9ac7c865c94dbcd0db977607e837c1f3f45107472789b21ceef4365d06fe39eb`。

因此先前从 DL3DV 小型 JSON 得出的数值结论可以复核沿用，但不能把前缀文件的 byte hash 当成上传包文件 hash，也不能由这两个 JSON 语义相同推出两个 ZIP 的其余文件相同。

## Split 核查与实验状态

读取了完整 central directory 和所有非图像元数据文件类型后，归档中未发现 train/test 子目录、split 列表、`transforms_train/test`、README 或含划分说明的文件；两份相机 JSON 只列出相机记录，没有 split 字段。这次“包内没有显式官方 split”的结论基于完整上传包，而非早先不足 18 MB 的下载前缀。它不代表课程外部页面或论文代码没有数据划分协议。

下一步应将选择的固定划分写入公开 manifest，明确标注若采用项目自定义 fallback，不能称为包内官方 split。三种初始化共享完全相同的训练/测试清单；测试 RGB 不进入初始化、几何点云融合或 3DGS 训练。

数据已经到位并通过归档完整性检查。之后主流程已安全解包并实际完成三场景标准化，所有784张RGB的输入 hash、相机一一对应、旋转和训练轨迹非退化检查均通过。Waymo 完成去畸变和960×640缩放，其他场景保留256×256。固定每8帧测试1帧后，train/test 分别为174/25、267/39、244/35；公开列表见 `configs/splits/`，完整检查结果见 `results/data_audit/preparation.json`。

初始化、训练及九项 GPU 实验仍待服务器执行。源图像抽样预览见 `results/data_audit/source_contact_sheet.png`，它展示输入数据，不是新视角重建结果。
