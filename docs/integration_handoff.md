# Part 1 与 Part 2 交接

本工作树整合 IADC 的 Part 1 和课程服务器上组员的 Part 2。两个实验各自保留原始结果及复现所需的源码版本。

| 实验 | 原始源码 | VGGT 初始化参数 |
| --- | --- | --- |
| Part 1 三种初始化比较 | `bde9244f692cc497355a307826ce1ca25f39cd99` | IADC 配置的 `vggt_offload_features=true` 通过矩阵入口转成 `--vggt-offload-features` |
| Part 2 VGGT full/sparse | `codex/part2-snapshot`，快照提交 `bd2e3ec` | 正式 runner 使用 `--neural-backend offload` |

前一个选项将预测头特征放到 CPU，后一个选项逐层使用 CPU 和磁盘卸载。两种方式都保留完整的输入视角。按原实验选用对应选项。

## 实验结果及已完成的对照

- `output/handoff/part1-iadc-results.zip` 有归档时的 Part 1 覆盖情况、正式指标、初始化、抽样渲染及大型 checkpoint 的原服务器路径和 SHA-256。
- `output/handoff/part2-snapshot-20261115-200104.zip` 有组员提交前的 17 个代码和文档文件的 SHA-256，以及归档时的 Part 2 日志和指标。Re10k-1 的配对已完成，Waymo 的 Full 已评估，Sparse 当时在训练。
- 归档包含采集时的状态。后续实验的实时状态以两台服务器原运行目录为准。

三场景的 dense split hash 在当前 IADC 与组员 Part 2 中一致。Part 2 已有的 Full 是组员在课程服务器训练的，位姿 ATE 也按 sparse 的共同原始帧另外计算，因此保留原配对。Part 1 的正式结果仍可独立用于三初始化比较。

## 运行版本

`p4.train` 会校验 checkpoint 绑定的 `p4/*.py` 内容、manifest 及训练图像。恢复或重渲染已有 checkpoint 时，从上表对应的提交建立独立工作树，并使用原实验的预处理数据和环境。

`scripts/part2_formal.py` 是组员在课程服务器上继续原实验的入口，依赖原工作树的数据、审核记录、预检和 Re10k 基线，并要求源码指纹一致。该流程及历史运行证据保存在 Part 2 快照。合并工作树可用于共同开发和查看归档；从合并版启动一轮新的 Part 2 时，需要先按该版本重建验证记录和对应基线。
