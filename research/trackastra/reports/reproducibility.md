# Fine-tuning 后续验证报告

> 本报告归档原本地实验。仓库只保存脱敏的 [数值结果](../results/)、配置模板和脚本；`model.pt`、CTC TIFF、数据及缓存留在本地，不随仓库发布。

## 范围与数据

仅使用既定参数 10 Epochs / 89 Samples / LR 1e-5 做 seed 重复；没有扩大超参数搜索。训练为 01，评价为 02 图像与 02_GT/TRA Mask。所有 CT、TRA、AOGM 都是 **02 验证序列结果，不是独立测试集成绩**。

- 三次新训练 seed：43、44、20839；均从 general_2d 启动，`resume: false`，并使用各自独立的最佳 `model.pt`。已有 seed 42 网格结果仅作参照，不计入三次统计。
- 数据目录相同：01 图像 92 帧、02 图像 92 帧、02_GT/TRA Mask 92 帧。手动与自动缓存中的图像、GT Mask 和 GT 轨迹内容一致。
- 手动 Exp 2 与 seed 42 网格实验的训练模型 `config.yaml` SHA256 相同；训练参数除 seed 外无数值差异。手动 seed 20839，网格 seed 42。输出名、目录、时间戳只影响保存位置，但相对路径 `.cache` 因工作目录不同而指向不同缓存。
- 手动推理与自动化推理均为 `Trackastra.from_folder`、02 图像、02_GT/TRA Mask、greedy、batch size 1。手动及自动 CTC 均由 `graph_to_ctc` 导出并由 py-ctcmetrics 评价。

## 手动 Exp 2 与自动化模型

|来源|Seed|实际 Epoch|最佳 Epoch|最佳 Validation Loss|CT|TRA|AOGM|
|---|---:|---:|---:|---:|---:|---:|---:|
|手动 Exp 2|20839|8|7|0.006172|0.886792|0.996589|98.5|
|自动网格候选|42|6|5|0.000890|0.897196|0.996624|97.5|
|相同 seed 重跑（项目根目录缓存）|20839|4|3|0.007421|0.897196|0.996624|97.5|
|相同 seed 重跑（手动缓存）|20839|8|7|0.006172|0.886792|0.996589|98.5|

手动训练和 seed 42 自动训练的 `model.pt` SHA256 不同，损失轨迹与早停轮数也不同。固定 seed 20839 的项目根目录缓存重跑仍未复现手动结果，因此不能将 CT 差异仅归因于 seed。

**已定位的复现差异：** 手动运行使用 `scripts/.cache`，自动运行使用项目根目录 `.cache`。两份缓存保存的 `WRRandomCrop._rng` 状态不同。Trackastra 在 `wrfeat.py` 中以 `np.random.RandomState()` 创建这个独立随机状态，`crop_size: [256,256]` 使训练和验证数据都会裁剪；仅设置全局 seed 不会固定该状态。缓存原始图像和标注内容一致。

**受控复现成功：** 固定 seed 20839 并指定手动缓存后，8 个实际 epoch、最佳第 7 个 epoch、Validation Loss、CT、TRA、AOGM 与手动 Exp 2 完全一致；最佳 `model.pt` 的 SHA256 也完全一致（`976a6eea642c14982aba0771ab3f930979573f54ad4ce8fcdd00ec086f1a7c0b`）。这证明缓存中保存的裁剪随机状态足以解释本次手动实验与自动化重跑的差异。

官方 `Trainer` 未启用确定性算法选项，因此不能把本次精确复现推广为所有机器和环境下的位级确定性保证。

## 三个随机种子的稳定性

|Seed|实际 Epoch|最佳 Epoch|最佳 Validation Loss|CT|TRA|AOGM|
|---:|---:|---:|---:|---:|---:|---:|
|43|2|1|0.013152|0.886792|0.996589|98.5|
|44|2|1|0.031201|0.897196|0.996624|97.5|
|20839|4|3|0.007421|0.897196|0.996624|97.5|

三次的 CT = **0.893728 ± 0.006007**，TRA = **0.996612 ± 0.000020**，AOGM = **97.833333 ± 0.577350**；标准差为样本标准差（n−1，n=3）。CT 范围 0.886792–0.897196。三次中有 2/3 达到原网格最高 CT。这些统计固定使用项目根目录的缓存随机状态，且样本数只有 3。

## 追踪图与错误连接

六个 CT/TRA 并列模型的节点集合和连接集合完全相同：各 2509 个节点、2462 条连接。连接权重并不完全相同；详细差值在 `graph_summary.csv` 和各模型的 `edges.csv`。
六个模型的 CTC 导出文件并非逐字节相同，轨迹数字 ID 的分配会变化；但以 e10_s89_lr1e-05_172ea7de19d3 为参照，六组的 92 帧前景相同，且均能通过一致的一对一轨迹 ID 映射，使 Mask 和 `res_track.txt` 的关系完全相同。语义等价检验结果见 `ctc_equivalence.csv`。
按 GT 轨迹构建的连接集合有 2461 条；六个模型均有 1 条错误连接、0 条缺失连接。手动 Exp 2 有 2 条错误连接、0 条缺失连接。GT 中另有 11 条跨帧间隔的 parent 关系未纳入逐帧边比较；正式 CT、TRA、AOGM 仍以 py-ctcmetrics 的结果为准。

六个模型共同的错误边：(6,43) → (7,45)。
手动 Exp 2 相比自动网格候选额外的错误边：(5,42) → (6,43)；不在 GT 中。seed 43 的边集合与手动 Exp 2 相同。

## 保存的结果

- `seed_repeats_summary.csv`、`seed_stats.json`：三次重复训练和统计量。
- `graph_summary.csv`、`graphs/<模型>/nodes.csv`、`edges.csv`、`false_positive_edges.csv`、`false_negative_edges.csv`：完整图与错误边。
- `ctc_equivalence.csv`：CTC 数字标签重映射后的 Mask 与轨迹表等价性。
- `manual_vs_selected_edges.csv`：手动与网格候选的逐边差异；`manual_exp2_metrics.csv`：重新计算的手动指标。
- `provenance.json`、`cache_comparison.json`：配置、checkpoint 指纹和缓存对照。
- 原训练目录和 `02_RES` 均保留；本报告只写入新的验证目录。
