# Trackastra Greedy 与 ILP 对照实验

> 本报告归档原本地实验。四组数值见 [对比表](../results/greedy_vs_ilp.csv)，逐边变化见 [changed_edges.csv](../results/changed_edges.csv)。模型权重、CTC TIFF、数据和缓存未上传；可用 [对照脚本](../scripts/compare_greedy_ilp.py)及本地模型重建。

## 实验范围与接口核实

本实验只对已安装的官方 `general_2d` 和此前选定的 Fine-tuned `e10_s89_lr1e-05_172ea7de19d3/attempt_01/train/model.pt` 做推理。两种模型均使用同一 Fluo-N2DH-GOWT1 `02` 图像（92 帧）和 `02_GT/TRA` Mask（92 帧），`batch_size=1`、CUDA、默认图像归一化与默认候选图参数。没有启动训练，也没有修改已有模型、官方源码或以前的结果。全部指标均为 **02 验证序列成绩，不是独立测试集成绩**。

本地 Trackastra 0.5.6（Git `aa57a95`）的正式接口为 `Trackastra.track_from_disk(imgs_path, masks_path, mode="greedy" | "ilp")`。ILP 分支实际调用 `track_ilp(candidate_graph, ilp_config="gt")`。`gt` 参数为：节点权重 0、节点常数 -10、边权重 -1、边常数 0、出现代价 0.25、消失代价 0.5、分裂代价 0.25；每个节点至多一个父节点、两个子节点。Trackastra 未向 `motile.Solver.solve()` 传额外求解参数，因此使用 Motile 默认的 `backend=Any`、1 线程、无超时。环境中安装了 `motile 1.0.1`、`ilpy 0.6.0`、`gurobipy 13.0.3`；小图求解测试与正式两组 ILP 均成功，正式运行日志均显示 `OPTIMAL`。`backend=Any` 的实际内部后端没有由当前接口报告，故不宣称特定后端。

Greedy 使用官方默认阈值 0.5，允许分裂。py-ctcmetrics 1.3.3 使用 `--tra --ct --bc 1 -n 1`；`--bc 1` 同时输出严格的 BC(0) 和允许 ±1 帧的 BC(1)。BC 不依赖我们的逐边近似分析。

## 四组正式结果

| 模型 | 追踪 | TRA ↑ | CT ↑ | AOGM ↓ | BC(0) ↑ | BC(1) ↑ | 节点 | 边 | 错误边 FP / 缺失边 FN | 分裂数 / 逐帧 GT 精确分裂 | 完整追踪耗时 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| general_2d | Greedy | 0.996537 | 0.867925 | 100.0 | 0.0 | 0.0 | 2509 | 2461 | 1 / 1 | 0 / 0 | 6.426 s |
| general_2d | ILP | **0.999359** | **0.964912** | **18.5** | 0.4 | 0.4 | 2517 | 2463 | 2 / 0 | 1 / 1 | 6.661 s |
| Fine-tuned | Greedy | 0.996624 | 0.897196 | 97.5 | 0.4 | 0.4 | 2509 | 2462 | 1 / 0 | 1 / 1 | 5.451 s |
| Fine-tuned | ILP | 0.999290 | 0.910714 | 20.5 | 0.4 | 0.4 | 2517 | 2465 | 4 / 0 | 1 / 1 | 6.531 s |

“完整追踪耗时”只包含 `track_from_disk`，不含模型加载、CTC 文件导出和 py-ctcmetrics。计入后处理与评价的总耗时依次为 41.318、42.057、40.583、42.237 秒；其中评价本身约 34 秒。上述时间是每组一次的墙钟测量，不作为稳定的速度基准。

## 同一候选图的求解器计时

每个模型只预测一次，再将 **同一张候选图** 分别交给 Greedy 和 ILP。重新计算出的四张结果图与正式推理的节点、边集合均完全相同。general_2d 候选图有 2517 节点、2465 边；Fine-tuned 候选图有 2517 节点、2468 边。

| 模型 | 预测 | 候选图构建 | Greedy 求解 | ILP 求解 |
|---|---:|---:|---:|---:|
| general_2d | 4.946 s | 0.081 s | 0.0126 s | 1.261 s |
| Fine-tuned | 4.114 s | 0.076 s | 0.0119 s | 1.035 s |

因此在这段 92 帧数据上，ILP 的求解步骤明显慢于 Greedy；总追踪调用中的差距因图像预测、数据读取和其他工作而缩小。每个求解器仅测一次，不能推断其它序列的典型耗时。

## 节点、错误连接与分裂

两组 Greedy 图相对对应 ILP 图均缺少同样 8 个节点：`(2,41)`、`(6,44)`、`(10,46)`、`(17,48)`、`(19,49)`、`(20,50)`、`(46,64)`、`(48,54)`，其中元组为 `(帧, GT/TRA Mask 标签)`。这些节点在候选图及输入 GT/TRA Mask 中存在，但没有被 Greedy 的选中边触及；官方 Greedy 实现只将选中边的端点放入结果图。ILP 的节点常数 -10 使它保留全部 2517 个候选节点。py-ctcmetrics 的 `AOGM_FN` 因而由两组 Greedy 的 8 降为两组 ILP 的 0。**这项检测保留差异是 TRA/AOGM 大幅改善的主要原因**，不能全部归因于 ILP 找到了更正确的连接。

四组共同的错误边是 `(6,43) → (7,45)`。相对于 Greedy，general_2d 的 ILP 增加了一条正确分裂边 `(82,22) → (83,24)`，也增加一条错误边 `(5,42) → (6,43)`。Fine-tuned 的 ILP 没有新增正确边，而增加三条错误边：`(1,40) → (2,41)`、`(5,42) → (6,43)`、`(19,49) → (20,50)`。这些 ILP 新增边的权重分别可在各组 `edges.csv` 查到；后两条低于 Greedy 的 0.5 阈值，但 ILP 的整体目标仍可选择它们。因此，ILP 在本实验中提高正式 CTC 指标的同时，**逐边错误连接数上升**。

GT 轨迹表有 4 次分裂事件。py-ctcmetrics 的 BC(0)/BC(1) 分别是 general_2d Greedy 0、其 ILP 0.4、Fine-tuned 两种追踪均 0.4。按逐帧相邻 GT 边能直接核对的精确双子分裂只有 `(82,22) → {(83,24),(83,25)}`：general_2d Greedy 漏掉一条女儿边，其 ILP 补上；Fine-tuned 两组均已正确连接。GT 还有 11 条跨时间间隔的 parent 关系，不能放进默认 `delta_t=1` 的逐边集合；正式 BC 和 TRA 仍以 py-ctcmetrics 为准。

## 结论与边界

在固定的 **02 验证序列** 上，ILP 对两个模型都提高了 CT 和 TRA：general_2d 的 CT `+0.096988`、TRA `+0.002822`；Fine-tuned 的 CT `+0.013518`、TRA `+0.002666`。四组中 `general_2d + ILP` 的 CT、TRA 最高，AOGM 最低。Fine-tuned 在 Greedy 下优于 general_2d Greedy，但在 ILP 下低于 general_2d ILP；模型排序依赖追踪解码方式。

ILP 的改进主要来自保留孤立的 GT 检测节点；general_2d ILP 还恢复了一次可逐帧验证的分裂。ILP 并没有减少错误边：general_2d 由 1 增为 2，Fine-tuned 由 1 增为 4。BC 只在 general_2d 上改善，Fine-tuned 不变。结论限于此序列、GT/TRA Mask、默认 `gt` ILP 代价和单次推理，不应外推为 ILP 在独立测试集或所有图像上更好。

## 可复查文件

- `comparison.csv`、`summary.json`：四组指标、图数量、运行时间。
- `provenance.json`：输入、模型 checkpoint SHA256、软件和追踪设置。
- `solver_benchmark.json`：同一候选图的计时及与正式结果图的一致性。
- `mode_differences.csv`、`changed_edges.csv`：同一模型的模式差异及所有发生变化的边。
- 各组目录中的 `02_RES/`、`ctc_metrics.csv`、`metrics.log`、`nodes.csv`、`edges.csv`、`false_positive_edges.csv`、`false_negative_edges.csv`、`splits.csv`、`result.json`：完整 CTC 结果与逐图检查数据。
- 重运行脚本：`scripts/compare_greedy_ilp.py`；默认跳过已经完成且 checkpoint 指纹匹配的四组结果。
