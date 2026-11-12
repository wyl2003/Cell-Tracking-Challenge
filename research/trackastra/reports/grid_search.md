# Trackastra Fine-tuning 与 8 组 Grid Search

本表仅使用 Fluo-N2DH-GOWT1 `01` 训练，使用 `02` 图像和 `02_GT/TRA` Mask 验证。CT、TRA 和 AOGM 均为 **02 验证序列结果，不是独立测试集成绩**。推理为 Greedy，`batch_size=1`；所有实验从同一官方 `general_2d` 权重独立开始，`resume: false`，保存 Validation Loss 最低的 `model.pt`。

| 实验 | 请求 Epoch | 实际 Epoch | 最佳 Epoch | Samples | LR | 最佳 Val Loss | CT ↑ | TRA ↑ | AOGM ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e10_s16_lr1e-05_f55dd22773bd | 10 | 5 | 4 | 16 | 1e-5 | 0.068140 | 0.897196 | 0.996624 | 97.5 |
| e10_s89_lr1e-05_172ea7de19d3 | 10 | 6 | 5 | 89 | 1e-5 | **0.000890** | 0.897196 | 0.996624 | 97.5 |
| e10_s89_lr5e-06_061c63c3a050 | 10 | 5 | 4 | 89 | 5e-6 | 0.004663 | 0.897196 | 0.996624 | 97.5 |
| e2_s16_lr1e-05_3ff88d75f518 | 2 | 2 | 2 | 16 | 1e-5 | 1.317067 | 0.897196 | 0.996624 | 97.5 |
| e2_s89_lr1e-05_6ab88bdbbbc7 | 2 | 2 | 2 | 89 | 1e-5 | 0.016232 | 0.897196 | 0.996624 | 97.5 |
| e2_s89_lr5e-06_5ae22a0d4fb2 | 2 | 2 | 2 | 89 | 5e-6 | 0.149374 | 0.897196 | 0.996624 | 97.5 |
| e10_s16_lr5e-06_9e653ae3ead8 | 10 | 5 | 4 | 16 | 5e-6 | 0.665823 | 0.867925 | 0.996537 | 100.0 |
| e2_s16_lr5e-06_733af018e2cd | 2 | 2 | 2 | 16 | 5e-6 | 2.017452 | 0.867925 | 0.996537 | 100.0 |

六组在 CT、TRA 和 AOGM 上并列。它们的 2509 个追踪节点和 2462 条连接完全相同；CTC Mask 的数字轨迹 ID 可能不同，但能通过一致的一对一重映射得到相同的追踪结果。六组共同的错误连接为 `(6,43) → (7,45)`。选择 `e10_s89_lr1e-05_172ea7de19d3` 作为后续模型，是因为它在这六组中有最低的 Validation Loss；这项选择本身仍基于同一验证集，不能作为独立泛化证据。

原始数值和各 checkpoint SHA256 见 [grid_search.csv](../results/grid_search.csv)；图对照与 CTC 等价性见 [graph_summary.csv](../results/graph_summary.csv) 和 [ctc_equivalence.csv](../results/ctc_equivalence.csv)。随机种子与缓存复现结果见 [reproducibility.md](reproducibility.md)。
