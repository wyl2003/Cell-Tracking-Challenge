# 分割改进报告——多尺度细胞恢复（Multi-scale Cell Recovery）

**课程：** COMP9444 Neural Networks and Deep Learning  
**项目：** Cell Tracking Challenge（细胞追踪挑战）  
**数据集：** Fluo-N2DH-GOWT1  
**负责部分：** 细胞分割改进（Segmentation Improvement）及实验评估

## 1. 研究目标（Objective）

我负责改进细胞追踪流程中的分割（Segmentation）部分，重点解决较小、低对比度或被模型漏检的细胞。

项目的 Baseline 使用 **Cellpose-SAM v2** 进行细胞实例分割（Instance Segmentation），随后使用 **Trackastra** 完成细胞追踪（Cell Tracking）。

虽然 Baseline 已经取得了较好的表现，但通过错误分析发现，Cellpose 在原始推理尺度下仍然会完全漏检部分细胞。

因此，本部分工作的目标是在尽量保留原有分割结果的基础上，恢复这些漏检细胞，提高整体分割质量。

## 2. Baseline 与错误分析（Baseline and Error Analysis）

Baseline 使用以下 Cellpose 参数：

| 参数 | 设置 |
|---|---|
| 模型 | Cellpose-SAM v2 (`cpsam_v2`) |
| Diameter | 48 |
| Flow threshold | 0.4 |
| Cell probability threshold | 0.0 |
| Minimum instance size | 200 pixels |

在 Sequence 01 上，Baseline 的官方 **SEG 分数为 0.8878**。

通过对 144 个 GT/SEG 人工标注细胞进行详细分析，发现其中有 **7 个细胞与任何 Baseline 预测实例都没有重叠**，说明这些细胞被模型完全漏检。

进一步观察发现，这些漏检细胞集中在多个连续时间帧中的相似图像区域，并且具有不同于周围正常细胞的尺寸、形状或内部亮度特征。

例如，在 **Frame 014** 中，Baseline 完全漏检了一个人工标注细胞，其 IoU 为 0。

当我们将 Cellpose 的推理参数 `diameter` 从 48 调整为 32 后，模型成功识别该细胞，得到 **IoU = 0.7907**。

通过控制变量实验，进一步确认这一改善来自 `diameter` 的变化，而不是降低 `min_size` 参数。

这一发现为后续多尺度分割方法提供了依据。

## 3. 尝试过的改进方法（Methods Investigated）

在确定最终方案之前，我们尝试了多种分割优化方法，包括图像预处理、形态学后处理以及模型推理参数调整。

以下是 Sequence 01 上的主要实验结果：

| 方法 | 官方 SEG | 实验结论 |
|---|---:|---|
| 原始 Baseline | 0.88782 | 对照组 |
| CLAHE（clip limit = 1.0） | 0.88764 | 无明显改善 |
| 形态学腐蚀（Erosion 1px） | 0.88036 | 性能下降 |
| 形态学膨胀（Dilation 1px） | 0.86495 | 性能下降 |
| Cell probability threshold = 1.0 | 0.88610 | 无改善 |
| Cell probability threshold = -1.0 | 0.87730 | 性能下降 |
| **Multi-scale Cell Recovery** | **0.89839** | **性能提升** |

这些实验表明：

简单的图像增强、统一调整细胞边界，以及修改细胞概率阈值，都没有有效提高原始模型的分割性能。

相比之下，**恢复原本漏检的细胞，比进一步修改已经较为准确的细胞轮廓更加有效**。

因此，我们最终选择了多尺度细胞恢复（Multi-scale Cell Recovery）作为主要改进方案。

## 4. 提出的改进方法：多尺度细胞恢复（Multi-scale Cell Recovery）

该方法结合两个不同推理尺度下的 Cellpose 预测结果：

- **主尺度（Primary scale）：** `diameter = 48`
- **辅助尺度（Auxiliary scale）：** `diameter = 32`

首先，使用原始参数 `diameter=48` 生成 Baseline 分割结果。

随后，使用 `diameter=32` 进行第二次 Cellpose 推理，以寻找原始尺度可能遗漏的细胞。

对于辅助尺度预测得到的每一个细胞实例，计算它与 Baseline 中已有细胞区域的重叠比例：

\[
\text{Overlap Ratio}=\frac{|\text{Candidate}\cap\text{Baseline Foreground}|}{|\text{Candidate}|}
\]

只有同时满足以下三个条件，才会将候选细胞加入最终结果：

1. 与 Baseline 的重叠比例不超过 **0.10（10%）**。
2. 候选实例面积至少为 **200 pixels**。
3. 去除与已有细胞重叠的区域后，剩余面积仍然至少为 **200 pixels**。

合并过程中，所有 Baseline 的原始像素和实例 ID 都会被保留，不会覆盖原本已经分割出来的细胞。

符合条件的新增实例将被分配新的实例 ID。

这一方法有以下特点：

- 不需要重新训练 Cellpose 模型。
- 保留原有分割结果，降低破坏已有准确细胞轮廓的风险。
- 能够利用较小推理尺度寻找额外细胞。
- 实际推理过程中不需要读取 Ground Truth。
- 可以直接生成标准实例分割 Mask，交给后续 Tracking Pipeline 使用。

所有参数均在 Sequence 01 的开发过程中确定，并在 Sequence 02 上保持不变，用于验证方法的泛化能力。

## 5. 实验设置（Experimental Setup）

我们使用 Fluo-N2DH-GOWT1 数据集的两个序列进行实验：

- Sequence 01：92 帧图像。
- Sequence 02：92 帧图像。

两个序列均生成完整的分割 Mask，并交给 Trackastra 进行细胞追踪。

为了保证 Baseline 与改进方法之间的比较公平，Tracking 阶段采用完全一致的配置：

| 参数 | 设置 |
|---|---|
| Segmentation model | Cellpose-SAM v2 |
| Tracking model | Trackastra `general_2d` |
| Tracking mode | `greedy` |
| Evaluation | CTC 官方评价指标 |

使用的官方评价指标包括：

- **SEG（Segmentation）：** 衡量分割结果与人工标注的重合程度。
- **DET（Detection）：** 衡量细胞检测结果的质量。
- **TRA（Tracking）：** 衡量细胞追踪结果的质量。
- **OP_CTB：** 综合评价指标，由 SEG 和 TRA 的平均值计算得到。

\[
OP_{CTB}=\frac{SEG+TRA}{2}
\]

## 6. 实验结果（Results）

### 6.1 Sequence 01

| 指标 | Baseline | Multi-scale | 提升 |
|---|---:|---:|---:|
| DET | 0.9818 | **0.9824** | +0.0006 |
| SEG | 0.8878 | **0.8984** | +0.0106 |
| TRA | 0.9822 | **0.9829** | +0.0007 |
| OP_CTB | 0.9350 | **0.9406** | +0.0056 |

在 Sequence 01 的完整 92 帧中，多尺度方法总共增加了 **52 个预测实例**。

其中，在具有 GT/SEG 人工标注的 30 帧中，新增了 16 个实例。

通过对这些新增实例进行分析，我们确认其中有两个成功恢复了 Baseline 原本漏检的人工标注细胞：

- Frame 014：IoU = **0.7907**
- Frame 016：IoU = **0.7305**

这些新增细胞直接改善了官方 SEG 分数。

Sequence 01 的 SEG 从 0.8878 提高到 0.8984，绝对提升约 **0.0106**，相对提升约 **1.19%**。

同时，DET、TRA 和综合指标 OP_CTB 也均获得小幅提升。

### 6.2 Sequence 02

| 指标 | Baseline | Multi-scale | 提升 |
|---|---:|---:|---:|
| DET | 0.9561 | **0.9600** | +0.0039 |
| SEG | 0.9193 | 0.9193 | 0.0000 |
| TRA | 0.9569 | **0.9611** | +0.0042 |
| OP_CTB | 0.9381 | **0.9402** | +0.0021 |

在 Sequence 02 的 92 帧中，多尺度方法共增加了 **37 个预测实例**，涉及 29 个时间帧。

与 Sequence 01 不同，Sequence 02 的官方 SEG 分数保持不变。

这表明新增实例没有进一步提高该序列中已标注细胞的分割质量。

不过，在将改进后的 Mask 输入 Trackastra 后，官方 DET 和 TRA 均有所提升，最终 OP_CTB 从 0.9381 提高到 0.9402。

因此，虽然该方法在不同序列上的分割收益并不完全一致，但在本次实验中，两组序列的综合 Tracking Pipeline 表现都得到了改善。

## 7. 讨论与局限性（Discussion and Limitations）

### 7.1 方法的主要优势

多尺度细胞恢复方法能够利用不同推理尺度之间的互补性，恢复原始尺度下漏检的细胞，同时保留已经准确分割的 Baseline 实例。

相较于重新训练模型，该方法实现简单，不需要额外的人工标注或训练过程，也便于集成到现有 Pipeline。

此外，实验结果表明，改善分割结果并不一定需要修改所有细胞的边界。对于已经具有较高分割精度的模型，有针对性地恢复遗漏实例可能更加有效。

### 7.2 新增实例可能产生误检

需要注意的是，**新增预测实例并不一定都是真实细胞**。

在 Sequence 01 的 30 个标注帧中，我们使用 GT/TRA 和 IoU ≥ 0.5 的自定义实例匹配方法进行了额外诊断。

结果显示：

- Baseline F1：0.7613
- Multi-scale F1：0.7529

自定义 F1 下降的原因是：新增的 16 个实例都未能在该匹配规则下达到 IoU ≥ 0.5，因此增加了未匹配预测实例的数量。

然而，该自定义 F1 与官方 CTC DET 使用的评价规则不同，两者不能直接等同。

在正式的 CTC 评价中，Sequence 01 和 Sequence 02 的 DET 均有所提高。

因此，我们不能仅凭自定义 F1 判断最终 Tracking Pipeline 的效果，而应同时参考官方 DET、SEG 和 TRA。

### 7.3 不同序列上的改善程度不同

Multi-scale 方法使 Sequence 01 的 SEG 明显提高，但 Sequence 02 的 SEG 保持不变。

这表明方法的分割收益与原始模型漏检细胞的类型和分布有关。

对于 Sequence 01 中的特殊形态或较小目标，辅助尺度能够提供有效的额外预测。

但对于 Sequence 02，新增预测并没有进一步改善已标注细胞的分割结果。

因此，不能认为多尺度方法在所有数据集或序列上都必然提高 SEG。

### 7.4 实验数据的使用限制

Sequence 01 在方法开发过程中被用于错误分析和参数选择，因此其结果属于开发集上的实验表现。

Sequence 02 则用于固定参数后的独立验证。

由于目前只有两个序列，仍然需要更多数据来验证该方法的普遍适用性。

### 7.5 后续可能的改进方向

未来可以考虑进一步研究：

- **候选实例置信度（Candidate Confidence）：** 利用模型输出判断新增细胞的可信程度。
- **时间一致性（Temporal Consistency）：** 检查新增细胞是否在相邻时间帧中持续存在。
- **误检过滤（False Positive Filtering）：** 进一步减少可能的错误新增实例。

这些方向有望在保留多尺度恢复能力的同时，进一步改善检测和追踪性能。

## 8. 总结（Conclusion）

本次工作针对 Cellpose-SAM v2 在部分细胞上的漏检问题，提出了 **Multi-scale Cell Recovery（多尺度细胞恢复）** 方法。

该方法结合 `diameter=48` 和 `diameter=32` 两种推理尺度，在保留原始分割结果的基础上，自动补充低重叠的候选细胞。

实验表明：

- 在 **Sequence 01** 上，SEG 从 0.8878 提升到 **0.8984**，OP_CTB 从 0.9350 提升到 **0.9406**。
- 在 **Sequence 02** 上，SEG 保持不变，但 DET、TRA 和 OP_CTB 均有所提升，其中 OP_CTB 从 0.9381 提升到 **0.9402**。
- 两组序列均在不重新训练 Cellpose 模型的情况下，实现了整体 Pipeline 综合指标的改善。

总体而言，本次实验说明，利用不同推理尺度提供的互补预测，可以有效恢复部分漏检细胞，并改善细胞追踪系统的整体表现。

同时，实验也表明，该方法在不同序列上的分割收益存在差异，新增实例仍可能包含误检，需要结合检测和追踪指标进行综合评价。