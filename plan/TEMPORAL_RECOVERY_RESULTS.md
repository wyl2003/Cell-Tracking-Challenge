# 时间感知的漏检细胞恢复：补充实验报告

**COMP9444 Cell Tracking Challenge — Fluo-N2DH-GOWT1**

## 1. 研究目标

研究能否利用当前帧前后相邻帧的信息，恢复现有 Cellpose-SAM 多尺度分割流程中漏检的细胞实例。

本方法属于**无需训练、基于规则的时空后处理方法**，并不是重新训练的神经网络模型。

## 2. 方法

1. **重建多尺度分割结果**：结合 Cellpose 在 `diameter=48` 和 `diameter=32` 下的输出，使用重叠比例阈值 `0.10` 和最小面积 `200 pixels`，保留所有 Baseline 实例。
2. **寻找可能的单帧漏检**：在当前帧 `t` 的前一帧 `t-1` 和后一帧 `t+1` 中，寻找空间位置接近、面积相似的细胞。如果它们的插值中心附近在当前帧没有已有实例，就将其视为候选。
3. **生成并筛选候选 Mask**：把相邻帧的实例 Mask 平移至插值中心，取其交集作为较保守的候选区域，再检查形状重叠、已占据像素比例、最小面积以及局部荧光强度对比度。
4. **合并结果**：对通过筛选的候选分配新的实例 ID，添加至当前帧，不覆盖已有细胞。整个恢复过程**不使用 Ground Truth（GT）**。

**实现脚本：** `src/run_temporal_recovery.py`  
**可视化与逐实例消融脚本：** `src/analyze_temporal_events.py`

### 默认参数

| 参数 | 设置 |
|---|---:|
| 最小实例面积 | 200 pixels |
| 前后两帧最大中心位移 | 24 pixels |
| 当前帧已有实例的中心排除半径 | 15 pixels |
| 相邻帧实例最小面积比例 | 0.60 |
| 对齐后形状的最小 IoU | 0.50 |
| 与当前已有 Mask 的最大重叠比例 | 0.10 |
| 最小局部亮度对比度比例 | 1.15 |

以上参数是实验使用的**启发式初始设置**，尚未证明是最优超参数。

## 3. 实验设置与整体结果

Fluo-N2DH-GOWT1 的 Sequence 01 和 Sequence 02 均包含 **92 帧**，图像尺寸为 **1024 × 1024**。

后续追踪器固定使用 Trackastra `general_2d` 模型与 `greedy` 模式，通过项目现有评价脚本计算 DET、SEG、TRA 以及综合指标：

\[
OP_{CTB}=\frac{SEG+TRA}{2}
\]

为保证对比公平，我们重新构建了仅使用 Multi-scale 的对照结果，并在两个序列上**均复现了原先四位小数精度的评价结果**。

| 序列 | 方法 | DET | SEG | TRA | OP_CTB |
|---|---|---:|---:|---:|---:|
| 01 | Baseline | 0.9818 | 0.8878 | 0.9822 | 0.9350 |
| 01 | Multi-scale Recovery | 0.9824 | 0.8984 | 0.9829 | 0.9406 |
| 01 | Multi-scale + Temporal Recovery | 0.9823 | 0.8984 | 0.9828 | 0.9406 |
| 02 | Baseline | 0.9561 | 0.9193 | 0.9569 | 0.9381 |
| 02 | Multi-scale Recovery | 0.9600 | 0.9193 | 0.9611 | 0.9402 |
| 02 | Multi-scale + Temporal Recovery | 0.9603 | 0.9193 | 0.9615 | 0.9404 |

Temporal Recovery 在每个序列中都额外恢复了 **3 个实例，共 6 个**。这些新增实例是在原 Multi-scale 方法已恢复的 **52 个（Sequence 01）和 37 个（Sequence 02）** 实例的基础上增加的。

表中分数保留四位小数，因此显示的 OP_CTB 相同，**不代表未四舍五入的真实数值完全一致**。

## 4. 时间恢复事件与逐实例消融实验

我们采用 **Leave-one-out Ablation（逐实例移除消融）**：每次从完整 Temporal Recovery 结果中只删除**一个**新增实例，然后重新运行 Trackastra 并评价。

注意：删除一个实例可能改变追踪器在其他位置的关联结果，因此最终分数变化**不等同于**该实例一定是真实细胞或误检。

### 4.1 Sequence 01

| 实验设置 | 移除的帧 | 实例面积（pixels） | DET | TRA | OP_CTB |
|---|---:|---:|---:|---:|---:|
| 仅 Multi-scale | — | — | 0.9824 | 0.9829 | 0.9406 |
| 完整 Temporal Recovery | — | — | 0.9823 | 0.9828 | 0.9406 |
| 移除事件 01 | 008 | 266 | 0.9823 | 0.9828 | 0.9406 |
| 移除事件 02 | 018 | 1422 | 0.9824 | 0.9829 | 0.9406 |
| 移除事件 03 | 089 | 1884 | 0.9824 | 0.9829 | 0.9406 |

**结果分析：**

- 移除事件 01 后，四位小数精度下的指标没有变化。
- 单独移除事件 02 或事件 03 后，指标恢复到原 Multi-scale 水平。
- 这**不能证明**事件 02 和事件 03 都是错误检出，因为不同事件可能通过追踪器产生相互影响。

### 4.2 Sequence 02

| 实验设置 | 移除的帧 | 实例面积（pixels） | DET | TRA | OP_CTB |
|---|---:|---:|---:|---:|---:|
| 仅 Multi-scale | — | — | 0.9600 | 0.9611 | 0.9402 |
| 完整 Temporal Recovery | — | — | 0.9603 | 0.9615 | 0.9404 |
| 移除事件 01 | 012 | 215 | 0.9604 | 0.9615 | 0.9404 |
| 移除事件 02 | 015 | 250 | 0.9604 | 0.9615 | 0.9404 |
| 移除事件 03 | 037 | 1712 | 0.9599 | 0.9610 | 0.9401 |

**结果分析：**

- **Frame 037 的事件 03 是最有价值的候选。** 删除它会使 TRA 和 OP_CTB 下降，不仅低于完整 Temporal Recovery，也略低于仅使用 Multi-scale 的结果。
- 另外两个面积较小、从可视化看疑似位于图像边界的候选，被单独删除时对四位小数精度的 TRA 没有影响，但 DET 略有上升。
- **尚未测试同时删除事件 01 和 02**，因此不能据此推断两者一起删除的效果。

## 5. 讨论与集成建议

实验表明，通过相邻帧插值可以恢复一些看起来合理的单帧漏检实例。其中 Sequence 02 的 Frame 037 是一个面积为 **1712 pixels** 的较大候选，原有 Multi-scale 方法没有补充这个实例。

不过，整体改善**较小，而且依赖具体序列**：

- Sequence 01：TRA 从 **0.9829 下降到 0.9828**（−0.0001）。
- Sequence 02：TRA 从 **0.9611 提高到 0.9615**（+0.0004）。
- 两组序列的 **SEG 均保持不变**。

本次仅产生 6 个时间恢复候选。因此，实验尚不能证明该方法具有稳定的泛化收益，也没有证明它能够解决细胞分裂识别失败的问题。

### 推荐给 Tracking 组员的集成方案

**建议继续将已经验证有效的 Multi-scale Recovery 作为默认 Segmentation 输出。** Temporal Recovery 可以作为一个**可选的实验模块**保留，并在报告中展示实现思路、评价结果及消融分析，但目前不建议默认启用，也不应声称它在所有序列上都优于 Multi-scale。

后续若有额外数据或逐事件的细胞分裂分析结果，可以再评估是否值得集成 Temporal Recovery。

## 6. 复现方法（PowerShell，项目根目录下运行）

首先生成 Temporal Recovery 的分割 Mask：

```powershell
python -m src.run_temporal_recovery `
  --baseline runs/cpsam/01 `
  --small runs/multiscale/seq01_d32_full `
  --images data/Fluo-N2DH-GOWT1/01 `
  --out runs/temporal_recovery/01

python -m src.run_temporal_recovery `
  --baseline runs/cpsam/02 `
  --small runs/multiscale/seq02_d32 `
  --images data/Fluo-N2DH-GOWT1/02 `
  --out runs/temporal_recovery/02
```

按照脚本设计，**重新运行前，输出目录必须不存在或为空**。

下面以 Sequence 02 为例，运行 Trackastra 并评价：

```powershell
python -m src.run_trackastra `
  --img data/Fluo-N2DH-GOWT1/02 `
  --masks runs/temporal_recovery/02 `
  --out runs/02_RES_temporal_trackastra `
  --model general_2d --mode greedy

python -m src.evaluate `
  --gt data/Fluo-N2DH-GOWT1/02_GT `
  --res runs/02_RES_temporal_trackastra
```

实验生成的 `runs/` 和数据集目录 `data/` 不属于源代码文件，**不应该加入 Git 提交**。复现实验还需要使用与项目 Baseline 一致的输入数据和软件依赖。
