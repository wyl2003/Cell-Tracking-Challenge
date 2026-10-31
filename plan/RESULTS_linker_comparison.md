# 链接器对比实验：Trackastra（SOTA）vs 我们的 IoU + 匈牙利

> 全部数字都是在这台机器（RTX 3060 Laptop，torch 2.14.1+cu126，trackastra 0.5.6）上现跑的，
> 不是文献引的。复现命令在第 5 节。给组员看的版本。

---

## 0. 一句话结论

**在这个数据集上，"换成 SOTA 追踪器" 不会提高分数，因为瓶颈不在链接器。**

同一份 Cellpose `cpsam` 掩膜，只替换链接器：01 的 AOGM 从 424 变成 422，TRA 都是 **0.9821**；
02 反而从 1230 变成 1245（变差）。连**在 CTC 数据上训练过的** Trackastra 专用权重也是平手。

误差的 64–74% 是 **FN（细胞根本没被检出）**，那是分割侧的事，任何链接器都救不了。

---

## 1. 实验设置

唯一变量是链接器，输入掩膜完全相同（`runs/cpsam/{01,02}`，零样本 `cpsam` 输出）：

| 链接器 | 说明 | 学没学 |
|---|---|---|
| **ours**：IoU + 匈牙利 | 本项目 `src/track.py`，帧间 IoU 矩阵 → `linear_sum_assignment`，IoU<0.2 丢弃；分裂用"子细胞 ≥50% 落在母 mask 内"的规则 | 无 |
| **Trackastra `general_2d`** | ECCV 2024，ISBI 2024 第七届 CTC 冠军；transformer 预测两两关联，`greedy` 模式 | 有（通用活细胞数据） |
| **Trackastra `ctc`** | 同一模型，在 **CTC 训练数据**上训练的权重 —— **属于泄漏条件下的上界参考**，不能当作公平比较 | 有（含 CTC） |

还需要一个"完美分割"条件下的对照，把链接器质量和检测器质量分开：

| 设置 | 输入掩膜 |
|---|---|
| `02*` | `data/Fluo-N2DH-GOWT1/02_GT/TRA`（金标准轨迹掩膜，完美分割） |

---

## 2. 主结果（真实工作点：cpsam 掩膜）

| 序列 | 链接器 | AOGM ↓ | TRA | CT | 分裂 对/真/错 | 轨迹数（真值） |
|---|---|---|---|---|---|---|
| 01 | ours | 424 | 0.9821 | 0.5507 | 0 / 2 / 0 | 41（28） |
| 01 | astra `general_2d` | **422** | 0.9821 | **0.6207** | 0 / 2 / 0 | **30**（28） |
| 01 | astra `ctc`（泄漏） | 424.5 | 0.9820 | 0.6207 | 0 / 2 / 0 | 30 |
| 02 | **ours** | **1230** | **0.9574** | 0.2985 | 0 / 4 / 7 | 76（58） |
| 02 | astra `general_2d` | 1245 | 0.9569 | **0.3276** | 0 / 4 / 1 | **58**（58） |
| 02 | astra `ctc`（泄漏） | 1249 | 0.9567 | 0.3167 | 0 / 4 / 4 | 62 |

DET / SEG 两行完全不变（同一批掩膜），差别全在 TRA 的分子里。

**AOGM 差值换算成分数**：01 从 424 → 422 等于 TRA +0.0001；02 从 1230 → 1245 等于 TRA −0.0005。
**在四位小数上，三个链接器是同一个数。**

---

## 3. 但 Trackastra 确实更好 —— 好在"轨迹质量"，不在总分

这条比上面那张表重要：

| 序列 | 指标 | ours | astra `general_2d` |
|---|---|---|---|
| 01 | 轨迹数（真值 28） | 41 | **30** |
| 01 | TF（轨迹碎不碎） | 0.9470 | **0.9723** |
| 01 | CT（谱系对不对） | 0.5507 | **0.6207** |
| 02 | 轨迹数（真值 58） | 76 | **58** |
| 02 | TF | 0.9142 | **0.9304** |
| 02 | 误报分裂 | 7 | **1** |

它把 02 的轨迹数**正好压到真值的 58 条**，误报分裂从 7 次降到 1 次。
**这是真本事，但 TRA 几乎不动**（02 甚至略降）。

原因：AOGM 里"多断一条边"只值 1.5 个单位，而"漏一个细胞"值 10 个单位。
一条被碎成 3 段的轨迹，只扣 3 分；漏掉一个细胞，扣 10 分。
**所以聚合指标对轨迹质量是几乎失明的 —— 和它对分裂失明是同一个毛病。**

这正好是报告里现成的第二条论据：**TRA 0.98 不等于 98 分的追踪**。
建议报告里并列报 TF / CT / 轨迹数，而不是只报 TRA。

---

## 4. 完美分割下谁赢？（本节是最干净的一条证据）

输入换成**金标准轨迹掩膜**（02 序列，分割误差为零），只比链接：

| 链接器 | AOGM | TRA | CT | TF | 轨迹数 | FN | EA |
|---|---|---|---|---|---|---|---|
| **ours** | **56.5** | **0.9980** | 0.6043 | 0.9398 | 81 | **0** | 37 |
| astra `general_2d` | 100.0 | 0.9965 | **0.8679** | **1.0000** | 48 | 8 | **12** |

两条都要看：

1. **我们的 TRA 反而更高**（0.9980 vs 0.9965）。原因是 astra 把 8 个细胞**丢掉了**
   （FN = 8：图里没被任何窗口连上的节点不会被写进结果），而我们的追踪器"每个检出都会进结果"。
   **简单方法在这一项上没有输。**
2. astra 的谱系（CT 0.868 vs 0.604）和轨迹完整性（TF 1.0000）明显更好。
   也就是说 **Trackastra 的强项是"连对的边"，弱项是"保证不漏节点"**。

**"传统方法一定打不过 SOTA" 在这个数据上不成立。** 差距是有的，但方向不是单调的，
而且被检测侧的误差完全盖住了。

---

## 5. 分裂：0/2 和 0/4，三个链接器全灭

| 序列 | 真值分裂 | ours | astra `general_2d` | astra `ctc` |
|---|---|---|---|---|
| 01 | 2 | 0 对 | 0 对 | 0 对 |
| 02 | 4 | 0 对 | 0 对 | 0 对 |

**连在 CTC 数据上训练过的权重也一次没对。** 02 上它预测了 4 次分裂，位置全错。

这条把"分裂要不要换成学习模型"这个问题**实证性地关掉了**：

> 分裂错误不是链接策略的问题。当两个子细胞在分割结果里还没有分成两个独立对象时，
> 任何链接器都无法把这次分裂判对。当前的 6 次分裂失败属于**分割侧的时相问题**
> （分裂帧前后掩膜粘连 / 面积突变），不属于关联侧。

这也解释了为什么 `--division-containment` 规则"调多少次都是 0 次触发"——
规则没有错到值得调，是输入里没有可判的信号。

---

## 6. 复现

```bash
# SOTA 链接器（新增脚本，接口和 run_track.py 一致）
.venv/Scripts/python -m src.run_trackastra --img data/Fluo-N2DH-GOWT1/01 --masks runs/cpsam/01 \
    --out runs/01_RES_astra_g2d --model general_2d --mode greedy

# 我们的链接器
.venv/Scripts/python -m src.run_track --masks runs/cpsam/01 --out runs/01_RES_fixed --iou 0.2

# 完美分割对照
.venv/Scripts/python -m src.run_track --masks data/Fluo-N2DH-GOWT1/02_GT/TRA --out runs/02_RES_gtmask_ours
.venv/Scripts/python -m src.run_trackastra --img data/Fluo-N2DH-GOWT1/02 \
    --masks data/Fluo-N2DH-GOWT1/02_GT/TRA --out runs/02_RES_gtmask_astra --model general_2d

# 打分 + 六项分解
.venv/Scripts/python -m src.diagnostics --gt data/Fluo-N2DH-GOWT1/01_GT --res runs/01_RES_astra_g2d
```

耗时：Trackastra 在 3060 上约 **1 分钟/序列**（含特征提取；关联本身只 3–5 秒），
比我们的追踪器（约 90 秒）**更快**。

---

## 7. 没做成的（如实记录）

* **`mode="ilp"` 没跑起来。** Trackastra 的 ILP 模式用 `motile`，需要 Gurobi 或 SCIP：
  - Gurobi：免费许可有模型尺寸上限，直接报 `Model too large for size-limited license`
  - SCIP：`PySCIPOpt 6.3.0` 在这台 Windows 机器上 `ImportError: DLL load failed`
  所以上表全部是 `greedy` 模式的结果。**论文里必须写这一句**——ILP 通常比 greedy 略好，
  我们的"平手"结论在 ILP 下没有被验证。想补这一行的话，去 Colab/Linux 上跑。
* 只跑了 2D `cpsam` 掩膜这一种输入（除第 4 节的完美分割对照）。
* 分裂事件只有 6 个，任何"0/2 vs 0/4"的差别都**没有统计意义**，只能给逐事件表。

---

## 8. 这次的代码改动

1. `src/track.py`：删掉关联候选里一处**错误的过滤条件**
   （`t not in frame_map.get(frame, {})`——`frame_map` 的 key 是 mask 标签，不是轨迹 ID，
   导致轨迹 ID 恰好等于某个 mask 标签时被误排除）。
   **注意：这个 bug 只在 `--max-gap > 0` 时才发作**，而那条路径本来就产出 CTC 非法的结果，
   所以**修复前后的 max_gap=0 输出逐位相同，已报的 0.9821 / 0.9574 不受影响**。
   之前我说"轨迹碎可能是这个 bug 造成的"，实测**不成立**，此处更正。
2. `src/run_trackastra.py`：新增，把 Trackastra 接进现有 pipeline，输出与 `run_track.py`
   完全同格式的 `<seq>_RES`（`mask###.tif` + `res_track.txt`），可直接用 `src.evaluate` 打分。
3. 环境：`.venv` 里加了 `trackastra==0.5.6`（连带 motile / zarr / scikit-image 等），
   numpy / torch / cellpose 版本未变，`cpsam` 仍可正常运行。
