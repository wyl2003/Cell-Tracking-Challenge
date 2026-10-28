# COMP9444 26T3 · 项目 15「Cell Tracking Challenge」实施方案

> ⚠️ **部分内容已被替代（2026-10-28）**：组内已拿到 `Fluo-N2DH-GOWT1` 数据集（两段 92 帧，含 GT/TRA、GT/SEG、ST/SEG），数据选型由 HeLa 改为 GOWT1。数据集事实、基线实测分数、错误分工与人员安排请看 **[group_plan_review.md](group_plan_review.md)**。本文档中关于 GOWT1 不成立的部分：§1 的数据集选择、「细胞密集粘连」相关的错误分析、以及需要训练 4 个模型的实验矩阵。其余（时间表、报告页数预算、评分对照、CTC 规范细节）仍然有效。

> 依据：`COMP9444_Project_Descriptions.pdf` 第 15 页（项目 015）、WebCMS 上的 Project Marking Criteria、Project Report Template、Group Project Schedule；以及 celltrackingchallenge.net 的 2D 数据集页、评测方法页、py-ctcmetrics 官方仓库。文中所有课程分数/页数限制都来自这些文件，数据集大小/链接来自 CTC 官网。

---

## 1. 结论先行（如果只读一段）

**主数据集选 `Fluo-N2DL-HeLa`（2D 荧光，182 MB，两段各 ~92 帧），加 `Fluo-N2DH-SIM+`（合成，有完美分割标注）做第二个数据集；分割用「Cellpose 零样本」当强 baseline + 「微调 U-Net」当主方法；追踪用「IoU + 匈牙利 + 分裂检测 + 帧间隔补全」这条可解释的追踪器；评测一律用官方 `py-ctcmetrics` 报 DET / SEG / TRA / OP_CTB；实验做成 8 行左右的分割×追踪消融表。绝对不要碰 3D 数据集（1.2 GB+、上千帧、显存和时间都不够）。**

理由很直接：
- 课程评分里 **EDA 3 分 + 结果 3 分 + 讨论 2 分 = 8/15**，全在 notebook 里，靠的是「数据讲得清楚 + 对比做得干净」，不是靠模型多新。2D HeLa 帧数少、下载小、可视化好看，最容易把这三块写满。
- TRA 指标对**谱系（分裂）**单独计分，而大多数 student baseline 都栽在分裂上——这是最容易做出「自己的贡献」的点，也正好对应 rubric 里「Methods 3 分」要求的"clearly delineate which parts are your own work"。
- 报告只有 4 页，方法越简单越写得完。

---

## 2. 课程硬性要求（从 WebCMS 抓到的原文要点）

总分 **28 + 3**：

| 交付物 | 分值 | 关键约束 |
|---|---|---|
| Project Notebook | 15 | **只能交一个 notebook**，所有成员的工作汇总进去 |
| Summary Report | 5 | **最多 4 页**（不含参考文献），超页部分不计分 |
| Project Presentation | 8 | 个人 3 分（贡献 1.5 + 表达 1.5）+ 小组 5 分 |
| Group Class Participation | 3 | **week 7–9 三次 monitoring sessions**，按个人计 |

Notebook 15 分细分：Intro/Motivation 1 · 数据来源 1 · **EDA 3** · 模型方法 3 · 结果 3 · 讨论 2 · 写作 2。

报告 5 分细分：Introduction 0.5 · Literature Review 0.5 · Models/Methods 0.5 · Experimental Setup 0.5 · Results 0.5 · Conclusions 0.5 · 细节充分性 1 · 整体质量 1。模板（docx）已给：Abstract / Keywords / Introduction / Related Work / Methods / Experiments / Results / Conclusion / References。

两个容易被忽略的扣分点，rubric 写得很明确：
1. **「If building on previous work, identify the source and clearly delineate which parts are your own work」** —— 用 Cellpose 预训练权重就必须写清楚哪些是别人的、你改了什么。
2. **「Do NOT submit multiple notebooks」** —— 5 个人各自的实验必须提前合并成一个，别到最后一天合并。

> 待你们自己确认的两件事（我这边查不到）：
> - **具体 due date**：WebCMS 首页显示 "There is nothing due!"，Course Outline 已迁到 ECOS。去 Moodle / 公告里确认报告和展示的截止时间。
> - **分组登记**：`Group Project Schedule` 里你们那一组（F09A，含 Yile Wang / Shu Lian / Enjia Wu）的 **Project ID、Project Title、Group Name 三栏还是空的（组名那格还是占位符 "name"）**，组人数标 5 但只列了 3 个学号。交之前记得跟 tutor 确认已经登记成 015 号项目，否则可能按错项目评分。

（`Project 12 – Dataset Guide` 是给放射报告生成项目用的 CheXpert 说明，和你们无关，不用管。）

---

## 3. 数据集怎么选

CTC 官网 2D 数据集页给出的原始信息（大小为主站标注的训练包体积）：

| 数据集 | 模态 | 训练包 | 标注情况 | 建议用途 |
|---|---|---|---|---|
| **Fluo-N2DL-HeLa** | 荧光，HeLa 核，稠密 | 182 MB | 有 gold 轨迹（TRA）+ silver 分割（SEG） | **主数据集**：分裂多、细胞密，最能体现追踪难度 |
| **Fluo-N2DH-SIM+** | 计算机合成，HL60 核 | 91 MB | 合成数据提供**完美分割掩膜** | **第二数据集**：干净的上限实验 + 证明方法不是只对一种模态有效 |
| PhC-C2DH-U373 | 相差，少细胞、形变大 | 40 MB | silver SEG + gold TRA | 备选：难例/泛化，细胞少、跑得快 |
| DIC-C2DH-HeLa | DIC，对比度差 | 37 MB | silver SEG + gold TRA | 备选：最难，适合放进"局限性"讨论 |
| BF-C2DL-HSC / MuSC | 明场，微孔 | 1.2–1.6 GB，~1000 帧 | — | **不建议**（下载/算力/时间） |
| 全部 3D 数据集 | 3D | 更大 | — | **不建议** |

下载（直接 `curl` 即可，主站可用）：

```bash
base=https://data.celltrackingchallenge.net
curl -O $base/training-datasets/Fluo-N2DL-HeLa.zip     # 182 MB
curl -O $base/training-datasets/Fluo-N2DH-SIM+.zip     # 91 MB
unzip -q Fluo-N2DL-HeLa.zip -d data/
```

⚠️ **我实测的网络问题**：`data.celltrackingchallenge.net` 从你这台机器（国内网络）**完全超时不可达**，而 `celltrackingchallenge.net` 主站正常。所以别在本地死磕下载——去 UNSW 校园网 / 学校 VPN，或者让在澳洲的同学下好传给你。下载失败时不要以为是 URL 错了。

⚠️ **合规**（CTC 官网 "Conditions of use"，原文写得很硬）：
- **禁止 clone / 转载数据集**，包括参考标注。所以 **绝对不要把 `data/` 提交到 GitHub**（体积也会让 repo 爆掉），`.gitignore` 里加 `data/`。
- 报告/演示里引用 Maška et al. 2023（Nature Methods）并注明数据来自 CTC。

---

## 4. 技术方案（三层，按你们的工期设计）

### Layer A：实例分割（每帧）

| 方案 | 做法 | 定位 |
|---|---|---|
| A0 传统 | 阈值 + 形态学 + 距离变换 watershed 分水岭（`skimage.segmentation.watershed`） | 下限 baseline，几乎零成本，能把"深度学习到底贡献多少"讲清楚 |
| A1 零样本 | **Cellpose**（`model_type='nuclei'`，diameter 按各数据集细胞直径调） | 强 baseline，不训练，半天出结果 |
| A2 主方法 | **微调 U-Net / UNet++**（`segmentation_models_pytorch` 或自己写），随机裁剪 256×256，Dice+BCE，用距离变换 watershed 或 affinities 把语义图切成实例 | 主方法，能训练、能消融、能画 loss 曲线 |

> Cellpose 版本间 API 有差异（v3 `models.Cellpose(...)` vs v4 `models.CellposeModel(...)`），装完先看 `cellpose.__version__` 再抄文档，别照抄博客。

**把分割结果喂给我写的追踪器**：你的 segmenter 只需要每帧存一张 16-bit label 图（背景 0，细胞 1..N）：

```python
from src import ctc_io
for frame, mask in enumerate(cellpose_masks):      # mask: int label image
    ctc_io.write_mask(f"runs/cellpose/01/mask{frame:03d}.tif", mask)
```

### Layer B：追踪（`src/track.py`，已写好并测过）

1. 相邻帧按 **IoU + 匈牙利最优分配**（`scipy.optimize.linear_sum_assignment`）关联；
2. `--max-gap`：允许轨迹消失 1–2 帧再连上（漏检补救，直接降 FN）；
3. **分裂检测**：母细胞区域里出现两个子细胞、面积和约等于母细胞 → 母轨迹结束、两个子轨迹以 `parent=母` 新建。

这三步正好对应三个消融开关（`--iou`、`--max-gap`、`--no-divisions` / `--division-containment`），实验表天然就有了。

想再加一层：给每个轨迹加**恒速预测**（用上两帧位移外推质心再做匹配），对快速运动的 U373 / MuSC 有明显帮助。这是"你自己的贡献"最好写的一块。

### Layer C：评测（`src/evaluate.py`）

```bash
pip install -r requirements.txt
python -m src.evaluate --gt data/Fluo-N2DL-HeLa/01_GT --res runs/01_RES
# DET / SEG / TRA / OP_CTB
```

内部调用 CTC 官方 Python 实现 `py-ctcmetrics`（`ctc_evaluate`）。我在这台机器上已经装好并**端到端验证过**：合成一段"1 个细胞移动 → 分裂成 2 个"的 4 帧序列，用完美分割喂进追踪器，得到 `DET=1.0000 SEG=1.0000 TRA=1.0000`，0 基和 1 基文件名两种编号都对。

指标要点（写报告要用的）：
- **SEG**：逐帧实例 IoU 匹配后的平均 Jaccard；**TRA**：把预测的细胞谱系图"编辑"成 GT 图所需代价 AOGM 的归一化分数，`TRA = 1 − min(AOGM, AOGM₀)/AOGM₀`；**OP_CTB = (SEG+TRA)/2**。
- AOGM 默认权重：漏检 `wFN=10`、过分割 `wNS=5`、误检 `wFP=1`、删边 `wED=1`、加边 `wEA=1.5`。**含义很重要**：漏一个细胞罚 10，多检一个只罚 1 → **调参时优先保召回**；分裂边错一处 1.5，所以谱系必须单独消融。

**数据划分（rubric 明确要写）**：只用 training 序列（challenge set 没有 GT）。建议 `01` 训练/调参、`02` 测试，即 leave-one-sequence-out；两个数据集都这么做，报告里画一张 split 表。

---

## 5. 实验矩阵（直接变成 notebook 的一张表）

| 编号 | 分割 | 追踪 | 想证明什么 |
|---|---|---|---|
| B0 | **GT SEG（oracle）** | 我的追踪器 | 追踪上限，把分割误差和关联误差分开 |
| B1 | 阈值 + watershed | IoU | 传统下限 |
| B2 | Cellpose 零样本 | IoU | 强现成 baseline |
| M1 | 微调 U-Net | IoU | 分割的贡献（B2→M1） |
| M2 | 微调 U-Net | + gap closing | 关联/漏检补救的贡献 |
| M3 | 微调 U-Net | + 分裂检测 | 谱系的贡献（TRA 涨、SEG 不变，最漂亮的一组） |
| M4 | 微调 U-Net | + 恒速预测 | 快速运动的贡献（选做） |
| X1 | 最佳配置迁到 SIM+ / U373 | — | 泛化性 |

每行都报 `DET / SEG / TRA / OP_CTB`（01 和 02 分开报 + 平均）。B0 那一行基本能拿到 ~0.99 的 TRA，是"追踪器本身没问题、掉分来自分割"的有力证据，答辩很好讲。

---

## 6. Notebook 结构（对着 15 分逐条给分）

```
1. Introduction & Motivation            → 1 分（为什么细胞追踪重要：癌症转移/免疫/药物筛选，引 2–3 篇）
2. Data Sources                         → 1 分（数据集 URL、许可、帧数、分辨率、下载方式）
3. Exploratory Data Analysis            → 3 分（拿满的关键）
   3.1 序列长度 / 图像尺寸 / 位深 / 每帧细胞数分布（直方图）
   3.2 细胞面积、形状、位移速度分布（箱线图）；细胞密度随时间曲线
   3.3 分裂事件统计：每帧分裂数、轨迹长度分布（从 man_track.txt 的 parent 列直接算）
   3.4 挑战点可视化：稠密区、粘连、消失再出现、silver 标注噪声示例
4. Methods                              → 3 分（分割网络结构图 + 追踪算法伪代码 + 损失函数 + 超参表）
   明确写：哪些是 Cellpose 预训练权重、哪些是你们改的
5. Experiments & Results                → 3 分（第 5 节那张表 + 训练曲线 + 预测可视化 + 逐帧 TRA/SEG 曲线）
6. Discussion                           → 2 分（消融解释了哪一项贡献多少、失败案例、silver SEG 的噪声影响、局限）
7. Conclusion & Future Work
附录：完整代码（一个 notebook 内，用 section 折叠，不要拆成多个文件提交）
```

写作 2 分（markdown 标题层级、代码可读、图表有 caption）建议一开始就按上面的编号写死，别最后补。

---

## 7. 4 页报告怎么写（5 分）

| 部分 | 分值 | 页数预算 | 内容 |
|---|---|---|---|
| Abstract + Keywords | — | 0.2 页 | 方法一句话 + 关键数字（TRA/SEG 提升） |
| Introduction | 0.5 | 0.4 页 | 问题、动机、贡献列表（3 条 bullet） |
| Related Work | 0.5 | 0.4 页 | Cellpose/StarDist、U-Net 分割、CTC 基准（Maška 2023、Ulman 2017）+ AOGM（Matula 2015） |
| Methods | 0.5 | 0.8 页 | 分割 + 追踪 + 分裂检测，公式只留 IoU 和 AOGM 概念 |
| Experiments | 0.5 | 0.5 页 | 数据集、split、超参、指标定义 |
| Results | 0.5 | 0.8 页 | 主表 + 1–2 张图 |
| Conclusion | 0.5 | 0.4 页 | 强项、局限、future work |
| References | — | 不限 | 6–10 篇 |

"细节充分性 1 分"和"整体质量 1 分"是白送的：**用官方模板（Overleaf/Word 的 IEEE 样式）**、图有编号和 caption、数字和 notebook 里完全一致、不做语法错误。这 2 分比 Results 那 0.5 分好拿多了。

---

## 8. 展示（8 分，按人计分）

- 个人 3 分 = 贡献 1.5 + 表达 1.5 → **5 个人都要讲**，讲自己做的部分，别让一个人讲完。
- 小组 5 分 = 问题陈述 1 + 数据/EDA 1 + 方法 1 + 结果与误差分析 1 + 幻灯片质量 1。
- 用课程给的 `COMP9444_Presentation_TeamName.pptx` 模板。
- 建议结构（10–12 分钟）：问题与动机 1' → 数据与 EDA 2' → 分割方法 2' → 追踪与分裂 2' → 结果表 + 消融 2' → 失败案例 1' → 分工与总结 1'。
- 误差分析一定放：一张"漏检帧"和一张"错误分裂边"的可视化，比多讲两个模型都加分。

---

## 9. 5 人分工（互不阻塞）

| 角色 | 主责 | 交叉 review |
|---|---|---|
| A 数据/EDA | 下载、格式转换、EDA 全部图表、split 定义 | 报告 Experiments 段 |
| B 分割 | A0/A1/A2 三个分割器，产出统一格式 mask 目录 | 结果表的 SEG 列 |
| C 追踪 | IoU/匈牙利/gap/分裂/恒速，产出所有 RES | 结果表的 TRA 列 |
| D 评测/可视化 | 跑 `ctc_evaluate`、汇总 CSV、训练曲线、轨迹图、视频 | 全部数字复核 |
| E 报告/展示 | 4 页报告 + 幻灯片 + 合并 notebook | 时间线推进 |

约定（避免最后一晚合并爆炸）：
- **统一目录约定**：`data/<dataset>/<seq>{,_GT}`、`runs/<tag>/<seq>{,_RES}`、`results/<tag>.csv`。
- notebook **从一开始就只有一个**，用 git 分节 + 每次改完立刻推送，或统一用 `jupytext` 的 `.py` 版本做 git 冲突管理。
- 每天 10 分钟同步：谁阻塞了、明天的表格哪一行能填上。

---

## 10. 时间表（现在已经是监控期内，压缩到 4 周）

今天 10/27，week 7–9 的 monitoring sessions 就是接下来这几周，所以别再"先学理论"。

| 时间 | 交付 |
|---|---|
| 第 1 周 | 数据下好（校园网）+ 跑通我这套 scaffold（B0 行出数）+ EDA 3.1–3.3 图表 + notebook 骨架按第 6 节写死 |
| 第 2 周 | Cellpose 零样本（B2）+ 微调 U-Net（M1）+ 统一 mask 目录；报告 Related Work 初稿 |
| 第 3 周 | 全消融（M2/M3/M4）+ 第二数据集（X1）+ 误差分析图 + 报告初稿 + 幻灯片骨架 |
| 第 4 周 | 数字定稿、notebook 合并、报告压到 4 页、演练 2 次 |

每个 monitoring session 前准备：一张当前结果表（哪几行已有数、哪几行空）+ 一张下周计划。参与分 3 分是按"有没有推进"给的，空表也去讲，但要讲清卡在哪。

---

## 11. 坑清单（按踩坑概率排序）

1. **拿 challenge set 当测试集**：测试集没有 GT，只有本地 training 序列能评分。官方 leaderboard 要注册 + FTP 上传 + 交命令行程序，课程项目**不需要**走这条路。
2. **帧编号 0 基 / 1 基不统一**：官方指标代码按文件名数字索引掩膜，1 基序列会整体错一帧。`src/evaluate.py` 已经在临时目录里统一成 0 基，别自己再手改文件名。
3. **silver SEG 的噪声**：真实数据集的分割标注是 silver（官网脚注 ✱ 说明），所以 SEG 分数有天花板，别把 0.02 的 SEG 差异当结论；写进 Discussion 反而是加分项。
4. **只调 IoU 阈值不看分裂**：TRA 掉分常常全是分裂边错的（每处 1.5）。先跑 `--no-divisions` 对比，一眼看出谱系贡献。
5. **过分割/欠分割方向选错**：FN 权重 10 远大于 FP 权重 1，宁可稍微过分割。
6. **数据进 git**：CTC 明令禁止转载，`.gitignore` 加 `data/`、`runs/`、`*.zip`。
7. **多个 notebook 提交**：rubric 直接写禁止，最后合并极易出错，早期就单文件。
8. **报告超 4 页**：超页内容不计分，不是扣分是直接不看。
9. **组信息没登记**：见第 2 节，先去确认项目 ID 已登记为 015。

---

## 12. 已经给你准备好的代码（都跑过）

```
README.md                 用法
requirements.txt          numpy / scipy / tifffile / py-ctcmetrics
src/ctc_io.py             CTC 格式读写 + 一致性检查（res_track.txt ↔ mask 标签对齐）
src/track.py              IoU+匈牙利 + gap closing + 分裂检测
src/run_track.py          mask 目录 → <seq>_RES（也可 --from-gt 做 oracle 上限）
src/evaluate.py           <seq>_GT + <seq>_RES → DET/SEG/TRA/OP_CTB
tests/test_pipeline.py    合成序列端到端测试
```

验证结果（本机实跑）：

```
0-based names: DET=1.0000 SEG=1.0000 TRA=1.0000
1-based names: DET=1.0000 SEG=1.0000 TRA=1.0000
pipeline: ok (perfect masks + correct lineage -> SEG=DET=TRA=1.0)
```

也就是说：**分割给得准的前提下，追踪器和评测链路是通的**，剩下的分数全靠你们把分割做好、把对比做全、把故事讲清楚。

---

## 13. 还没确认 / 需要你们决定的事

1. **due date 与展示日期**（WebCMS 上无 due date，需查 Moodle/公告）。
2. **组信息是否已登记为 015**（schedule 里那三栏是空的）。
3. 第二数据集用 `SIM+`（干净但合成）还是 `U373`（真实但细胞少）——取决于你们想要"泛化/模态差异"还是"快速迭代"。建议 SIM+，因为它的完美分割标注可以做"分割误差 = 0"的上限实验。
4. 有没有 GPU。没有的话：Cellpose 零样本 + 微调小 U-Net（256² 裁剪、~10 epoch）在 CPU 上会非常慢，优先用 Colab/Kaggle 免费 GPU，或只做 HeLa 一个数据集 + 更充分的消融。
