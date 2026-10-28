
COMP9444 Project 015 Cell Tracking 分工方案
Oct 16, 2026 ·
总体思路
评分标准奖励的是问题驱动，而不是新架构：EDA 占 3 分，和 Models/Methods 一样多；Methods 要求标清楚哪些是 own work；Results 要求和 SOTA 对比。
Scope：只做 Fluo-N2DL-HeLa 一个 dataset，只做 2D。它细胞密集、division 多，tracking 部分才有东西可比。
Baseline 直接用开源 SOTA：zero-shot Cellpose 或 StarDist 做 segmentation，Hungarian matching 做 tracking。这本身就满足 "comparison with SOTA"。
每人认领一类错误：用 EDA 和 error analysis 找出 baseline 在 HeLa 上具体错在哪，每人针对一类错误做改进，用 ablation 证明有效。
改进互相独立：每个改进单独接在 baseline 上，某一个没做成不影响其他人；做了但没提升，把原因分析清楚，同样能写进 Discussion。
训练 4 个 model：fine-tuned Cellpose、temporal refinement、LSTM tracker、mitosis detector。
接口第一周定死：mask 统一用 CTC 格式（16-bit TIFF，每个细胞一个 label），tracker 读 mask、输出 res_track.txt。
HeLa 上的难点与错误类型
CTC 的 TRA 指标基于 AOGM，会把错误拆成几类操作分别计数，每类错误正好对应一位负责人。下表是预期的难点，需要 #1 用 EDA 验证和量化。
难点
对应的 AOGM 错误
负责人
细胞密集，相邻细胞核粘连被合并成一个
NS
#2、#3
mitosis 时细胞核变亮变小，外观突变，分割出错
FN / FP
#2
某一帧漏检，导致 track 断开
FN，以及随之缺失的连接 EA
#3
密集区域连错细胞（ID switch）
EA / ED
#4
division 后 parent→daughter 关系漏掉或连错
EC，以及 division edge 上的 EA
#5
缩写：FN 漏检，FP 多检，NS 需要拆开的合并检测，EA 缺失的连接，ED 多余的连接，EC 连接类型错误（比如把 division 当成普通延续）。
角色总览

baseline pipeline · 3 个阶段，4 个改进模块
每个改进模块只替换或增加 baseline 的一步，所以可以单独做 ablation。额外职责：#1 整合 notebook；#2 整合 report，并讲 presentation 的结尾总结；#3 统一 slides 模板；#4、#5 技术工作最重，不承担整合工作。
每人详细分工
#1：EDA、Evaluation、Notebook 整合
技术：解析 CTC 数据；做 EDA；搭 eval（SEG / DET / TRA + AOGM error breakdown）；对 baseline 做 error analysis；把 tracks 可视化成视频
Notebook：§1 Introduction、§2 Data Sources、§3 EDA、§5.1 主结果表、§6 中的 error analysis 可视化；统一整个 notebook 的结构、markdown 标题和代码风格
Report：Introduction（约 0.3 页）；Experimental Setup 里的 data split 和 metrics；Results 的主表和一段总述
Slides（3 张，开场）：problem statement；dataset 和 EDA 发现的难点；baseline 的 error breakdown，每类错误分别引出后面一位讲者
产出的图：pipeline 概览图（report 和 slides 共用）；四张 EDA 图（密度随时间变化、相邻帧位移和细胞核大小的比例、division 频率、mitosis 前后亮度变化），每张请对应模块的负责人确认；error breakdown 柱状图
#2：Segmentation fine-tune + Report 整合
技术：用 Cellpose 或 StarDist 的 zero-shot 结果做 baseline，再 fine-tune：对密集区域和 mitotic 细胞的 crop 过采样；对比只用 GT 训练和 GT + ST silver truth 训练
Notebook：§4.1 Segmentation、§5.2 segmentation ablation
Report：Lit review 中 segmentation 的部分（U-Net、Cellpose、StarDist）；Methods 的 segmentation 小节（约 0.25 页）；Results 中一段分析；起草 Conclusions；整合全文并控制在 4 页以内
Slides（2 张 + 结尾 1 张）：密集区域和 mitotic 细胞的分割错误示例 → fine-tune 策略 → 按区域拆分的 SEG / DET 对比；结尾一张讲主结果表、takeaways 和 limitations
产出的图：zero-shot 和 fine-tuned 在密集区域的 mask 对比图；按区域（密集 vs 稀疏、mitotic vs interphase）拆分的 SEG / DET 表
#3：Temporal refinement + Slides 统一
技术：用 mask propagation 或 ConvLSTM，把上一帧的图像和 mask 作为额外输入，修复单帧漏检和细胞粘连
Notebook：§4.2、§5.3
Report：Lit review 中 ConvLSTM 和 Arbelle & Raviv（ISBI 2019）的部分；Methods 的 temporal 小节（约 0.25 页）；Results 中一段分析
Slides（2 张）：一个因单帧漏检导致 track 断开的例子 → 用上一帧信息修正的方法 → 修正前后的 FN、NS 对比
额外职责：确定 slides 模板，统一配色、colormap 和字体，这直接关系到 "quality of slides" 那 1 分
产出的图：修正前后的示例帧；FN / NS 数量的柱状图
#4：Association（Hungarian + LSTM）
技术：先写 Hungarian baseline（cost 用 IoU 或 centroid distance），它是全组 baseline 的一部分，要最早交付；再做 LSTM tracker（motion + appearance），算出的分数作为 cost，接回同一个 Hungarian
Notebook：§4.3、§5.4
Report：Lit review 中 u-track（Jaqaman et al., 2008）和 Milan et al.（2017，RNN-based MOT）的部分；Methods 的 association 小节；Results 中一段分析
Slides（2 张）：密集区域 ID switch 的示例 → LSTM tracker 结构图 → EA / ED 数量和 TRA 的对比
产出的图：LSTM tracker 结构图；修正前后的 track 可视化
#5：Division（mitosis detector）
技术：把连续几帧的局部 patch 叠起来输入小 CNN，判断是否发生 division，并建立 parent→两个 daughter 的连接
Notebook：§4.4、§5.5
Report：Lit review 中能处理 division 的 learned tracker（例如 Trackastra，Gallusser & Weigert, ECCV 2024）；Methods 的 division 小节；Results 中一段分析
Slides（2 张）：division 漏检的示例 → detector 的输入和输出 → division precision / recall 和 EC 数量的对比
产出的图：mitosis patch 序列示例（判断正确和错误的各几例）；precision / recall
所有人共同负责
Discussion：每人写自己模块的 strengths、limitations、future work，各 1–2 句。notebook 里由 #1 整合，report 里由 #2 整合
Hyperparameters：每人提供自己 model 的 learning rate、epochs、batch size、input size 等，汇总成 Experimental Setup 里的一张表
交叉校对：每人校对下一位负责的部分，#5 校对 #1
Notebook 结构
规定只能交一个 notebook：开发时每人在自己的 notebook 里做，按约定的 CTC 格式接口输入输出，最后由 #1 合并。
Section
内容
负责人
1
Introduction & Problem Statement
#1
2
Data Sources
#1
3
Exploratory Data Analysis
#1
4.0
Pipeline overview & baseline
#1 / #2 / #4
4.1
Segmentation fine-tuning
#2
4.2
Temporal refinement
#3
4.3
Learned association (LSTM)
#4
4.4
Mitosis detection
#5
5.1
Main results table
#1
5.2–5.5
各模块 ablation
#2–#5
6
Discussion & error analysis
#1 整合，每人贡献
Report 页数分配（上限 4 页）
合计约 3.8 页，留一点余量。图最多放两张：pipeline 图和一张最有代表性的 error analysis 图，其余的图留给 notebook 和 slides。
Section
页数
负责人
Introduction
~0.3
#1
Literature Review
~0.5
每人写自己模块 2–3 句，#2 整合
Models / Methods
~1.2
#1 写 pipeline 概述并配图；#2–#5 各约 0.25 页
Experimental Setup
~0.4
#1 写 data split 和 metrics；hyperparameter 表由每人提供
Results
~1.1
#1 写主表和总述；#2–#5 各一段 ablation 分析
Conclusions
~0.3
#2 起草，每人贡献一条 limitation
主结果表设计
中间四行每行只在 baseline 上加一个模块，不做累加，这样每个人的贡献可以单独归因；最后一行是全部模块合在一起的结果。
设置
SEG
DET
TRA
Baseline（zero-shot Cellpose + Hungarian）
+ fine-tuned segmentation（#2）
+ temporal refinement（#3）
+ LSTM association（#4）
+ mitosis detector（#5）
Full system
Data split：HeLa 的 training data 只有 01、02 两个 sequence。用 01 训练、02 测试，再交换做一次，形成 2-fold，report 的 Experimental Setup 里写清楚。
和 SOTA 对比的口径：CTC leaderboard 的分数是在不公开的 test set 上算的，我们在 training sequence 上评测，两者不能直接比，report 里说明一下。
Presentation 顺序
每人都有一段完整的"问题 → 方法 → 结果"，对应个人分里的 visible contribution。slides 数量按每人 2–3 张估算，具体按课程规定的时长调整，尽量让每人讲话时间相近。
顺序
讲者
内容
slides
对应的 group 评分项
1
#1
problem、dataset、EDA、baseline 的 error breakdown
3
problem statement、data exploration
2
#2
segmentation fine-tune
2
methods
3
#3
temporal refinement
2
methods
4
#4
association（Hungarian + LSTM）
2
methods
5
#5
division（mitosis detector）
2
methods
6
#2
主结果表、takeaways、limitations
1
results / error analysis
slides 模板、配色和图表风格由 #3 统一。
时间线

时间线 · 4 个阶段，3 次 monitoring session
三次 monitoring session 合计最多 3 分，按个人计分，所以每次 session 时每人都要有能展示的进度。最关键的是 week 7 前跑通 baseline（#2 的 zero-shot + #4 的 Hungarian + #1 的 eval），拿到 error breakdown 后大家才知道各自的改进空间。
