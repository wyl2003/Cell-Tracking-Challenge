# IoU + Hungarian Baseline：实验与优化对比

本报告保留 Original Baseline、Optimized Baseline 和相关消融。没有训练新网络；不包含 Trackastra 或 ILP 实验。实验完成于 2026-11-05，代码整理及搬移后复查于 2026-11-06。

## 1. 来源与贡献边界

- 原始基线：Wang Yile / [wyl2003/Cell-Tracking-Challenge](https://github.com/wyl2003/Cell-Tracking-Challenge)，基线提交 `4022d4ea75b927bae90a47663e3903c0bba8db2b`。
- 后续追踪研究负责人：ZIHAO LI（GitHub：tuzzzzi）。代码实现、实验运行和分析整理使用 AI 辅助，汇报时按课程要求披露。
- Original Baseline：直接调用仓库原有的 `src/track.py`，该文件及其历史提交保留。
- Optimized Baseline：新增 `src/baseline_tracking/track.py`；输入输出和评分适配放在同一子目录，避免改变组员现有命令的行为。
- 修改范围：新增基线子模块、测试、配置、结果和本报告；现有文件只追加 README 入口与忽略规则。

## 2. 实验设置

数据为 Fluo-N2DH-GOWT1 的 01、02，各 92 帧。先在 Mac 上复现原版分割和追踪，再固定本机保存的 Cellpose 分割结果比较不同追踪设置。各方法逐实例检查区域不变，避免把分割差异计为关联改进。

本机分割调用保持组员原参数；Cellpose 4.2.1.1 忽略 `model_type="cpsam"`，实际权重为默认 `cpsam_v2`。分割阶段为 Apple MPS / torch 2.15.0。追踪对比在同一 CPU 环境中运行，具体版本见 [环境记录](baseline_tracking/environment-lock.txt)。

核心参数：IoU 阈值 0.20，max_gap=0，子细胞覆盖比例阈值 0.50，两个子细胞总面积与母细胞面积比允许 0.50～1.80。六组完整参数见 [configurations.json](baseline_tracking/configurations.json)。这是实验时的配置记录；benchmark 中设置与其对应，命令行 `--max-distance` 可覆盖运动距离参数，实际配置仍写入每次输出。

## 3. 组员报告与本机 Original Baseline 复现

| 序列 | 来源 | TRA ↑ | CT ↑ | TF ↑ | 轨迹数 |
|---|---|---:|---:|---:|---:|
| 01 | 组员原报告 | 0.982100 | 0.550700 | 0.947000 | 41 |
| 01 | 本机原版复现 | 0.982143 | 0.550725 | 0.946986 | 41 |
| 02 | 组员原报告 | 0.957400 | 0.298500 | 0.914200 | 76 |
| 02 | 本机原版复现 | 0.957580 | 0.285714 | 0.898355 | 82 |

01 的主要结果接近，02 轨迹数 76 对 82，不能宣称严格一致。组员使用 RTX 3060 / torch 2.14.1+cu126，本机使用 MPS / torch 2.15.0；缺少组员原始 masks 与权重校验，无法确定各个差异的来源。这部分差异不作为优化效果。原始逐项对比见 [baseline_comparison.json](baseline_tracking/baseline_comparison.json)。

## 4. 固定输入：Original 与 Optimized

| 序列 | 方法 | TRA ↑ | CT ↑ | TF ↑ | 轨迹数 | 正确分裂 ↑ | 误报分裂 ↓ | 追踪秒数 ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 01 | Original Baseline | 0.982143 | 0.550725 | 0.946986 | 41 | 0 | 0 | 17.02 |
| 01 | Optimized Baseline（保留分裂） | 0.982143 | 0.550725 | 0.946986 | 41 | 0 | 0 | 7.31 |
| 01 | Optimized（关闭分裂） | 0.982143 | 0.550725 | 0.946986 | 41 | 0 | 0 | 7.67 |
| 02 | Original Baseline | 0.957580 | 0.285714 | 0.898355 | 82 | 0 | 9 | 26.92 |
| 02 | Optimized Baseline（保留分裂） | 0.957580 | 0.285714 | 0.898355 | 82 | 0 | 9 | 8.94 |
| 02 | Optimized（关闭分裂） | 0.957857 | 0.305344 | 0.905602 | 73 | 0 | 0 | 8.93 |

- 保留原重叠分裂规则时，两段准确性指标与本机 Original 一致。单次追踪计时 01 为 17.02→7.31 秒（约 2.3 倍加速），02 为 26.92→8.94 秒（约 3.0 倍加速）。
- 02 关闭分裂后：误报 9→0，TRA 0.957580→0.957857，CT 0.285714→0.305344，TF 0.898355→0.905602。关闭分裂是原版已有开关，这里验证其影响，不是新发明的分裂模型。
- 所有方法正确分裂均为 0；关闭分裂只减少误报，未提高分裂召回。
- 时间只包含追踪计算，不含分割、文件读写和评分，是本机单次测量，不是稳定性能或跨硬件保证。未单独测量每项修改的加速贡献。

## 5. 匈牙利匹配前后改了什么

仍使用 SciPy 的 `linear_sum_assignment`，没有修改匈牙利算法本身。

| 环节 | Original | Optimized |
|---|---|---|
| IoU 计算 | 逐对细胞重复扫描图像 | 集中统计标签交集，再计算 IoU，减少重复扫描 |
| 合格候选 | 全部候选先匹配，再过滤低于阈值的连接 | 匹配前屏蔽低于阈值或无重叠的候选 |
| 不匹配 | 矩形分配或事后过滤留下未匹配对象 | 增加代表不匹配的虚拟列 |
| 目标 | 先优化总 IoU，再过滤 | 先最大化合法配对数，再在配对数量相同时最大化得分 |

示例：阈值为 0.20，暂不考虑分裂。

| 得分 | C | D |
|---|---:|---:|
| A | 0.90 | 0.50 |
| B | 0.40 | 0.19（不合格） |

Original 先选择 A→C、B→D（总分 1.09），再删除 B→D，只剩一个有效配对。Optimized 先排除 B→D，再选择 A→D、B→C，得到两个合法配对。这是机制示例，真实两段默认设置没有观察到准确率提升。更多合法配对也不保证每条都是真实连接，仍需用官方指标验证。

原版分裂规则在一对一匹配后寻找第二个未被占用的子细胞，要求两个子细胞各有至少 50% 面积与上一帧母细胞重叠，且总面积比例在范围内。Optimized 默认保留这类重叠规则；输入缺少子细胞时，关联器无法凭空恢复真实分裂。

## 6. 可选运动与几何分裂消融

运动预测采用恒速度：预测位置 = 当前位置 +（当前位置 − 上次位置）。hybrid 得分为 0.6×平移后 IoU + 0.3×exp(−距离/30) + 0.1×面积相似度，包含距离和面积门槛，参数为手工设置而非训练。新的 motion 分裂规则依据预测位置、面积、子细胞平衡性及加权中心筛选。

| 序列 | 可选方法 | TRA ↑ | CT ↑ | TF ↑ | 误报分裂 ↓ | 追踪秒数 ↓ |
|---|---|---:|---:|---:|---:|---:|
| 01 | 运动预测，关闭分裂 | 0.982143 | 0.550725 | 0.946986 | 0 | 11.89 |
| 01 | IoU＋几何分裂 | 0.982164 | 0.542857 | 0.946986 | 1 | 7.32 |
| 01 | 运动预测＋几何分裂 | 0.982164 | 0.542857 | 0.946986 | 1 | 12.19 |
| 02 | 运动预测，关闭分裂 | 0.957857 | 0.303030 | 0.905602 | 0 | 14.25 |
| 02 | IoU＋几何分裂 | 0.957857 | 0.303030 | 0.905602 | 1 | 8.82 |
| 02 | 运动预测＋几何分裂 | 0.957857 | 0.300752 | 0.905602 | 1 | 14.48 |

运动预测没有稳定提升，还增加耗时；新几何分裂在 01 增加一个误报并降低 CT。因此它们保留为消融，不作为默认最佳方案。

## 7. 指标与限制

TRA 衡量检测与连接的综合正确性；CT 衡量完整轨迹恢复；TF 衡量真值轨迹中被正确恢复的连续片段比例，取最佳片段后平均。三者越高越好，轨迹数量本身没有越多或越少越好的含义。

这两段属于已有公开序列上的探索性实验，不是独立盲测。分裂输入诊断显示，6 次真实分裂在标注子细胞首次出现时均至少缺失一个子细胞；覆盖诊断不替代官方分裂评分。下一步应区分分割漏检和关联错误，再研究延迟子细胞连接。

原版 gap closing 会产生缺少中间帧 mask 的轨迹；优化版明确拒绝 max_gap≠0，未声称实现补帧。评估使用完整序列和 CTC 合法输出检查。

## 8. 运行与验证

安装与追踪测试：

```bash
python -m pip install -r requirements-baseline.txt
python -m pytest tests/test_baseline_tracking.py tests/test_baseline_tracking_metrics.py -q
python tests/test_pipeline.py
```

对同一份分割运行六组设置（输出目录必须不存在），两个序列分别运行：

```bash
python -m src.baseline_tracking.benchmark --masks runs/cpsam/01 --gt data/Fluo-N2DH-GOWT1/01_GT --out runs/baseline_comparison_01
python -m src.baseline_tracking.benchmark --masks runs/cpsam/02 --gt data/Fluo-N2DH-GOWT1/02_GT --out runs/baseline_comparison_02
```

单独运行 Optimized 默认设置：

```bash
python -m src.baseline_tracking.run_track --masks runs/cpsam/01 --out runs/optimized_01_RES --iou 0.2
python -m src.baseline_tracking.diagnostics --gt data/Fluo-N2DH-GOWT1/01_GT --res runs/optimized_01_RES
```

benchmark 会保存每组配置、原始评分 JSON、summary.csv、输入与源码哈希及追踪输出。发布记录位于 [baseline_tracking/](baseline_tracking/)；evidence/ 中保存历史逐方法评分与输入哈希，机器私有绝对路径已替换。历史源码哈希对应搬移前文件，模块路径和文档字符串变更后的源码应以本分支 Git 提交及新运行 manifest 为准。

验证：21 项追踪测试通过；原组员流程测试在 0/1 起始帧命名下均 DET=SEG=TRA=1。搬移后的两序列六组真实数据复查记录见 [validation.json](baseline_tracking/validation.json)。

数据集、原始 mask、权重、虚拟环境及缓存均不在提交中；本次未测试 Windows 的完整依赖安装。
