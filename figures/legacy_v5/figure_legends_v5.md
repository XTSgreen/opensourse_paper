# 图 1–5 配套图注与统计说明

本文件对应 `figures_revision_v5` 中的成图。图内保留整图标题、子图主标题、坐标轴与常规图例；灰色副标题和较长的统计说明归入以下图注。

## 图 1　IOT framework for cell-state inference and prediction

公开单细胞与谱系观测经状态和特征构建、分析设计后，进入三个分别定义估计对象的任务。上行 UOT-IOT 以源、目标快照与目标态特征为输入，固定源边际，对目标边际施加 KL 锚定，估计转移代价参数与传输计划，并输出曲率、多重启和灵敏度诊断；通过合成真值、治疗时序及参数冻结后的外部转移队列评价。中行 GH-IOT 将参数分解为共享项与上下文偏移，以收缩和门控组织上下文信息，输出谱系历史对比与患者分支得分，通过留组评价、全流程置换、ARTEMIS、GDSC 和 PRRX1 扰动数据检验。下行 PERSIST-IOT 在预测时读取源端状态、细胞数、条件和时间间隔，将前向 IOT 生成器与观测校正 HMM 结合，分别输出未来状态组成和持续／检出概率；组成以交叉熵、主导状态命中率及校准评价，持续检出以 Brier 和 AUPRC 评价。箭头表示各任务内部的数据处理和评价顺序。

路线图中的方法关系依据正文的代价参数化、层级上下文与前瞻层方法部分重写。图中表达式为概念性简写，完整目标函数和假设以正文方法为准；图 1 为技术示意图，不含模拟生成的数值结果。

## 图 2　Identifiability and external validation

**a，Target-state curvature。** GSE228154 中五个目标态特征方向的经验 Hessian 曲率。横轴为以 10 为底的对数，空心灰圆和实心蓝圆分别表示平衡 OT 与半松弛 IOT，同一特征的两种结果以线连接；直接标注为半松弛 IOT 的原尺度曲率。灰色区域对应曲率不超过 10⁻¹⁰ 的范围。五个特征为增殖、EMT、干性、衰老和上皮特征，每个点表示一个方向的数值诊断，不代表一个生物学重复。

**b，Coefficient recovery。** 八状态合成系统中八个代价系数的真值与恢复值，横轴为真值，纵轴为恢复值，虚线为一致性线；灰色箭头连接同一系数在两种方法下的结果。红色外圈标出五个纯列特征方向。相应方向的 Pearson 相关系数为半松弛 IOT 1.000、平衡 OT −0.020。右侧逐系数显示绝对恢复误差，使用同一套方法颜色与填充编码；该部分由已有真值与估计值直接相减得到。

**c，KL-anchor sensitivity。** 八状态系统中纯列方向灵敏度范数随目标边际惩罚 μ 的变化，展示源文件中的全部八个扫描点。星形对应工作点 μ=0.5，灵敏度范数约为 0.6224；浅蓝色区域对应 μ=0.1–1.0。原参数扫描报告的方向排序 Spearman 相关系数不低于 0.90，该稳定性说明与灵敏度曲线分别解释。

**d，Unseen metastatic sites。** 在 GSE163558 训练后冻结参数及标准化规则，比较两个外部队列六个未见部位相对于独立边际基线的重建 MAE 改善。点为源结果记录的平均绝对改善，线为各部位 200 次 bootstrap 得到的 95% 区间；百分数为相对 MAE 降幅。GSE246662 包含 LM1–LM3，GSE183904 包含 s19、s20、s38。六个部位的区间均高于零，相对误差降低约 64%–82%；六个部位均为正的精确二项检验 P=0.031。

## 图 3　Context-dependent performance and lineage history

**a，Performance across public tasks。** 十种方法在 macsGESTALT、耐药时序和外部谱系三个条件状态组成任务中的名次，评价指标为 DM-NLL／细胞，名次 1 为最佳。圆、方和三角标记区分任务，纵向微偏移用于分开重合点；水平细线表示每种方法的最低至最高名次，黑色菱形及右侧数值表示三个任务的平均名次。紫色沿用原图的 IOT 家族方法标识，浅紫色区域表示前三名。三个任务为比较单位，图中名次范围不作为统计置信区间；全局门控配置的平均名次为 1.33。

**b，Lineage-specific direction shifts。** GSE171940 中每只小鼠从上皮谱系到 ever-EMT 谱系的侵袭型 EMT 分支得分配对变化，TENA 为五只小鼠、Cdh 为三只小鼠。小标记和箭头表示配对个体，大菱形为各系统均值。两个系统的方向向量余弦相似度为 −0.796；保留预处理、先验构建与外层留组顺序的 1,000 次单侧全流程历史标签置换分别得到 P=0.01299 和 P=0.02997。

**c，Branch-level history effects。** 六个预设分支中 ever-EMT 谱系减上皮谱系的配对平均差异。细线与端帽为配对 t 方法的 95% 置信区间，粗线为均值 ±1 标准误，圆与方分别表示 TENA 和 Cdh；实心对应原配对检验 P<0.05，空心对应 P≥0.05。该组 P 值沿用源文件的未校正配对检验，精确数值列于文末统计表。统计单位为小鼠，TENA 的自由度为 4，Cdh 的自由度为 2。

## 图 4　Patient programmes and experimental evidence

**a，IFN/HLA and response；b，Invasive EMT and response。** ARTEMIS 队列中 79 名具有可用肿瘤数据的患者，RD 为 34 人、pCR 为 45 人。每个小点表示一名患者，半小提琴表示组内得分密度，空心菱形与误差线表示患者均值及源文件记录的 10,000 次患者 bootstrap 95% 区间。IFN/HLA 的 pCR−RD 均值差为 +0.2604，BH q=0.01219，AUROC=0.6967；侵袭型 EMT 的均值差为 +0.002970，BH q=0.82078，AUROC=0.53399。

**c，Branch specificity of treatment response。** 六个分支的患者级 pCR−RD 平均差异及 bootstrap 95% 区间，BH q 沿用原始统计文件的校正结果与检验族。IFN/HLA 的完整区间为 [0.10354, 0.40374]，成图已将其上端完整纳入横轴范围。红色突出 IFN/HLA，其余分支采用灰色。

**d，EGFR-TKI response in GDSC。** 五种 EGFR-TKI 的细胞系 EMT 得分与 ln(IC50) 的 Spearman 相关，点为相关系数，行内数字为系数及各药物可用细胞系数量；虚线为五个系数的中位数 0.2535。各药物 n 为 513–520，均 P<10⁻⁵，精确 P 值列于文末统计表。各药物中的细胞系可有重叠，五个药物结果分别报告。

**e，Paired PRRX1 perturbation。** GSE164488 的 MDCK-NBL2 细胞在第 4 天、TGFβ 条件下比较 siPRRX1 与 siCTR。按生物学重复编号配对，使用全部三个重复；半透明小点为每个配对重复的分支得分差，菱形为配对差的平均值，误差线为平均值 ±t₀.₉₇₅,₂×SE。红色实心菱形表示原配对 t 检验 P<0.05，灰色空心菱形表示 P≥0.05。新增的重复层展示及区间由原始样本特征表计算，六个分支的均值和配对 P 均复现已有结果。Epithelial 为 P=0.02063，Invasive EMT 为 P=0.01601；全部结果列于文末统计表。

## 图 5　Future-state composition and persistence prediction

图 5 采用三行组合，第一行显示组成性能，第二行显示校准，第三行显示持续检出。当前 d 为状态偏差、e 为 Brier 差、f 为 AUPRC。

**a，Future-state cross-entropy；b，Dominant-state agreement。** 比较 PERSIST-IOT 与去传输对照在开发集五折外层 OOF 和锁定外部 E1 的未来状态组成表现。开发集状态交叉熵从 4.8509 降至 1.0370，外部 E1 从 4.5831 降至 0.9859，相对下降分别为 78.6% 和 78.5%；主导状态命中率从 49.70% 增至 64.43%、从 47.50% 增至 68.20%，分别增加 14.7 和 20.7 个百分点。组成指标对应目标端检出的克隆与时间窗预测案例，开发集为 4,867，外部为 7,267，重复出现的克隆和时间窗按原评价方案处理。

**c，Predicted vs observed composition。** 外部 E1 的 DMSO 和 ispinesib 各包含 28 个预测时间窗、六种状态，共 336 个条件／窗口／状态条目；全部条目均绘制。半透明点为窗口层组成，点大小随该窗口预测案例数变化，两条件使用同一大小映射；黑边大标记为各条件下按窗口等权计算的六个状态均值。圆形表示 DMSO、方形表示 ispinesib，虚线为预测与观测一致。插图放大低丰度区域，位置经过数据坐标核对，未覆盖主图观测点。

**d，State-specific calibration bias。** 偏差定义为观测组成减预测组成。蓝圆、橙红方形分别表示 DMSO 和 ispinesib 的 28 窗口平均偏差；灰色菱形与区间为合并两条件共 56 个窗口的平均偏差及 1,000 次窗口 bootstrap 的 95% 区间，沿用 v4 的随机种子和计算方式。合并区间排除零时菱形填实。分条件点为描述性均值，未附加新的显著性检验。窗口 bootstrap 描述既有窗口集合中的变化，重叠时间窗不作为独立生物学重复解释。

**e，Persistence detection (Δ Brier)。** PERSIST-IOT 减各分区最佳监督模型的 Brier 差，负值表示 PERSIST-IOT 的 Brier 更低。开发集差异为 +0.004815，95% 区间 [0.002691, 0.007193]；外部 E1 为 −0.001279，95% 区间 [−0.002188, −0.000381]。区间沿用已有 10,000 次配对克隆 bootstrap，开发集比较对象为 Extra Trees，外部为 HistGradientBoosting。

**f，Persistence detection (AUPRC)。** PERSIST-IOT 在开发集及外部 E1 的 AUPRC 分别为 0.78065、0.71560；对应最佳监督结果分别为 Random Forest 的 0.80982、HistGradientBoosting 的 0.73275。最佳监督比较对象按当前指标分别选取。检测评价使用开发集 38,010 和外部 109,704 个预测案例，组成与检测终点分别报告。训练、外层 OOF 和锁定外部 E1 均沿用既有锁定运行结果，绘图阶段不重新拟合预测模型。

## 源数据对应

| 子图 | 数据文件及主要字段 |
| :--- | :--- |
| 2a | `output/nature_stat_redesign/source_data_v3/Fig2a_curvature.csv`；`feature`、`balanced_ot`、`semi_relaxed_iot` |
| 2b | 同目录 `Fig2b_recovery.csv`；`coefficient`、`truth`、两种恢复值、`pure_column_direction` |
| 2c | 同目录 `Fig2c_sensitivity.csv`；`mu`、`direction_sensitivity_norm` |
| 2d | 同目录 `Fig2d_external_validation.csv`；`cohort`、`site`、`gain_mean`、`ci_low/high`、`relative_gain`、`n_bootstrap` |
| 3a | 同目录 `Fig3a_all_method_ranks.csv`；`method`、`task`、`rank`、`our_method` |
| 3b | 同目录 `Fig3b_lineage_paired_values.csv`、`Fig3b_lineage_context_statistics.csv` |
| 3c | 同目录 `Fig3c_lineage_effects_with_ci.csv`；`mean_difference_a_minus_b`、`se`、`ci_low/high`、`p` |
| 4a–c | 同目录 `Fig4ab_patient_scores.csv`、`Fig4ab_descriptive_mean_ci.csv`、`Fig4c_patient_effects.csv` |
| 4d | 同目录 `Fig4d_gdsc_egfr_tki.csv`；`drug`、`spearman_rho`、`n_cell_lines`、`p` |
| 4e | `output/artemis_emt_branches/mechanism_v2/perturbation/gse164488_dog_sample_features.tsv` 与同目录 `gse164488_prrx1_perturbation_effects.tsv`；按 `replicate` 配对 |
| 5a–b | `output/nature_stat_redesign/source_data_v3/Fig5ab_state_metrics.csv`；按 `partition`、`method` 读取组成指标 |
| 5c–d | 同目录 `Fig5c_external_all_windows.csv` 与 `Fig5c_external_condition_state_summary.csv`；全部窗口及均值 |
| 5e | 同目录 `Fig5d_brier_paired_bootstrap.csv`；配对克隆 bootstrap 差异和区间 |
| 5f | 同目录 `Fig5d_detection_boundary.csv`；`metric=AUPRC` 的分区、方法和数值 |

新增派生表为 `output/nature_stat_redesign/revision_v5/Fig4e_prrx1_paired_replicates.csv` 和 `Fig5d_condition_and_pooled_bias.csv`；原有源数据文件保留原值。所有图中统计区间、P 与 q 按上述定义分别解释。

## 精确统计表

开发集固定划分使用源克隆家族分组的五折划分，划分种子为 20260826；绘图抖动及图 5d 的窗口 bootstrap 种子为 20260905。以下表格直接读取同一源数据。

### 配对历史效应

| 系统 | 分支 | 平均差 | 95% CI | 配对 P |
| :--- | :--- | :--- | :--- | :--- |
| TENA | EPI | +0.6026 | [-0.6310, 1.8362] | 0.246528 |
| TENA | INV_EMT | -0.5078 | [-0.8589, -0.1567] | 0.015924 |
| TENA | INF_EMT | -0.4932 | [-0.7233, -0.2632] | 0.00399567 |
| TENA | IFN_HLA | -0.1729 | [-0.6103, 0.2645] | 0.334108 |
| TENA | PROLIF | +0.1885 | [-0.5123, 0.8894] | 0.496655 |
| TENA | DORM_STRESS | -0.2588 | [-0.8767, 0.3592] | 0.309594 |
| Cdh | EPI | -3.3055 | [-5.9639, -0.6471] | 0.033207 |
| Cdh | INV_EMT | +1.4965 | [0.5004, 2.4926] | 0.023107 |
| Cdh | INF_EMT | +0.1318 | [-2.3614, 2.6250] | 0.841236 |
| Cdh | IFN_HLA | -0.0712 | [-2.6251, 2.4827] | 0.915494 |
| Cdh | PROLIF | +0.2501 | [-2.1864, 2.6866] | 0.701868 |
| Cdh | DORM_STRESS | +0.2818 | [-3.5112, 4.0748] | 0.77955 |

### GDSC 药敏相关

| 药物 | 细胞系 n | Spearman ρ | P |
| :--- | :--- | :--- | :--- |
| Afatinib | 520 | 0.30924 | 5.49804e-13 |
| Erlotinib | 513 | 0.21072 | 1.47022e-06 |
| Gefitinib | 515 | 0.25350 | 5.3993e-09 |
| Osimertinib | 515 | 0.28461 | 4.71046e-11 |
| Lapatinib | 520 | 0.20274 | 3.15266e-06 |

### PRRX1 配对扰动

| 分支 | 生物学配对 n | 平均差 | 配对 P |
| :--- | :--- | :--- | :--- |
| EPI | 3 | +0.32339 | 0.0206272 |
| INV_EMT | 3 | -0.48911 | 0.0160106 |
| INF_EMT | 3 | +0.00924 | 0.893445 |
| IFN_HLA | 3 | +0.94603 | 0.0974389 |
| PROLIF | 3 | -0.32380 | 0.194579 |
| DORM_STRESS | 3 | +0.09321 | 0.23238 |
