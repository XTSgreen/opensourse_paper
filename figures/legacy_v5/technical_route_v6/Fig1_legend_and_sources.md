# Figure 1 | Technical route for inverse optimal transport

公开单细胞、谱系和患者数据经状态表示及分析设计进入三个相互关联的分析任务。左列以 GSE228154 为例展示数据准备，包括 3,208 个细胞、1,820 个已保存表达基因的 PCA，D0、D3、D6、D9 的十状态组成，以及五个程序在十个状态中的评分。PCA 由 R 对保存的 log 归一化表达矩阵进行中心化后计算，全部细胞均纳入；该图是用于展示输入数据的二维投影，原有聚类标签沿用已有状态文件。状态程序热图对每个程序在十个状态之间标准化，颜色为程序内 z 分数。

**分析 1，传输代价的可辨识性与外部转移。** UOT-IOT 固定源边际，对目标边际施加 KL 锚定，并根据参数化代价估计方向参数 θ 与传输计划 πθ。原生示意中的源、目标堆叠条分别使用真实 D0、D9 状态比例；中间的四阶方格表示耦合关系示意。右侧依次展示 GSE228154 的五个目标态方向经验曲率、八状态合成系统的八个系数恢复结果、同一合成系统的八个 KL 惩罚扫描点，以及两个独立队列六个未见部位的冻结参数验证。系数恢复图中的红色外圈表示五个纯列方向；灵敏度图的星形表示 μ=0.5，背景区间为 μ=0.1–1.0。外部验证点表示 MAE 平均改善，线段为源结果中的 200 次 bootstrap 95% 区间。

**分析 2，情境与谱系历史。** GH-IOT 将代价方向分解为共享参数与受到收缩约束的情境偏移，以门控组合训练组 IOT 先验和情境残差，并在留组边界内评价；图中的 g₁、g₂、…、gₖ 与 gₕ 示意训练组和留出组。谱系图使用 GSE171940 的全部八只小鼠，TENA 为五只、Cdh 为三只，小点与连线为配对个体，大菱形为系统均值。患者分布图使用 ARTEMIS 中有可用肿瘤评分的 79 位患者，其中 RD 34 位、pCR 45 位；小点为患者，小提琴为组内密度，菱形和线段为均值及已有患者 bootstrap 95% 区间。下方热图包含全部 79 位患者的六个分支，先按结局分组，再在组内按 IFN/HLA 得分排序；每个分支在患者之间标准化，显示色阶在 ±3 处饱和，完整 z 分数保存在数据表中。患者结果用于分支读出与治疗结局关联的评价。

**分析 3，源端信息下的前瞻预测。** PERSIST-IOT 读取源状态 s₀、细胞数 n₀、条件 c 与预测时间间隔 Δt，通过 IOT 前向生成器与观测校正的持续状态 HMM，输出未来组成及持续／检出概率；生成器节点与 HMM 节点为模型关系示意。前瞻评价沿用既有五折外层 OOF 与锁定外部 E1 结果。外部组成校准图纳入 DMSO、ispinesib 各 28 个窗口和六种状态，共 336 个窗口／状态条目，半透明小点为窗口条目、黑边大点为条件下的等权窗口均值，虚线为一致性线。交叉熵图比较 PERSIST-IOT 与去传输对照。Brier 图为 PERSIST-IOT 减各分区最佳监督模型的配对差及既有 10,000 次克隆 bootstrap 95% 区间；开发集比较对象为 Extra Trees，外部 E1 为 HistGradientBoosting，负值表示 PERSIST-IOT 的 Brier 更低。

图中的三个编号表示分析层次，各任务使用与其估计目标相应的输入；实线箭头表示任务内部的计算关系。统计图使用原有数据与结果表，图 2–5 的完整统计定义和检验结果见 `paper/figures_revision_v5/figure_legends_v5.md`。

## R 小图与数据对应

所有小图均由 `script/nature_stat_redesign/render_route_panels_v6.R` 通过用户指定的 R 4.6.1 绘制。`R_panels` 内分别保存 SVG、PDF 和 600 dpi PNG，图中 SVG 已内嵌于 draw.io 文件。数据副本与新生成的 PCA 坐标在本目录的 `source_data` 中。

| 编号 | 小图 | 输入数据或计算方式 |
| :--- | :--- | :--- |
| 01 | Expression PCA | GSE228154_log_expression.mtx.gz 与 GSE228154_cells.csv；R `irlba::prcomp_irlba`，中心化，不缩放基因，全体细胞 |
| 02 | Treatment-state composition | GSE228154_cells.csv，按时间点与状态计数后归一化 |
| 03 | State × programme features | GSE228154_state_programmes.csv，原有全基因状态质心程序评分；按程序标准化 |
| 04 | Empirical curvature | Fig2a_curvature.csv，五个经验曲率方向 |
| 05 | Synthetic coefficient recovery | Fig2b_recovery.csv，八个合成系数及原恢复值 |
| 06 | KL-anchor sensitivity | Fig2c_sensitivity.csv，八个既有扫描点 |
| 07 | Frozen transfer | Fig2d_external_validation.csv，GSE246662 与 GSE183904 的六个部位 |
| 08 | Lineage history | Fig3b_lineage_paired_values.csv，八只小鼠的配对结果 |
| 09 | Treatment response | Fig4ab_patient_scores.csv 与 Fig4ab_descriptive_mean_ci.csv，79 位患者及已有区间 |
| 10 | Patient × programme profiles | Fig4ab_patient_scores.csv 的六个分支；派生完整 z 分数在 ARTEMIS_R_heatmap_z_scores.csv |
| 11 | External composition calibration | Fig5c_external_all_windows.csv 与 Fig5c_external_condition_state_summary.csv，336 条目与 12 个条件／状态均值 |
| 12 | Composition performance | Fig5ab_state_metrics.csv，两个分区、两种方法 |
| 13 | Persistence detection | Fig5d_brier_paired_bootstrap.csv，配对 Brier 差及原有区间 |

## 文件与编辑

**Fig1_technical_route_v6.drawio** 为主编辑文件，分区、文字、参数关系、状态节点和连接线为 draw.io 原生对象；13 张 R 小图以矢量 SVG 嵌入，可以整体移动、缩放和替换，小图中的统计元素可通过配套 R 脚本重新绘制。

**Fig1_technical_route_v6.pdf** 与 **Fig1_technical_route_v6.svg** 采用 183 mm 图宽，约 199.3 mm 图高；**Fig1_technical_route_v6.png** 为 2700 × 2940 像素预览。图内保留整图标题、模块主标题、坐标轴和常规图例，较长的方法与统计说明集中于本图注。

在项目根目录先运行 `python script/nature_stat_redesign/prepare_route_data_v6.py`，再使用 `D:/R-4.6.1/bin/Rscript.exe --vanilla script/nature_stat_redesign/render_route_panels_v6.R` 重画小图，最后运行 `python script/nature_stat_redesign/build_technical_route_drawio_v6.py` 生成 draw.io 文件；导出 PNG 时使用 draw.io 的标准导出，不勾选 PNG 中嵌入图表副本。
