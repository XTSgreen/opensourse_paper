# CURRENT_FIGURE_NOTES — 当前版 Figure 1–5 统计口径说明

本文件是 `figures/rendered/QA_REPORT.json` 中 `statistics_document` 字段指向的统计说明，对应 `scripts/render_current_v5.py` 生成的 Figure 1–5。

- **风格基准**：`figures_revision_v5` + `technical_route_v6`（配色、字体、面板布局、点线样式、标记编码沿用旧版，不沿用旧版数值）。
- **数据基准**：本仓库 `results/` 下的当前 benchmark 结果表；`figures/legacy_v5/` 只作为风格素材与 Figure 1 输入描述面板来源，不参与任何 benchmark 面板的数值。
- **生成入口**：`python scripts/render_figures.py`（委托 `scripts/render_current_v5.py`）；面板到脚本、输入文件及输入 SHA-256 的机器可读清单见 `results/reviewer_tables/figure_source_manifest.csv`。
- **可编辑源**：Figure 1 保留 draw.io 源文件 `figures/rendered/Figure1.drawio`，由 `scripts/legacy/build_technical_route_drawio_v6.py --current-benchmarks` 生成后经 draw.io CLI 导出 PDF/SVG/PNG。

## 1. 通用统计约定

| 约定 | 说明 |
| --- | --- |
| 交叉熵下限 | `1e-12`（`src/iot_benchmark/metrics.py::cross_entropy`，默认 `floor=1e-12`）。旧版 v5 图按 `1e-8` 下限计算，因此去传输对照的交叉熵与旧图不同，见第 6 节。 |
| Sinkhorn divergence | `epsilon=0.05`，`iterations=1000`，按 `ab − 0.5·aa − 0.5·bb` 计算并截断于 0（`metrics.py::sinkhorn_divergence`）。 |
| 传输矩阵 L1 | 行归一化转移矩阵按真实端行质量的加权逐行绝对差之和（`metrics.py::transition_weighted_l1`）。 |
| 命运相关 | 估计与真实 fate 矩阵展平后的 Pearson/Spearman；top-1 为逐行 argmax 命中率（`metrics.py::fate_correlations`）。 |
| 系数恢复 | 归一化后的余弦，标准差归一化下的符号一致率与相对 L2（`metrics.py::coefficient_recovery`）。 |
| 方向稳定性 | 种子间方向向量两两余弦的均值/最小值/方差（`metrics.py::direction_stability`）。 |
| 校准误差 | 10 分箱 top-1 置信度校准误差（`metrics.py::expected_calibration_error`）。 |
| 缺失值 | 不做插补。方法没有相应输出时在图内显示 `NA` 文字或留空（例如 CellRank 2 没有 coupling 对应的传输矩阵指标）。 |
| bootstrap | 区间次数与单位由各结果文件记录；百分位 bootstrap 区间可能不含点估计，绘图直接画端点。 |
| 组图约束 | 每个面板 `save()` 时校验内容不越出画布（`content_fits_canvas`），任一越界时渲染脚本返回非零。 |

## 2. Figure 1 — IOT framework for cell-state inference and prediction

技术路线图沿用 v6 draw.io 版式：3 个输入描述面板与状态条为**不变量**（数据来源为 `figures/legacy_v5/technical_route_v6/source_data/` 下的 GSE228154 输入文件，属于实验输入而非结果），其余 10 个 mini panel 全部由当前结果重绘：

| mini panel | 来源结果 | 内容 |
| --- | --- | --- |
| `04_curvature` | `results/B1_known_truth_final/b1_all_runs.csv` | 六场景纯列方向最小曲率，symlog |
| `05_synthetic_recovery` | 同上 | 合成系数真值 vs 恢复值 |
| `06_anchor_sensitivity` | 同上 | 重启方向方差随样本量变化，symlog |
| `07_external_transfer` | `results/external_direction_validation/frozen_direction_external_validation.csv` | 六个未见位点的 MAE 改善及 95% 区间 |
| `08_lineage_pairs` | `results/B2_lineage_transition/b2_method_summary.csv` | 谱系面板传输 L1 |
| `09_patient_response` | `results/direction_stability/output_direction_seed_stability.csv` | output-proxy 方向余弦 |
| `10_patient_programme_heatmap` | `results/direction_stability/direction_parameter_availability.csv` | 方法是否具有可比较的显式方向参数 |
| `11_external_calibration` | `results/B3_prospective_composition/external_dynamics/b3_calibration_summary.csv` | 期望校准误差及 95% 区间 |
| `12_state_performance` | `results/B3_prospective_composition/b3_state_prediction_metrics.csv` | 克隆级交叉熵（开发集 OOF vs 锁定外部 E1） |
| `13_detection_brier` | `results/B3_prospective_composition/external_dynamics/b3_external_all_runs.csv` | 群体组成交叉熵（面板均值再按数据集平均） |

导出格式：PDF/SVG/PNG（PNG 1.5×），`Figure1.drawio` 为可编辑源。

## 3. Figure 2 — Identifiability and external validation

| 面板 | 指标与聚合 | 关键数值（当前结果） |
| --- | --- | --- |
| a Target-state curvature | `b1_all_runs.csv` 中 `pure_column_minimum_curvature` 按场景×方法的均值；symlog `linthresh=1e-5`，不做对数下限伪造 | Hard OT ≈ 0（−2.7e-10…+3.0e-11）；UOT-IOT 0.217–0.250 |
| b Coefficient recovery | 全部 B1 运行的 `theta_true` vs `theta_hat` 散点（hard_ot 空心、soft_iot 实心），虚线为一致性线 | 无插补；非有限值直接报错 |
| c Stability across sample sizes | `restart_direction_variance` 按样本量（400/2000/10000）×方法的均值；symlog `linthresh=1e-12` | Hard OT ≈ 0.116–0.117；UOT-IOT ≈ 1.5e-13–2.5e-13 |
| d Unseen metastatic sites | 冻结方向外部验证：`gain`（`conditional_transition_mae_gain_over_independence`）及 200 次 bootstrap 95% 区间；GSE246662 的 LM1–LM3、GSE183904 的 s19/s20/s38 | 增益 0.0223–0.0736；相对降幅 16.2%–46.1%；6/6 区间不含零 |

外部验证的声明边界由 `results/external_direction_validation/manifest.json` 固定：只声明“冻结方向在未见队列与位点上的预测效度”，不声明已知真值恢复、全系数符号独立确认或因果效应。

## 4. Figure 3 — Transition recovery and stability across methods

| 面板 | 指标与聚合 | 说明 |
| --- | --- | --- |
| a Lineage transition recovery | `b2_method_summary.csv::transition_weighted_l1_mean`，方法条带 + 数据集形状编码 | UOT-IOT 0.0107–0.3322；对照方法 0.064–1.285；无 coupling 输出的方法显示 NA |
| b Prediction–stability relationship | B2（传输 L1）与 `output_direction_seed_stability.csv`（`pairwise_cosine_mean`）按数据集×方法内连接；只保留有 output-proxy 的 4 个方法（UOT-IOT、MIOFlow、PRESCIENT、TIGON） | 纵轴下限固定 0.65；图注明确 output-proxy 稳定性不等同于符号参数可识别性 |
| c Fate correlation | `b2_method_summary.csv::fate_spearman_mean` 方法条带 | 四个数据集；NA 不做插补 |

## 5. Figure 4 — Future-state composition and calibration

| 面板 | 指标与聚合 | 关键数值（当前结果） |
| --- | --- | --- |
| a Clone-level cross-entropy | `b3_state_prediction_metrics.csv` 中 `status=success` 行；PERSIST-no-IOT（灰）vs PERSIST-IOT（紫） | 开发集 OOF：6.956 → 1.037；锁定外部 E1：6.550 → 0.986 |
| b Clone-level Sinkhorn divergence | 同上 | 开发集：0.178 → 0.102；E1：0.197 → 0.111 |
| c Population composition error | `external_dynamics/b3_external_all_runs.csv`：先按数据集×方法×panel 取种子均值，再按数据集平均面板均值；seeds 嵌套于 panel | 五个方法（development-target-transfer、source-carry-forward、MIOFlow、PRESCIENT、TIGON）；面板为 GSE140802 与 GSE239651 expt2 |
| d Population transport distance | 同上，指标为 `sinkhorn_divergence` | 缺失比较组合直接报错，不静默跳过 |
| e Dominant-state agreement | 同上，指标为 `top1_accuracy` | 同上 |
| f Calibration and 95% intervals | `b3_calibration_summary.csv`：`expected_calibration_error` 及 10,000 次 bootstrap 区间（bootstrap 单位为 held-out panel 聚类，seeds 一起保留；10 分箱） | 百分位区间可能不含点估计，绘图直接画端点 |

组成指标评价单位：开发集 4,867、外部 E1 7,267 个目标时点检出克隆；重复克隆与重叠时间窗按各评价脚本既定方案处理。

## 6. Figure 5 — External transfer and computational performance

| 面板 | 指标与聚合 | 关键数值（当前结果） |
| --- | --- | --- |
| a Frozen-direction transfer | 外部验证 `relative_gain × 100`：位点散点 + 队列均值菱形；GSE246662 与 GSE183904 各 3 个位点 | 同 Figure 2d |
| b Observed computation times | `results/B2_lineage_transition/b2_all_runs.csv::runtime_seconds`，过滤 `runtime_seconds > 0`，log 横轴，菱形为方法中位数 | 中位数：UOT-IOT 1.09 s、WOT 3.39、LineageOT 3.68、PRESCIENT 5.57、moscot 9.17、MIOFlow 12.45、CellRank 2 17.48、TIGON 33.04 |
| c Coupling recovery | `b2_method_summary.csv::coupling_relative_frobenius_mean` 方法条带 | NA 不插补 |

## 7. 与 legacy v5 图的关键口径差异

以下差异是刻意的：当前图以本仓库当前结果为准，不复用旧图数值。旧数值存档于 `figures/legacy_v5/source_data/` 与 `figures/legacy_v5/figure_legends_v5.md`。

1. **外部验证任务不同**：v5 图 2d 为重建 MAE 改善（0.098–0.124，相对 64%–82%）；当前面板为 `conditional_transition_mae_gain_over_independence`（0.0223–0.0736，相对 16.2%–46.1%），两者不能直接比较。
2. **交叉熵下限修正**：旧版按 `1e-8` 计算，去传输对照为开发集 4.8509、E1 4.5831；当前按 `1e-12`，对照改为 PERSIST-no-IOT，为 6.956 与 6.550。PERSIST-IOT 输出未触及下限，数值保持 1.0370 与 0.9859。
3. **面板集合重排**：图 3–5 现对应方法学 benchmark（传输恢复/稳定性、前瞻组成/校准、外部迁移/运行时）；v5 的 context/patient 结构面板不再出现在主图。
4. **Figure 1 mini panel**：结构与 v6 相同，但 10 个结果 mini panel 全部重绘为当前数值；3 个输入描述面板为不变量。
