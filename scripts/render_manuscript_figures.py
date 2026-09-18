"""Assemble Nature Methods manuscript figures from frozen benchmark and biology tables.

Figure 3 merges the previous repo Figures 3-5 panels under one display item.
Figures 4 and 5 redraw the frozen biology analyses in the same v5/v6 visual system.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

import render_current_v5 as rc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "manuscript"
LEGACY = "figures/legacy_v5/source_data"
SOURCES: dict[str, set[str]] = {}
QA: list[dict] = []

BRANCH_ORDER = ["EPI", "INV_EMT", "INF_EMT", "IFN_HLA", "PROLIF", "DORM_STRESS"]
BRANCH_LABELS = ["Epithelial", "Invasive EMT", "Inflammatory EMT", "IFN/HLA", "Proliferation", "Dormancy/stress"]
SYSTEM_COLORS = {"TENA": rc.BLUE, "Cdh": rc.ORANGE}


def load(panel: str, path: str) -> pd.DataFrame:
    SOURCES.setdefault(panel, set()).add(path)
    frame = pd.read_csv(ROOT / path)
    if frame.empty:
        raise ValueError(f"Empty source: {path}")
    return frame


def save(fig, name: str) -> None:
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    fits = bb.x0 >= -.01 and bb.y0 >= -.01 and bb.x1 <= w + .01 and bb.y1 <= h + .01
    QA.append({"figure": name, "content_fits_canvas": bool(fits), "bounds_inches": list(bb.bounds),
               "width_mm": w * 25.4, "height_mm": h * 25.4})
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches=None, dpi=300, facecolor="white")
    plt.close(fig)


def clone_cross_entropy(ax) -> None:
    locked = load("Figure3c", rc.LOCKED)
    locked = locked[locked.status.eq("success")]
    for i, dataset in enumerate(sorted(locked.dataset.unique())):
        row = locked[locked.dataset.eq(dataset)].set_index("method")
        control, value = row.loc["PERSIST-no-IOT", "cross_entropy"], row.loc["PERSIST-IOT", "cross_entropy"]
        ax.plot([control, value], [i, i], color="#D0D0D0", lw=2.6, solid_capstyle="round")
        for v, color in [(control, rc.GREY), (value, rc.PURPLE)]:
            ax.scatter(v, i, s=38, color=color, edgecolors="white", lw=.6, zorder=3)
            ax.annotate(f"{v:.3f}", (v, i), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=6, color=color)
    ax.set_yticks([0, 1], ["Development\nouter OOF", "Locked\nexternal E1"], fontsize=6)
    ax.set_ylim(1.55, -.6)
    ax.margins(x=.17)
    rc.grid(ax)
    ax.set_xlabel("Clone-level cross-entropy", fontsize=7)


def calibration_strip(ax) -> None:
    cal = load("Figure3e", rc.CAL)
    order = [m for m in rc.METHODS if m in set(cal.method)]
    for i, method in enumerate(order):
        for j, dataset in enumerate(sorted(cal.evaluation_dataset.unique())):
            row = cal[cal.method.eq(method) & cal.evaluation_dataset.eq(dataset)].iloc[0]
            y = i + (j - .5) * .2
            ax.plot([row.ci95_lower, row.ci95_upper], [y, y], color=[rc.BLUE, rc.ORANGE][j], lw=1)
            ax.scatter(row.expected_calibration_error, y, s=22, color=[rc.BLUE, rc.ORANGE][j],
                       marker=["o", "s"][j], edgecolors="white", lw=.4, zorder=3)
    ax.set_yticks(range(len(order)), [rc.METHODS[m] for m in order], fontsize=5.8)
    ax.set_ylim(len(order) - .4, -.6)
    ax.set_xlabel("State-wise calibration error ↓", fontsize=7)
    rc.grid(ax)


def runtime_strip(ax) -> None:
    runs = load("Figure3f", rc.RUNS)
    order = [m for m in rc.METHODS if m in set(runs.method)]
    for i, method in enumerate(order):
        values = runs[runs.method.eq(method)].runtime_seconds.dropna().to_numpy()
        values = values[values > 0]
        color = rc.PURPLE if method == "UOT-IOT" else rc.GREY
        ax.scatter(values, i + np.linspace(-.16, .16, len(values)), s=9, color=color, alpha=.40, lw=0)
        ax.scatter(np.median(values), i, s=24, color=color, marker="D", edgecolors="white", lw=.4, zorder=4)
    ax.set_yticks(range(len(order)), [rc.METHODS[m] for m in order], fontsize=6)
    ax.set_ylim(len(order) - .4, -.6)
    ax.set_xscale("log")
    ax.set_xlabel("Runtime per run (s, log)", fontsize=7)


def figure3() -> None:
    b2 = load("Figure3a", rc.B2)
    stability = load("Figure3b", rc.STABILITY)
    load("Figure3b", rc.B2)
    fig = plt.figure(figsize=(183 / 25.4, 190 / 25.4))
    gs = fig.add_gridspec(2, 3, left=.17, right=.975, top=.92, bottom=.11, wspace=.62, hspace=.85)
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    d, e, f = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    for ax, letter in zip([a, b, c, d, e, f], "abcdef"):
        rc.panel(ax, letter, dx=-.42 if letter == "a" else -.13)
    rc.method_strip(a, b2, "transition_weighted_l1_mean", "Transition-matrix L1 ↓")
    joined = b2.merge(stability, on=["dataset", "method"], how="inner")
    colors = {"UOT-IOT": rc.PURPLE, "mioflow": rc.BLUE, "prescient": rc.GREEN, "tigon": rc.ORANGE}
    for method, part in joined.groupby("method"):
        for _, row in part.iterrows():
            b.scatter(row.transition_weighted_l1_mean, row.pairwise_cosine_mean, s=24, color=colors[method],
                      marker=rc.SHAPES[row.dataset], edgecolors="white", lw=.4)
    b.set_xlabel("Transition-matrix L1 ↓", fontsize=7)
    b.set_ylabel("Output-proxy cosine ↑", fontsize=7)
    b.set_ylim(.65, 1.025)
    b.legend(handles=[Line2D([], [], marker="o", ls="", color=colors[m], label=rc.METHODS[m]) for m in colors],
             loc="upper left", bbox_to_anchor=(-.07, -.25), ncol=2, fontsize=5.6, columnspacing=.6, handletextpad=.3)
    rc.grid(b)
    clone_cross_entropy(c)
    pop = load("Figure3d", rc.POP)
    rc.population_strip(d, pop, "top1_accuracy", "Population top-1 accuracy ↑")
    d.legend(handles=[Line2D([], [], marker="o", ls="", color=rc.BLUE, label="GSE140802"),
                      Line2D([], [], marker="s", ls="", color=rc.ORANGE, label="GSE239651 expt2")],
             loc="upper left", bbox_to_anchor=(-.02, -.22), ncol=1, fontsize=5.8, handletextpad=.3)
    calibration_strip(e)
    runtime_strip(f)
    rc.title(fig, "Method benchmark and prospective prediction",
             [(a, "Lineage transition recovery"), (b, "Prediction-stability relationship"),
              (c, "Clone-level future composition"), (d, "Population dominant-state agreement"),
              (e, "Calibration"), (f, "Observed computation times")])
    save(fig, "Figure3")


def half_violin(ax, values: np.ndarray, x: float, color: str, width: float = .30) -> None:
    kde = gaussian_kde(values)
    grid = np.linspace(values.min(), values.max(), 128)
    density = kde(grid)
    density = density / density.max() * width
    ax.fill_betweenx(grid, x, x + density, color=color, alpha=.22, lw=0)
    ax.plot(x + density, grid, color=color, lw=.7, alpha=.85)


def figure4() -> None:
    paired = load("Figure4a", f"{LEGACY}/Fig3b_lineage_paired_values.csv")
    effects = load("Figure4b", f"{LEGACY}/Fig3c_lineage_effects_with_ci.csv")
    fig = plt.figure(figsize=(183 / 25.4, 98 / 25.4))
    gs = fig.add_gridspec(1, 2, left=.13, right=.975, top=.87, bottom=.17, wspace=.42)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    rc.panel(a, "a", dx=-.16)
    rc.panel(b, "b", dx=-.16)
    for system in ["TENA", "Cdh"]:
        part = paired[paired.system.eq(system)]
        for _, row in part.iterrows():
            a.plot([0, 1], [row.epithelial_lineage, row.ever_emt_lineage], color=SYSTEM_COLORS[system],
                   alpha=.65, lw=.9, marker="o", ms=3.4)
        mean = part[["epithelial_lineage", "ever_emt_lineage"]].mean()
        a.plot([0, 1], [mean.epithelial_lineage, mean.ever_emt_lineage], color=SYSTEM_COLORS[system],
               lw=2.0, marker="D", ms=6, zorder=4)
    a.set_xticks([0, 1], ["Epithelial\nlineage", "Ever-EMT\nlineage"], fontsize=6.5)
    a.set_ylabel("Invasive-EMT score", fontsize=7)
    a.set_xlim(-.25, 1.35)
    a.legend(handles=[Line2D([], [], color=SYSTEM_COLORS[s], lw=1.2, marker="D", ms=4,
                             label=f"{s} (n = {len(paired[paired.system.eq(s)])} mice)") for s in ["TENA", "Cdh"]],
             loc="upper left", fontsize=6, handletextpad=.4)
    rc.grid(a)
    for system in ["TENA", "Cdh"]:
        part = effects[effects.system.eq(system)].set_index("branch")
        for i, branch in enumerate(BRANCH_ORDER):
            row = part.loc[branch]
            y = i + (.22 if system == "Cdh" else -.22)
            b.plot([row.ci_low, row.ci_high], [y, y], color=SYSTEM_COLORS[system], lw=.9)
            b.plot([row.mean_difference_a_minus_b - row.se, row.mean_difference_a_minus_b + row.se], [y, y],
                   color=SYSTEM_COLORS[system], lw=2.4)
            filled = bool(row.p < .05)
            b.scatter(row.mean_difference_a_minus_b, y, s=20, zorder=3,
                      color=SYSTEM_COLORS[system] if filled else "white",
                      edgecolors=SYSTEM_COLORS[system], lw=.8)
    b.axvline(0, color=rc.GREY, lw=.5, ls="--")
    b.set_yticks(range(len(BRANCH_ORDER)), BRANCH_LABELS, fontsize=6)
    b.invert_yaxis()
    b.set_xlabel("Ever-EMT minus epithelial lineage (paired difference)", fontsize=7)
    rc.grid(b)
    rc.title(fig, "Lineage history changes the direction of state change", [])
    save(fig, "Figure4")


def patient_panel(ax, scores: pd.DataFrame, descriptive: pd.DataFrame, feature: str, ylabel: str) -> None:
    groups = ["RD", "pCR"]
    colors = {"RD": rc.GREY, "pCR": rc.PURPLE}
    for i, group in enumerate(groups):
        values = scores.loc[scores.outcome.eq(group), feature].to_numpy()
        half_violin(ax, values, i, colors[group])
        jitter = (np.arange(len(values)) % 17) / 16 * .16 - .08
        ax.scatter(np.full(len(values), i) + jitter, values, s=7, color=colors[group], alpha=.55, lw=0, zorder=3)
        row = descriptive[descriptive.feature.eq(feature) & descriptive.outcome.eq(group)].iloc[0]
        ax.plot([i - .11, i + .11], [row["mean"], row["mean"]], color=colors[group], lw=1.4, zorder=4)
        ax.plot([i, i], [row.mean_ci_low, row.mean_ci_high], color=colors[group], lw=1.0, zorder=4)
        ax.scatter(i, row["mean"], s=26, marker="D", color=colors[group], edgecolors="white", lw=.5, zorder=5)
    ax.set_xticks([0, 1], ["RD\n(n = 34)", "pCR\n(n = 45)"], fontsize=6.5)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_xlim(-.45, 1.45)
    rc.grid(ax)


def figure5() -> None:
    scores = load("Figure5ab", f"{LEGACY}/Fig4ab_patient_scores.csv")
    descriptive = load("Figure5ab", f"{LEGACY}/Fig4ab_descriptive_mean_ci.csv")
    effects = load("Figure5c", f"{LEGACY}/Fig4c_patient_effects.csv")
    fig = plt.figure(figsize=(183 / 25.4, 140 / 25.4))
    gs = fig.add_gridspec(2, 2, left=.13, right=.975, top=.90, bottom=.10, wspace=.42, hspace=.55)
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    for ax, letter in zip([a, b, c], "abc"):
        rc.panel(ax, letter, dx=-.16 if letter != "c" else -.075)
    patient_panel(a, scores, descriptive, "score_IFN_HLA", "IFN/HLA score")
    patient_panel(b, scores, descriptive, "score_INV_EMT", "Invasive-EMT score")
    part = effects.set_index("feature")
    for i, branch in enumerate(BRANCH_ORDER):
        row = part.loc[f"score_{branch}"]
        highlight = branch == "IFN_HLA"
        color = rc.RED if highlight else rc.DARKGREY
        c.plot([row.bootstrap_95_low, row.bootstrap_95_high], [i, i], color=color, lw=1.1)
        filled = bool(row.bh_q_value < .05)
        c.scatter(row.difference_pCR_minus_RD, i, s=22, zorder=3, color=color if filled else "white",
                  edgecolors=color, lw=.8)
        c.annotate(f"q = {row.bh_q_value:.3g}", (row.bootstrap_95_high, i), xytext=(6, 0),
                   textcoords="offset points", va="center", fontsize=5.8, color=color)
    c.axvline(0, color=rc.GREY, lw=.5, ls="--")
    c.set_yticks(range(len(BRANCH_ORDER)), BRANCH_LABELS, fontsize=6)
    c.invert_yaxis()
    c.set_xlim(-.15, .56)
    c.set_xlabel("pCR minus RD (branch score)", fontsize=7)
    rc.grid(c)
    rc.title(fig, "Patient programmes and treatment response", [])
    save(fig, "Figure5")


def figure_s_pseudotime() -> None:
    e1 = load("FigureS1a", "results/pseudotime/e1_synthetic.csv")
    e2 = load("FigureS1c", "results/pseudotime/e2_gse228154.csv")
    e3 = load("FigureS1d", "results/pseudotime/e3_panels.csv")
    e6 = load("FigureS1d", "results/pseudotime/e6_controls.csv")
    fig = plt.figure(figsize=(183 / 25.4, 150 / 25.4))
    gs = fig.add_gridspec(2, 2, left=.15, right=.975, top=.90, bottom=.12, wspace=.55, hspace=.62)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    c, d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    for ax, letter in zip([a, b, c, d], "abcd"):
        rc.panel(ax, letter, dx=-.18)
    for method, color, marker, label in [("soft_iot", rc.BLUE, "o", "Semi-relaxed"),
                                         ("hard_ot", rc.DARKGREY, "s", "Hard OT")]:
        quiet = e1[e1.method.eq(method) & e1.dropout.eq(0.0)].groupby("sample_count").ordering.mean()
        a.plot(quiet.index, quiet.values, color=color, marker=marker, ms=4, lw=1.2, label=label)
        noisy = e1[e1.method.eq(method) & e1.dropout.gt(0)].ordering.mean()
        a.scatter([2000], [noisy], facecolors="white", edgecolors=color, marker=marker, s=28, zorder=3)
    a.set_xscale("log")
    a.set_ylim(0, 1.05)
    a.set_xlabel("Sample count (log)", fontsize=7)
    a.set_ylabel("Known-truth ordering (ρ)", fontsize=7)
    a.legend(loc="lower right", fontsize=6)
    a.annotate("open: 15% dropout", (2000, 0.45), fontsize=5.6, color=rc.DARKGREY, ha="center")
    rc.grid(a)
    diagnostics = e1.groupby("method")[["direction_dispersion", "pure_column_curvature"]].mean()
    positions = np.arange(2)
    width = .34
    for offset, (column, color, label) in zip([-.17, .17], [("direction_dispersion", rc.PURPLE, "Direction dispersion"),
                                                             ("pure_column_curvature", rc.BLUE, "Pure-column curvature")]):
        values = [diagnostics.loc["hard_ot", column], diagnostics.loc["soft_iot", column]]
        b.bar(positions + offset, values, width=width, color=color, alpha=.85, label=label)
        for x, value in zip(positions + offset, values):
            b.annotate(f"{value:.3g}", (x, value), xytext=(0, 3), textcoords="offset points",
                       ha="center", fontsize=5.6, color=color)
    b.set_yscale("log")
    b.set_xticks(positions, ["Hard OT", "Semi-relaxed"], fontsize=6.5)
    b.set_ylabel("Diagnostic value (log)", fontsize=7)
    b.legend(loc="lower center", fontsize=5.8, ncol=1)
    rc.grid(b)
    order = ["soft_iot", "hard_ot", "frozen_uot_direction", "marginal_prevalence"]
    labels = ["Semi-relaxed", "Hard OT", "Frozen\ndirection", "Marginal\nprevalence"]
    colors = [rc.PURPLE, rc.DARKGREY, rc.BLUE, rc.ORANGE]
    values = e2.set_index("method").loc[order, "ordering_state"].to_numpy(float)
    aucs = e2.set_index("method").loc[order, "time_auc"].to_numpy(float)
    c.bar(np.arange(len(order)), values, color=colors, alpha=.85)
    for x, (value, auc) in enumerate(zip(values, aucs)):
        c.annotate(f"ρ {value:.2f}\nAUC {auc:.2f}", (x, value), xytext=(0, 4), textcoords="offset points",
                   ha="center", fontsize=5.8)
    c.set_xticks(np.arange(len(order)), labels, fontsize=6)
    c.set_ylim(0, .62)
    c.set_ylabel("GSE228154 state ordering (ρ)", fontsize=7)
    rc.grid(c)
    primary_synth = float(e1[e1.method.eq("soft_iot")].ordering.mean())
    primary_gse = float(e2[e2.method.eq("soft_iot")].iloc[0].ordering_state)
    primary_panel = float(e3[(e3.dataset.eq("gse140802_t2_t16")) & e3.method.eq("uot_iot")].ordering.mean())
    controls = e6.set_index(["dataset", "control"]).ordering.to_dict()
    groups = [
        ("Synthetic", [("primary", primary_synth), ("permuted\ntruth", controls[("synthetic_chain", "permuted_truth")])]),
        ("GSE228154", [("primary", primary_gse), ("shuffled\nlabels", controls[("gse228154", "shuffled_target_labels")])]),
        ("GSE140802", [("primary", primary_panel), ("shuffled\ncoupling", controls[("gse140802_t2_t16", "shuffled_coupling")])]),
    ]
    x = 0
    xticks, xticklabels = [], []
    for name, entries in groups:
        start = x
        for label, value in entries:
            color = rc.BLUE if label == "primary" else rc.GREY
            d.bar([x], [abs(value)], color=color, alpha=.85, width=.7)
            d.annotate(f"{abs(value):.2f}", (x, abs(value)), xytext=(0, 3), textcoords="offset points",
                       ha="center", fontsize=5.8)
            xticks.append(x)
            xticklabels.append(label)
            x += 1
        d.annotate(name, ((start + x - 1) / 2, 1.0), xycoords=("data", "axes fraction"),
                   ha="center", va="top", fontsize=6)
        x += .8
    d.set_xticks(xticks, xticklabels, fontsize=5.8)
    d.set_ylim(0, 1.05)
    d.set_ylabel("|ordering| (ρ)", fontsize=7)
    d.legend(handles=[Line2D([], [], color=rc.BLUE, lw=4, label="Primary"),
                      Line2D([], [], color=rc.GREY, lw=4, label="Control")],
             loc="upper right", fontsize=5.8)
    rc.grid(d)
    rc.title(fig, "Pseudotime as an audited readout",
             [(a, "Known-truth recovery"), (b, "Attribution diagnostics"),
              (c, "Real-data ordering versus marginal baseline"), (d, "Negative controls")])
    save(fig, "FigureS_pseudotime")


def figure_ed_mu_scan() -> None:
    path = "../output/22_m7_uot_identifiability/m7_results.json"
    SOURCES.setdefault("ED1_mu_scan", set()).add(path)
    payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
    grid = list(payload["synthetic"]["mu_grid"])
    values = list(payload["synthetic"]["dnorm"])
    working = float(payload["synthetic"]["dnorm_at_working_point"])
    fig = plt.figure(figsize=(110 / 25.4, 80 / 25.4))
    ax = fig.add_subplot(111)
    ax.plot(grid, values, color=rc.BLUE, marker="o", ms=4, lw=1.2)
    ax.scatter([0.5], [working], marker="*", s=70, color=rc.RED, zorder=3)
    ax.annotate(f"μ = 0.5, {working:.3f}", (0.5, working), xytext=(6, 6), textcoords="offset points",
                fontsize=6, color=rc.RED)
    ax.set_xscale("log")
    ax.set_xlabel("Target-marginal penalty μ (log)", fontsize=7)
    ax.set_ylabel("Sensitivity norm", fontsize=7)
    rc.grid(ax)
    rc.title(fig, "Target-marginal penalty sensitivity", [])
    save(fig, "ED1_mu_sensitivity")


def figure_ed_calibration() -> None:
    cal = load("ED2_calibration", rc.CAL)
    order = [m for m in rc.METHODS if m in set(cal.method)]
    datasets = sorted(cal.evaluation_dataset.unique())
    fig = plt.figure(figsize=(120 / 25.4, 85 / 25.4))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=.34, right=.98, top=.84, bottom=.16)
    for i, method in enumerate(order):
        for j, dataset in enumerate(datasets):
            row = cal[cal.method.eq(method) & cal.evaluation_dataset.eq(dataset)].iloc[0]
            y = i + (j - .5) * .22
            ax.plot([row.ci95_lower, row.ci95_upper], [y, y], color=[rc.BLUE, rc.ORANGE][j], lw=1)
            ax.scatter(row.expected_calibration_error, y, s=22, color=[rc.BLUE, rc.ORANGE][j],
                       marker=["o", "s"][j], edgecolors="white", lw=.4, zorder=3)
    ax.set_yticks(range(len(order)), [rc.METHODS[m] for m in order], fontsize=6)
    ax.set_ylim(len(order) - .4, -.6)
    ax.set_xlabel("State-wise calibration error ↓", fontsize=7)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=rc.BLUE, label="GSE140802"),
                       Line2D([], [], marker="s", ls="", color=rc.ORANGE, label="GSE239651 expt2")],
              loc="upper right", fontsize=6)
    rc.grid(ax)
    rc.title(fig, "Calibration across external panels", [])
    save(fig, "ED2_calibration")


def figure_ed_biology() -> None:
    gdsc = load("ED3_gdsc", f"{LEGACY}/Fig4d_gdsc_egfr_tki.csv")
    prrx1_path = "figures/legacy_v5/perturbation/gse164488_prrx1_perturbation_effects.tsv"
    SOURCES.setdefault("ED3_prrx1", set()).add(prrx1_path)
    prrx1 = pd.read_csv(ROOT / prrx1_path, sep="\t")
    prrx1 = prrx1[prrx1.contrast.eq("siPRRX1_minus_siCTR")]
    fig = plt.figure(figsize=(183 / 25.4, 80 / 25.4))
    gs = fig.add_gridspec(1, 2, left=.13, right=.975, top=.86, bottom=.18, wspace=.42)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    rc.panel(a, "a", dx=-.16)
    rc.panel(b, "b", dx=-.16)
    a.scatter(gdsc.drug, gdsc.spearman_rho, s=30, color=rc.BLUE, zorder=3)
    a.axhline(gdsc.spearman_rho.median(), color=rc.GREY, lw=.7, ls="--")
    for _, row in gdsc.iterrows():
        a.annotate(f"n = {int(row.n_cell_lines)}", (row.drug, row.spearman_rho), xytext=(0, 5),
                   textcoords="offset points", ha="center", fontsize=5.6, color=rc.DARKGREY)
    a.set_ylim(0, .4)
    a.set_ylabel("Spearman ρ (EMT vs ln IC50)", fontsize=7)
    a.tick_params(axis="x", labelsize=6)
    rc.grid(a)
    branches = ["EPI", "INV_EMT", "INF_EMT", "IFN_HLA", "PROLIF", "DORM_STRESS"]
    labels = ["Epithelial", "Invasive EMT", "Inflammatory EMT", "IFN/HLA", "Proliferation", "Dormancy/stress"]
    part = prrx1.set_index("branch").loc[branches]
    tcrit = 4.302652729911275
    for i, branch in enumerate(branches):
        row = part.loc[branch]
        se = abs(row.mean_difference_a_minus_b / row.paired_t) if row.paired_t else 0.0
        low, high = row.mean_difference_a_minus_b - tcrit * se, row.mean_difference_a_minus_b + tcrit * se
        color = rc.BLUE if branch == "EPI" else rc.GREY
        b.plot([low, high], [i, i], color=color, lw=1)
        filled = bool(row.paired_p < .05)
        b.scatter(row.mean_difference_a_minus_b, i, s=22, zorder=3,
                  color=color if filled else "white", edgecolors=color, lw=.8)
        b.annotate(f"P = {row.paired_p:.3g}", (high, i), xytext=(5, 0), textcoords="offset points",
                   va="center", fontsize=5.6, color=color)
    b.axvline(0, color=rc.GREY, lw=.5, ls="--")
    b.set_yticks(range(len(branches)), labels, fontsize=6)
    b.invert_yaxis()
    b.set_xlim(-1.1, 1.3)
    b.set_xlabel("siPRRX1 minus siCTR (paired difference)", fontsize=7)
    rc.grid(b)
    rc.title(fig, "Orthogonal biology: drug sensitivity and perturbation", [])
    save(fig, "ED3_biology_orthogonal")


def output_name(panel: str) -> str:
    mapping = {
        "Figure3": "Figure3", "Figure4": "Figure4", "Figure5": "Figure5",
        "FigureS": "FigureS_pseudotime", "ED1": "ED1_mu_sensitivity",
        "ED2": "ED2_calibration", "ED3": "ED3_biology_orthogonal",
    }
    for prefix, name in mapping.items():
        if panel.startswith(prefix):
            return name
    raise KeyError(panel)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    figure3()
    figure4()
    figure5()
    figure_s_pseudotime()
    figure_ed_mu_scan()
    figure_ed_calibration()
    figure_ed_biology()
    records = []
    for panel, paths in sorted(SOURCES.items()):
        figure = output_name(panel)
        for path in sorted(paths):
            records.append({
                "figure_panel": panel,
                "script": "scripts/render_manuscript_figures.py",
                "input": path,
                "input_sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
                "output_pdf": f"figures/manuscript/{figure}.pdf",
                "output_png": f"figures/manuscript/{figure}.png",
            })
    pd.DataFrame(records).to_csv(OUT / "source_manifest.csv", index=False)
    report = {"style_reference": "figures_revision_v5 + technical_route_v6",
              "data_authority": "current benchmark results and frozen biology source tables",
              "legacy_numbers_used_for_benchmark_panels": False,
              "checks": QA}
    (OUT / "QA_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    failed = [row for row in QA if row.get("content_fits_canvas") is False]
    print(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
