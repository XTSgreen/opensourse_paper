# -*- coding: utf-8 -*-
"""
Statistical figure set v5 — annotation-led visual refinements of v4.

Academic Figure Skill Asset Confirmation (verified against assets/figures/)
(a) Fig2a curvature dumbbell      -> BarComparison (visual system: dumbbell + direct labels) -> param inherit
(b) Fig2b rescue-arrow scatter    -> MarginalDensity (inset/annotation system) + basic scatter -> param inherit
(c) Fig2c sensitivity curve       -> LineTrend (marker/line/star system)                     -> param inherit
(d) Fig2d site forest             -> BarAblation (CI lollipop system)                        -> param inherit
(e) Fig3a rank strip              -> GroupedBarChart (points-overlay + badge system)         -> param inherit
(f) Fig3b paired slopegraph       -> PairedBoxScatter (paired-identity system, R->py)        -> param inherit
(g) Fig3c unified branch forest   -> BarAblation (CI lollipop system)                        -> param inherit
(h) Fig4a/b raincloud             -> Violin + StackedBarScatter (jitter + mean CI system)    -> param inherit
(i) Fig4c branch forest           -> BarAblation                                             -> param inherit
(j) Fig4d1 GDSC lollipop          -> BarComparison                                           -> param inherit
(k) Fig4e PRRX1 paired-replicate forest   -> BarAblation                                             -> param inherit
(l) Fig5a/b improvement dumbbells -> LineTrend (marker system)                               -> param inherit
(m) Fig5c calibration + zoom inset-> MarginalDensity (inset system)                          -> param inherit
(n) Fig5d detection endpoint      -> BarAblation                                             -> param inherit
RULE: all panels are param-inherit drawing functions that copy Class A/B/C values from the
named production assets; no panel claims "native run" because every panel maps user data.
"""

import json
import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.stats import gaussian_kde, t as student_t, ttest_rel
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
parser = argparse.ArgumentParser(description="Regenerate the original v5 statistical figures without changing their visual design.")
parser.add_argument("--source-data", default=os.path.join(ROOT, "output", "nature_stat_redesign", "source_data_v3"))
parser.add_argument("--perturbation-data", default=os.path.join(ROOT, "output", "artemis_emt_branches", "mechanism_v2", "perturbation"))
parser.add_argument("--output", default=os.path.join(ROOT, "paper", "figures_revision_v5"))
parser.add_argument("--derived-data", default=os.path.join(ROOT, "output", "nature_stat_redesign", "revision_v5"))
args = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])
SRC, PERT, OUT = args.source_data, args.perturbation_data, args.output
os.makedirs(OUT, exist_ok=True)
DERIVED = args.derived_data
os.makedirs(DERIVED, exist_ok=True)

SEED = 20260905

# ============================================================================
# Academic Figure Skill Typography Baseline — COPY VERBATIM
# ============================================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# ============================================================================
# Academic Figure Skill Nature/Cell/Science Color Palette — COPY VERBATIM
# ============================================================================
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED = "#B2182B"
GREY = "#999999"
BLACK = "#222222"

# ============================================================================
# Academic Figure Skill Export Baseline — COPY VERBATIM
# ============================================================================
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})


def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


# ----------------------------------------------------------------------------
# local helpers (Class B scaling: 183 mm double column, 7pt base)
# ----------------------------------------------------------------------------
MM = 1 / 25.4
BLUE, RED, GREEN, ORANGE, PURPLE, DARKGREY = CATEGORICAL
SOFT_RED = CATEGORICAL_EXTENDED[7]      # #D6604D  (Ispinesib)
SOFT_BLUE = CATEGORICAL_EXTENDED[6]     # #4393C3
TICK_GREY = "#555555"
LIGHT_GRID = "#E0E0E0"


def new_fig(w_mm=183, h_mm=115):
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    return fig


def panel_label(ax, letter, dx=-0.13, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left", color=BLACK)


def figure_titles(fig, title, panels):
    fig.suptitle(title, x=0.53, y=0.99, fontsize=11, fontweight="bold", color=BLACK)
    for ax, heading in panels:
        ax.set_title(heading, loc="left", fontsize=7.5, fontweight="bold", pad=8, color=BLACK)






def light_xgrid(ax):
    ax.grid(axis="x", color=LIGHT_GRID, lw=0.3, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)


def fmt_p(p):
    if p < 1e-4:
        s = f"{p:.1e}".split("e")
        return f"P={s[0]}×10$^{{{int(s[1])}}}$"
    return f"P={p:.3g}"




def fmt_ci(v):
    return f"{v:+.3f}" if abs(v) < 1 else f"{v:+.2f}"


QA_LOG = []


def qa(name, cond, detail=""):
    QA_LOG.append((name, "PASS" if cond else "FAIL", detail))
    print(f"[QA] {'PASS' if cond else 'FAIL'}  {name}  {detail}")


# ============================================================================
# FIGURE 2 — identifiability: defect, rescue, working window, frozen transfer
# layout: 2x2 asymmetric; (a) hero dumbbell | (b) rescue-arrow scatter
#                    (c) sensitivity curve | (d) 6/6 unseen-site forest
# ============================================================================
def fig2():
    curv = pd.read_csv(os.path.join(SRC, "Fig2a_curvature.csv"))
    rec = pd.read_csv(os.path.join(SRC, "Fig2b_recovery.csv"))
    sens = pd.read_csv(os.path.join(SRC, "Fig2c_sensitivity.csv"))
    ext = pd.read_csv(os.path.join(SRC, "Fig2d_external_validation.csv"))

    # ---- Step 5.5 data validation ----
    qa("Fig2a balanced curvature <=1e-9", bool((curv.balanced_ot < 1e-9).all()))
    qa("Fig2a IOT curvature in [0.08,0.22]",
       bool(curv.semi_relaxed_iot.between(0.08, 0.22).all()))
    pure_m = rec.pure_column_direction.astype(bool).values
    r_soft = np.corrcoef(rec.truth[pure_m], rec.semi_relaxed_iot[pure_m])[0, 1]
    r_hard = np.corrcoef(rec.truth[pure_m], rec.balanced_ot[pure_m])[0, 1]
    qa("Fig2b pure-column r_soft~1.000 / r_hard~-0.020",
       abs(r_soft - 1) < 5e-3 and abs(r_hard + 0.020) < 5e-3, f"r={r_soft:.3f}/{r_hard:.3f}")
    qa("Fig2c star value mu=0.5",
       abs(sens.loc[sens.mu == 0.5, "direction_sensitivity_norm"].iloc[0] - 0.6224) < 1e-3)
    qa("Fig2d all CI > 0", bool((ext.ci_low > 0).all()))
    qa("Fig2d relative gain 64-82%", bool(ext.relative_gain.between(0.63, 0.83).all()))

    fig = new_fig(183, 148)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0],
                          left=0.17, right=0.975, top=0.885, bottom=0.14,
                          wspace=0.45, hspace=0.63)
    axA = fig.add_subplot(gs[0, 0])
    gsB = gs[0, 1].subgridspec(1, 2, width_ratios=[2.15, 1.0], wspace=0.56)
    axB = fig.add_subplot(gsB[0])
    axBE = fig.add_subplot(gsB[1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # ---------------- (a) curvature dumbbell, log10 ----------------
    d = curv.copy()
    feats = d.feature.tolist()
    y = np.arange(len(feats))[::-1].astype(float)
    lo = np.log10(d.balanced_ot.values)
    hi = np.log10(d.semi_relaxed_iot.values)
    panel_label(axA, "a")
    axA.axvspan(-13.2, -10, color="#E8E8E8", alpha=0.55, lw=0, zorder=0)
    for yi, l, h, f in zip(y, lo, hi, feats):
        axA.plot([l, h], [yi, yi], color="#C9C9C9", lw=1.1, zorder=1)
        axA.scatter(l, yi, s=34, facecolors="white", edgecolors=DARKGREY,
                    lw=1.0, zorder=3)
        axA.scatter(h, yi, s=42, facecolors=BLUE, edgecolors="white",
                    lw=0.7, zorder=4)
        axA.annotate(f"{d.semi_relaxed_iot[d.feature == f].iloc[0]:.3f}",
                     (h, yi), textcoords="offset points", xytext=(7, -2),
                     fontsize=6.2, color=BLUE, fontweight="bold")
    axA.set_yticks(y)
    axA.set_yticklabels(feats)
    axA.set_xlim(-13.2, 0.9)
    axA.set_ylim(-0.65, 4.65)
    axA.set_xlabel("Empirical curvature (log$_{10}$)")
    light_xgrid(axA)
    axA.scatter([], [], s=34, facecolors="white", edgecolors=DARKGREY, lw=1.0,
                label="Balanced OT (hard marginal)")
    axA.scatter([], [], s=42, facecolors=BLUE, edgecolors="white", lw=0.7,
                label="Semi-relaxed IOT (KL anchor)")
    axA.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.21), fontsize=6.0,
               handletextpad=0.3, borderpad=0, frameon=False)

    # ---------------- (b) rescue-arrow scatter ----------------
    pure = rec.pure_column_direction.astype(bool).values
    panel_label(axB, "b")
    axB.plot([-1.15, 1.15], [-1.15, 1.15], ls="--", color=GREY, lw=0.6,
             alpha=0.6, zorder=1)
    for _, row in rec.iterrows():
        arr = FancyArrowPatch((row.truth, row.balanced_ot), (row.truth, row.semi_relaxed_iot),
                              connectionstyle="arc3,rad=0.06", arrowstyle="-|>",
                              mutation_scale=6, color=GREY, alpha=0.55, lw=0.6, zorder=2)
        axB.add_patch(arr)
    axB.scatter(rec.truth, rec.balanced_ot, s=24, facecolors="white",
                edgecolors=DARKGREY, lw=0.9, zorder=3)
    axB.scatter(rec.truth, rec.semi_relaxed_iot, s=28, facecolors=BLUE,
                edgecolors="white", lw=0.6, zorder=4)
    axB.scatter(rec.truth[pure], rec.semi_relaxed_iot[pure], s=46,
                facecolors="none", edgecolors=RED, lw=1.0, zorder=5)
    axB.set_xlim(-1.28, 1.28)
    axB.set_ylim(-1.28, 1.28)
    axB.set_aspect("equal")
    axB.set_anchor("N")
    axB.set_xticks([-1, 0, 1])
    axB.set_yticks([-1, 0, 1])
    axB.set_xlabel("True coefficient", fontsize=7)
    axB.set_ylabel("Recovered coefficient", fontsize=7, labelpad=2)
    err_y = np.arange(len(rec))[::-1]
    err_hard = np.abs(rec.balanced_ot - rec.truth)
    err_soft = np.abs(rec.semi_relaxed_iot - rec.truth)
    axBE.hlines(err_y, err_soft, err_hard, color="#D1D1D1", lw=0.7)
    axBE.scatter(err_hard, err_y, s=16, facecolors="white", edgecolors=DARKGREY, lw=0.7)
    axBE.scatter(err_soft, err_y, s=18, color=BLUE, edgecolors="white", lw=0.3, zorder=3)
    axBE.set_yticks(err_y, [f"θ{i+1}" for i in range(len(rec))], fontsize=5.5)
    axBE.set_xlim(-0.045, 1.05)
    axBE.set_ylim(-0.7, 7.7)
    axBE.set_xticks([0, 1])
    axBE.set_xlabel("Absolute\nerror", fontsize=6.5)
    axBE.tick_params(length=2, pad=2, labelsize=5.5)
    axB.legend(handles=[Line2D([], [], marker="o", ls="", mfc="none", mec=RED,
                               label="Pure-column direction")], loc="upper left",
               bbox_to_anchor=(-0.16, -0.29), fontsize=5.8, handletextpad=0.3)

    # ---------------- (c) sensitivity curve ----------------
    panel_label(axC, "c")
    axC.axvspan(0.1, 1.0, color=BLUE, alpha=0.07, lw=0, zorder=0)
    axC.plot(sens.mu, sens.direction_sensitivity_norm, "-", color=BLUE, lw=1.3,
             zorder=2)
    axC.scatter(sens.mu, sens.direction_sensitivity_norm, s=22, color=BLUE,
                zorder=3)
    axC.scatter([0.5], [0.6224], marker="*", s=150, color=RED, edgecolors="white",
                lw=0.5, zorder=4)
    axC.legend(handles=[Patch(fc=mpl.colors.to_rgba(BLUE, 0.07), ec="none", label="μ = 0.1–1.0"),
                        Line2D([], [], marker="*", ls="", color=RED, ms=8, label="μ = 0.5")],
               loc="upper right", fontsize=6.0, handlelength=1.3, handletextpad=0.5)
    axC.set_xlim(-0.12, 5.2)
    axC.set_ylim(0.15, 0.80)
    axC.set_xlabel("Target-marginal penalty μ")
    axC.set_ylabel("Direction sensitivity ‖d(μ)‖")

    # ---------------- (d) unseen-site forest ----------------
    panel_label(axD, "d")
    rows = ext.copy()
    ypos, labels, colors, sep = [], [], [], []
    y = 0.0
    prev_cohort = None
    for _, r in rows.iterrows():
        if prev_cohort is not None and r.cohort != prev_cohort:
            y -= 0.9
            sep.append(y + 0.45)
        y -= 1.0
        ypos.append(y)
        labels.append(r.site)
        colors.append(BLUE if r.cohort == "GSE246662" else ORANGE)
        prev_cohort = r.cohort
    ypos = np.array(ypos)
    for yi, (_, r), c in zip(ypos, rows.iterrows(), colors):
        axD.plot([r.ci_low, r.ci_high], [yi, yi], color=c, lw=1.6, solid_capstyle="butt")
        axD.scatter(r.gain_mean, yi, s=34, color=c, edgecolors="white", lw=0.6, zorder=3)
        axD.annotate(f"{r.relative_gain*100:.0f}%", (r.ci_high, yi),
                     textcoords="offset points", xytext=(6, -2), fontsize=6.2,
                     color=c, fontweight="bold")
    for sy in sep:
        axD.axhline(sy, color="#DDDDDD", lw=0.5)
    axD.set_yticks(ypos)
    axD.set_yticklabels(labels)
    axD.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="GSE246662"),
                        Line2D([], [], marker="o", ls="", color=ORANGE, label="GSE183904")],
               loc="upper left", bbox_to_anchor=(0, -0.24), ncol=2,
               fontsize=5.8, handletextpad=0.3, columnspacing=0.8)
    axD.set_xlim(0.085, 0.137)
    axD.set_ylim(ypos.min() - 1.15, ypos.max() + 0.5)
    axD.set_xlabel("Reduction in reconstruction MAE", fontsize=7)
    light_xgrid(axD)

    figure_titles(fig, "Identifiability and external validation", [
        (axA, "Target-state curvature"), (axB, "Coefficient recovery"),
        (axC, "KL-anchor sensitivity"), (axD, "Unseen metastatic sites")])
    save_all(fig, "Fig2_identifiability_and_validation")


def save_all(fig, name):
    stem = os.path.join(OUT, name)
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    qa(f"{name} content fits fixed canvas", bb.x0 >= -0.01 and bb.y0 >= -0.01
       and bb.x1 <= w + 0.01 and bb.y1 <= h + 0.01,
       f"bounds={bb.bounds}; canvas={w:.3f} x {h:.3f} in")
    with mpl.rc_context({"savefig.bbox": None}):
        for ext in ("pdf", "svg", "png"):
            fig.savefig(f"{stem}.{ext}", bbox_inches=None, dpi=300, facecolor="white")
        fig.savefig(f"{stem}.tiff", bbox_inches=None, dpi=600, facecolor="white",
                    pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"[saved] {name} (.pdf/.svg/.png/.tiff)")


# ============================================================================
# FIGURE 3 — context decomposition & lineage history (schematic dropped)
# (a) rank strip across 3 tasks  | (b) TENA/Cdh paired slopegraph
# (c) unified branch-effect forest, both systems
# ============================================================================
def fig3():
    ranks = pd.read_csv(os.path.join(SRC, "Fig3a_all_method_ranks.csv"))
    paired = pd.read_csv(os.path.join(SRC, "Fig3b_lineage_paired_values.csv"))
    stats3 = pd.read_csv(os.path.join(SRC, "Fig3b_lineage_context_statistics.csv"))
    eff = pd.read_csv(os.path.join(SRC, "Fig3c_lineage_effects_with_ci.csv"))

    qa("Fig3a global-alpha mean rank 1.33",
       abs(ranks.loc[ranks.method == "Gated_IOT_global_alpha", "rank"].mean() - 1.33) < 0.01)
    cos = stats3.loc[stats3.statistic == "direction_cosine_TENA_vs_Cdh", "value"].iloc[0]
    qa("Fig3b cosine -0.80", abs(cos + 0.7962) < 0.01, f"cos={cos:.3f}")
    p_t = stats3.loc[stats3.statistic == "TENA_history_label_OOF_permutation_p", "value"].iloc[0]
    p_c = stats3.loc[stats3.statistic == "Cdh_history_label_OOF_permutation_p", "value"].iloc[0]
    qa("Fig3b perm P 0.013/0.030", abs(p_t - 0.0130) < 1e-3 and abs(p_c - 0.0300) < 1e-3)
    tena_del = (paired[paired.system == "TENA"].eval("ever_emt_lineage - epithelial_lineage"))
    cdh_del = (paired[paired.system == "Cdh"].eval("ever_emt_lineage - epithelial_lineage"))
    qa("Fig3b TENA all negative / Cdh all positive",
       bool((tena_del < 0).all()) and bool((cdh_del > 0).all()))
    sig = eff[eff.p < 0.05]
    qa("Fig3c significant branches", len(sig) == 4, ", ".join(sig.branch))

    fig = new_fig(183, 159)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[0.95, 1.25],
                          left=0.25, right=0.975, top=0.895, bottom=0.14,
                          wspace=0.50, hspace=0.56)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    # ---------------- (a) rank strip ----------------
    disp = {
        "Gated_IOT_global_alpha": "Gated-IOT global α",
        "Gated_IOT_DM": "Gated-IOT DM",
        "pure_IOT": "Pure IOT",
        "Gated_IOT_square_loss": "Gated-IOT squared loss",
        "IOT_HRT": "IOT-HRT",
        "supervised_DM": "Supervised DM",
        "zero_centered_ridge": "Zero-centred ridge",
        "empirical_centered_ridge": "Empirical-centred ridge",
        "target_marginal": "Target marginal",
        "state_persistence": "State persistence",
    }
    task_shape = {
        "GSE173958_macsgestalt_locked_state": ("o", "macsGESTALT"),
        "GSE228154_lineage_grouped_state": ("s", "Resistance time course"),
        "GSE239651_external_state": ("^", "External lineage"),
    }
    our = set(ranks[ranks.our_method].method.unique())
    mr = ranks.groupby("method")["rank"].mean().sort_values()
    order = mr.index.tolist()
    panel_label(axA, "a", dx=-0.32)
    axA.axvspan(0.55, 3.45, color=PURPLE, alpha=0.06, lw=0, zorder=0)
    axA.text(2.0, 0.12, "top-3", fontsize=6.0, color=PURPLE,
             ha="center", va="bottom", fontweight="bold")
    for yi, m in enumerate(order):
        yy = len(order) - 1 - yi
        is_our = m in our
        c = PURPLE if is_our else GREY
        sub = ranks[ranks.method == m]
        axA.plot([sub["rank"].min(), sub["rank"].max()], [yy, yy],
                 color=c, alpha=0.25, lw=0.9, zorder=1)
        for _, r in sub.iterrows():
            sh, _ = task_shape[r.task]
            offset = {"o": 0.19, "s": 0.0, "^": -0.19}[sh]
            axA.scatter(r["rank"], yy + offset, marker=sh, s=18,
                        facecolors=c if is_our else "white",
                        edgecolors=c, lw=0.8, zorder=3)
        axA.scatter(mr[m], yy, marker="D", s=19, color=BLACK, zorder=4)
        axA.annotate(f"{mr[m]:.2f}", (10.55, yy), fontsize=6.2, color=c,
                     fontweight="bold" if is_our else "normal", va="center")
    axA.set_yticks(range(len(order)))
    axA.set_yticklabels([disp[m] for m in order[::-1]], fontsize=6.2)
    for tick, m in zip(axA.get_yticklabels(), order[::-1]):
        if m in our:
            tick.set_color(PURPLE)
            tick.set_fontweight("bold")
    axA.set_xlim(0.55, 11.3)
    axA.set_ylim(-0.7, len(order) - 0.4)
    axA.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    axA.set_xlabel("Rank within task  (1 = best)")
    handles = [plt.Line2D([], [], marker=sh, ls="", color=BLACK, markerfacecolor="white",
                          markersize=5, label=lab)
               for sh, lab in task_shape.values()]
    handles.append(plt.Line2D([], [], marker="D", ls="", color=BLACK, markersize=5,
                              label="mean rank"))
    axA.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.40, -0.24),
               ncol=2, fontsize=5.6, handletextpad=0.2, columnspacing=0.7,
               borderpad=0.2, frameon=False)

    # ---------------- (b) TENA / Cdh slopegraph ----------------
    panel_label(axB, "b")
    sys_style = {"TENA": (BLUE, "o", "TENA (n=5)"), "Cdh": (RED, "s", "Cdh (n=3)")}
    for system, (c, mk, lab) in sys_style.items():
        d = paired[paired.system == system]
        for _, r in d.iterrows():
            arr = FancyArrowPatch((0, r.epithelial_lineage), (1, r.ever_emt_lineage),
                                  arrowstyle="-|>", mutation_scale=7, color=c,
                                  lw=1.1, alpha=0.75, zorder=2,
                                  shrinkA=2.2, shrinkB=2.2)
            axB.add_patch(arr)
        axB.scatter(np.zeros(len(d)), d.epithelial_lineage, s=26, color=c,
                    marker=mk, edgecolors="white", lw=0.5, zorder=3)
        axB.scatter(np.ones(len(d)), d.ever_emt_lineage, s=26, color=c,
                    marker=mk, edgecolors="white", lw=0.5, zorder=3)
        axB.scatter([0, 1], [d.epithelial_lineage.mean(), d.ever_emt_lineage.mean()],
                    s=70, marker="D", facecolors=c, edgecolors=BLACK, lw=0.7, zorder=4)
    axB.set_xlim(-0.45, 1.75)
    axB.set_ylim(0.90, 3.5)
    axB.set_xticks([0, 1])
    axB.set_xticklabels(["Epithelial\nlineage", "ever-EMT\nlineage"])
    axB.set_ylabel("Invasive-EMT branch score")
    handles = [plt.Line2D([], [], marker=mk, ls="", color=c, markersize=5, label=lab)
               for c, mk, lab in sys_style.values()]
    handles.append(plt.Line2D([], [], marker="D", ls="", color=GREY, markersize=5,
                              label="group mean"))
    axB.legend(handles=handles, loc="upper left", fontsize=6.0, handletextpad=0.15,
               borderpad=0.2)

    # ---------------- (c) unified branch forest ----------------
    panel_label(axC, "c", dx=-0.055)
    branches = ["EPI", "INV_EMT", "INF_EMT", "IFN_HLA", "PROLIF", "DORM_STRESS"]
    blab = {"EPI": "Epithelial", "INV_EMT": "Invasive EMT", "INF_EMT": "Inflammatory EMT",
            "IFN_HLA": "IFN/HLA", "PROLIF": "Proliferation", "DORM_STRESS": "Dormancy/stress"}
    yb = np.arange(len(branches))[::-1].astype(float)
    for system, off, c, mk in (("TENA", 0.16, BLUE, "o"), ("Cdh", -0.16, RED, "s")):
        d = eff[eff.system == system].set_index("branch").loc[branches]
        for yi, (_, r) in zip(yb + off, d.iterrows()):
            sig_i = r.p < 0.05
            axC.errorbar(r.mean_difference_a_minus_b, yi,
                         xerr=[[r.mean_difference_a_minus_b - r.ci_low],
                               [r.ci_high - r.mean_difference_a_minus_b]],
                         fmt=mk, ms=5, color=c, ecolor=c, elinewidth=1.2, capsize=2.2,
                         mfc=c if sig_i else "white", mew=1.0, mec=c,
                         alpha=1.0 if sig_i else 0.62, zorder=3)
            axC.plot([r.mean_difference_a_minus_b - r.se, r.mean_difference_a_minus_b + r.se],
                     [yi, yi], color=c, lw=3.0, alpha=0.7, zorder=2)
    axC.axvline(0, ls="--", color=GREY, lw=0.7, alpha=0.8)
    axC.set_yticks(yb)
    axC.set_yticklabels([blab[b] for b in branches])
    axC.set_xlim(-6.6, 4.6)
    axC.set_ylim(-0.75, len(branches) - 0.2)
    axC.set_xlabel("Mean branch-score difference, ever-EMT − epithelial lineage", fontsize=7)
    light_xgrid(axC)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, markersize=5, label="TENA (n=5)"),
               plt.Line2D([], [], marker="s", ls="", color=RED, markersize=5, label="Cdh (n=3)")]
    handles += [Line2D([], [], marker="o", ls="", color=DARKGREY, mfc=DARKGREY,
                       ms=4, label="P < 0.05"),
                Line2D([], [], marker="o", ls="", color=DARKGREY, mfc="white",
                       ms=4, label="P ≥ 0.05")]
    axC.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.16),
               ncol=4, fontsize=5.8, handletextpad=0.2, columnspacing=1.0)

    figure_titles(fig, "Context-dependent performance and lineage history", [
        (axA, "Performance across public tasks"), (axB, "Lineage-specific direction shifts"),
        (axC, "Branch-level history effects")])
    save_all(fig, "Fig3_context_and_lineage_history")


# ============================================================================
# FIGURE 4 — pre-treatment immune visibility & response triangulation
# (a) HERO raincloud IFN/HLA   | (b) null raincloud invasive-EMT
# (c) branch forest            | (d1) GDSC lollipop + (d2) PRRX1 perturbation
# ============================================================================
def fig4():
    pat = pd.read_csv(os.path.join(SRC, "Fig4ab_patient_scores.csv"))
    desc = pd.read_csv(os.path.join(SRC, "Fig4ab_descriptive_mean_ci.csv"))
    effc = pd.read_csv(os.path.join(SRC, "Fig4c_patient_effects.csv"))
    gdsc = pd.read_csv(os.path.join(SRC, "Fig4d_gdsc_egfr_tki.csv"))
    prrx = pd.read_csv(os.path.join(PERT, "gse164488_prrx1_perturbation_effects.tsv"), sep="\t")

    n_rd = (pat.outcome == "RD").sum()
    n_pcr = (pat.outcome == "pCR").sum()
    qa("Fig4 patients 34 RD + 45 pCR", n_rd == 34 and n_pcr == 45 and len(pat) == 79,
       f"RD={n_rd}, pCR={n_pcr}")
    ifn_q = effc.loc[effc.feature == "score_IFN_HLA", "bh_q_value"].iloc[0]
    inv_q = effc.loc[effc.feature == "score_INV_EMT", "bh_q_value"].iloc[0]
    ifn_d = effc.loc[effc.feature == "score_IFN_HLA", "difference_pCR_minus_RD"].iloc[0]
    ifn_auc = effc.loc[effc.feature == "score_IFN_HLA", "auroc_pCR"].iloc[0]
    qa("Fig4 IFN q=0.0122, diff +0.260, AUROC 0.697",
       abs(ifn_q - 0.01219) < 1e-3 and abs(ifn_d - 0.2604) < 1e-3 and abs(ifn_auc - 0.6967) < 1e-3)
    qa("Fig4 INV q=0.821", abs(inv_q - 0.8208) < 1e-3)
    med = gdsc.spearman_rho.median()
    qa("Fig4d median rho 0.253 / all P<1e-5", abs(med - 0.2535) < 1e-3 and bool((gdsc.p < 1e-5).all()))
    inv_pr = prrx[(prrx.contrast == "siPRRX1_minus_siCTR") & (prrx.branch == "INV_EMT")].iloc[0]
    qa("Fig4d2 siPRRX1 INV-EMT P=0.016", abs(inv_pr.paired_p - 0.01601) < 1e-3)

    rng = np.random.default_rng(SEED)
    fig = new_fig(183, 218)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.12, 1.0, 1.0],
                          left=0.19, right=0.975, top=0.918, bottom=0.095,
                          wspace=0.54, hspace=0.55)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])
    axD1 = fig.add_subplot(gs[2, 0])
    axD2 = fig.add_subplot(gs[2, 1])

    def raincloud(ax, col, hero=True):
        d = pat[["outcome", col]].dropna().copy()
        desc_f = desc[desc.feature == col].set_index("outcome")
        cats = ["RD", "pCR"]
        cols = {"RD": GREY, "pCR": ORANGE}
        for xi, cat in enumerate(cats):
            v = d.loc[d.outcome == cat, col].values
            c = cols[cat]
            kde = gaussian_kde(v)
            kde.set_bandwidth(kde.factor * 0.62)
            ys = np.linspace(v.min(), v.max(), 140)
            dens = kde(ys)
            dens = dens / dens.max() * 0.30
            ax.fill_betweenx(ys, xi + 0.045, xi + 0.045 + dens, color=c,
                             alpha=0.32 if hero else 0.22, lw=0, zorder=1)
            jit = rng.uniform(-0.22, -0.055, len(v))
            ax.scatter(xi + jit, v, s=13, color=c, alpha=0.75 if hero else 0.5,
                       edgecolors="white", lw=0.3, zorder=3)
            m = desc_f.loc[cat, "mean"]
            lo_, hi_ = desc_f.loc[cat, "mean_ci_low"], desc_f.loc[cat, "mean_ci_high"]
            ax.errorbar(xi + 0.055, m, yerr=[[m - lo_], [hi_ - m]], fmt="D", ms=5.5,
                        color=BLACK, ecolor=BLACK, elinewidth=1.0, capsize=2.5,
                        mfc="white", mew=1.1, zorder=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"RD\nn={n_rd}", f"pCR\nn={n_pcr}"])
        ax.set_xlim(-0.62, 1.62)
        ax.set_ylim(-0.02, max(float(pat[col].max()), float(desc_f.mean_ci_high.max())) * 1.12)
        ax.set_ylabel("IFN/HLA programme score" if hero else "Invasive-EMT programme score")
        ax.set_title("")

    panel_label(axA, "a")
    raincloud(axA, "score_IFN_HLA", hero=True)
    axA.legend(handles=[Line2D([], [], marker="o", ls="", color=GREY, ms=3, label="Patient"),
                        Line2D([], [], marker="D", color=BLACK, mfc="white", ms=4,
                               label="Mean (95% CI)")],
               loc="upper center", bbox_to_anchor=(0.50, -0.25), ncol=2,
               fontsize=5.8, columnspacing=0.7, handletextpad=0.4)

    panel_label(axB, "b")
    raincloud(axB, "score_INV_EMT", hero=False)

    # ---------------- (c) branch forest ----------------
    panel_label(axC, "c", dx=-0.055)
    order = ["score_EPI", "score_INV_EMT", "score_INF_EMT", "score_IFN_HLA",
             "score_PROLIF", "score_DORM_STRESS"]
    blab = {"score_EPI": "Epithelial", "score_INV_EMT": "Invasive EMT",
            "score_INF_EMT": "Inflammatory EMT", "score_IFN_HLA": "IFN/HLA",
            "score_PROLIF": "Proliferation", "score_DORM_STRESS": "Dormancy/stress"}
    e = effc.set_index("feature").loc[order]
    yb = np.arange(len(order))[::-1].astype(float)
    for yi, (feat, r) in zip(yb, e.iterrows()):
        hero_i = feat == "score_IFN_HLA"
        c = RED if hero_i else DARKGREY
        axC.errorbar(r.difference_pCR_minus_RD, yi,
                     xerr=[[r.difference_pCR_minus_RD - r.bootstrap_95_low],
                           [r.bootstrap_95_high - r.difference_pCR_minus_RD]],
                     fmt="o", ms=5.2, color=c, ecolor=c, elinewidth=1.4, capsize=2.5,
                     mfc=c, mec="white", mew=0.5, zorder=3)
        axC.annotate(f"q = {r.bh_q_value:.3f}", (0.445, yi), fontsize=6.0,
                     color=c if hero_i else GREY,
                     fontweight="bold" if hero_i else "normal", va="center",
                     annotation_clip=False)
    axC.axvline(0, ls="--", color=GREY, lw=0.7, alpha=0.8)
    axC.set_yticks(yb)
    axC.set_yticklabels([blab[f] for f in order])
    axC.set_xlim(-0.24, 0.55)
    qa("Fig4c complete patient confidence intervals", bool((e.bootstrap_95_low >= -0.24).all()
       and (e.bootstrap_95_high <= 0.55).all()))
    axC.set_ylim(-0.75, 6.05)
    axC.set_xlabel("Mean difference, pCR − RD  (bootstrap 95% CI)")
    light_xgrid(axC)

    # ---------------- (d1) GDSC lollipop ----------------
    panel_label(axD1, "d")
    g = gdsc.sort_values("spearman_rho")
    yg = np.arange(len(g)).astype(float)
    axD1.hlines(yg, 0, g.spearman_rho, color="#CCCCCC", lw=1.1, zorder=1)
    axD1.scatter(g.spearman_rho, yg, s=34, color=BLUE, edgecolors="white", lw=0.6,
                 zorder=3)
    for yi, (_, r) in zip(yg, g.iterrows()):
        axD1.text(0.58, yi, f"{r.spearman_rho:.3f}   n={r.n_cell_lines}",
                   ha="right", va="center", fontsize=5.7, color=TICK_GREY)
    axD1.axvline(med, ls="--", color=ORANGE, lw=0.9)
    axD1.legend(handles=[Line2D([], [], color=ORANGE, ls="--", lw=1,
                               label=f"Median ρ = {med:.3f}")],
                loc="upper left", fontsize=6.0)
    axD1.set_yticks(yg)
    axD1.set_yticklabels(g.drug)
    axD1.set_xlim(0, 0.61)
    axD1.set_xticks([0, 0.2, 0.4, 0.6])
    axD1.set_ylim(-0.6, len(g) + 1.0)
    axD1.set_xlabel("Spearman ρ, EMT vs ln(IC50)", fontsize=7)
    light_xgrid(axD1)

    # ---------------- (d2) PRRX1 perturbation ----------------
    panel_label(axD2, "e")
    p = prrx[prrx.contrast == "siPRRX1_minus_siCTR"].copy()
    raw = pd.read_csv(os.path.join(PERT, "gse164488_dog_sample_features.tsv"), sep="\t")
    raw = raw[(raw.cell_line == "MDCK-NBL2") & (raw.time_days == 4)]
    treated = raw[raw.condition == "TGFb_siPRRX1"].set_index("replicate").sort_index()
    control = raw[raw.condition == "TGFb_siCTR"].set_index("replicate").sort_index()
    qa("Fig4e matched three biological replicates", len(treated) == 3 and treated.index.equals(control.index))
    b2 = {"EPI": "Epithelial", "INV_EMT": "Invasive EMT", "INF_EMT": "Inflammatory EMT",
          "IFN_HLA": "IFN/HLA", "PROLIF": "Proliferation", "DORM_STRESS": "Dormancy/stress"}
    p["lab"] = p.branch.map(b2)
    yp = np.arange(len(p))[::-1].astype(float)
    replicate_rows = []
    ci_extent = []
    for yi, (_, r) in zip(yp, p.iterrows()):
        sig_i = r.paired_p < 0.05
        c = RED if sig_i else DARKGREY
        diffs = (treated[r.branch] - control[r.branch]).to_numpy()
        half_ci = student_t.ppf(0.975, 2) * diffs.std(ddof=1) / np.sqrt(3)
        mean = diffs.mean()
        qa(f"Fig4e {r.branch} reproduces saved mean and paired P",
           np.isclose(mean, r.mean_difference_a_minus_b, atol=1e-10) and
           np.isclose(ttest_rel(treated[r.branch], control[r.branch]).pvalue, r.paired_p, atol=1e-10))
        axD2.scatter(diffs, yi + np.array([-0.15, 0, 0.15]), s=16, marker="o",
                     color=c, alpha=0.4, edgecolors="white", lw=0.3, zorder=2)
        axD2.errorbar(mean, yi, xerr=half_ci, fmt="D", ms=4.3,
                      color=c, mfc=c if sig_i else "white", mec=c, capsize=2.2,
                      elinewidth=1.0, mew=0.8, zorder=4)
        ci_extent += [mean - half_ci, mean + half_ci]
        for rep, diff in zip(treated.index, diffs):
            replicate_rows.append(dict(branch=r.branch, replicate=int(rep), difference=diff,
                                       mean=mean, ci_low=mean-half_ci, ci_high=mean+half_ci,
                                       paired_p=r.paired_p))
    pd.DataFrame(replicate_rows).to_csv(os.path.join(DERIVED, "Fig4e_prrx1_paired_replicates.csv"), index=False)
    axD2.axvline(0, color=GREY, lw=0.7, ls="--")
    axD2.set_yticks(yp)
    axD2.set_yticklabels(p.lab, fontsize=6.0)
    axD2.set_xlim(min(ci_extent) - 0.15, max(ci_extent) + 0.15)
    axD2.set_ylim(-0.7, 6.3)
    axD2.set_xlabel("Δ branch score\n(siPRRX1 − siCTR)", fontsize=7)
    axD2.tick_params(axis="x", labelsize=6)
    axD2.legend(handles=[Line2D([], [], marker="o", ls="", color=GREY, ms=3,
                               label="Paired replicate"),
                        Line2D([], [], marker="D", color=DARKGREY, mfc="white", ms=4,
                               label="Mean (95% CI)")],
                loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=5.7,
                ncol=2, columnspacing=0.6, handletextpad=0.3)

    figure_titles(fig, "Patient programmes and experimental evidence", [
        (axA, "IFN/HLA and response"), (axB, "Invasive EMT and response"),
        (axC, "Branch specificity of treatment response"),
        (axD1, "EGFR-TKI response in GDSC"), (axD2, "Paired PRRX1 perturbation")])
    save_all(fig, "Fig4_patient_programmes")


# ============================================================================
# FIGURE 5 — performance (a,b), calibration (c,d), detection (e,f)
# (a) CE drop dumbbells | (b) top-1 agreement dumbbells
# (c) Calibration scatter with low-abundance inset; (d) condition and pooled bias
# (e) Paired Brier difference | (f) detection AUPRC
# ============================================================================
def fig5():
    met = pd.read_csv(os.path.join(SRC, "Fig5ab_state_metrics.csv"))
    wins = pd.read_csv(os.path.join(SRC, "Fig5c_external_all_windows.csv"))
    summ = pd.read_csv(os.path.join(SRC, "Fig5c_external_condition_state_summary.csv"))
    brier = pd.read_csv(os.path.join(SRC, "Fig5d_brier_paired_bootstrap.csv"))
    bnd = pd.read_csv(os.path.join(SRC, "Fig5d_detection_boundary.csv"))
    bnd = bnd[bnd.metric == "AUPRC"].copy()

    dev = met[met.partition == "development_outer_oof"].set_index("method")
    ext = met[met.partition == "locked_external_e1"].set_index("method")
    ce_d = (dev.loc["PERSIST-no-IOT", "state_cross_entropy"],
            dev.loc["PERSIST-IOT", "state_cross_entropy"])
    ce_e = (ext.loc["PERSIST-no-IOT", "state_cross_entropy"],
            ext.loc["PERSIST-IOT", "state_cross_entropy"])
    t1_d = (dev.loc["PERSIST-no-IOT", "state_top1_agreement"],
            dev.loc["PERSIST-IOT", "state_top1_agreement"])
    t1_e = (ext.loc["PERSIST-no-IOT", "state_top1_agreement"],
            ext.loc["PERSIST-IOT", "state_top1_agreement"])
    drop_d = 1 - ce_d[1] / ce_d[0]
    drop_e = 1 - ce_e[1] / ce_e[0]
    qa("Fig5a CE drops 78.6%/78.5%", abs(drop_d - 0.7862) < 2e-3 and abs(drop_e - 0.7849) < 2e-3,
       f"{drop_d:.3f}/{drop_e:.3f}")
    qa("Fig5b top-1 +14.7/+20.7 pp",
       abs((t1_d[1] - t1_d[0]) - 0.1473) < 2e-3 and abs((t1_e[1] - t1_e[0]) - 0.2070) < 2e-3)
    qa("Fig5c 56 windows", wins.groupby(["condition", "window"]).ngroups == 56)
    qa("Fig5e paired CIs exclude 0",
       bool(brier.ci_low.iloc[0] > 0 and brier.ci_high.iloc[1] < 0))
    qa("Fig5f AUPRC roles", set(bnd.role.unique()) == {"PERSIST-IOT", "Best supervised"}
       and len(bnd) == 4 and set(bnd.metric.unique()) == {"AUPRC"})
    _v_ext = float(bnd.query("partition == 'locked_external_e1' and role == 'PERSIST-IOT'").value.iloc[0])
    _v_dev = float(bnd.query("partition == 'development_outer_oof' and role == 'Best supervised'").value.iloc[0])
    qa("Fig5f AUPRC values match source",
       abs(_v_ext - 0.7156) < 2e-3 and abs(_v_dev - 0.8098) < 2e-3,
       f"ext {_v_ext:.4f} / dev {_v_dev:.4f}")
    agg = wins.groupby(["condition", "state"])["observed"].mean().sort_index()
    agg_ref = summ.set_index(["condition", "state"])["observed"].sort_index()
    qa("Fig5d window->summary aggregation", bool(np.allclose(agg.values, agg_ref.values, atol=1e-6)))

    rng = np.random.default_rng(SEED)
    fig = new_fig(183, 195)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.55, 1.13, 0.55],
                          left=0.18, right=0.975, top=0.915, bottom=0.12,
                          wspace=0.50, hspace=0.67)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axF = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[2, 0])
    axE = fig.add_subplot(gs[2, 1])

    rows = ["Development\nouter OOF", "Locked\nexternal E1"]

    # ---------------- (a) CE drop ----------------
    panel_label(axA, "a")
    for yi, (ctl, per, drop) in enumerate([(ce_d[0], ce_d[1], drop_d),
                                           (ce_e[0], ce_e[1], drop_e)]):
        y = 1 - yi
        axA.plot([ctl, per], [y, y], color="#D0D0D0", lw=2.6, solid_capstyle="round",
                 zorder=1)
        axA.scatter(ctl, y, s=36, color=GREY, edgecolors="white", lw=0.6, zorder=3)
        axA.scatter(per, y, s=44, color=PURPLE, edgecolors="white", lw=0.6, zorder=3)
        axA.annotate(f"{ctl:.2f}", (ctl, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=6.0, color=TICK_GREY)
        axA.annotate(f"{per:.2f}", (per, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=6.2, color=PURPLE,
                     fontweight="bold")
        axA.annotate(f"−{drop*100:.1f}%", (per, y), textcoords="offset points",
                     xytext=(14, -3), fontsize=6.6, color=PURPLE, fontweight="bold")
    axA.set_yticks([1, 0])
    axA.set_yticklabels(rows, fontsize=6.4)
    axA.set_xlim(0.2, 6.4)
    axA.set_ylim(-0.55, 1.6)
    axA.set_xticks([1, 2, 3, 4, 5])
    axA.set_xlabel("State cross-entropy", fontsize=7)
    light_xgrid(axA)
    handles = [plt.Line2D([], [], marker="o", ls="", color=GREY, markersize=5, label="Without transport"),
               plt.Line2D([], [], marker="o", ls="", color=PURPLE, markersize=5, label="PERSIST-IOT")]
    axA.legend(handles=handles, loc="upper left", fontsize=5.8, handletextpad=0.3,
               borderpad=0.2, bbox_to_anchor=(-0.03, -0.40), ncol=2, columnspacing=0.8)

    # ---------------- (b) top-1 agreement ----------------
    panel_label(axB, "b")
    for yi, (ctl, per) in enumerate([(t1_d[0], t1_d[1]), (t1_e[0], t1_e[1])]):
        y = 1 - yi
        axB.plot([ctl, per], [y, y], color="#D0D0D0", lw=2.6, solid_capstyle="round",
                 zorder=1)
        axB.scatter(ctl, y, s=36, color=GREY, edgecolors="white", lw=0.6, zorder=3)
        axB.scatter(per, y, s=44, color=PURPLE, edgecolors="white", lw=0.6, zorder=3)
        axB.annotate(f"+{(per-ctl)*100:.1f} pp", (per, y), textcoords="offset points",
                     xytext=(8, -2), fontsize=6.6, color=PURPLE, fontweight="bold")
    axB.set_yticks([1, 0])
    axB.set_yticklabels(rows, fontsize=6.4)
    axB.set_xlim(0.42, 0.80)
    axB.set_ylim(-0.55, 1.6)
    axB.set_xticks([0.45, 0.55, 0.65, 0.75])
    axB.set_xticklabels(["45%", "55%", "65%", "75%"])
    axB.set_xlabel("Top-1 agreement", fontsize=7)
    light_xgrid(axB)

    # ---------------- (c) HERO calibration ----------------
    panel_label(axC, "c")
    con_col = {"DMSO": BLUE, "ispinesib": SOFT_RED}
    con_mk = {"DMSO": "o", "ispinesib": "s"}
    lim = 0.92
    for cond, d in wins.groupby("condition"):
        c = con_col[cond]
        size = 5 + 18 * np.sqrt(d.n_cases / wins.n_cases.max())
        axC.scatter(d.predicted, d.observed, s=size, marker=con_mk[cond], color=c,
                    alpha=0.50, edgecolors="white", lw=0.3, zorder=2,
                    label=f"{cond} (window)")
    for cond, d in summ.groupby("condition"):
        c = con_col[cond]
        axC.scatter(d.predicted, d.observed, s=43, marker=con_mk[cond], color=c,
                    edgecolors=BLACK, lw=0.9, zorder=4,
                    label=f"{cond} (condition mean)")
    axC.plot([-0.02, lim], [-0.02, lim], ls="--", color=GREY, lw=0.7, alpha=0.7,
             zorder=1)
    axC.set_xlim(-0.03, lim)
    axC.set_ylim(-0.03, lim)
    axC.set_aspect("equal")
    axC.set_anchor("C")
    axC.set_xlabel("Predicted future composition", fontsize=7)
    axC.set_ylabel("Observed future composition", fontsize=7)
    # zoom inset for low-abundance states — placed in the emptiest corner
    pts = np.c_[np.r_[wins.predicted.values, summ.predicted.values],
                np.r_[wins["observed"].values, summ["observed"].values]]
    axins = axC.inset_axes([0.04, 0.66, 0.34, 0.31])
    norm_x, norm_y = (wins.predicted + 0.03) / 0.95, (wins.observed + 0.03) / 0.95
    hidden = norm_x.between(0.04, 0.38) & norm_y.between(0.66, 0.97)
    qa("Fig5c inset covers no main-panel observations", int(hidden.sum()) == 0)
    axins.set_facecolor("white")
    axins.set_zorder(5)
    for cond, d in wins.groupby("condition"):
        c = con_col[cond]
        size = 5 + 18 * np.sqrt(d.n_cases / wins.n_cases.max())
        axins.scatter(d.predicted, d.observed, s=size * 0.8, marker=con_mk[cond],
                      color=c, alpha=0.42, edgecolors="white", lw=0.2, zorder=2)
    axins.plot([-0.02, lim], [-0.02, lim], ls="--", color=GREY, lw=0.6, alpha=0.9,
               zorder=4)
    axins.set_xlim(-0.012, 0.13)
    axins.set_ylim(-0.012, 0.13)
    axins.tick_params(labelsize=5, length=1.5, pad=1.2)
    axins.set_xticks([0, 0.1])
    axins.set_yticks([0, 0.1])
    for s in axins.spines.values():
        s.set_visible(True)
        s.set_color("#BBBBBB")
        s.set_linewidth(0.5)
    axC.indicate_inset_zoom(axins, edgecolor=GREY, alpha=0.65, lw=0.6)
    axC.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2,
               fontsize=5.2, handletextpad=0.2, borderpad=0.2,
               columnspacing=0.6, labelspacing=0.3, markerscale=0.75)

    # ---------------- (e) Brier difference ----------------
    panel_label(axD, "e")
    yb = [1, 0]
    for yi, (_, r) in zip(yb, brier.iterrows()):
        axD.errorbar(r.difference_persist_minus_supervised, yi,
                     xerr=[[r.difference_persist_minus_supervised - r.ci_low],
                           [r.ci_high - r.difference_persist_minus_supervised]],
                     fmt="o", ms=5, color=PURPLE, ecolor=PURPLE, elinewidth=1.3,
                     capsize=2.5, mfc=PURPLE, mec="white", mew=0.5, zorder=3)
    axD.axvline(0, ls="--", color=GREY, lw=0.7)
    axD.set_yticks(yb)
    axD.set_yticklabels(["Dev OOF", "Ext E1"], fontsize=6.4)
    axD.set_xlim(-0.0038, 0.0092)
    axD.set_ylim(-0.45, 1.45)
    axD.set_xlabel("Δ Brier\n(PERSIST-IOT − best supervised)", fontsize=7)
    axD.tick_params(axis="x", labelsize=6)
    light_xgrid(axD)

    # ---------------- (f) detection AUPRC ----------------
    panel_label(axE, "f")
    b = bnd.pivot_table(index="partition", columns="role", values="value")
    yb2 = [1, 0]
    for yi, part in zip(yb2, ["development_outer_oof", "locked_external_e1"]):
        v_sup = b.loc[part, "Best supervised"]
        v_per = b.loc[part, "PERSIST-IOT"]
        axE.plot([v_sup, v_per], [yi, yi], color="#D0D0D0", lw=1.6, zorder=1)
        axE.scatter(v_sup, yi, s=30, color=ORANGE, edgecolors="white", lw=0.5, zorder=3)
        axE.scatter(v_per, yi, s=34, color=PURPLE, edgecolors="white", lw=0.5, zorder=3)
        axE.annotate(f"{v_sup:.3f}", (v_sup, yi), xytext=(0, 7), textcoords="offset points",
                     ha="center", fontsize=6, color=ORANGE)
        axE.annotate(f"{v_per:.3f}", (v_per, yi), xytext=(0, -12), textcoords="offset points",
                     ha="center", fontsize=6, color=PURPLE)
    axE.set_yticks(yb2)
    axE.set_yticklabels(["Dev OOF", "Ext E1"], fontsize=6.4)
    axE.set_xlim(0.70, 0.825)
    axE.set_xticks([0.72, 0.76, 0.80])
    axE.set_ylim(-0.55, 1.55)
    axE.set_xlabel("AUPRC", fontsize=7)
    handles = [plt.Line2D([], [], marker="o", ls="", color=PURPLE, markersize=5, label="PERSIST-IOT"),
               plt.Line2D([], [], marker="o", ls="", color=ORANGE, markersize=5, label="Best supervised")]
    axE.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.03, -0.37),
               fontsize=5.6, handletextpad=0.3, borderpad=0.2, ncol=2, columnspacing=0.6)

    # ---------------- (d) per-state calibration bias ----------------
    panel_label(axF, "d")
    states = sorted(wins.state.unique())
    yf = np.arange(len(states))[::-1].astype(float)
    bias_rows = []
    for yi, st in zip(yf, states):
        d = wins[wins.state == st]
        vals = (d["observed"] - d["predicted"]).values
        boots = np.array([rng.choice(vals, len(vals), replace=True).mean()
                          for _ in range(1000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        m = vals.mean()
        sig_i = lo > 0 or hi < 0
        c = DARKGREY
        axF.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt="D", ms=4.8, color=c,
                     ecolor=c, elinewidth=1.3, capsize=2.5, mfc=c if sig_i else "white",
                     mec=c, mew=1.0, zorder=3)
        bias_rows.append(dict(state=int(st), condition="Pooled", n_windows=len(vals),
                              mean_bias=m, ci_low=lo, ci_high=hi))
        for cond, offset in [("DMSO", 0.20), ("ispinesib", -0.20)]:
            cvals = d.loc[d.condition == cond, "observed"] - d.loc[d.condition == cond, "predicted"]
            axF.scatter(cvals.mean(), yi + offset, s=22, marker=con_mk[cond],
                         color=con_col[cond], edgecolors="white", lw=0.4, zorder=4)
            bias_rows.append(dict(state=int(st), condition=cond, n_windows=len(cvals),
                                  mean_bias=cvals.mean(), ci_low=np.nan, ci_high=np.nan))
    pd.DataFrame(bias_rows).to_csv(os.path.join(DERIVED, "Fig5d_condition_and_pooled_bias.csv"), index=False)
    axF.axvline(0, ls="--", color=GREY, lw=0.7, alpha=0.8)
    axF.set_yticks(yf)
    axF.set_yticklabels([f"State {s}" for s in states], fontsize=6.4)
    axF.set_ylim(-0.7, 6.3)
    axF.set_xlim(-0.075, 0.085)
    axF.set_xticks([-0.05, 0, 0.05])
    axF.set_xlabel("Mean observed − predicted", fontsize=7)
    light_xgrid(axF)
    axF.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="DMSO mean"),
                        Line2D([], [], marker="s", ls="", color=SOFT_RED, ms=4, label="ispinesib mean"),
                        Line2D([], [], marker="D", color=DARKGREY, mfc="white", ms=4,
                               label="Pooled mean (95% CI)")],
               loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=5.4,
               handletextpad=0.4, labelspacing=0.3, ncol=2, columnspacing=0.5)

    figure_titles(fig, "Future-state composition and persistence prediction", [
        (axA, "Future-state cross-entropy"), (axB, "Dominant-state agreement"),
        (axC, "Predicted vs observed composition"), (axF, "State-specific calibration bias"),
        (axD, "Persistence detection (Δ Brier)"), (axE, "Persistence detection (AUPRC)")])
    save_all(fig, "Fig5_future_state_prediction")


# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Statistical figure set v5 — layered data and revised layout (academic-figure-skill)")
    print("=" * 70)
    fig2()
    fig3()
    fig4()
    fig5()
    print("-" * 70)
    n_fail = sum(1 for _, s, _ in QA_LOG if s == "FAIL")
    print(f"QA summary: {len(QA_LOG)} checks, {n_fail} failures")
    manifest = {
        "generated_by": "script/nature_stat_redesign/plot_statistical_figures_v5.py",
        "skill": "academic-figure-skill (codex)",
        "random_seed": SEED,
        "manuscript_claim_authority": "paper/中文正文.docx",
        "manuscript_modified": False,
        "figure_scope": "Fig2-Fig5 revisions; Fig1 is generated separately with native draw.io objects",
        "titles": "figure titles and panel main titles retained; grey subtitles removed",
        "Fig5_layout": "three rows: composition performance (a,b), calibration (c,d), persistence detection (e,f)",
        "statistical_caliber_sources": [
            "output/nature_stat_redesign/source_data_v3/*.csv",
            "output/artemis_emt_branches/mechanism_v2/perturbation/gse164488_prrx1_perturbation_effects.tsv",
            "output/artemis_emt_branches/mechanism_v2/perturbation/gse164488_dog_sample_features.tsv",
        ],
        "figures": [
            "Fig2_identifiability_and_validation",
            "Fig3_context_and_lineage_history",
            "Fig4_patient_programmes",
            "Fig5_future_state_prediction",
        ],
        "formats": ["PDF vector", "SVG editable text", "PNG 300 dpi", "TIFF LZW 600 dpi"],
        "qa_log": [{"check": n, "status": s, "detail": d} for n, s, d in QA_LOG],
    }
    with open(os.path.join(OUT, "figure_build_manifest_v5.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[saved] figure_build_manifest_v5.json")
    sys.exit(1 if n_fail else 0)
