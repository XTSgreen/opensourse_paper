# Academic Figure Skill Asset Confirmation
# All panels inherit the user's plot_statistical_figures_v5.py visual system.
# Layouts, semantic colours, marker styles and fixed-canvas export are retained.
# Quantitative inputs come from the current benchmark result tables.
"""Current evidence in the original v5/v6 visual design; no model fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
    "figure.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": .6, "ytick.major.width": .6, "legend.frameon": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
})
BLUE, RED, GREEN, ORANGE, PURPLE, DARKGREY = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
GREY, BLACK = "#999999", "#222222"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/rendered"
SOURCES: dict[str, set[str]] = {}
QA: list[dict] = []
METHODS = {"UOT-IOT": "UOT-IOT", "wot": "Waddington-OT", "moscot": "moscot",
           "lineageot": "LineageOT", "cellrank2": "CellRank 2", "tigon": "TIGON",
           "mioflow": "MIOFlow", "prescient": "PRESCIENT",
           "development-target-transfer": "Development transfer",
           "source-carry-forward": "Source carry-forward"}
DATASETS = {"gse140802_t2_t16": "GSE140802 t2→t16", "gse140802_t2_t9": "GSE140802 t2→t9",
            "gse239651_transition_panel": "GSE239651", "macsgestalt_transition_panel": "macsGESTALT"}
SHAPES = dict(zip(DATASETS, ["o", "s", "^", "D"]))
SCENARIOS = {
    "identifiable_soft_marginal": "Identifiable",
    "hard_marginal_flat_direction": "Flat direction",
    "equivalent_fit_multiple_parameters": "Equivalent fits",
    "shared_plus_context_offset": "Context offset",
    "observation_noise_and_dropout": "Noise / dropout",
    "nonlinear_or_hidden_confounding_misspecification": "Misspecified",
}
B1 = "results/B1_known_truth_final/b1_all_runs.csv"
B2 = "results/B2_lineage_transition/b2_method_summary.csv"
RUNS = "results/B2_lineage_transition/b2_all_runs.csv"
STABILITY = "results/direction_stability/output_direction_seed_stability.csv"
EXTERNAL = "results/external_direction_validation/frozen_direction_external_validation.csv"
LOCKED = "results/B3_prospective_composition/b3_state_prediction_metrics.csv"
POP = "results/B3_prospective_composition/external_dynamics/b3_external_all_runs.csv"
CAL = "results/B3_prospective_composition/external_dynamics/b3_calibration_summary.csv"


def load(panel, path):
    SOURCES.setdefault(panel, set()).add(path)
    frame = pd.read_csv(ROOT / path)
    if frame.empty:
        raise ValueError(f"Empty source: {path}")
    return frame


def title(fig, text, axes):
    fig.suptitle(text, x=.53, y=.99, fontsize=11, fontweight="bold", color=BLACK)
    for ax, heading in axes:
        ax.set_title(heading, loc="left", fontsize=7.5, fontweight="bold", pad=8, color=BLACK)


def panel(ax, letter, dx=-.13):
    ax.text(dx, 1.04, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left", color=BLACK)


def grid(ax):
    ax.grid(axis="x", color="#E0E0E0", lw=.3, alpha=.55)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    fits = bb.x0 >= -.01 and bb.y0 >= -.01 and bb.x1 <= w+.01 and bb.y1 <= h+.01
    QA.append({"figure": name, "content_fits_canvas": bool(fits), "bounds_inches": list(bb.bounds),
               "width_mm": w*25.4, "height_mm": h*25.4})
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches=None, dpi=300, facecolor="white")
    plt.close(fig)


def shape_legend(ax, y=-.24):
    handles = [Line2D([], [], marker=SHAPES[d], ls="", mfc="white", mec=DARKGREY,
                      ms=4, label=DATASETS[d]) for d in DATASETS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-.02, y), ncol=2,
              fontsize=5.8, handletextpad=.3, columnspacing=.8)


def method_strip(ax, frame, metric, xlabel, *, legend=True):
    """v5 rank-strip geometry; actual errors replace ranks, without imputation."""
    order = [m for m in METHODS if m in set(frame.method)]
    for i, m in enumerate(order):
        part = frame[frame.method.eq(m)]
        finite = part[np.isfinite(part[metric])]
        c = PURPLE if m == "UOT-IOT" else GREY
        if not finite.empty:
            ax.plot([finite[metric].min(), finite[metric].max()], [i, i], color=c, alpha=.3, lw=.9)
        else:
            ax.text(.99, i, "NA", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6, color=GREY)
        for _, row in finite.iterrows():
            offset = list(DATASETS).index(row.dataset)*.13-.195
            ax.scatter(row[metric], i+offset, marker=SHAPES[row.dataset], s=20,
                       facecolors=c if m == "UOT-IOT" else "white", edgecolors=c, lw=.8, zorder=3)
    ax.set_yticks(range(len(order)), [METHODS[m] for m in order], fontsize=6.5)
    ax.set_ylim(len(order)-.35, -.7)
    ax.set_xlabel(xlabel, fontsize=7)
    grid(ax)
    if legend:
        shape_legend(ax)


def figure2():
    b1 = load("Figure2a", B1)
    for p in "bc":
        load("Figure2"+p, B1)
    ext = load("Figure2d", EXTERNAL)
    fig = plt.figure(figsize=(183/25.4, 148/25.4))
    gs = fig.add_gridspec(2, 2, left=.17, right=.975, top=.885, bottom=.14, wspace=.45, hspace=.63)
    a, b, c, d = [fig.add_subplot(gs[i,j]) for i,j in [(0,0),(0,1),(1,0),(1,1)]]
    for ax, letter in zip([a,b,c,d], "abcd"):
        panel(ax, letter)
    # symlog keeps exact zero curvatures visible; no fabricated log floor.
    curv = b1.groupby(["scenario", "method"]).pure_column_minimum_curvature.mean().unstack()
    for i, scenario in enumerate(SCENARIOS):
        lo, hi = curv.loc[scenario, ["hard_ot", "soft_iot"]]
        a.plot([lo, hi], [i,i], color="#C9C9C9", lw=1.1)
        a.scatter(lo,i,s=30,facecolors="white",edgecolors=DARKGREY,lw=1)
        a.scatter(hi,i,s=36,color=BLUE,edgecolors="white",lw=.6,zorder=3)
    a.set_xscale("symlog", linthresh=1e-5)
    a.set_yticks(range(6), list(SCENARIOS.values()), fontsize=6)
    a.set_ylim(5.6,-.6)
    a.set_xlabel("Minimum curvature (symlog)", fontsize=7)
    grid(a)
    a.legend(handles=[Line2D([],[],marker="o",ls="",mfc="white",mec=DARKGREY,label="Hard OT"),
                      Line2D([],[],marker="o",ls="",color=BLUE,label="UOT-IOT")],
             loc="upper left",bbox_to_anchor=(-.02,-.22),fontsize=6,ncol=2,handletextpad=.3)
    for method,color,filled in [("hard_ot",DARKGREY,False),("soft_iot",BLUE,True)]:
        part=b1[b1.method.eq(method)]
        truth=np.concatenate([json.loads(v) for v in part.theta_true])
        fitted=np.concatenate([json.loads(v) for v in part.theta_hat])
        if not np.isfinite(np.r_[truth,fitted]).all():
            raise ValueError("Nonfinite coefficient in B1")
        b.scatter(truth,fitted,s=8,facecolors=color if filled else "none",edgecolors=color,
                  lw=.35,alpha=.38,zorder=3 if filled else 2)
    lim=max(abs(b.get_xlim()[0]),abs(b.get_xlim()[1]),abs(b.get_ylim()[0]),abs(b.get_ylim()[1]))
    b.plot([-lim,lim],[-lim,lim],ls="--",color=GREY,lw=.6)
    b.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel="True coefficient",ylabel="Recovered coefficient")
    b.set_aspect("equal")
    b.tick_params(labelsize=6)
    summary=b1.groupby(["sample_count","method"]).restart_direction_variance.agg(["mean","std"]).reset_index()
    for method,color,marker in [("hard_ot",DARKGREY,"s"),("soft_iot",BLUE,"o")]:
        part=summary[summary.method.eq(method)]
        c.plot(part.sample_count,part["mean"],color=color,marker=marker,ms=4,lw=1.2)
    c.set_xscale("log")
    c.set_yscale("symlog",linthresh=1e-12)
    c.set_ylim(0,summary["mean"].max()*2)
    c.set_yticks([0,1e-12,1e-8,1e-4,1e-1])
    c.set_xlabel("Sample count (log)",fontsize=7)
    c.set_ylabel("Restart variance (symlog)",fontsize=7)
    for i,row in ext.iterrows():
        color=BLUE if row.validation_cohort=="GSE246662" else ORANGE
        d.errorbar(row.gain,i,xerr=[[row.gain-row.ci95_lower],[row.ci95_upper-row.gain]],
                   fmt="o",color=color,ms=4,capsize=2,lw=1.2)
    d.set_yticks(range(len(ext)),[s.split("_")[-1] for s in ext.site],fontsize=6)
    d.invert_yaxis()
    d.set_xlabel("Reduction in transition MAE",fontsize=7)
    grid(d)
    d.legend(handles=[Line2D([],[],marker="o",ls="",color=BLUE,label="GSE246662"),
                      Line2D([],[],marker="o",ls="",color=ORANGE,label="GSE183904")],
             loc="upper left",bbox_to_anchor=(0,-.24),ncol=2,fontsize=5.8,handletextpad=.3,columnspacing=.8)
    title(fig,"Identifiability and external validation",[(a,"Target-state curvature"),(b,"Coefficient recovery"),
          (c,"Stability across sample sizes"),(d,"Unseen metastatic sites")])
    save(fig,"Figure2")


def figure3():
    b2=load("Figure3a",B2)
    stability=load("Figure3b",STABILITY)
    load("Figure3b",B2)
    load("Figure3c",B2)
    fig=plt.figure(figsize=(183/25.4,159/25.4))
    gs=fig.add_gridspec(2,2,width_ratios=[1.25,1],height_ratios=[.95,1.25],
                       left=.25,right=.975,top=.895,bottom=.14,wspace=.50,hspace=.56)
    a,b,c=fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])
    for ax,l in zip([a,b,c],"abc"):
        panel(ax,l,dx=-.32 if l=="a" else -.13)
    method_strip(a,b2,"transition_weighted_l1_mean","Transition-matrix L1 ↓")
    joined=b2.merge(stability,on=["dataset","method"],how="inner")
    colors={"UOT-IOT":PURPLE,"mioflow":BLUE,"prescient":GREEN,"tigon":ORANGE}
    for m,part in joined.groupby("method"):
        for _,r in part.iterrows():
            b.scatter(r.transition_weighted_l1_mean,r.pairwise_cosine_mean,s=24,
                      color=colors[m],marker=SHAPES[r.dataset],edgecolors="white",lw=.4)
    b.set_xlabel("Transition-matrix L1 ↓",fontsize=7)
    b.set_ylabel("Output-proxy cosine ↑",fontsize=7)
    b.set_ylim(.65,1.025)
    b.legend(handles=[Line2D([],[],marker="o",ls="",color=colors[m],label=METHODS[m]) for m in colors],
             loc="upper left",bbox_to_anchor=(-.05,-.24),ncol=2,fontsize=5.8,columnspacing=.6,handletextpad=.3)
    method_strip(c,b2,"fate_spearman_mean","Fate Spearman correlation ↑",legend=False)
    title(fig,"Transition recovery and stability across methods",[(a,"Lineage transition recovery"),
          (b,"Prediction–stability relationship"),(c,"Fate correlation across all four panels")])
    fig.text(.25,.038,"Output-proxy stability is distinct from signed-parameter identifiability. NA denotes no coupling output.",fontsize=5.8,color=DARKGREY)
    save(fig,"Figure3")


def population_strip(ax, frame, metric, label):
    # Each symbol is a dataset mean of panel means; seeds stay nested in panels.
    per_panel=frame.groupby(["dataset","method","panel"])[metric].mean().reset_index()
    summary=per_panel.groupby(["dataset","method"])[metric].mean().reset_index()
    order=["development-target-transfer","source-carry-forward","mioflow","prescient","tigon"]
    datasets=sorted(summary.dataset.unique())
    for i,m in enumerate(order):
        vals=summary[summary.method.eq(m)]
        ax.plot([vals[metric].min(),vals[metric].max()],[i,i],color="#D0D0D0",lw=1)
        for j,ds in enumerate(datasets):
            r=vals[vals.dataset.eq(ds)]
            if len(r)!=1:
                raise ValueError(f"Missing population comparison {m}, {ds}")
            ax.scatter(r[metric].iloc[0],i+(j-.5)*.20,s=25,marker=["o","s"][j],
                       color=[BLUE,ORANGE][j],edgecolors="white",lw=.4,zorder=3)
    ax.set_yticks(range(len(order)),[METHODS[m] for m in order],fontsize=5.8)
    ax.set_ylim(len(order)-.4,-.6)
    ax.set_xlabel(label,fontsize=7)
    grid(ax)


def figure4():
    locked=load("Figure4a",LOCKED)
    load("Figure4b",LOCKED)
    locked=locked[locked.status.eq("success")]
    pop=load("Figure4c",POP)
    for p in "de":load("Figure4"+p,POP)
    cal=load("Figure4f",CAL)
    fig=plt.figure(figsize=(183/25.4,195/25.4))
    gs=fig.add_gridspec(3,2,height_ratios=[.55,1.13,.85],left=.18,right=.975,top=.915,bottom=.12,
                       wspace=.70,hspace=.67)
    axes=[fig.add_subplot(gs[i,j]) for i,j in [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]]
    for ax,l in zip(axes,"abcdef"):panel(ax,l)
    for ax,metric in zip(axes[:2],["cross_entropy","sinkhorn_divergence"]):
        for i,ds in enumerate(sorted(locked.dataset.unique())):
            row=locked[locked.dataset.eq(ds)].set_index("method")
            ctl,value=row.loc["PERSIST-no-IOT",metric],row.loc["PERSIST-IOT",metric]
            ax.plot([ctl,value],[i,i],color="#D0D0D0",lw=2.6,solid_capstyle="round")
            for v,col in [(ctl,GREY),(value,PURPLE)]:
                ax.scatter(v,i,s=38,color=col,edgecolors="white",lw=.6,zorder=3)
                ax.annotate(f"{v:.3f}",(v,i),xytext=(0,7),textcoords="offset points",ha="center",fontsize=6,color=col)
        ax.set_yticks([0,1],["Development\nouter OOF","Locked\nexternal E1"],fontsize=6)
        ax.set_ylim(1.55,-.6)
        ax.margins(x=.17)
        grid(ax)
    axes[0].set_xlabel("Clone-level cross-entropy",fontsize=7)
    axes[1].set_xlabel("Clone-level Sinkhorn divergence",fontsize=7)
    axes[0].legend(handles=[Line2D([],[],marker="o",ls="",color=GREY,label="Without transport"),
                           Line2D([],[],marker="o",ls="",color=PURPLE,label="PERSIST-IOT")],
                   loc="upper left",bbox_to_anchor=(-.03,-.43),ncol=2,fontsize=5.8,columnspacing=.8,handletextpad=.3)
    for ax,metric,label in zip(axes[2:5],["cross_entropy","sinkhorn_divergence","top1_accuracy"],
            ["Population cross-entropy ↓","Population Sinkhorn divergence ↓","Population top-1 accuracy ↑"]):
        population_strip(ax,pop,metric,label)
    order=["development-target-transfer","source-carry-forward","mioflow","prescient","tigon"]
    for i,m in enumerate(order):
        for j,ds in enumerate(sorted(cal.evaluation_dataset.unique())):
            r=cal[cal.method.eq(m)&cal.evaluation_dataset.eq(ds)].iloc[0]
            # Percentile bootstrap CIs can exclude the point estimate: draw endpoints directly.
            yy=i+(j-.5)*.2
            axes[5].plot([r.ci95_lower,r.ci95_upper],[yy,yy],color=[BLUE,ORANGE][j],lw=1)
            axes[5].scatter(r.expected_calibration_error,yy,s=22,color=[BLUE,ORANGE][j],
                            marker=["o","s"][j],edgecolors="white",lw=.4,zorder=3)
    axes[5].set_yticks(range(5),[METHODS[m] for m in order],fontsize=5.8)
    axes[5].set_ylim(4.6,-.6)
    axes[5].set_xlabel("State-wise calibration error ↓",fontsize=7)
    grid(axes[5])
    fig.legend(handles=[Line2D([],[],marker="o",ls="",color=BLUE,label="GSE140802"),
                        Line2D([],[],marker="s",ls="",color=ORANGE,label="GSE239651 expt2")],
               loc="lower center",bbox_to_anchor=(.57,.035),ncol=2,fontsize=6)
    title(fig,"Future-state composition and calibration",list(zip(axes,["Clone-level cross-entropy",
          "Clone-level transport distance","Population composition error","Population transport distance",
          "Dominant-state agreement","Calibration and 95% intervals"])))
    save(fig,"Figure4")


def figure5():
    ext=load("Figure5a",EXTERNAL)
    b2=load("Figure5c",B2)
    runs=load("Figure5b",RUNS)
    fig=plt.figure(figsize=(183/25.4,159/25.4))
    gs=fig.add_gridspec(2,2,width_ratios=[1.25,1],height_ratios=[.95,1.25],
                       left=.25,right=.975,top=.895,bottom=.14,wspace=.50,hspace=.56)
    a,b,c=fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])
    for ax,l in zip([a,b,c],"abc"):panel(ax,l)
    for i,(cohort,part) in enumerate(ext.groupby("validation_cohort",sort=False)):
        color=[BLUE,ORANGE][i]
        a.scatter(part.relative_gain*100,i+np.linspace(-.15,.15,len(part)),s=26,color=color,alpha=.65,edgecolors="white",lw=.5)
        a.scatter(part.relative_gain.mean()*100,i,s=45,color=color,marker="D",edgecolors=BLACK,lw=.6,zorder=4)
    a.set_yticks([0,1],["GSE246662\n(n = 3 sites)","GSE183904\n(n = 3 sites)"],fontsize=6)
    a.set_ylim(1.55,-.6)
    a.set_xlabel("Relative MAE reduction (%)",fontsize=7)
    grid(a)
    a.legend(handles=[Line2D([],[],marker="o",ls="",color=GREY,label="Individual site"),
                      Line2D([],[],marker="D",ls="",color=DARKGREY,label="Cohort mean")],
             loc="upper left",bbox_to_anchor=(0,-.23),fontsize=6)
    order=[m for m in METHODS if m in set(runs.method)]
    for i,m in enumerate(order):
        part=runs[runs.method.eq(m)]
        vals=part.runtime_seconds.dropna().to_numpy()
        vals=vals[vals>0]
        color=PURPLE if m=="UOT-IOT" else GREY
        b.scatter(vals,i+np.linspace(-.16,.16,len(vals)),s=9,color=color,alpha=.40,lw=0)
        b.scatter(np.median(vals),i,s=24,color=color,marker="D",edgecolors="white",lw=.4,zorder=4)
    b.set_yticks(range(len(order)),[METHODS[m] for m in order],fontsize=6)
    b.set_ylim(len(order)-.4,-.6)
    b.set_xscale("log")
    b.set_xlabel("Runtime per run (s, log)",fontsize=7)
    method_strip(c,b2,"coupling_relative_frobenius_mean","Relative coupling error ↓")
    title(fig,"External transfer and computational performance",[(a,"Frozen-direction transfer"),
          (b,"Observed computation times"),(c,"Coupling recovery across lineage panels")])
    save(fig,"Figure5")


def mini(name, width, height, draw):
    """Match the v6 mini-panel's exact placement size and Arial visual system."""
    route=OUT/"technical_route_current"
    dest=route/"R_panels"
    dest.mkdir(parents=True,exist_ok=True)
    with mpl.rc_context({"font.size":5.5,"axes.labelsize":5.7,"xtick.labelsize":5.2,
                         "ytick.labelsize":5.2,"legend.fontsize":5.2}):
        fig,ax=plt.subplots(figsize=(width*183/1800/25.4,height*183/1800/25.4))
        draw(ax)
        fig.tight_layout(pad=.35)
        for ext in ("svg","pdf","png"):
            fig.savefig(dest/f"{name}.{ext}",dpi=600,facecolor="white")
        plt.close(fig)


def figure1(drawio):
    route=OUT/"technical_route_current"
    original=ROOT/"figures/legacy_v5/technical_route_v6"
    # The three input-description panels and state bars are unchanged data inputs.
    for sub in ["source_data","R_panels"]:
        shutil.copytree(original/sub,route/sub,dirs_exist_ok=True)
    for name in ["GSE228154_cells.csv","GSE228154_log_expression.mtx.gz","GSE228154_state_programmes.csv"]:
        SOURCES.setdefault("Figure1_inputs",set()).add(f"figures/legacy_v5/technical_route_v6/source_data/{name}")
    b1=load("Figure1_identifiability",B1)
    ext=load("Figure1_transfer",EXTERNAL)
    b2=load("Figure1_transition",B2)
    stab=load("Figure1_stability",STABILITY)
    cal=load("Figure1_calibration",CAL)
    locked=load("Figure1_composition",LOCKED)
    pop=load("Figure1_population",POP)
    def curvature(ax):
        t=b1.groupby(["scenario","method"]).pure_column_minimum_curvature.mean().unstack()
        for i,s in enumerate(SCENARIOS):
            l,h=t.loc[s,["hard_ot","soft_iot"]]
            ax.plot([l,h],[i,i],color="#C5C5C5",lw=1)
            ax.scatter(l,i,s=10,facecolors="white",edgecolors=GREY,lw=.5)
            ax.scatter(h,i,s=11,color=BLUE,lw=0)
        ax.set_yticks(range(6),list(SCENARIOS.values()),fontsize=5)
        ax.invert_yaxis();ax.set_xscale("symlog",linthresh=1e-5);ax.set_xlabel("Curvature (symlog)")
        ax.set_xlim(0,float(t.soft_iot.max())*1.3)
        ax.set_xticks([0,1e-4,1e-2])
    mini("04_curvature",380,200,curvature)
    def recovery(ax):
        for m,col in [("hard_ot",GREY),("soft_iot",BLUE)]:
            part=b1[b1.method.eq(m)]
            x=np.concatenate([json.loads(v) for v in part.theta_true]);y=np.concatenate([json.loads(v) for v in part.theta_hat])
            ax.scatter(x,y,s=2,color=col,alpha=.25,lw=0)
        ax.plot([-1,1],[-1,1],ls="--",color=GREY,lw=.4)
        ax.set(xlabel="True coefficient",ylabel="Recovered")
    mini("05_synthetic_recovery",380,200,recovery)
    def stability(ax):
        t=b1.groupby(["sample_count","method"]).restart_direction_variance.mean().unstack()
        ax.plot(t.index,t.hard_ot,color=GREY,marker="s",ms=2,lw=.7,label="Hard OT")
        ax.plot(t.index,t.soft_iot,color=BLUE,marker="o",ms=2,lw=.7,label="UOT-IOT")
        ax.set_xscale("log");ax.set_yscale("symlog",linthresh=1e-12)
        ax.set_ylim(0,float(t.hard_ot.max())*2)
        ax.set_yticks([0,1e-8,1e-1])
        ax.set(xlabel="Sample count (log)",ylabel="Variance (symlog)");ax.legend(fontsize=5,loc="center right")
    mini("06_anchor_sensitivity",380,200,stability)
    def transfer(ax):
        for i,r in ext.iterrows():
            col=BLUE if r.validation_cohort=="GSE246662" else ORANGE
            ax.plot([r.ci95_lower,r.ci95_upper],[i,i],color=col,lw=1)
            ax.scatter(r.gain,i,s=10,color=col)
        ax.set_yticks(range(len(ext)),[s.split("_")[-1] for s in ext.site]);ax.invert_yaxis()
        ax.set_xlabel("MAE reduction (95% CI)")
    mini("07_external_transfer",380,200,transfer)
    def transitions(ax):
        for i,m in enumerate(["UOT-IOT","wot","moscot","lineageot","mioflow","prescient","tigon"]):
            p=b2[b2.method.eq(m)].transition_weighted_l1_mean.dropna()
            ax.scatter(p,np.full(len(p),i),s=8,color=PURPLE if m=="UOT-IOT" else GREY,alpha=.7)
        ax.set_yticks(range(7),[METHODS[m] for m in ["UOT-IOT","wot","moscot","lineageot","mioflow","prescient","tigon"]],fontsize=5)
        ax.invert_yaxis();ax.set_xlabel("Transition L1")
    mini("08_lineage_pairs",380,255,transitions)
    def proxies(ax):
        for i,m in enumerate(["UOT-IOT","mioflow","prescient","tigon"]):
            p=stab[stab.method.eq(m)].pairwise_cosine_mean
            ax.scatter(p,np.full(len(p),i),s=10,color=PURPLE if m=="UOT-IOT" else GREY)
        ax.set_yticks(range(4),["UOT-IOT","MIOFlow","PRESCIENT","TIGON"]);ax.invert_yaxis()
        ax.set_xlabel("Output-proxy cosine")
    mini("09_patient_response",380,255,proxies)
    def capability(ax):
        av=load("Figure1_capability","results/direction_stability/direction_parameter_availability.csv")
        vals=av.explicit_comparable_direction.astype(int).to_numpy()[None,:]
        from matplotlib.colors import ListedColormap
        ax.imshow(vals,cmap=ListedColormap(["#EEEEEE",PURPLE]),vmin=0,vmax=1,aspect="auto")
        labels=[str(x).replace("Waddington-OT","WOT").replace("CellRank2","CellRank 2") for x in av.method]
        ax.set_xticks(range(len(labels)),labels,fontsize=5);ax.set_yticks([])
        for i,v in enumerate(vals[0]):ax.text(i,0,"Yes" if v else "NA",ha="center",va="center",fontsize=5,color="white" if v else DARKGREY)
        for s in ax.spines.values():s.set_visible(False)
    mini("10_patient_programme_heatmap",790,150,capability)
    def calibration(ax):
        for j,ds in enumerate(sorted(cal.evaluation_dataset.unique())):
            part=cal[cal.evaluation_dataset.eq(ds)].set_index("method")
            for i,m in enumerate(["development-target-transfer","source-carry-forward","mioflow","prescient","tigon"]):
                r=part.loc[m];y=i+(j-.5)*.2
                ax.plot([r.ci95_lower,r.ci95_upper],[y,y],color=[BLUE,ORANGE][j],lw=.7)
                ax.scatter(r.expected_calibration_error,y,s=8,color=[BLUE,ORANGE][j],marker=["o","s"][j])
        ax.set_yticks(range(5),["Dev. transfer","Carry-forward","MIOFlow","PRESCIENT","TIGON"])
        ax.invert_yaxis();ax.set_xlabel("Calibration error (95% CI)")
    mini("11_external_calibration",430,335,calibration)
    def ce(ax):
        for i,ds in enumerate(["GSE239651_expt1","GSE239651_expt2"]):
            part=locked[locked.dataset.eq(ds)&locked.status.eq("success")].set_index("method")
            a,b=part.loc["PERSIST-no-IOT","cross_entropy"],part.loc["PERSIST-IOT","cross_entropy"]
            ax.plot([a,b],[i,i],color="#C5C5C5",lw=1)
            ax.scatter(a,i,s=12,color=GREY);ax.scatter(b,i,s=12,color=PURPLE)
        ax.set_yticks([0,1],["Dev OOF","External E1"]);ax.set_ylim(1.6,-.6);ax.set_xlabel("Clone-level cross-entropy")
    mini("12_state_performance",420,170,ce)
    def population(ax):
        t=pop.groupby(["dataset","method","panel"]).cross_entropy.mean().groupby(["dataset","method"]).mean().reset_index()
        for j,ds in enumerate(sorted(t.dataset.unique())):
            p=t[t.dataset.eq(ds)].set_index("method")
            for i,m in enumerate(["development-target-transfer","source-carry-forward","mioflow","prescient","tigon"]):
                ax.scatter(p.loc[m,"cross_entropy"],i+(j-.5)*.15,s=7,color=[BLUE,ORANGE][j],marker=["o","s"][j])
        ax.set_yticks(range(5),["Dev. transfer","Carry-forward","MIOFlow","PRESCIENT","TIGON"],fontsize=5)
        ax.invert_yaxis();ax.set_xlabel("Population cross-entropy")
    mini("13_detection_brier",420,170,population)
    subprocess.run([sys.executable,str(ROOT/"scripts/legacy/build_technical_route_drawio_v6.py"),
                    "--route-dir",str(route),"--current-benchmarks"],check=True)
    drawio_file=route/"Fig1_technical_route_v6.drawio"
    for fmt in ["pdf","svg","png"]:
        cmd=[drawio,"--export","--format",fmt,"--output",str(OUT/f"Figure1.{fmt}"),str(drawio_file)]
        if fmt=="pdf":cmd.extend(["--crop"])
        if fmt=="png":cmd.extend(["--scale","1.5"])
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=120,
                         creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0)
        if p.returncode or not (OUT/f"Figure1.{fmt}").exists():
            raise RuntimeError(f"draw.io {fmt} export failed: {p.stdout} {p.stderr}")
    shutil.copy2(drawio_file,OUT/"Figure1.drawio")
    QA.append({"figure":"Figure1","editable_drawio":True,"visual_template":"technical_route_v6",
               "current_result_mini_panels":10,"unchanged_input_description_panels":3})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drawio",default=shutil.which("drawio") or "C:/Program Files/draw.io/draw.io.exe")
    parser.add_argument("--figures",nargs="+",type=int,default=[1,2,3,4,5],choices=[1,2,3,4,5])
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    for n in args.figures:
        print(f"Rendering Figure {n} with current results and v5/v6 styling",flush=True)
        if n==1:figure1(args.drawio)
        else:globals()[f"figure{n}"]()
    records=[]
    for p,paths in sorted(SOURCES.items()):
        fig=p.split("_")[0][:7]
        for path in sorted(paths):
            records.append({"figure_panel":p,"script":"scripts/render_current_v5.py","input":path,
                            "input_sha256":hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
                            "output_pdf":f"figures/rendered/{fig}.pdf","output_png":f"figures/rendered/{fig}.png"})
    manifest=ROOT/"results/reviewer_tables/figure_source_manifest.csv"
    current=pd.DataFrame(records)
    if len(args.figures)!=5 and manifest.exists():
        old=pd.read_csv(manifest)
        keep=~old.figure_panel.str.match("^("+"|".join(f"Figure{n}" for n in args.figures)+")")
        current=pd.concat([old.loc[keep],current],ignore_index=True)
    current.to_csv(manifest,index=False)
    report={"style_reference":"figures_revision_v5 + technical_route_v6","data_authority":"current benchmark results",
            "historical_numeric_results_used_for_benchmark_panels":False,"checks":QA,
            "cross_entropy_floor":1e-12,"statistics_document":"figures/CURRENT_FIGURE_NOTES.md"}
    (OUT/"QA_REPORT.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    failed=[q for q in QA if q.get("content_fits_canvas") is False]
    print(json.dumps(report,indent=2))
    return 1 if failed else 0


if __name__=="__main__":raise SystemExit(main())
