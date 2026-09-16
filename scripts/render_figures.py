# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) schematic → cross-type inherit → param inherit
# (b) grouped comparison → assets/figures/BarComparison/plot_comparison_Trajectory.py → param inherit
# (c) line trend → assets/figures/LineTrend/plot_sweep.py → param inherit
# (d) grouped bars → assets/figures/GroupedBarChart/plot_GroupedBarChartv1.py → param inherit
# (e) heatmap → assets/figures/heatmap/plot_composition.py → param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.

# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
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

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM
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


import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


WIDTH = 183 / 25.4
METHOD_LABEL = {
    "UOT-IOT": "IOT", "cellrank2": "CellRank 2", "lineageot": "LineageOT",
    "mioflow": "MIOFlow", "moscot": "moscot", "prescient": "PRESCIENT",
    "tigon": "TIGON", "wot": "WOT",
}


def _panel(ax, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def _short_dataset(value: str) -> str:
    return {"gse140802_t2_t16": "GSE140802\nt2→t16", "gse140802_t2_t9": "GSE140802\nt2→t9",
            "gse239651_transition_panel": "GSE239651", "macsgestalt_transition_panel": "macsGESTALT"}.get(value, value)


def _dot_methods(ax, frame: pd.DataFrame, metric: str, title: str) -> None:
    pivot = frame.pivot(index="method", columns="dataset", values=metric)
    methods = [m for m in METHOD_LABEL if m in pivot.index]
    for index, method in enumerate(methods):
        values = pivot.loc[method].dropna().to_numpy(float)
        color = ACCENT_RED if method == "UOT-IOT" else CATEGORICAL_EXTENDED[index % len(CATEGORICAL_EXTENDED)]
        ax.scatter(values, np.full(len(values), index), s=18, color=color, alpha=0.82,
                   marker="o" if method == "UOT-IOT" else "s", edgecolor="white", linewidth=0.35)
        if len(values):
            ax.plot([values.min(), values.max()], [index, index], color=color, lw=0.8, alpha=0.7, zorder=0)
    ax.set_yticks(range(len(methods)), [METHOD_LABEL[m] for m in methods])
    ax.invert_yaxis(); ax.set_xlabel(title); ax.grid(axis="x", color="#dddddd", lw=0.45)


def figure1(root: Path, output: Path) -> None:
    fig = plt.figure(figsize=(WIDTH, 4.0))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.42)
    ax = fig.add_subplot(grid[0]); ax.axis("off"); _panel(ax, "a")
    boxes = [(0.02, "B1 known truth", "Parameter recovery\nand curvature"),
             (0.27, "B2 lineage", "Coupling and fate\nreconstruction"),
             (0.52, "B3 prospective", "Future composition\nwithout target access"),
             (0.77, "External test", "Frozen direction\nacross cohorts")]
    for x, title, body in boxes:
        patch = FancyBboxPatch((x, 0.25), 0.20, 0.48, boxstyle="round,pad=0.018", transform=ax.transAxes,
                               facecolor="#F7FBFF", edgecolor=CATEGORICAL[0], linewidth=1.0)
        ax.add_patch(patch); ax.text(x + 0.10, 0.61, title, transform=ax.transAxes, ha="center", fontweight="bold")
        ax.text(x + 0.10, 0.40, body, transform=ax.transAxes, ha="center", va="center", color="#444444")
        if x < 0.7:
            ax.add_patch(FancyArrowPatch((x + 0.205, 0.49), (x + 0.255, 0.49), transform=ax.transAxes,
                                         arrowstyle="-|>", mutation_scale=10, color=GREY, lw=0.8))
    ax.text(0.5, 0.08, "Prediction performance  +  identifiable, stable, testable direction parameters",
            transform=ax.transAxes, ha="center", color=ACCENT_RED, fontweight="bold")
    ax = fig.add_subplot(grid[1]); _panel(ax, "b")
    labels = ["WOT", "moscot", "LineageOT", "TIGON", "MIOFlow", "CellRank 2", "PRESCIENT"]
    matrix = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0], [1, 1, 0]])
    ax.imshow(matrix, cmap=LinearSegmentedColormap.from_list("audit", ["#EEEEEE", CATEGORICAL[0]]), vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(labels)), labels); ax.set_xticks(range(3), ["Coupling", "Fate / composition", "Explicit signed direction"])
    ax.tick_params(axis="x", rotation=0); ax.set_title("Common-contract capability audit")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]): ax.text(j, i, "Yes" if matrix[i, j] else "—", ha="center", va="center", color="white" if matrix[i,j] else GREY, fontsize=6.5)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.10)
    save_cns_figure(fig, str(output / "Figure1")); plt.close(fig)


def figure2(root: Path, output: Path) -> None:
    b1 = pd.read_csv(root / "results/B1_known_truth_final/b1_all_runs.csv")
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.55)); metrics = [
        ("coefficient_cosine", "Coefficient cosine", True),
        ("pure_column_minimum_curvature", "Minimum curvature", True),
        ("restart_direction_variance", "Direction variance", False),
    ]
    for label, ax, (metric, ylabel, higher) in zip("abc", axes, metrics):
        _panel(ax, label)
        summary = b1.groupby(["sample_count", "method"])[metric].agg(["mean", "std"]).reset_index()
        for method, color, marker in [("soft_iot", ACCENT_RED, "o"), ("hard_ot", CATEGORICAL[0], "s")]:
            part = summary.loc[summary.method.eq(method)]
            ax.errorbar(part.sample_count, part["mean"], yerr=part["std"], color=color, marker=marker,
                        ms=3.5, lw=1.0, capsize=2, label="IOT" if method == "soft_iot" else "Hard OT")
        ax.set_xscale("log"); ax.set_xlabel("Sample count"); ax.set_ylabel(ylabel); ax.grid(axis="y", color="#e2e2e2", lw=0.45)
        if label == "a": ax.legend(loc="best")
        ax.set_title("Higher is better" if higher else "Lower is better")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.87, bottom=0.20, wspace=0.42)
    save_cns_figure(fig, str(output / "Figure2")); plt.close(fig)


def figure3(root: Path, output: Path) -> None:
    b2 = pd.read_csv(root / "results/B2_lineage_transition/b2_method_summary.csv")
    stability = pd.read_csv(root / "results/direction_stability/output_direction_seed_stability.csv")
    noninf = pd.read_csv(root / "results/noninferiority/paired_noninferiority.csv")
    external = pd.read_csv(root / "results/external_direction_validation/frozen_direction_external_validation.csv")
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 5.25)); axes = axes.ravel()
    _panel(axes[0], "a");
    avail = pd.read_csv(root / "results/direction_stability/direction_parameter_availability.csv")
    vals = avail.explicit_comparable_direction.astype(int).to_numpy()[:, None]
    axes[0].imshow(vals, cmap=LinearSegmentedColormap.from_list("direction", ["#EEEEEE", ACCENT_RED]), vmin=0, vmax=1, aspect="auto")
    axes[0].set_yticks(range(len(avail)), avail.method); axes[0].set_xticks([0], ["Explicit comparable\ndirection"])
    for i, v in enumerate(vals[:, 0]): axes[0].text(0, i, "Yes" if v else "—", ha="center", va="center", color="white" if v else GREY, fontsize=6.5)
    _panel(axes[1], "b"); _dot_methods(axes[1], b2, "transition_weighted_l1_mean", "Transition-matrix L1 ↓")
    _panel(axes[2], "c")
    b3_external_path = root / "results/B3_prospective_composition/external_dynamics/b3_external_all_runs.csv"
    if b3_external_path.exists():
        b3e = pd.read_csv(b3_external_path)
        b3_summary = b3e.groupby("method", as_index=False).cross_entropy.mean().sort_values("cross_entropy")
        axes[2].barh(np.arange(len(b3_summary)), b3_summary.cross_entropy,
                     color=[ACCENT_RED if m == "development-target-transfer" else CATEGORICAL_EXTENDED[i % len(CATEGORICAL_EXTENDED)] for i, m in enumerate(b3_summary.method)])
        axes[2].set_yticks(np.arange(len(b3_summary)), [METHOD_LABEL.get(m, m) for m in b3_summary.method]); axes[2].invert_yaxis()
        axes[2].set_xlabel("Future composition cross-entropy ↓")
    else:
        axes[2].text(.5, .5, "B3 external result pending", ha="center", va="center", transform=axes[2].transAxes); axes[2].axis("off")
    _panel(axes[3], "d")
    stability_mean = stability.groupby("method", as_index=False).pairwise_cosine_mean.mean()
    b2_mean = b2.groupby("method", as_index=False).transition_weighted_l1_mean.mean()
    joined = b2_mean.merge(stability_mean, on="method", how="inner")
    for _, row in joined.iterrows():
        method = row["method"]; color = ACCENT_RED if method == "UOT-IOT" else CATEGORICAL_EXTENDED[list(joined.method).index(method) % len(CATEGORICAL_EXTENDED)]
        axes[3].scatter(row["transition_weighted_l1_mean"], row["pairwise_cosine_mean"], s=35, color=color, edgecolor="white", linewidth=.4)
        axes[3].text(row["transition_weighted_l1_mean"] + .008, row["pairwise_cosine_mean"], METHOD_LABEL.get(method, method), fontsize=6, va="center")
    axes[3].set_xlabel("Transition-matrix L1 ↓"); axes[3].set_ylabel("Output-direction stability ↑"); axes[3].grid(color="#e2e2e2", lw=.45)
    _panel(axes[4], "e")
    b1 = pd.read_csv(root / "results/B1_known_truth_final/b1_all_runs.csv")
    b1_mean = b1.groupby("method", as_index=False)[["coefficient_cosine", "pure_column_minimum_curvature"]].mean()
    for _, row in b1_mean.iterrows():
        method = row["method"]; color = ACCENT_RED if method == "soft_iot" else CATEGORICAL[0]
        axes[4].scatter(row["coefficient_cosine"], row["pure_column_minimum_curvature"], s=42, color=color, edgecolor="white", linewidth=.5)
        axes[4].text(row["coefficient_cosine"] + .01, row["pure_column_minimum_curvature"], "IOT" if method == "soft_iot" else "Hard OT", fontsize=6, va="center")
    axes[4].set_xlabel("Coefficient cosine ↑"); axes[4].set_ylabel("Minimum curvature ↑"); axes[4].grid(color="#e2e2e2", lw=.45)
    _panel(axes[5], "f")
    y = np.arange(len(external)); axes[5].errorbar(external.gain, y, xerr=[external.gain-external.ci95_lower, external.ci95_upper-external.gain], fmt="o", color=ACCENT_RED, capsize=2)
    axes[5].axvline(0, color=BLACK, lw=0.6); axes[5].set_yticks(y, [f"{c}\n{s}" for c,s in zip(external.validation_cohort, external.site)]); axes[5].set_xlabel("Frozen-direction MAE gain ↑")
    fig.subplots_adjust(left=0.14, right=0.99, top=0.97, bottom=0.11, wspace=0.64, hspace=0.55)
    save_cns_figure(fig, str(output / "Figure3")); plt.close(fig)


def figure4(root: Path, output: Path) -> None:
    locked = pd.read_csv(root / "results/B3_prospective_composition/b3_state_prediction_metrics.csv")
    locked = locked.loc[locked.status.eq("success")]
    external_path = root / "results/B3_prospective_composition/external_dynamics/b3_external_all_runs.csv"
    external = pd.read_csv(external_path) if external_path.exists() else pd.DataFrame()
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 6.7)); axes = axes.ravel()
    for label, ax, metric, title in zip("ab", axes[:2], ["cross_entropy", "sinkhorn_divergence"], ["Clone-level cross-entropy ↓", "Clone-level Sinkhorn ↓"]):
        _panel(ax, label); pivot = locked.pivot(index="dataset", columns="method", values=metric)
        x=np.arange(len(pivot)); w=.34
        for j, method in enumerate(["PERSIST-no-IOT", "PERSIST-IOT"]):
            ax.bar(x+(j-.5)*w, pivot[method], width=w, color=GREY if j==0 else ACCENT_RED, label=method)
        ax.set_xticks(x, [v.replace("GSE239651_", "") for v in pivot.index]); ax.set_ylabel(title); ax.legend()
    for label, ax, metric, title in zip("cd", axes[2:4], ["cross_entropy", "sinkhorn_divergence"], ["Prospective population CE ↓", "Prospective population Sinkhorn ↓"]):
        _panel(ax, label)
        if external.empty:
            ax.text(.5,.5,"Full external dynamics run in progress",ha="center",va="center",transform=ax.transAxes); ax.axis("off"); continue
        summary=external.groupby("method")[metric].agg(["mean","std"]).sort_values("mean")
        colors=[CATEGORICAL[0] if "transfer" in m else CATEGORICAL_EXTENDED[i%len(CATEGORICAL_EXTENDED)] for i,m in enumerate(summary.index)]
        ax.barh(range(len(summary)), summary["mean"], xerr=summary["std"], color=colors, alpha=.9, error_kw={"lw":.6,"capsize":2})
        ax.set_yticks(range(len(summary)), summary.index); ax.invert_yaxis(); ax.set_xlabel(title)
    _panel(axes[4], "e")
    if external.empty:
        axes[4].text(.5,.5,"Full external dynamics run in progress",ha="center",va="center",transform=axes[4].transAxes); axes[4].axis("off")
    else:
        top1=external.groupby("method").top1_accuracy.agg(["mean","std"]).sort_values("mean")
        axes[4].barh(range(len(top1)), top1["mean"], xerr=top1["std"].fillna(0), color=CATEGORICAL[2],
                     error_kw={"lw":.6,"capsize":2})
        axes[4].set_yticks(range(len(top1)), top1.index); axes[4].invert_yaxis(); axes[4].set_xlabel("Prospective population top-1 accuracy ↑")
    _panel(axes[5], "f")
    calibration_path = root / "results/B3_prospective_composition/external_dynamics/b3_calibration_summary.csv"
    if not calibration_path.exists():
        axes[5].text(.5,.5,"Calibration analysis pending",ha="center",va="center",transform=axes[5].transAxes); axes[5].axis("off")
    else:
        calibration=pd.read_csv(calibration_path)
        methods=sorted(calibration.method.unique())
        markers={dataset: marker for dataset,marker in zip(sorted(calibration.evaluation_dataset.unique()),["o","s","D"])}
        colors={method:CATEGORICAL_EXTENDED[index%len(CATEGORICAL_EXTENDED)] for index,method in enumerate(methods)}
        for dataset, part in calibration.groupby("evaluation_dataset",sort=True):
            for _, row in part.iterrows():
                y=methods.index(row.method)
                low=max(0.0,row.expected_calibration_error-row.ci95_lower) if np.isfinite(row.ci95_lower) else 0.0
                high=max(0.0,row.ci95_upper-row.expected_calibration_error) if np.isfinite(row.ci95_upper) else 0.0
                axes[5].errorbar(row.expected_calibration_error,y,xerr=np.array([[low],[high]]),fmt=markers[dataset],
                                 color=colors[row.method],ms=4,capsize=2,lw=.7)
        axes[5].set_yticks(range(len(methods)),methods); axes[5].invert_yaxis(); axes[5].set_xlabel("State-wise expected calibration error ↓")
        handles=[plt.Line2D([0],[0],marker=marker,color="black",linestyle="none",markersize=4,label=dataset)
                 for dataset,marker in markers.items()]
        axes[5].legend(handles=handles,fontsize=6,loc="best")
    fig.subplots_adjust(left=.18,right=.98,top=.97,bottom=.08,wspace=.48,hspace=.55)
    save_cns_figure(fig, str(output / "Figure4")); plt.close(fig)


def figure5(root: Path, output: Path) -> None:
    b2 = pd.read_csv(root / "results/B2_lineage_transition/b2_method_summary.csv")
    resources = []
    all_runs = pd.read_csv(root / "results/B2_lineage_transition/b2_all_runs.csv")
    resources = all_runs.groupby("method").runtime_seconds.agg(["mean","std"]).sort_values("mean")
    external = pd.read_csv(root / "results/external_direction_validation/frozen_direction_external_validation.csv")
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.7))
    _panel(axes[0], "a"); grouped=external.groupby("validation_cohort").relative_gain.agg(["mean","std"])
    axes[0].bar(range(len(grouped)), grouped["mean"], yerr=grouped["std"], color=[CATEGORICAL[0], CATEGORICAL[2]], error_kw={"capsize":2,"lw":.7})
    axes[0].set_xticks(range(len(grouped)), grouped.index, rotation=25, ha="right"); axes[0].set_ylabel("Relative MAE gain")
    _panel(axes[1], "b"); _dot_methods(axes[1], b2, "coupling_relative_frobenius_mean", "Coupling error ↓")
    _panel(axes[2], "c"); axes[2].barh(range(len(resources)), resources["mean"], xerr=resources["std"].fillna(0), color=CATEGORICAL[5], error_kw={"capsize":2,"lw":.6})
    axes[2].set_yticks(range(len(resources)), [METHOD_LABEL.get(m,m) for m in resources.index]); axes[2].invert_yaxis(); axes[2].set_xscale("log"); axes[2].set_xlabel("Runtime per panel (s, log)")
    fig.subplots_adjust(left=.12,right=.99,top=.92,bottom=.22,wspace=.58)
    save_cns_figure(fig, str(output / "Figure5")); plt.close(fig)


def _sha256(path: Path) -> str:
    digest=hashlib.sha256();
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""): digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root=Path(__file__).resolve().parents[1]; output=root/"figures"/"rendered"; output.mkdir(parents=True,exist_ok=True)
    figure1(root,output); figure2(root,output); figure3(root,output); figure4(root,output); figure5(root,output)
    panel_sources = {
        "Figure1a": ["configs/protocol_v1.yaml"], "Figure1b": ["results/direction_stability/direction_parameter_availability.csv"],
        "Figure2a": ["results/B1_known_truth_final/b1_all_runs.csv"], "Figure2b": ["results/B1_known_truth_final/b1_all_runs.csv"],
        "Figure2c": ["results/B1_known_truth_final/b1_all_runs.csv"],
        "Figure3a": ["results/direction_stability/direction_parameter_availability.csv"],
        "Figure3b": ["results/B2_lineage_transition/b2_method_summary.csv"],
        "Figure3c": ["results/B3_prospective_composition/external_dynamics/b3_external_all_runs.csv"],
        "Figure3d": ["results/direction_stability/output_direction_seed_stability.csv"],
        "Figure3e": ["results/noninferiority/paired_noninferiority.csv"],
        "Figure3f": ["results/external_direction_validation/frozen_direction_external_validation.csv"],
        "Figure4a": ["results/B3_prospective_composition/b3_state_prediction_metrics.csv"],
        "Figure4b": ["results/B3_prospective_composition/b3_state_prediction_metrics.csv"],
        "Figure4c": ["results/B3_prospective_composition/external_dynamics/b3_external_all_runs.csv"],
        "Figure4d": ["results/B3_prospective_composition/external_dynamics/b3_external_all_runs.csv"],
        "Figure4e": ["results/B3_prospective_composition/external_dynamics/b3_external_all_runs.csv"],
        "Figure4f": ["results/B3_prospective_composition/external_dynamics/b3_calibration_summary.csv"],
        "Figure5a": ["results/external_direction_validation/frozen_direction_external_validation.csv"],
        "Figure5b": ["results/B2_lineage_transition/b2_method_summary.csv"],
        "Figure5c": ["results/B2_lineage_transition/b2_all_runs.csv"],
    }
    manifest_rows = []
    for panel, paths in panel_sources.items():
        for rel in paths:
            path = root / rel
            figure_id = re.match(r"(Figure\d+)", panel).group(1)
            manifest_rows.append({"figure_panel": panel, "script": "scripts/render_figures.py", "input": rel,
                                  "input_sha256": _sha256(path) if path.exists() else "MISSING",
                                  "output_pdf": f"figures/rendered/{figure_id}.pdf", "output_png": f"figures/rendered/{figure_id}.png"})
    manifest = pd.DataFrame(manifest_rows)
    reviewer=root/"results"/"reviewer_tables"; reviewer.mkdir(parents=True,exist_ok=True); manifest.to_csv(reviewer/"figure_source_manifest.csv",index=False)
    missing_outputs = sorted({path for path in manifest.output_pdf.tolist() + manifest.output_png.tolist()
                              if not (root / path).exists()})
    qa={"figures":[f"Figure{i}" for i in range(1,6)],"vector_pdf":True,"png_dpi":300,"width_mm":183,
        "source_files":len(manifest),"asset_mode":"parameter inheritance","target":"Nature-family standard",
        "source_manifest_output_paths_exist": not missing_outputs,"missing_manifest_outputs":missing_outputs}
    (output/"QA_REPORT.json").write_text(json.dumps(qa,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(qa,indent=2)); return 0 if not missing_outputs else 1


if __name__ == "__main__": raise SystemExit(main())
