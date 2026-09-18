"""Collect existing measurements for the R mini-panels in the draw.io route."""
from pathlib import Path
import gzip
import json
import shutil

import numpy as np
import pandas as pd
from scipy import sparse, io

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/technical_route_v6/source_data"
OUT.mkdir(parents=True, exist_ok=True)
PREP = ROOT / "output/03_data_preparation"
SRC = ROOT / "output/nature_stat_redesign/source_data_v3"

states = pd.read_csv(PREP / "output/states.csv")
meta = np.load(PREP / "data/prep_meta.npz", allow_pickle=True)
expression = sparse.load_npz(PREP / "data/prep_expression.npz")
assert np.array_equal(states.state.to_numpy(), meta["states"])
assert expression.shape == (len(states), len(meta["genes"]))
states.to_csv(OUT / "GSE228154_cells.csv", index=False)
with gzip.open(OUT / "GSE228154_log_expression.mtx.gz", "wb") as handle:
    io.mmwrite(handle, expression)
annotation = json.loads((PREP / "output/state_annotation.json").read_text(encoding="utf-8"))
features = ["epithelial", "proliferation", "emt", "stemness", "senescence"]
pd.DataFrame(annotation["state_scores"])[features].rename_axis("state").to_csv(
    OUT / "GSE228154_state_programmes.csv")

for name in [
    "Fig2a_curvature.csv", "Fig2b_recovery.csv", "Fig2c_sensitivity.csv",
    "Fig2d_external_validation.csv", "Fig3b_lineage_paired_values.csv",
    "Fig4ab_patient_scores.csv", "Fig4ab_descriptive_mean_ci.csv",
    "Fig4c_patient_effects.csv", "Fig5c_external_all_windows.csv",
    "Fig5c_external_condition_state_summary.csv", "Fig5ab_state_metrics.csv",
    "Fig5d_brier_paired_bootstrap.csv",
]:
    shutil.copy2(SRC / name, OUT / name)

print(f"Collected {expression.shape[0]} cells × {expression.shape[1]} genes and existing figure data into {OUT}")
