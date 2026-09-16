"""Paired uncertainty estimates and pre-registered B1 gate decisions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    replicates: int = 10000,
    seed: int = 20260916,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) < 2:
        raise ValueError("paired bootstrap requires at least two finite pairs")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(replicates, len(differences)))
    means = differences[indices].mean(axis=1)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(differences.mean()), float(lower), float(upper)


def analyze_b1(input_csv: str | Path, output_dir: str | Path) -> dict[str, object]:
    frame = pd.read_csv(input_csv)
    keys = ["scenario", "sample_size", "sample_count", "seed"]
    metrics = [
        "heldout_coupling_relative_frobenius",
        "coefficient_cosine",
        "coefficient_sign_accuracy",
        "pure_column_minimum_curvature",
        "restart_direction_variance",
    ]
    soft = frame.loc[frame["method"] == "soft_iot", keys + metrics].set_index(keys)
    hard = frame.loc[frame["method"] == "hard_ot", keys + metrics].set_index(keys)
    common = soft.index.intersection(hard.index)
    if len(common) != len(soft) or len(common) != len(hard):
        raise ValueError("B1 soft and hard runs are not completely paired")
    records = []
    for scenario in sorted(frame["scenario"].unique()):
        scenario_index = [index for index in common if index[0] == scenario]
        for metric in metrics:
            # Positive differences always favor soft IOT.
            if metric in {"heldout_coupling_relative_frobenius", "restart_direction_variance"}:
                differences = hard.loc[scenario_index, metric].to_numpy() - soft.loc[scenario_index, metric].to_numpy()
            else:
                differences = soft.loc[scenario_index, metric].to_numpy() - hard.loc[scenario_index, metric].to_numpy()
            mean, lower, upper = paired_bootstrap_interval(differences)
            records.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "positive_favors": "soft_iot",
                    "paired_n": len(differences),
                    "mean_difference": mean,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    statistics = pd.DataFrame(records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    statistics.to_csv(output_dir / "b1_paired_statistics.csv", index=False)

    equivalent = statistics[statistics["scenario"] == "equivalent_fit_multiple_parameters"].set_index("metric")
    all_data = statistics.groupby("metric").agg(
        mean_difference=("mean_difference", "mean"),
        worst_lower=("ci95_lower", "min"),
        worst_upper=("ci95_upper", "max"),
    )
    g4 = bool(
        equivalent.loc["coefficient_cosine", "ci95_lower"] > 0
        and equivalent.loc["pure_column_minimum_curvature", "ci95_lower"] > 0
    )
    g5 = bool(equivalent.loc["restart_direction_variance", "ci95_lower"] > 0)
    result = {
        "benchmark": "B1_known_truth",
        "paired_runs": int(len(common)),
        "bootstrap_replicates": 10000,
        "G4_parameter_recovery": {
            "pass": g4,
            "evidence": "equivalent-fit coefficient cosine and pure-column curvature paired CI lower bounds exceed zero",
        },
        "G5_direction_stability_B1_component": {
            "pass": g5,
            "evidence": "equivalent-fit hard-minus-soft restart direction variance paired CI lower bound exceeds zero",
        },
        "descriptive_all_scenarios": all_data.reset_index().to_dict(orient="records"),
    }
    (output_dir / "b1_gate_status.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
