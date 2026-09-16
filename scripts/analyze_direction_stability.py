"""Audit direction identifiability and compare seed-level output-direction stability."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from iot_benchmark.statistics import paired_bootstrap_interval


SEEDED_METHODS = ("mioflow", "prescient", "tigon")


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norm = np.linalg.norm(values)
    return values / norm if norm > 0 else values


def _pairwise_cosine(vectors: list[np.ndarray]) -> tuple[float, float, float]:
    values = [float(a @ b) for a, b in combinations(vectors, 2)]
    return float(np.mean(values)), float(np.min(values)), float(np.var(values, ddof=1))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data_root = root / "data" / "derived"
    result_root = root / "results" / "B2_lineage_transition"
    output = root / "results" / "direction_stability"
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for panel_path in sorted(data_root.glob("*.npz")):
        dataset = panel_path.stem
        panel = np.load(panel_path, allow_pickle=False)
        source = panel["source_counts"].sum(axis=0).astype(float)
        source /= source.sum()

        for method in SEEDED_METHODS:
            vectors = []
            for artifact in sorted((result_root / method).glob(f"{dataset}_seed*.npz")):
                prediction = np.load(artifact, allow_pickle=False)["predicted_population"]
                vectors.append(_unit(prediction - source))
            mean, minimum, variance = _pairwise_cosine(vectors)
            rows.append({"dataset": dataset, "method": method, "direction_type": "output_change_proxy",
                         "seed_n": len(vectors), "pairwise_cosine_mean": mean,
                         "pairwise_cosine_min": minimum, "pairwise_cosine_variance": variance})

        vectors = []
        for artifact in sorted((result_root / "uot_iot").glob(f"{dataset}_seed*.npz")):
            prediction = np.load(artifact, allow_pickle=False)["state_prediction"].mean(axis=0)
            vectors.append(_unit(prediction - source))
        mean, minimum, variance = _pairwise_cosine(vectors)
        rows.append({"dataset": dataset, "method": "UOT-IOT", "direction_type": "output_change_proxy",
                     "seed_n": len(vectors), "pairwise_cosine_mean": mean,
                     "pairwise_cosine_min": minimum, "pairwise_cosine_variance": variance})

    stability = pd.DataFrame(rows)
    stability.to_csv(output / "output_direction_seed_stability.csv", index=False)

    paired = []
    pivot = stability.pivot(index="dataset", columns="method", values="pairwise_cosine_mean")
    for method in SEEDED_METHODS:
        differences = (pivot["UOT-IOT"] - pivot[method]).to_numpy(dtype=float)
        mean, lower, upper = paired_bootstrap_interval(differences)
        paired.append({"comparison": f"UOT-IOT_minus_{method}", "paired_dataset_n": len(differences),
                       "mean_difference": mean, "ci95_lower": lower, "ci95_upper": upper,
                       "positive_favors": "UOT-IOT"})
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(output / "output_direction_paired_statistics.csv", index=False)

    availability = pd.DataFrame([
        ("UOT-IOT", True, "gauge-normalized signed transition-feature coefficients"),
        ("Waddington-OT", False, "transport coupling without an identified signed feature coefficient"),
        ("moscot", False, "transport coupling without an identified signed feature coefficient"),
        ("LineageOT", False, "coupling and lineage constraints without a common signed feature coefficient"),
        ("CellRank 2", False, "fate probabilities without a common signed transition-feature coefficient"),
        ("TIGON", False, "neural vector field parameters are not identifiable as a common signed feature coefficient"),
        ("MIOFlow", False, "neural flow parameters are not identifiable as a common signed feature coefficient"),
        ("PRESCIENT", False, "potential-network weights are not identifiable as a common signed feature coefficient"),
    ], columns=["method", "explicit_comparable_direction", "reason"])
    availability.to_csv(output / "direction_parameter_availability.csv", index=False)

    b1_gate = json.loads((root / "results" / "B1_known_truth_final" / "b1_gate_status.json").read_text(encoding="utf-8"))
    parameter_stability = pd.read_csv(result_root / "b2_direction_stability.csv")
    manifest = {
        "gate_pass": bool(
            b1_gate["G5_direction_stability_B1_component"]["pass"]
            and parameter_stability["direction_pairwise_cosine_min"].min() >= 0.95
            and (paired_frame["ci95_lower"] > 0).all()
        ),
        "known_truth_component_pass": bool(b1_gate["G5_direction_stability_B1_component"]["pass"]),
        "real_data_parameter_direction_min_cosine": float(parameter_stability["direction_pairwise_cosine_min"].min()),
        "output_proxy_comparisons_with_positive_ci": int((paired_frame["ci95_lower"] > 0).sum()),
        "output_proxy_comparisons_total": int(len(paired_frame)),
        "interpretation": (
            "The output-change proxy is comparable across seeded methods but is not a learned feature coefficient. "
            "Methods lacking an explicit comparable direction are recorded as structurally non-estimable, not unstable."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
