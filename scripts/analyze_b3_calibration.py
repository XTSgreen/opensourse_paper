"""Estimate state-wise calibration error for prospective population predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    if not path.is_absolute() and (root / path).exists():
        return root / path
    normalized = value.replace("\\", "/")
    marker = "/iot_reproducibility/"
    index = normalized.lower().find(marker)
    if index >= 0:
        relative = Path(normalized[index + len(marker):])
        candidate = root / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(value)


def _ece(predicted: np.ndarray, observed: np.ndarray, bins: int = 10) -> float:
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    observed = np.asarray(observed, dtype=float).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(predicted, edges[1:-1], right=False), bins - 1)
    return float(sum(
        np.mean(assignments == index) * abs(predicted[assignments == index].mean() - observed[assignments == index].mean())
        for index in range(bins) if np.any(assignments == index)
    ))


def _bootstrap_panel_ci(frame: pd.DataFrame, bins: int, replicates: int, seed: int) -> tuple[float, float]:
    panels = frame.panel.unique()
    if len(panels) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    panel_values = []
    for panel in panels:
        rows = frame.loc[frame.panel.eq(panel)]
        predicted = np.concatenate(rows.predicted.to_numpy())
        observed = np.concatenate(rows.observed.to_numpy())
        panel_values.append((predicted, observed))
    for index in range(replicates):
        selected = rng.integers(0, len(panel_values), size=len(panels))
        predicted = np.concatenate([panel_values[item][0] for item in selected])
        observed = np.concatenate([panel_values[item][1] for item in selected])
        estimates[index] = _ece(predicted, observed, bins)
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    root = args.package_root.resolve()
    manifest = json.loads((root / "data" / "derived" / "b3" / "panel_manifest.json").read_text(encoding="utf-8"))
    panel_map = {item["panel"]: item for item in manifest["panels"] if item["status"] == "included"}
    run_path = root / "results" / "B3_prospective_composition" / "external_dynamics" / "b3_external_all_runs.csv"
    runs = pd.read_csv(run_path)
    rows = []
    for record in runs.loc[runs.status.eq("success")].to_dict(orient="records"):
        panel = panel_map[str(record["panel"])]
        artifact_path = _resolve(root, str(record["artifact"]))
        truth_path = _resolve(root, str(panel["truth_path"]))
        artifact = np.load(artifact_path, allow_pickle=False)
        truth = np.load(truth_path, allow_pickle=False)
        predicted = np.asarray(artifact["predicted_population"], dtype=float).reshape(-1)
        counts = np.asarray(truth["target_counts"], dtype=float).sum(axis=0)
        observed = counts / counts.sum()
        if predicted.shape != observed.shape:
            raise ValueError(f"state dimension mismatch in {panel['panel']} / {record['method']}")
        rows.append({
            "evaluation_dataset": panel.get("evaluation_dataset", "GSE239651_expt2"),
            "panel": str(record["panel"]), "condition": panel.get("condition", record.get("condition")),
            "time_days": panel.get("time_days", record.get("time_days")), "method": str(record["method"]),
            "seed": record.get("seed"), "n_states": len(predicted),
            "expected_calibration_error": _ece(predicted, observed, args.bins),
            "predicted": predicted, "observed": observed,
        })
    by_prediction = pd.DataFrame(rows)
    output = root / "results" / "B3_prospective_composition" / "external_dynamics"
    output.mkdir(parents=True, exist_ok=True)
    serializable = by_prediction.drop(columns=["predicted", "observed"])
    serializable.to_csv(output / "b3_calibration_by_prediction.csv", index=False)
    summary_rows = []
    for (dataset, method), group in by_prediction.groupby(["evaluation_dataset", "method"], sort=True):
        pooled = _ece(np.stack(group.predicted), np.stack(group.observed), args.bins)
        lower, upper = _bootstrap_panel_ci(group, args.bins, args.bootstrap, 20260916)
        summary_rows.append({
            "evaluation_dataset": dataset, "method": method, "expected_calibration_error": pooled,
            "ci95_lower": lower, "ci95_upper": upper, "n_panels": int(group.panel.nunique()),
            "n_predictions": int(len(group)), "n_states_per_prediction_mean": float(group.n_states.mean()),
            "bootstrap_unit": "held-out panel as cluster; seeds retained together; cross-validation folds may share training observations",
            "bins": args.bins, "bootstrap_replicates": args.bootstrap,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "b3_calibration_summary.csv", index=False)
    print(json.dumps({"prediction_rows": len(by_prediction), "summary_rows": len(summary),
                      "datasets": sorted(by_prediction.evaluation_dataset.unique()), "methods": sorted(by_prediction.method.unique())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
