"""Create reviewer-facing B2 tables and quantify UOT-IOT seed stability."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from iot_benchmark.metrics import direction_stability


def _finite_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: float(value)
        for key, value in payload.items()
        if isinstance(value, (int, float)) and np.isfinite(value)
    }


def _status_rows(status: pd.DataFrame, *, default_seed: object) -> list[dict[str, object]]:
    """Resolve metrics referenced by an adapter status table."""
    rows: list[dict[str, object]] = []
    for record in status.to_dict(orient="records"):
        row = {
            "dataset": record["dataset"],
            "method": record["method"],
            "seed": record.get("seed", default_seed),
            "status": record["status"],
            "runtime_seconds": record["runtime_seconds"],
            "artifact": record["artifact"],
        }
        if record["status"] == "success":
            row.update(_finite_metrics(Path(record["metrics"])))
        rows.append(row)
    return rows


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    result_root = root / "results" / "B2_lineage_transition"
    rows: list[dict[str, object]] = []

    status = pd.read_csv(result_root / "core_method_status.csv")
    rows.extend(_status_rows(status, default_seed="deterministic_or_method_default"))

    neural_path = result_root / "neural_method_status.csv"
    neural = pd.read_csv(neural_path) if neural_path.exists() else pd.DataFrame()
    if not neural.empty:
        rows.extend(_status_rows(neural, default_seed="method_default"))

    iot = pd.read_csv(result_root / "uot_iot" / "uot_iot_all_runs.csv")
    iot["status"] = "success"
    rows.extend(iot.to_dict(orient="records"))
    all_runs = pd.DataFrame(rows)
    all_runs.to_csv(result_root / "b2_all_runs.csv", index=False)

    numeric = [
        "coupling_relative_frobenius",
        "transition_weighted_l1",
        "fate_pearson",
        "fate_spearman",
        "fate_top1_accuracy",
        "runtime_seconds",
    ]
    for column in numeric:
        if column not in all_runs:
            all_runs[column] = np.nan
    summary = (
        all_runs.groupby(["dataset", "method"], as_index=False)[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary.to_csv(result_root / "b2_method_summary.csv", index=False)

    stability_rows = []
    for dataset in sorted(iot["dataset"].unique()):
        artifacts = [Path(value) for value in iot.loc[iot["dataset"] == dataset, "artifact"]]
        directions = np.stack([np.load(path, allow_pickle=False)["direction"] for path in artifacts])
        stability_rows.append({"dataset": dataset, "method": "UOT-IOT", **direction_stability(directions)})
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(result_root / "b2_direction_stability.csv", index=False)

    manifest = {
        "benchmark": "B2_lineage_transition",
        "core_external_runs": int(len(status)),
        "core_external_success": int((status["status"] == "success").sum()),
        "neural_external_runs": int(len(neural)),
        "neural_external_success": int((neural["status"] == "success").sum()) if not neural.empty else 0,
        "uot_iot_runs": int(len(iot)),
        "datasets": sorted(all_runs["dataset"].unique().tolist()),
        "methods": sorted(all_runs["method"].unique().tolist()),
        "direction_stability_scope": "UOT-IOT only; external methods do not expose a comparable signed feature direction",
    }
    (result_root / "b2_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
