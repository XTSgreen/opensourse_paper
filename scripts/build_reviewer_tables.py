"""Assemble machine-readable reviewer tables from standard benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


SCHEMA = [
    "dataset", "benchmark", "task", "split", "repeat", "seed", "method", "method_version",
    "metric", "value", "status", "runtime_seconds", "peak_memory_gb", "failure_reason", "artifact",
]


def _portable_path(value: object, root: Path) -> object:
    if not isinstance(value, str) or not value:
        return value


def _resolve_artifact(value: object, root: Path) -> Path | None:
    """Resolve old absolute result paths after moving the package to another machine."""

    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        candidate = root / candidate
        return candidate if candidate.exists() else None
    portable = value.replace("\\", "/")
    for marker in ("results/", "data/"):
        index = portable.find(marker)
        if index >= 0:
            candidate = root / Path(portable[index:])
            return candidate if candidate.exists() else None
    return None
    candidate = Path(value)
    if not candidate.is_absolute():
        return value
    try:
        return str(candidate.resolve().relative_to(root.resolve()))
    except ValueError:
        return value


def _rows_from_wide(frame: pd.DataFrame, *, benchmark: str, task: str, split: str, id_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    available = [column for column in metric_columns if column in frame]
    long = frame.melt(id_vars=id_columns, value_vars=available, var_name="metric", value_name="value")
    long["benchmark"] = benchmark
    long["task"] = task
    long["split"] = split
    return long


def _metric_json_rows(status: pd.DataFrame, benchmark: str, task: str, root: Path) -> list[dict[str, object]]:
    rows = []
    for record in status.to_dict(orient="records"):
        base = {
            "dataset": record.get("dataset"), "benchmark": benchmark, "task": task,
            "split": "retrospective_reconstruction", "repeat": np.nan, "seed": record.get("seed", np.nan),
            "method": record.get("method"), "method_version": "see_method_status",
            "status": record.get("status"), "runtime_seconds": record.get("runtime_seconds", np.nan),
            "peak_memory_gb": np.nan, "failure_reason": record.get("failure_reason", ""), "artifact": record.get("artifact", ""),
        }
        metric_path = _resolve_artifact(record.get("metrics"), root)
        if record.get("status") == "success" and metric_path is not None:
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    rows.append({**base, "metric": metric, "value": value})
        else:
            rows.append({**base, "metric": "method_status", "value": np.nan})
    return rows


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = root / "results" / "reviewer_tables"
    output.mkdir(parents=True, exist_ok=True)
    frames = []

    b1_path = root / "results" / "B1_known_truth_final" / "b1_all_runs.csv"
    if b1_path.exists():
        b1 = pd.read_csv(b1_path).rename(columns={"scenario": "dataset"})
        b1["repeat"] = b1.get("sample_count", np.nan)
        b1["method_version"] = "iot-benchmark-0.1.0"
        b1["status"] = "success"
        b1["peak_memory_gb"] = np.nan
        b1["failure_reason"] = ""
        b1["artifact"] = str(b1_path)
        long = _rows_from_wide(
            b1, benchmark="B1_known_truth", task="known_truth_parameter_and_coupling_recovery", split="synthetic",
            id_columns=["dataset", "repeat", "seed", "method", "method_version", "status", "runtime_seconds", "peak_memory_gb", "failure_reason", "artifact"],
            metric_columns=["heldout_coupling_relative_frobenius", "coefficient_cosine", "coefficient_sign_accuracy", "pure_column_minimum_curvature", "restart_direction_variance"],
        )
        frames.append(long)

    b2_root = root / "results" / "B2_lineage_transition"
    b2_rows = []
    for name in ["core_method_status.csv", "neural_method_status.csv"]:
        path = b2_root / name
        if path.exists():
            b2_rows.extend(_metric_json_rows(pd.read_csv(path), "B2_lineage_transition", "transition_recovery", root))
    iot_path = b2_root / "uot_iot" / "uot_iot_all_runs.csv"
    if iot_path.exists():
        iot = pd.read_csv(iot_path)
        iot["repeat"] = np.nan
        iot["method_version"] = "iot-benchmark-0.1.0"
        iot["status"] = "success"
        iot["peak_memory_gb"] = np.nan
        iot["failure_reason"] = ""
        long = _rows_from_wide(
            iot, benchmark="B2_lineage_transition", task="transition_recovery", split="retrospective_reconstruction",
            id_columns=["dataset", "repeat", "seed", "method", "method_version", "status", "runtime_seconds", "peak_memory_gb", "failure_reason", "artifact"],
            metric_columns=["coupling_relative_frobenius", "transition_weighted_l1", "fate_pearson", "fate_spearman", "fate_top1_accuracy"],
        )
        frames.append(long)
    if b2_rows:
        frames.append(pd.DataFrame(b2_rows))

    b3_import = root / "results" / "B3_prospective_composition" / "b3_state_prediction_metrics.csv"
    if b3_import.exists():
        b3 = pd.read_csv(b3_import)
        b3 = b3.loc[b3["status"].eq("success")].copy()
        b3["repeat"] = np.nan
        b3["seed"] = 20260916
        b3["method_version"] = "locked_primary_rerun"
        b3["runtime_seconds"] = np.nan
        b3["peak_memory_gb"] = np.nan
        b3["failure_reason"] = ""
        b3["artifact"] = str(b3_import)
        long = _rows_from_wide(
            b3, benchmark="B3_prospective_composition", task="clone_level_future_state_composition", split="locked_existing",
            id_columns=["dataset", "repeat", "seed", "method", "method_version", "status", "runtime_seconds", "peak_memory_gb", "failure_reason", "artifact"],
            metric_columns=["cross_entropy", "sinkhorn_divergence", "top1_accuracy", "brier_score", "expected_calibration_error"],
        )
        frames.append(long)

    external_path = root / "results" / "B3_prospective_composition" / "external_dynamics" / "b3_external_all_runs.csv"
    if external_path.exists():
        b3e = pd.read_csv(external_path)
        b3e["repeat"] = b3e["panel"].astype(str) + "::" + b3e["condition"].astype(str)
        b3e["method_version"] = "see_method_status"
        b3e["peak_memory_gb"] = np.nan
        b3e["artifact"] = b3e["artifact"].fillna("")
        long = _rows_from_wide(
            b3e, benchmark="B3_prospective_composition", task="population_future_state_composition", split="expt1_to_locked_expt2",
            id_columns=["dataset", "repeat", "seed", "method", "method_version", "status", "runtime_seconds", "peak_memory_gb", "failure_reason", "artifact"],
            metric_columns=["cross_entropy", "sinkhorn_divergence", "top1_accuracy", "multiclass_brier"],
        )
        long["split"] = np.where(
            long["repeat"].astype(str).str.contains("_holdout_", regex=False),
            "held_out_animal", "expt1_to_locked_expt2",
        )
        frames.append(long)

    all_runs = pd.concat(frames, ignore_index=True, sort=False)
    for column in SCHEMA:
        if column not in all_runs:
            all_runs[column] = np.nan
    all_runs = all_runs[SCHEMA]
    all_runs["artifact"] = all_runs["artifact"].map(lambda value: _portable_path(value, root))
    all_runs["repeat"] = all_runs["repeat"].map(lambda value: "" if pd.isna(value) else str(value))
    all_runs.to_csv(output / "benchmark_all_runs.csv", index=False)
    all_runs.to_parquet(output / "benchmark_all_runs.parquet", index=False)
    summary = (
        all_runs.loc[all_runs["status"].eq("success")]
        .groupby(["benchmark", "task", "dataset", "method", "metric"], as_index=False)
        .agg(mean=("value", "mean"), std=("value", "std"), n=("value", "count"))
    )
    summary.to_csv(output / "benchmark_summary.csv", index=False)

    interpretability = []
    b1_stats = root / "results" / "B1_known_truth_final" / "b1_paired_statistics.csv"
    if b1_stats.exists():
        part = pd.read_csv(b1_stats)
        part.insert(0, "benchmark", "B1_known_truth")
        interpretability.append(part)
    b2_stability = b2_root / "b2_direction_stability.csv"
    if b2_stability.exists():
        part = pd.read_csv(b2_stability)
        part.insert(0, "benchmark", "B2_lineage_transition")
        interpretability.append(part)
    output_stability = root / "results" / "direction_stability" / "output_direction_seed_stability.csv"
    if output_stability.exists():
        part = pd.read_csv(output_stability)
        part.insert(0, "benchmark", "B2_lineage_transition_output_proxy")
        interpretability.append(part)
    external_direction = root / "results" / "external_direction_validation" / "frozen_direction_external_validation.csv"
    if external_direction.exists():
        part = pd.read_csv(external_direction)
        part.insert(0, "benchmark", "external_frozen_direction")
        interpretability.append(part)
    interpretability_table = pd.concat(interpretability, ignore_index=True, sort=False) if interpretability else pd.DataFrame()
    interpretability_table.to_csv(output / "interpretability_summary.csv", index=False)

    protocol_stats = root / "results" / "B1_known_truth_final" / "b1_paired_statistics.csv"
    paired_frames = []
    if protocol_stats.exists():
        part = pd.read_csv(protocol_stats)
        part.insert(0, "benchmark", "B1_known_truth")
        paired_frames.append(part)
    noninferiority = root / "results" / "noninferiority" / "paired_noninferiority.csv"
    if noninferiority.exists():
        paired_frames.append(pd.read_csv(noninferiority))
    direction_paired = root / "results" / "direction_stability" / "output_direction_paired_statistics.csv"
    if direction_paired.exists():
        part = pd.read_csv(direction_paired)
        part.insert(0, "benchmark", "B2_lineage_transition_output_proxy")
        paired_frames.append(part)
    paired_table = pd.concat(paired_frames, ignore_index=True, sort=False) if paired_frames else pd.DataFrame()
    paired_table.to_csv(output / "paired_statistics.csv", index=False)

    registry = yaml.safe_load((root / "configs" / "methods" / "method_registry.yaml").read_text(encoding="utf-8"))["methods"]
    method_rows = [{"method": method, **properties} for method, properties in registry.items()]
    pd.DataFrame(method_rows).to_csv(output / "method_status.csv", index=False)

    resources = all_runs[["benchmark", "dataset", "method", "seed", "runtime_seconds", "peak_memory_gb", "status", "artifact"]].drop_duplicates()
    resources.to_csv(output / "compute_resources.csv", index=False)
    datasets = yaml.safe_load((root / "configs" / "datasets" / "dataset_registry.yaml").read_text(encoding="utf-8"))["datasets"]
    dataset_rows = [{"dataset": name, **spec} for name, spec in datasets.items()]
    pd.json_normalize(dataset_rows).to_csv(output / "table_S1_dataset_contract.csv", index=False)
    pins = yaml.safe_load((root / "external_methods" / "SOURCE_PINS.yaml").read_text(encoding="utf-8"))["methods"]
    pd.json_normalize([{"method": name, **spec} for name, spec in pins.items()]).to_csv(output / "table_S2_method_versions.csv", index=False)
    all_runs.to_csv(output / "table_S3_all_runs_long.csv", index=False)
    paired_table.to_csv(output / "table_S4_noninferiority_and_paired.csv", index=False)
    if interpretability:
        interpretability_table.to_csv(output / "table_S5_direction_and_interpretability.csv", index=False)
    all_runs.loc[~all_runs["status"].eq("success")].to_csv(output / "table_S6_failure_audit.csv", index=False)
    calibration = root / "results" / "B3_prospective_composition" / "external_dynamics" / "b3_calibration_summary.csv"
    if calibration.exists():
        pd.read_csv(calibration).to_csv(output / "table_S8_b3_calibration.csv", index=False)
    figure_manifest = output / "figure_source_manifest.csv"
    if figure_manifest.exists():
        pd.read_csv(figure_manifest).to_csv(output / "table_S7_figure_sources.csv", index=False)
    pseudotime_summary = root / "results" / "pseudotime" / "pseudotime_summary.csv"
    if pseudotime_summary.exists():
        pd.read_csv(pseudotime_summary).to_csv(output / "table_S9_pseudotime.csv", index=False)
    print(json.dumps({"all_run_rows": len(all_runs), "summary_rows": len(summary), "methods": len(method_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
