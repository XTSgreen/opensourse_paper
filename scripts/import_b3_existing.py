"""Standardize the locked PERSIST-IOT experiment into the frozen B3 schema."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from iot_benchmark.metrics import (
    cross_entropy,
    expected_calibration_error,
    multiclass_brier,
    sinkhorn_divergence,
    top1_accuracy,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_cost(dictionary_path: Path) -> np.ndarray:
    centers = np.load(dictionary_path, allow_pickle=False)["cluster_centers"].astype(float)
    cost = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    positive = cost[cost > 0]
    return cost / np.median(positive)


def _evaluate_predictions(path: Path, partition: str, cost: np.ndarray) -> list[dict[str, object]]:
    frame = pd.read_parquet(path)
    prediction_columns = [f"state_prob_{index}" for index in range(6)]
    truth_columns = [f"target_state_prob_{index}" for index in range(6)]
    rows = []
    for method, group in frame.groupby("method", sort=True):
        prediction = group[prediction_columns].to_numpy(dtype=float)
        truth = group[truth_columns].to_numpy(dtype=float)
        mask = (
            np.all(np.isfinite(prediction), axis=1)
            & np.all(np.isfinite(truth), axis=1)
            & (prediction.sum(axis=1) > 0)
            & (truth.sum(axis=1) > 0)
        )
        if not np.any(mask):
            rows.append({
                "dataset": str(group["dataset_id"].iloc[0]),
                "partition": partition,
                "method": method,
                "status": "not_applicable",
                "reason": "method does not emit a future-state probability vector",
                "state_n": 0,
            })
            continue
        prediction, truth = prediction[mask], truth[mask]
        sinkhorn = np.asarray(
            [sinkhorn_divergence(p, y, cost, epsilon=0.1, iterations=200) for p, y in zip(prediction, truth)]
        )
        rows.append({
            "dataset": str(group["dataset_id"].iloc[0]),
            "partition": partition,
            "method": method,
            "status": "success",
            "reason": "",
            "state_n": int(mask.sum()),
            "cross_entropy": cross_entropy(prediction, truth),
            "sinkhorn_divergence": float(sinkhorn.mean()),
            "top1_accuracy": top1_accuracy(prediction, truth),
            "brier_score": multiclass_brier(prediction, truth),
            "expected_calibration_error": expected_calibration_error(prediction, truth, bins=10),
        })
    return rows


def main() -> int:
    package_root = Path(__file__).resolve().parents[1]
    project_root = package_root.parent
    source = project_root / "output" / "future_state_prediction_IOT" / "persistence_sota_v2" / "locked_primary_rerun"
    dictionaries = project_root / "script" / "future_state_prediction_IOT" / "persistence_sota_v2" / "data" / "manifests"
    output = package_root / "results" / "B3_prospective_composition"
    output.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("development_outer_oof_predictions.parquet", "development_outer_oof", "gse239651_expt1_frozen_state_dictionary.npz"),
        ("external_e1_predictions.parquet", "locked_external_e1", "gse239651_expt2_frozen_state_dictionary.npz"),
    ]
    rows: list[dict[str, object]] = []
    input_hashes = {}
    for prediction_name, partition, dictionary_name in jobs:
        prediction_path = source / prediction_name
        dictionary_path = dictionaries / dictionary_name
        rows.extend(_evaluate_predictions(prediction_path, partition, _state_cost(dictionary_path)))
        input_hashes[str(prediction_path.relative_to(project_root))] = _sha256(prediction_path)
        input_hashes[str(dictionary_path.relative_to(project_root))] = _sha256(dictionary_path)

    standardized = pd.DataFrame(rows)
    standardized.to_csv(output / "b3_state_prediction_metrics.csv", index=False)
    for name in [
        "development_outer_oof_metrics.csv",
        "external_e1_metrics.csv",
        "method_ranking.csv",
        "paired_clone_bootstrap_brier.csv",
        "leakage_and_data_contract_audit.json",
        "finalization_manifest.json",
        "provisional_decision.json",
        "RUN_REPORT.md",
    ]:
        shutil.copy2(source / name, output / f"legacy_{name}")
        input_hashes[str((source / name).relative_to(project_root))] = _sha256(source / name)

    manifest = {
        "benchmark": "B3_prospective_composition",
        "status": "imported_locked_PERSIST_experiment",
        "partitions": [job[1] for job in jobs],
        "metrics_recomputed_from_predictions": [
            "cross_entropy",
            "sinkhorn_divergence",
            "top1_accuracy",
            "brier_score",
            "expected_calibration_error",
        ],
        "sinkhorn_ground_cost": "Euclidean distance between frozen day-0 state centroids, normalized by median positive distance",
        "input_sha256": input_hashes,
        "limitation": "External dynamic SOTA methods remain to be evaluated prospectively under the same frozen split.",
    }
    (output / "b3_import_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(standardized.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
