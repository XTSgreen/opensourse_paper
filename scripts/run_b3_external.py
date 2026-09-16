"""Run prospective external dynamics methods on physically separated B3 panels."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from iot_benchmark.external_evaluation import evaluate_prospective_population_artifact


SEEDS = [20260916, 20260917, 20260918, 20260919, 20260920]


def _run(command: list[str], log: Path, timeout: int) -> dict[str, object]:
    started = time.perf_counter()
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        output = (process.stdout or "") + ("\n" + process.stderr if process.stderr else "")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(output, encoding="utf-8")
        return {"status": "success" if process.returncode == 0 else "failed", "returncode": process.returncode,
                "runtime_seconds": time.perf_counter() - started, "failure_reason": "" if process.returncode == 0 else output[-2000:]}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "returncode": None, "runtime_seconds": time.perf_counter() - started,
                "failure_reason": f"timeout after {timeout} seconds"}


def _cost(project_root: Path, state_count: int, panel_name: str) -> np.ndarray:
    panel_cost = Path(__file__).resolve().parents[1] / "data" / "derived" / "b3" / f"state_cost_{panel_name}.csv"
    if panel_cost.exists():
        cost = pd.read_csv(panel_cost).to_numpy(dtype=float)
        if cost.shape == (state_count, state_count):
            return cost
    packaged = Path(__file__).resolve().parents[1] / "data" / "derived" / "b3" / "state_cost.csv"
    if packaged.exists():
        cost = pd.read_csv(packaged).to_numpy(dtype=float)
        if cost.shape == (state_count, state_count):
            return cost
    path = project_root / "script" / "future_state_prediction_IOT" / "persistence_sota_v2" / "data" / "manifests" / "gse239651_expt2_frozen_state_dictionary.npz"
    if path.exists():
        centers = np.load(path, allow_pickle=False)["cluster_centers"]
        cost = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        if cost.shape == (state_count, state_count):
            return cost / np.median(cost[cost > 0])
    identity = np.eye(state_count)
    return ((identity[:, None, :] - identity[None, :, :]) ** 2).sum(axis=2)


def _resolve_package_path(root: Path, value: str | Path) -> Path:
    """Resolve a manifest path without requiring the original machine layout."""

    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--methods", nargs="+", default=["source-carry-forward", "development-target-transfer", "prescient", "tigon", "mioflow"])
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--mioflow-epochs", type=int, default=50)
    parser.add_argument("--prescient-epochs", type=int, default=100)
    parser.add_argument("--tigon-iterations", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="reuse completed panel-method-seed rows from a prior checkpoint or final file")
    args = parser.parse_args()
    root = args.package_root.resolve()
    manifest = json.loads((root / "data" / "derived" / "b3" / "panel_manifest.json").read_text(encoding="utf-8"))
    legacy = root / "external_methods" / "envs_runtime" / "neural_legacy" / "Scripts" / "python.exe"
    modern = root / "external_methods" / "envs_runtime" / "neural_modern" / "Scripts" / "python.exe"
    adapters = root / "external_methods" / "adapters"
    output_root = root / "results" / "B3_prospective_composition" / "external_dynamics"
    records = []
    completed_keys: set[tuple[str, str, int]] = set()
    if args.resume:
        for checkpoint in (output_root / "b3_external_status.partial.csv", output_root / "b3_external_all_runs.csv"):
            if checkpoint.exists():
                prior = pd.read_csv(checkpoint)
                records = prior.to_dict(orient="records")
                completed_keys = {
                    (str(row["panel"]), str(row["method"]), int(row["seed"]))
                    for row in records
                    if str(row.get("status")) == "success"
                }
                break
    for panel in [item for item in manifest["panels"] if item["status"] == "included"]:
        state_count = len(np.load(_resolve_package_path(root, panel["train_path"]), allow_pickle=False)["state_names"])
        cost = _cost(root.parent, state_count, panel["panel"])
        train_path = _resolve_package_path(root, panel["train_path"])
        test_input_path = _resolve_package_path(root, panel["test_input_path"])
        truth_path = _resolve_package_path(root, panel["truth_path"])
        for method in args.methods:
            method_seeds = SEEDS if method in {"prescient", "tigon", "mioflow"} else [SEEDS[0]]
            for seed in method_seeds:
                key = (str(panel["panel"]), method, int(seed))
                if key in completed_keys:
                    continue
                artifact = output_root / method / f"{panel['panel']}_seed{seed}.npz"
                log = output_root / "logs" / method / f"{panel['panel']}_seed{seed}.log"
                common = ["--input", str(train_path), "--initial-input", str(test_input_path), "--output", str(artifact), "--seed", str(seed)]
                if method in {"source-carry-forward", "development-target-transfer"}:
                    command = ["python", str(adapters / "run_prospective_baseline.py"), *common, "--method", method]
                elif method == "prescient":
                    command = [str(legacy), str(adapters / "run_prescient.py"), *common, "--epochs", str(args.prescient_epochs)]
                elif method == "tigon":
                    command = [str(legacy), str(adapters / "run_tigon.py"), *common, "--source-root", str(root / "external_methods" / "sources" / "tigon"), "--iterations", str(args.tigon_iterations)]
                elif method == "mioflow":
                    command = [str(modern), str(adapters / "run_mioflow.py"), *common, "--epochs", str(args.mioflow_epochs)]
                else:
                    raise ValueError(f"unknown method: {method}")
                result = _run(command, log, args.timeout)
                row = {"dataset": panel.get("evaluation_dataset", "GSE239651_expt2"), "panel": panel["panel"], "condition": panel["condition"],
                       "time_days": panel["time_days"], "method": method, "seed": seed, "artifact": str(artifact), **result}
                if result["status"] == "success":
                    row.update(evaluate_prospective_population_artifact(truth_path, artifact, cost))
                records.append(row)
                completed_keys.add(key)
                output_root.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(records).to_csv(output_root / "b3_external_status.partial.csv", index=False)
    frame = pd.DataFrame(records)
    frame.to_csv(output_root / "b3_external_all_runs.csv", index=False)
    summary = frame.groupby(["dataset", "method"], as_index=False).mean(numeric_only=True)
    summary.to_csv(output_root / "b3_external_summary.csv", index=False)
    print(json.dumps({"runs": len(frame), "success": int((frame.status == "success").sum()), "failed": int((frame.status == "failed").sum())}, indent=2))
    return 0 if (frame.status == "success").all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
