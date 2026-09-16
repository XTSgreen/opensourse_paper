"""Run TIGON, MIOFlow and PRESCIENT on every B2 panel and protocol seed."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from iot_benchmark.external_evaluation import write_external_evaluation


SEEDS = [20260916, 20260917, 20260918, 20260919, 20260920]


def _run(command: list[str], log: Path, timeout: int) -> dict[str, object]:
    started = time.perf_counter()
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        text = (process.stdout or "") + ("\n" + process.stderr if process.stderr else "")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(text, encoding="utf-8")
        return {
            "status": "success" if process.returncode == 0 else "failed",
            "returncode": process.returncode,
            "runtime_seconds": time.perf_counter() - started,
            "failure_reason": "" if process.returncode == 0 else text[-2000:],
        }
    except subprocess.TimeoutExpired as error:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(str(error), encoding="utf-8")
        return {
            "status": "timeout", "returncode": None,
            "runtime_seconds": time.perf_counter() - started,
            "failure_reason": f"timeout after {timeout} seconds",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--mioflow-epochs", type=int, default=50)
    parser.add_argument("--prescient-epochs", type=int, default=100)
    parser.add_argument("--tigon-iterations", type=int, default=100)
    args = parser.parse_args()
    root = args.package_root.resolve()
    panels = sorted((root / "data" / "derived").glob("*_panel.npz")) + sorted(
        (root / "data" / "derived").glob("gse140802_*.npz")
    )
    neural_modern = root / "external_methods" / "envs_runtime" / "neural_modern" / "Scripts" / "python.exe"
    neural_legacy = root / "external_methods" / "envs_runtime" / "neural_legacy" / "Scripts" / "python.exe"
    adapters = root / "external_methods" / "adapters"
    output_root = root / "results" / "B2_lineage_transition"
    records = []
    for panel in panels:
        for seed in SEEDS:
            specs = [
                ("mioflow", neural_modern, adapters / "run_mioflow.py", ["--epochs", str(args.mioflow_epochs)]),
                ("prescient", neural_legacy, adapters / "run_prescient.py", ["--epochs", str(args.prescient_epochs)]),
                (
                    "tigon", neural_legacy, adapters / "run_tigon.py",
                    ["--source-root", str(root / "external_methods" / "sources" / "tigon"), "--iterations", str(args.tigon_iterations)],
                ),
            ]
            for method, executable, adapter, extra in specs:
                artifact = output_root / method / f"{panel.stem}_seed{seed}.npz"
                log = output_root / "logs" / method / f"{panel.stem}_seed{seed}.log"
                command = [str(executable), str(adapter), "--input", str(panel), "--output", str(artifact), "--seed", str(seed), *extra]
                result = _run(command, log, args.timeout)
                row = {"dataset": panel.stem, "method": method, "seed": seed, "artifact": str(artifact), "log": str(log), **result}
                if result["status"] == "success":
                    metric_path = artifact.with_name(artifact.stem + "_metrics.json")
                    write_external_evaluation(panel, artifact, metric_path)
                    row["metrics"] = str(metric_path)
                records.append(row)
                pd.DataFrame(records).to_csv(output_root / "neural_method_status.partial.csv", index=False)
    status = pd.DataFrame(records)
    status.to_csv(output_root / "neural_method_status.csv", index=False)
    summary = {
        "runs": len(status),
        "success": int((status.status == "success").sum()),
        "failed": int((status.status == "failed").sum()),
        "timeout": int((status.status == "timeout").sum()),
        "seeds": SEEDS,
        "mioflow_epochs": args.mioflow_epochs,
        "prescient_epochs": args.prescient_epochs,
        "tigon_iterations": args.tigon_iterations,
    }
    (output_root / "neural_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 and summary["timeout"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
