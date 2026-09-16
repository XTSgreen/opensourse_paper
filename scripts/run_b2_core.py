"""Run the first four official B2 adapters and retain all statuses."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from iot_benchmark.external_evaluation import write_external_evaluation


def run_command(command: list[str], log_path: Path, timeout: int = 3600) -> dict[str, object]:
    started = time.perf_counter()
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        output = (process.stdout or "") + ("\n" + process.stderr if process.stderr else "")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
        return {
            "status": "success" if process.returncode == 0 else "failed",
            "returncode": process.returncode,
            "runtime_seconds": time.perf_counter() - started,
            "failure_reason": None if process.returncode == 0 else output[-2000:],
        }
    except subprocess.TimeoutExpired as exc:
        output = ((exc.stdout or "") if isinstance(exc.stdout, str) else "") + ((exc.stderr or "") if isinstance(exc.stderr, str) else "")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
        return {
            "status": "timeout",
            "returncode": None,
            "runtime_seconds": time.perf_counter() - started,
            "failure_reason": f"timeout after {timeout} seconds",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    root = args.package_root.resolve()
    panels = sorted((root / "data" / "derived").glob("*_panel.npz")) + sorted(
        (root / "data" / "derived").glob("gse140802_*.npz")
    )
    legacy_python = root / "external_methods" / "envs_runtime" / "legacy_ot" / "Scripts" / "python.exe"
    modern_python = root / "external_methods" / "envs_runtime" / "modern_scverse" / "Scripts" / "python.exe"
    adapter_dir = root / "external_methods" / "adapters"
    result_root = root / "results" / "B2_lineage_transition"
    records: list[dict[str, object]] = []

    for panel in panels:
        dataset = panel.stem
        method_specs = (
            ("wot", legacy_python, adapter_dir / "run_wot.py"),
            ("moscot", modern_python, adapter_dir / "run_moscot.py"),
            ("lineageot", legacy_python, adapter_dir / "run_lineageot.py"),
        )
        for method, executable, adapter in method_specs:
            output = result_root / method / f"{dataset}.npz"
            log = result_root / "logs" / method / f"{dataset}.log"
            command = [str(executable), str(adapter), "--input", str(panel), "--output", str(output)]
            status = run_command(command, log, timeout=args.timeout)
            record = {"dataset": dataset, "method": method, "artifact": str(output), "log": str(log), **status}
            if status["status"] == "success":
                metric_path = output.with_name(output.stem + "_metrics.json")
                write_external_evaluation(panel, output, metric_path)
                record["metrics"] = str(metric_path)
            records.append(record)

        moscot_output = result_root / "moscot" / f"{dataset}.npz"
        cellrank_output = result_root / "cellrank2" / f"{dataset}.npz"
        log = result_root / "logs" / "cellrank2" / f"{dataset}.log"
        if moscot_output.exists():
            command = [
                str(modern_python), str(adapter_dir / "run_cellrank2.py"),
                "--panel", str(panel), "--coupling", str(moscot_output), "--output", str(cellrank_output),
            ]
            status = run_command(command, log, timeout=args.timeout)
        else:
            status = {"status": "not_applicable", "returncode": None, "runtime_seconds": 0.0, "failure_reason": "moscot dependency failed"}
        record = {"dataset": dataset, "method": "cellrank2", "artifact": str(cellrank_output), "log": str(log), **status}
        if status["status"] == "success":
            metric_path = cellrank_output.with_name(cellrank_output.stem + "_metrics.json")
            write_external_evaluation(panel, cellrank_output, metric_path)
            record["metrics"] = str(metric_path)
        records.append(record)

    result_root.mkdir(parents=True, exist_ok=True)
    status_frame = pd.DataFrame(records)
    status_frame.to_csv(result_root / "core_method_status.csv", index=False)
    summary = {
        "panels": len(panels),
        "runs": len(records),
        "success": int((status_frame["status"] == "success").sum()),
        "failed": int((status_frame["status"] == "failed").sum()),
        "timeout": int((status_frame["status"] == "timeout").sum()),
        "not_applicable": int((status_frame["status"] == "not_applicable").sum()),
    }
    (result_root / "core_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 and summary["timeout"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
