"""Stable top-level entry points for the paper reproducibility workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT


def _run(command: list[str], *, log_name: str, allow_nonzero: bool = False) -> None:
    log_root = ROOT / "results" / "logs" / "reproduction"
    log_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    payload = {
        "command": command,
        "returncode": process.returncode,
        "runtime_seconds": time.perf_counter() - started,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    (log_root / f"{log_name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if process.returncode != 0 and not allow_nonzero:
        raise RuntimeError(f"step failed: {log_name}; inspect {log_root / f'{log_name}.json'}")


def setup() -> None:
    _run([sys.executable, "-m", "pip", "install", "-e", ".[test]"], log_name="01_install_main")
    _run([sys.executable, "-m", "pytest", "-q"], log_name="01_test_main")
    _run([sys.executable, "scripts/audit_public_models.py"], log_name="01_audit_public_models")


def prepare() -> None:
    _run([sys.executable, "-m", "iot_benchmark.cli", "verify-protocol"], log_name="02_verify_protocol")
    _run([sys.executable, "scripts/normalize_packaged_paths.py"], log_name="02_normalize_packaged_paths")
    _run([sys.executable, "-m", "iot_benchmark.cli", "prepare-data", "--project-root", str(PROJECT)], log_name="02_prepare_b1_b2")
    _run([sys.executable, "scripts/prepare_b3_panels.py"], log_name="02_prepare_b3")
    _run([sys.executable, "scripts/prepare_b3_gse140802.py"], log_name="02_prepare_b3_gse140802")
    _run([sys.executable, "scripts/audit_velocity_inputs.py"], log_name="02_velocity_qualification")
    _run([sys.executable, "-m", "iot_benchmark.cli", "audit", "--project-root", str(PROJECT)], log_name="02_asset_audit")


def smoke() -> None:
    _run([sys.executable, "-m", "pytest", "-q"], log_name="03_tests")
    _run([sys.executable, "-m", "iot_benchmark.cli", "run-b1", "--mode", "smoke", "--output", "results/runtime/B1_smoke"], log_name="03_b1_smoke")


def benchmarks() -> None:
    _run([sys.executable, "-m", "iot_benchmark.cli", "run-b1", "--mode", "full", "--output", "results/B1_known_truth_final"], log_name="04_b1_full")
    _run([sys.executable, "-m", "iot_benchmark.cli", "analyze-b1", "--input", "results/B1_known_truth_final/b1_all_runs.csv", "--output", "results/B1_known_truth_final"], log_name="04_b1_statistics")
    _run([sys.executable, "scripts/run_b2_core.py"], log_name="04_b2_core")
    _run([sys.executable, "scripts/run_b2_iot.py"], log_name="04_b2_iot")
    _run([sys.executable, "scripts/run_b2_neural.py"], log_name="04_b2_neural")
    _run([sys.executable, "scripts/summarize_b2.py"], log_name="04_b2_summary")
    locked_b3 = ROOT / "results" / "B3_prospective_composition" / "b3_state_prediction_metrics.csv"
    if not locked_b3.exists():
        _run([sys.executable, "scripts/import_b3_existing.py"], log_name="04_b3_persist")
    else:
        _run([sys.executable, "-c", "print('packaged locked PERSIST-IOT results detected; import is reproducibly skipped')"], log_name="04_b3_persist_packaged")
    _run([sys.executable, "scripts/run_b3_external.py"], log_name="04_b3_external")
    _run([sys.executable, "scripts/analyze_b3_calibration.py"], log_name="04_b3_calibration")
    _run([sys.executable, "scripts/analyze_noninferiority.py"], log_name="04_noninferiority")
    _run([sys.executable, "scripts/analyze_direction_stability.py"], log_name="04_direction_stability")
    frozen_direction = ROOT / "results" / "external_direction_validation" / "frozen_direction_external_validation.csv"
    if not frozen_direction.exists():
        _run([sys.executable, "scripts/import_external_direction_validation.py"], log_name="04_external_direction")
    else:
        _run([sys.executable, "-c", "print('packaged frozen-direction validation detected; import is reproducibly skipped')"], log_name="04_external_direction_packaged")


def outputs() -> None:
    _run([sys.executable, "scripts/normalize_packaged_paths.py"], log_name="05_normalize_packaged_paths")
    _run([sys.executable, "scripts/audit_public_models.py"], log_name="05_audit_public_models")
    _run([sys.executable, "scripts/render_figures.py"], log_name="05_figures_1_to_5")
    _run([sys.executable, "scripts/build_reviewer_tables.py"], log_name="05_reviewer_tables")
    _run([sys.executable, "scripts/verify_release.py"], log_name="05_verify_release", allow_nonzero=True)
    _run([sys.executable, "scripts/normalize_packaged_paths.py"], log_name="05_finalize_portable_paths")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("setup", "prepare", "smoke", "benchmarks", "outputs", "all"))
    args = parser.parse_args()
    stages = {"setup": setup, "prepare": prepare, "smoke": smoke, "benchmarks": benchmarks, "outputs": outputs}
    selected = list(stages) if args.stage == "all" else [args.stage]
    for name in selected:
        stages[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
