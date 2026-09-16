"""Evaluate G1-G8 from artifacts without upgrading incomplete gates."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    required_families = {"wot", "moscot", "lineageot", "tigon", "mioflow", "cellrank2", "prescient"}
    successful = set()
    status_inputs = []
    for path in [results / "B2_lineage_transition" / "core_method_status.csv", results / "B2_lineage_transition" / "neural_method_status.csv"]:
        if path.exists():
            frame = pd.read_csv(path)
            status_inputs.append(str(path))
            successful.update(frame.loc[frame["status"].eq("success"), "method"].astype(str).str.lower())
    g1_fraction = len(required_families & successful) / len(required_families)

    b1_gate_path = results / "B1_known_truth_final" / "b1_gate_status.json"
    b1_gate = json.loads(b1_gate_path.read_text(encoding="utf-8")) if b1_gate_path.exists() else {}
    b3_table = results / "B3_prospective_composition" / "b3_state_prediction_metrics.csv"
    direction_manifest_path = results / "external_direction_validation" / "manifest.json"
    direction_manifest = json.loads(direction_manifest_path.read_text(encoding="utf-8")) if direction_manifest_path.exists() else {}
    stability_manifest_path = results / "direction_stability" / "manifest.json"
    stability_manifest = json.loads(stability_manifest_path.read_text(encoding="utf-8")) if stability_manifest_path.exists() else {}
    noninferiority_manifest_path = results / "noninferiority" / "manifest.json"
    noninferiority_manifest = json.loads(noninferiority_manifest_path.read_text(encoding="utf-8")) if noninferiority_manifest_path.exists() else {}
    g2 = all(_exists(path) for path in [
        results / "B1_known_truth_final" / "b1_all_runs.csv",
        results / "B2_lineage_transition" / "b2_method_summary.csv",
        b3_table,
    ])
    figures = [root / "figures" / "rendered" / f"Figure{index}.pdf" for index in range(1, 6)]
    reviewer = results / "reviewer_tables"
    reviewer_required = [
        "benchmark_summary.csv", "benchmark_all_runs.parquet", "paired_statistics.csv",
        "interpretability_summary.csv", "method_status.csv", "compute_resources.csv", "figure_source_manifest.csv",
        "table_S8_b3_calibration.csv",
    ]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
        git_commit = commit.stdout.strip() if commit.returncode == 0 else None
    except OSError:
        git_commit = None
    release_metadata = root / "release_metadata.json"
    release = json.loads(release_metadata.read_text(encoding="utf-8")) if release_metadata.exists() else {}
    release_tag_commit = None
    if release.get("release_tag"):
        tag_result = subprocess.run(
            ["git", "rev-list", "-n", "1", str(release["release_tag"])],
            cwd=root, capture_output=True, text=True, check=False,
        )
        release_tag_commit = tag_result.stdout.strip() if tag_result.returncode == 0 else None
    release_commit_matches_tag = bool(release.get("git_commit") and release.get("git_commit") == release_tag_commit)
    gates = {
        "G1_external_coverage": {"pass": g1_fraction >= 0.80, "coverage_fraction": g1_fraction, "successful_families": sorted(required_families & successful), "status_inputs": status_inputs},
        "G2_three_benchmarks": {"pass": g2},
        "G3_prediction_noninferiority": {
            "pass": bool(noninferiority_manifest.get("gate_pass", False)),
            "source": str(noninferiority_manifest_path),
            "benchmark_decisions": noninferiority_manifest.get("benchmark_decisions", {}),
        },
        "G4_parameter_recovery": {"pass": bool(b1_gate.get("G4_parameter_recovery", {}).get("pass", False)), "source": str(b1_gate_path)},
        "G5_direction_stability": {
            "pass": bool(stability_manifest.get("gate_pass", False)),
            "source": str(stability_manifest_path),
            "real_data_parameter_direction_min_cosine": stability_manifest.get("real_data_parameter_direction_min_cosine"),
            "output_proxy_comparisons_with_positive_ci": stability_manifest.get("output_proxy_comparisons_with_positive_ci", 0),
        },
        "G6_external_direction_test": {
            "pass": bool(direction_manifest.get("gate_pass", False)),
            "source": str(direction_manifest_path),
            "claim_scope": direction_manifest.get("claim_scope"),
            "sites_with_positive_ci": direction_manifest.get("sites_with_positive_ci", 0),
        },
        "G7_clean_reproduction": {"pass": all(_exists(path) for path in figures), "required_figures": [str(path) for path in figures]},
        "G8_public_release": {
            "pass": bool(git_commit and release.get("release_tag") and release.get("zenodo_doi") and release_commit_matches_tag and (root / "LICENSE").exists() and all(_exists(reviewer / name) for name in reviewer_required)),
            "git_commit": git_commit,
            "release_tag": release.get("release_tag"),
            "release_commit": release.get("git_commit"),
            "release_tag_commit": release_tag_commit,
            "release_commit_matches_tag": release_commit_matches_tag,
            "zenodo_doi": release.get("zenodo_doi"),
            "license_present": (root / "LICENSE").exists(),
        },
    }
    payload = {"all_pass": all(gate["pass"] for gate in gates.values()), "gates": gates}
    output = results / "FINAL_GATE_STATUS.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
