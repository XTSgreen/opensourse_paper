"""Standardize the frozen-theta cross-cohort validation with source hashes.

This evidence tests whether a direction vector learned on GSE163558 retains
predictive value after it is frozen and transferred to unseen cohorts.  It does
not claim known-truth coefficient recovery or independent recovery of every
coefficient sign.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    package_root = Path(__file__).resolve().parents[1]
    project_root = package_root.parent
    source_root = project_root / "output" / "transfer_08_generalization"
    sources = {
        "GSE246662": source_root / "output" / "validation_bootstrap_ci.json",
        "GSE183904": source_root / "output_g183904" / "validation_g183904.json",
    }
    training_direction = (
        project_root / "output" / "transfer_04_driver_inference" / "output_v2" / "drivers_v2.json"
    )
    output = package_root / "results" / "external_direction_validation"
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    hashes = {str(training_direction.relative_to(project_root)): _sha256(training_direction)}
    for cohort, path in sources.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        hashes[str(path.relative_to(project_root))] = _sha256(path)
        for record in payload["per_site"]:
            rows.append(
                {
                    "training_cohort": "GSE163558",
                    "validation_cohort": cohort,
                    "site": record["site"],
                    "direction_status": "frozen_before_validation",
                    "metric": "conditional_transition_mae_gain_over_independence",
                    "gain": record["gain_mean"],
                    "ci95_lower": record["ci"][0],
                    "ci95_upper": record["ci"][1],
                    "bootstrap_replicates": record["n_bootstrap"],
                    "ci_excludes_zero": bool(record["excl_zero"]),
                    "relative_gain": record["relative_gain"],
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "frozen_direction_external_validation.csv", index=False)
    passed = int(frame["ci_excludes_zero"].sum())
    manifest = {
        "claim_scope": "predictive validity of a frozen learned direction on unseen cohorts and sites",
        "excluded_claims": [
            "known-truth coefficient recovery",
            "independent confirmation of every coefficient sign",
            "causal effect identification",
        ],
        "training_cohort": "GSE163558",
        "validation_cohorts": sorted(frame["validation_cohort"].unique().tolist()),
        "independent_sites": int(len(frame)),
        "sites_with_positive_ci": passed,
        "gate_pass": bool(passed >= 1),
        "input_sha256": hashes,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
