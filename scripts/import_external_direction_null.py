"""Standardize the external-direction null control with source hashes.

The control re-evaluates the frozen GSE163558 direction and matched null
directions under an identical paired bootstrap.  The matched-norm Gaussian null
is the primary control; coefficient permutation retains a small positive gain
because several cost features carry redundant information and is reported as a
weak null.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    package_root = Path(__file__).resolve().parents[1]
    project_root = package_root.parent
    source = (
        project_root / "output" / "transfer_08_generalization" / "output"
        / "external_direction_null_control.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for record in payload["site_rows"]:
        rows.append(
            {
                "training_cohort": "GSE163558",
                "validation_cohort": record["cohort"],
                "site": record["site"].split("_")[-1],
                "observed_gain": record["observed"]["gain_mean"],
                "observed_ci_low": record["observed"]["gain_ci"][0],
                "observed_ci_high": record["observed"]["gain_ci"][1],
                "observed_excludes_zero": record["observed"]["excl_zero"],
                "gaussian_mean": record["gaussian"]["mean"],
                "gaussian_q025": record["gaussian"]["q025"],
                "gaussian_q975": record["gaussian"]["q975"],
                "gaussian_p_two_sided": record["gaussian"]["p_two_sided"],
                "gaussian_draws_excluding_zero": record["gaussian"]["draws_excluding_zero"],
                "permuted_mean": record["permuted"]["mean"],
                "permuted_q025": record["permuted"]["q025"],
                "permuted_q975": record["permuted"]["q975"],
                "permuted_p_two_sided": record["permuted"]["p_two_sided"],
                "permuted_draws_excluding_zero": record["permuted"]["draws_excluding_zero"],
                "negated_gain": record["negated"]["gain_mean"],
                "negated_ci_low": record["negated"]["gain_ci"][0],
                "negated_ci_high": record["negated"]["gain_ci"][1],
                "zero_direction_gain": record["zero_direction"]["gain_mean"],
                "n_bootstrap": record["n_bootstrap"],
                "n_gaussian": record["n_gaussian"],
                "n_permuted": record["n_permuted"],
            }
        )
    frame = pd.DataFrame(rows)
    output = package_root / "results" / "external_direction_validation"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "null_control.csv", index=False)
    manifest = {
        "control": "matched-norm null directions and weak permutation null under a paired bootstrap",
        "claim_scope": "the frozen-direction gain is specific to the learned direction, not to the baseline",
        "primary_null": "matched-norm Gaussian direction (100 draws)",
        "weak_null": "coefficient permutation (100 draws) retains a small positive gain because cost features are redundant",
        "reference_controls": ["reversed direction", "zero direction"],
        "protocol": payload["protocol"],
        "joint": payload["joint"],
        "n_sites": int(len(frame)),
        "source": str(source.relative_to(project_root)).replace("\\", "/"),
        "source_sha256": sha256(source),
        "output": "results/external_direction_validation/null_control.csv",
    }
    (output / "null_control_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
