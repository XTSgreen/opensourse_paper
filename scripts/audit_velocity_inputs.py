"""Record whether conditional scVelo and dynamo input contracts are satisfied."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    registry = yaml.safe_load((root / "configs" / "datasets" / "dataset_registry.yaml").read_text(encoding="utf-8"))
    rows = []
    for dataset, specification in registry["datasets"].items():
        layers = specification.get("velocity_layers_available", specification.get("velocity_layers_expected", False))
        raw_kinetics = specification.get("raw_kinetics", False)
        for method in ("scVelo", "dynamo"):
            if method == "scVelo":
                applicable = bool(layers)
                reason = "spliced and unspliced layers declared available" if applicable else "frozen public panel has no spliced/unspliced layers"
            else:
                applicable = bool(raw_kinetics or layers)
                reason = "raw kinetic inputs declared available" if applicable else "frozen public panel has neither metabolic-label kinetics nor velocity layers"
            rows.append({
                "dataset": dataset,
                "method": method,
                "status": "eligible_pending_qc" if applicable else "not_applicable",
                "reason": reason,
                "velocity_layers": bool(layers),
                "raw_kinetics": bool(raw_kinetics),
            })
    output = root / "results" / "audits"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "velocity_method_qualification.csv", index=False)
    payload = {
        "conditional_methods": ["scVelo", "dynamo"],
        "decision": "not_applicable" if not frame["status"].str.startswith("eligible").any() else "mixed",
        "records": rows,
    }
    (output / "velocity_method_qualification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(rows), "eligible": int(frame["status"].str.startswith("eligible").sum())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
