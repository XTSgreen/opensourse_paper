"""Audit that formal model implementations are self-contained in the package."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    package_root = Path(__file__).resolve().parents[1]
    project_root = package_root.parent
    model_root = package_root / "src" / "iot_benchmark" / "models"
    files = sorted(model_root.glob("*.py"))
    forbidden = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if re.search(r"(?:all_material|script[/\\]future_state_prediction_IOT|parents\(\)[^\n]*persistence_sota)", text):
            forbidden.append(str(path.relative_to(package_root)))
    origin_map = {
        "gh_backend.py": "script/future_state_prediction_IOT/code/future_prediction_models.py",
        "gh_iot.py": "script/future_state_prediction_IOT/code/canonical_models/gh_iot.py",
        "uot_iot_canonical.py": "script/future_state_prediction_IOT/code/canonical_models/uot_iot.py",
        "persist_iot.py": "script/future_state_prediction_IOT/code/canonical_models/persist_iot.py",
        "persist_features.py": "script/future_state_prediction_IOT/persistence_sota_v2/code/data_builders/forecast_cases.py",
        "persist_hmm.py": "script/future_state_prediction_IOT/persistence_sota_v2/code/models/persist_hmm.py",
        "prospective_iot.py": "script/future_state_prediction_IOT/persistence_sota_v2/code/models/prospective_iot.py",
        "persist_iot_estimator.py": "script/future_state_prediction_IOT/persistence_sota_v2/code/models/persist_iot_estimator.py",
    }
    mapping = []
    for name, origin in origin_map.items():
        current = model_root / name
        source = project_root / origin
        row = {"package_file": str(current.relative_to(package_root)), "origin_file": origin,
               "package_sha256": _sha256(current), "origin_available": source.exists()}
        if source.exists():
            row["origin_sha256"] = _sha256(source)
            row["hash_match"] = row["package_sha256"] == row["origin_sha256"]
        else:
            row["origin_sha256"] = None
            row["hash_match"] = None
        mapping.append(row)
    payload = {"self_contained": not forbidden, "forbidden_references": forbidden,
               "formal_models": ["UOT-IOT", "GH-IOT", "PERSIST-IOT"], "source_mapping": mapping,
               "audit_note": "Historical source hashes are checked when the provenance tree is present; the packaged model code is independently executable without it."}
    output = package_root / "results" / "audits" / "public_model_migration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["self_contained"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
