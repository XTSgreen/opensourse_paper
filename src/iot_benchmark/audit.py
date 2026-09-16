"""Read-only project and benchmark qualification audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .protocol import load_protocol, protocol_sha256


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def audit_project(project_root: str | Path, package_root: str | Path) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    package_root = Path(package_root).resolve()
    protocol = load_protocol(package_root / "configs" / "protocol_v1.yaml")
    dataset_registry = _load_yaml(package_root / "configs" / "datasets" / "dataset_registry.yaml")
    method_registry = _load_yaml(package_root / "configs" / "methods" / "method_registry.yaml")

    datasets: list[dict[str, Any]] = []
    for dataset, spec in dataset_registry["datasets"].items():
        path = project_root / spec["local_path"]
        datasets.append(
            {
                "dataset": dataset,
                "declared_status": spec["local_status"],
                "path": str(path),
                "exists": path.exists(),
                "kind": "directory" if path.is_dir() else "file" if path.is_file() else "missing",
            }
        )

    methods = []
    for method, spec in method_registry["methods"].items():
        record = {"method": method, **spec}
        if spec.get("implementation"):
            implementation = (package_root / spec["implementation"]).resolve()
            record["implementation_path"] = str(implementation)
            record["implementation_exists"] = implementation.exists()
        methods.append(record)

    required = set(protocol["methods"]["required_external_families"])
    registered = {record["method"] for record in methods}
    report = {
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_sha256(package_root / "configs" / "protocol_v1.yaml"),
        "project_root": str(project_root),
        "package_root": str(package_root),
        "datasets": datasets,
        "all_declared_local_assets_exist": all(item["exists"] for item in datasets),
        "methods": methods,
        "all_required_method_families_registered": required.issubset(registered),
        "adapter_pending_count": sum(item.get("status") == "adapter_pending" for item in methods),
    }
    return report


def write_audit(report: dict[str, Any], output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path
