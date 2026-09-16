"""Load, validate and hash the frozen revision protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_BENCHMARKS = {
    "B1_known_truth",
    "B2_lineage_transition",
    "B3_prospective_composition",
}
REQUIRED_EXTERNAL_FAMILIES = {
    "Waddington-OT",
    "moscot",
    "LineageOT_or_CoSpar",
    "TIGON",
    "TrajectoryNet_or_MIOFlow",
    "CellRank_2",
    "PRESCIENT",
}


def default_protocol_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "protocol_v1.yaml"


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:
    protocol_path = Path(path) if path else default_protocol_path()
    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: dict[str, Any]) -> None:
    if not isinstance(protocol, dict):
        raise ValueError("Protocol must be a mapping")
    if protocol.get("protocol_frozen") is not True:
        raise ValueError("Unified benchmark protocol must be frozen")
    benchmarks = set(protocol.get("benchmarks", {}))
    missing_benchmarks = REQUIRED_BENCHMARKS - benchmarks
    if missing_benchmarks:
        raise ValueError(f"Missing required benchmarks: {sorted(missing_benchmarks)}")
    methods = protocol.get("methods", {})
    external = set(methods.get("required_external_families", []))
    missing_methods = REQUIRED_EXTERNAL_FAMILIES - external
    if missing_methods:
        raise ValueError(f"Missing required method families: {sorted(missing_methods)}")
    if len(protocol.get("random_seeds", [])) < 5:
        raise ValueError("At least five frozen random seeds are required")
    information_modes = protocol.get("information_modes", {})
    prospective = information_modes.get("prospective_prediction", {})
    forbidden_true = [
        key
        for key in (
            "target_observations_allowed",
            "target_marginal_allowed",
            "future_clone_graph_allowed_when_method_contract_requires",
        )
        if prospective.get(key) is not False
    ]
    if forbidden_true:
        raise ValueError(f"Prospective information leakage in fields: {forbidden_true}")
    gates = protocol.get("gates", {})
    expected_gates = {f"G{i}" for i in range(1, 9)}
    observed_gates = {name.split("_", 1)[0] for name in gates}
    if expected_gates != observed_gates:
        raise ValueError("Protocol must define gates G1 through G8")


def protocol_sha256(path: str | Path | None = None) -> str:
    protocol_path = Path(path) if path else default_protocol_path()
    return hashlib.sha256(protocol_path.read_bytes()).hexdigest()


def write_lock(output: str | Path, path: str | Path | None = None) -> Path:
    protocol_path = Path(path) if path else default_protocol_path()
    protocol = load_protocol(protocol_path)
    payload = {
        "protocol_path": str(protocol_path.resolve()),
        "protocol_version": protocol["protocol_version"],
        "sha256": protocol_sha256(protocol_path),
        "bytes": protocol_path.stat().st_size,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
