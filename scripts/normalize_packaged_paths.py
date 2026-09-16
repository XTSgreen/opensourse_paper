"""Make packaged manifests and result tables portable across machines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _portable(value: object, root: Path) -> object:
    if not isinstance(value, str) or not value:
        return value
    candidate = Path(value)
    try:
        return str(candidate.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        pass
    normalized = value.replace("\\", "/")
    marker = "/iot_reproducibility/"
    index = normalized.lower().find(marker)
    if index >= 0:
        tail = normalized[index + len(marker):]
        if tail.startswith(("data/", "results/", "figures/")):
            return tail
    if candidate.is_absolute():
        return "[external absolute path omitted]"
    return value


def _normalize_json(value: object, root: Path, keys: set[str]) -> object:
    if isinstance(value, dict):
        return {
            key: _normalize_path_value(item, root, keys) if key in keys else _normalize_json(item, root, keys)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_json(item, root, keys) for item in value]
    return value


def _normalize_path_value(value: object, root: Path, keys: set[str]) -> object:
    if isinstance(value, str):
        return _portable(value, root)
    if isinstance(value, list):
        return [_normalize_path_value(item, root, keys) for item in value]
    if isinstance(value, dict):
        return _normalize_json(value, root, keys)
    return value


def _rewrite_json(path: Path, root: Path, keys: set[str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized = _normalize_json(payload, root, keys)
    path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")


def _rewrite_csv(path: Path, root: Path) -> None:
    frame = pd.read_csv(path)
    changed = False
    for column in frame.columns:
        if column in {"artifact", "log", "metrics", "path", "train_path", "test_input_path", "truth_path", "truth", "panel"}:
            normalized = frame[column].map(lambda value: _portable(value, root))
            changed = changed or not normalized.equals(frame[column])
            frame[column] = normalized
    if changed:
        frame.to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.package_root.resolve()

    json_targets = [
        root / "configs" / "protocol_v1.lock.json",
        root / "data" / "derived" / "panel_manifest.json",
        root / "data" / "derived" / "b3" / "panel_manifest.json",
    ]
    json_keys = {
        "path", "panel", "train_path", "test_input_path", "truth_path", "project_root", "package_root",
        "protocol_path", "required_figures", "status_inputs", "source", "coupling", "implementation_path",
    }
    for path in json_targets:
        if path.exists():
            _rewrite_json(path, root, json_keys)

    for path in sorted((root / "data" / "derived").rglob("*.csv")) + sorted((root / "results").rglob("*.csv")):
        if "partial" not in path.name:
            _rewrite_csv(path, root)

    for path in sorted((root / "results").rglob("*.json")):
        _rewrite_json(path, root, {
            "artifact", "log", "metrics", "path", "panel", "input", "initial_input", "output", "truth_path",
            "project_root", "package_root", "protocol_path", "required_figures", "status_inputs", "source",
            "coupling", "implementation_path",
        })

    print(json.dumps({"package_root": str(root), "status": "portable_paths_normalized"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
