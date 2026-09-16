"""Build frozen, method-neutral panels from the existing public-data assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("state-composition rows must have positive mass")
    return values / totals


def _save_panel(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _portable_path(path: Path, package_root: Path) -> str:
    """Return a package-relative artifact path when the artifact is packaged."""

    try:
        return str(path.resolve().relative_to(package_root.resolve()))
    except ValueError:
        return str(path)


def build_gse140802(project_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    package_root = output_dir.parent.parent
    packaged_source_dir = package_root / "data" / "processed" / "GSE140802"
    legacy_source_dir = project_root / "all_material" / "nature_methods_upgrade" / "outputs"
    outputs = []
    for target_time in (9, 16):
        filename = f"gse140802_in_vivo_t2_t{target_time}.npz"
        packaged_path = packaged_source_dir / filename
        source_path = packaged_path if packaged_path.exists() else legacy_source_dir / filename
        raw = np.load(source_path, allow_pickle=False)
        source_counts = np.asarray(raw["X_counts"], dtype=float)
        target_counts = np.asarray(raw["Y_counts"], dtype=float)
        panel_path = output_dir / f"gse140802_t2_t{target_time}.npz"
        record = _save_panel(
            panel_path,
            source_counts=source_counts,
            target_counts=target_counts,
            source_composition=_normalize_rows(source_counts),
            target_composition=_normalize_rows(target_counts),
            group=np.asarray(raw["mouse"]).astype(str),
            lineage_id=np.asarray(raw["clone_id"]).astype(str),
            state_names=np.asarray(raw["state_labels"]).astype(str),
            source_time=np.repeat(2, len(source_counts)),
            target_time=np.repeat(target_time, len(source_counts)),
        )
        record.update(
            {
                "dataset": "GSE140802",
                "source_asset": _portable_path(source_path, package_root),
                "source_asset_sha256": _sha256(source_path),
                "rows": int(len(source_counts)),
                "states": int(source_counts.shape[1]),
                "biological_units": int(len(np.unique(raw["mouse"]))),
                "information_modes": ["retrospective_reconstruction", "prospective_prediction"],
            }
        )
        outputs.append(record)
    return outputs


def _state_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in frame.columns if column.startswith(prefix)]
    return sorted(columns, key=lambda value: int(value.rsplit("_", 1)[1]))


def build_transition_csv(project_root: Path, output_dir: Path, filename: str, dataset_name: str) -> dict[str, Any]:
    package_root = output_dir.parent.parent
    packaged_path = package_root / "data" / "processed" / "GSE239651" / filename
    legacy_path = project_root / "data" / "future_state_prediction_IOT" / filename
    source_path = packaged_path if packaged_path.exists() else legacy_path
    frame = pd.read_csv(source_path)
    source_columns = _state_columns(frame, "source_state_")
    target_columns = _state_columns(frame, "target_state_")
    if len(source_columns) != len(target_columns):
        raise ValueError(f"state dimensions do not match in {source_path}")
    source_counts = frame[source_columns].to_numpy(dtype=float)
    target_counts = frame[target_columns].to_numpy(dtype=float)
    valid = (source_counts.sum(axis=1) > 0) & (target_counts.sum(axis=1) > 0)
    frame = frame.loc[valid].reset_index(drop=True)
    source_counts = source_counts[valid]
    target_counts = target_counts[valid]
    panel_path = output_dir / f"{dataset_name.lower()}_transition_panel.npz"
    record = _save_panel(
        panel_path,
        source_counts=source_counts,
        target_counts=target_counts,
        source_composition=_normalize_rows(source_counts),
        target_composition=_normalize_rows(target_counts),
        group=frame["sample_source"].astype(str).to_numpy(dtype=str),
        lineage_id=frame["lineage_id"].astype(str).to_numpy(dtype=str),
        window=frame["window"].astype(str).to_numpy(dtype=str),
        source_time=frame["source_time"].astype(str).to_numpy(dtype=str),
        target_time=frame["target_time"].astype(str).to_numpy(dtype=str),
        state_names=np.asarray([f"state_{index}" for index in range(len(source_columns))]),
    )
    record.update(
        {
            "dataset": dataset_name,
            "source_asset": _portable_path(source_path, package_root),
            "source_asset_sha256": _sha256(source_path),
            "rows": int(len(frame)),
            "states": int(len(source_columns)),
            "biological_units": int(frame["sample_source"].nunique()),
            "information_modes": ["retrospective_reconstruction"],
        }
    )
    return record


def verify_prospective_feature_boundary(panel_path: Path) -> dict[str, Any]:
    panel = np.load(panel_path, allow_pickle=False)
    prospective_keys = {"source_counts", "source_composition", "group", "lineage_id", "state_names", "source_time"}
    target_keys = {name for name in panel.files if name.startswith("target_")}
    leaked = sorted(prospective_keys & target_keys)
    return {
        "panel": str(panel_path),
        "prospective_feature_keys": sorted(prospective_keys & set(panel.files)),
        "target_label_keys": sorted(target_keys),
        "target_keys_in_prospective_features": leaked,
        "pass": len(leaked) == 0,
    }


def build_all_panels(project_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    package_root = output_dir.parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_gse140802(project_root, output_dir)
    records.append(
        build_transition_csv(
            project_root,
            output_dir,
            "gse239651_lineage_transition_records.csv",
            "GSE239651",
        )
    )
    records.append(
        build_transition_csv(
            project_root,
            output_dir,
            "macsgestalt_locked_transition_records.csv",
            "macsGESTALT",
        )
    )
    boundary_checks = [
        verify_prospective_feature_boundary(Path(record["path"]))
        for record in records
        if "prospective_prediction" in record["information_modes"]
    ]
    for record in records:
        record["path"] = _portable_path(Path(record["path"]), package_root)
    for check in boundary_checks:
        check["panel"] = _portable_path(Path(check["panel"]), package_root)
    manifest = {
        "schema_version": "1.0.0",
        "project_root": ".",
        "panels": records,
        "prospective_boundary_checks": boundary_checks,
        "all_boundary_checks_pass": all(check["pass"] for check in boundary_checks),
    }
    manifest_path = output_dir / "panel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
