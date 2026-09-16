"""Build strictly prospective expt1-training/expt2-evaluation population panels."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


STATE_COUNT = 6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_points(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probabilities = frame[[f"state_prob_{index}" for index in range(STATE_COUNT)]].to_numpy(dtype=float)
    counts = frame[[f"state_count_{index}" for index in range(STATE_COUNT)]].to_numpy(dtype=float)
    mask = np.all(np.isfinite(probabilities), axis=1) & (counts.sum(axis=1) > 0)
    return probabilities[mask], counts[mask]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _source_reference(path: Path, package_root: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(package_root.resolve())).replace("\\", "/")
    except ValueError:
        return "external_project/" + str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")


def main() -> int:
    package_root = Path(__file__).resolve().parents[1]
    project_root = package_root.parent
    packaged_panel_root = package_root / "data" / "processed" / "GSE239651"
    legacy_panel_root = project_root / "script" / "future_state_prediction_IOT" / "persistence_sota_v2" / "data" / "panels"
    expt1_name = "gse239651_expt1_lineage_panel.parquet"
    expt2_name = "gse239651_expt2_lineage_panel.parquet"
    expt1_path = packaged_panel_root / expt1_name
    expt2_path = packaged_panel_root / expt2_name
    if not expt1_path.exists() or not expt2_path.exists():
        expt1_path = legacy_panel_root / expt1_name
        expt2_path = legacy_panel_root / expt2_name
    expt1 = pd.read_parquet(expt1_path)
    expt2 = pd.read_parquet(expt2_path)
    output_root = package_root / "data" / "derived" / "b3"
    output_root.mkdir(parents=True, exist_ok=True)

    common_conditions = sorted(set(expt1["condition_id"]) & set(expt2["condition_id"]))
    records = []
    for condition in common_conditions:
        first = expt1.loc[expt1["condition_id"].eq(condition)]
        second = expt2.loc[expt2["condition_id"].eq(condition)]
        common_days = sorted(
            (set(first.loc[first["time_index"] > 0, "time_days"]) & set(second.loc[second["time_index"] > 0, "time_days"]))
        )
        source_frame = first.loc[first["time_index"].eq(0)]
        source, source_counts = _valid_points(source_frame)
        external_source, external_source_counts = _valid_points(second.loc[second["time_index"].eq(0)])
        for day in common_days:
            target, target_counts = _valid_points(first.loc[first["time_days"].eq(day)])
            external, external_counts = _valid_points(second.loc[second["time_days"].eq(day)])
            if min(len(source), len(target), len(external)) < 2:
                records.append({
                    "condition": condition, "time_days": day, "status": "excluded",
                    "reason": "fewer than two valid state-resolved lineages in at least one required partition",
                    "train_source_n": len(source), "train_target_n": len(target), "external_target_n": len(external),
                })
                continue
            name = f"gse239651_{_slug(condition)}_day{day:g}"
            train_path = output_root / "train" / f"{name}.npz"
            test_input_path = output_root / "test_input" / f"{name}.npz"
            truth_path = output_root / "evaluation_truth" / f"{name}.npz"
            train_path.parent.mkdir(parents=True, exist_ok=True)
            test_input_path.parent.mkdir(parents=True, exist_ok=True)
            truth_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                train_path,
                source_composition=source,
                target_composition=target,
                source_counts=source_counts,
                target_counts=target_counts,
                state_names=np.asarray([f"state_{index}" for index in range(STATE_COUNT)]),
                group=np.asarray([condition] * len(source)),
                train_dataset=np.asarray(["GSE239651_expt1"]),
                time_days=np.asarray([day], dtype=float),
            )
            np.savez_compressed(
                test_input_path,
                source_composition=external_source,
                source_counts=external_source_counts,
                state_names=np.asarray([f"state_{index}" for index in range(STATE_COUNT)]),
                dataset=np.asarray(["GSE239651_expt2"]),
                condition=np.asarray([condition]),
                source_time_days=np.asarray([0.0]),
                target_time_days=np.asarray([day], dtype=float),
            )
            np.savez_compressed(
                truth_path,
                target_composition=external,
                target_counts=external_counts,
                state_names=np.asarray([f"state_{index}" for index in range(STATE_COUNT)]),
                evaluation_dataset=np.asarray(["GSE239651_expt2"]),
                condition=np.asarray([condition]),
                time_days=np.asarray([day], dtype=float),
            )
            records.append({
                "panel": name, "train_path": str(train_path.relative_to(package_root)), "train_sha256": _sha256(train_path),
                "test_input_path": str(test_input_path.relative_to(package_root)), "test_input_sha256": _sha256(test_input_path),
                "truth_path": str(truth_path.relative_to(package_root)), "truth_sha256": _sha256(truth_path), "condition": condition,
                "time_days": day, "status": "included", "reason": "",
                "train_source_n": len(source), "train_target_n": len(target), "external_target_n": len(external),
                "information_boundary": "method process receives the train_path only; truth_path is opened only by the evaluator",
            })
    manifest = {
        "benchmark": "B3_prospective_composition",
        "training_dataset": "GSE239651_expt1",
        "locked_external_dataset": "GSE239651_expt2",
        "source_files": {
            _source_reference(expt1_path, package_root, project_root): _sha256(expt1_path),
            _source_reference(expt2_path, package_root, project_root): _sha256(expt2_path),
        },
        "panels": records,
    }
    (output_root / "panel_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(records).to_csv(output_root / "panel_manifest.csv", index=False)
    print(json.dumps({"included": sum(r["status"] == "included" for r in records), "total": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
