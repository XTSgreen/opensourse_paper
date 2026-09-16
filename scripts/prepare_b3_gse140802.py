"""Add held-out-animal prospective panels for GSE140802."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _identity_cost(path: Path, state_count: int) -> None:
    identity = np.eye(state_count)
    cost = ((identity[:, None, :] - identity[None, :, :]) ** 2).sum(axis=2)
    pd.DataFrame(cost).to_csv(path, index=False)


def _build_panel(source_path: Path, root: Path, holdout: str) -> dict[str, object]:
    source = np.load(source_path, allow_pickle=False)
    groups = np.asarray(source["group"]).astype(str)
    state_names = np.asarray(source["state_names"]).astype(str)
    train_mask = groups != holdout
    test_mask = groups == holdout
    if train_mask.sum() < 2 or test_mask.sum() < 2:
        raise ValueError(f"held-out animal split is too small for {source_path.name}")

    name = f"{source_path.stem}_holdout_{_slug(holdout)}"
    output_root = root / "data" / "derived" / "b3"
    train_path = output_root / "train" / f"{name}.npz"
    test_path = output_root / "test_input" / f"{name}.npz"
    truth_path = output_root / "evaluation_truth" / f"{name}.npz"
    for path in (train_path, test_path, truth_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    source_counts = np.asarray(source["source_counts"], dtype=float)
    target_counts = np.asarray(source["target_counts"], dtype=float)
    source_composition = np.asarray(source["source_composition"], dtype=float)
    target_composition = np.asarray(source["target_composition"], dtype=float)
    target_time = np.asarray(source["target_time"])
    np.savez_compressed(
        train_path,
        source_counts=source_counts[train_mask], source_composition=source_composition[train_mask],
        target_counts=target_counts[train_mask], target_composition=target_composition[train_mask],
        group=groups[train_mask], lineage_id=np.asarray(source["lineage_id"])[train_mask],
        state_names=state_names, source_time=np.asarray(source["source_time"])[train_mask],
        target_time=target_time[train_mask], train_dataset=np.asarray(["GSE140802"]),
    )
    np.savez_compressed(
        test_path,
        source_counts=source_counts[test_mask], source_composition=source_composition[test_mask],
        group=groups[test_mask], lineage_id=np.asarray(source["lineage_id"])[test_mask],
        state_names=state_names, source_time=np.asarray(source["source_time"])[test_mask],
        dataset=np.asarray(["GSE140802"]), heldout_group=np.asarray([holdout]),
    )
    np.savez_compressed(
        truth_path,
        target_counts=target_counts[test_mask], target_composition=target_composition[test_mask],
        state_names=state_names, evaluation_dataset=np.asarray(["GSE140802"]),
        heldout_group=np.asarray([holdout]), target_time=target_time[test_mask],
    )
    _identity_cost(output_root / f"state_cost_{name}.csv", len(state_names))
    return {
        "panel": name,
        "dataset": "GSE140802",
        "evaluation_dataset": "GSE140802",
        "train_path": _relative(train_path, root), "train_sha256": _sha256(train_path),
        "test_input_path": _relative(test_path, root), "test_input_sha256": _sha256(test_path),
        "truth_path": _relative(truth_path, root), "truth_sha256": _sha256(truth_path),
        "condition": f"heldout_animal_{holdout}",
        "time_days": int(np.asarray(target_time)[0]), "status": "included", "reason": "",
        "holdout_group": holdout, "split_type": "held_out_animal",
        "train_source_n": int(train_mask.sum()), "train_target_n": int(train_mask.sum()),
        "external_target_n": int(test_mask.sum()),
        "information_boundary": "training receives non-holdout animals; prediction receives held-out source states; future targets are evaluator-only",
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data" / "derived" / "b3" / "panel_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = {item["panel"]: item for item in manifest["panels"]}
    for source_path in sorted((root / "data" / "derived").glob("gse140802_*.npz")):
        source = np.load(source_path, allow_pickle=False)
        for holdout in sorted(np.unique(np.asarray(source["group"]).astype(str))):
            record = _build_panel(source_path, root, str(holdout))
            existing[record["panel"]] = record
    manifest["panels"] = [existing[key] for key in sorted(existing)]
    manifest["prospective_external_datasets"] = sorted({item.get("evaluation_dataset", "GSE239651_expt2") for item in manifest["panels"]})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(manifest["panels"]).to_csv(manifest_path.with_suffix(".csv"), index=False)
    print(json.dumps({"included": sum(item["status"] == "included" for item in manifest["panels"]),
                      "added_dataset": "GSE140802", "held_out_panels": sum(item.get("dataset") == "GSE140802" for item in manifest["panels"]),
                      "status": "ready"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
