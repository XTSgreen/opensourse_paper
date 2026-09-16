import json
from pathlib import Path

import numpy as np


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def test_b3_method_files_do_not_contain_future_truth():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data" / "derived" / "b3" / "panel_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["panels"]:
        if record["status"] != "included":
            continue
        with np.load(_path(root, record["train_path"]), allow_pickle=False) as train:
            assert "external_target_counts" not in train.files
            assert "external_target_composition" not in train.files
        with np.load(_path(root, record["test_input_path"]), allow_pickle=False) as test_input:
            forbidden = {name for name in test_input.files if name.startswith("target_") and name != "target_time_days"}
            assert not forbidden
        if record.get("split_type") == "held_out_animal":
            with np.load(_path(root, record["train_path"]), allow_pickle=False) as train, np.load(
                _path(root, record["test_input_path"]), allow_pickle=False
            ) as test_input:
                assert set(train["group"].astype(str)).isdisjoint(set(test_input["group"].astype(str)))
                assert np.allclose(test_input["source_composition"].sum(axis=1), 1.0)


def test_b3_truth_is_physically_separate():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data" / "derived" / "b3" / "panel_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["panels"]:
        if record["status"] == "included":
            assert _path(root, record["train_path"]).resolve() != _path(root, record["truth_path"]).resolve()
            assert _path(root, record["test_input_path"]).resolve() != _path(root, record["truth_path"]).resolve()
