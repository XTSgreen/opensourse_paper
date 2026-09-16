import numpy as np
import pandas as pd

from iot_benchmark.models import GHIOTModel, PersistIOTModel, UOTIOTModel
from iot_benchmark.models.persist_features import FeatureContract
from iot_benchmark.models.prospective_iot import gauge_project_cost, log_sinkhorn


def test_public_uot_and_gh_interfaces_are_operational():
    source = np.array([[8, 2], [2, 8], [7, 3], [3, 7]], dtype=float)
    target = np.array([[6, 4], [3, 7], [5, 5], [4, 6]], dtype=float)
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])
    uot = UOTIOTModel().fit(source, target, cost)
    q = np.vstack([uot.predict_state(row) for row in source])
    x = source / source.sum(axis=1, keepdims=True)
    gh = GHIOTModel().fit(x, source, target, q, cost)
    prediction, gate = gh.predict(x, q)
    assert prediction.shape == target.shape
    assert np.allclose(prediction.sum(axis=1), 1.0)
    assert np.all((gate >= 0.0) & (gate <= 1.0))


def test_public_persist_interface_does_not_read_target_columns_at_prediction():
    rows = []
    for lineage_index in range(8):
        source_state = np.array([0.8, 0.2]) if lineage_index % 2 == 0 else np.array([0.2, 0.8])
        target_state = source_state[::-1]
        for target_time, horizon in (("D3", 3.0), ("D7", 7.0), ("D14", 14.0)):
            rows.append({
                "case_id": f"d|r|c|L{lineage_index}|D0->{target_time}", "dataset_id": "d",
                "replicate_id": "r", "condition_id": "c", "lineage_id": f"L{lineage_index}",
                "source_time": "D0", "target_time": target_time, "horizon_days": horizon,
                "source_observed_count": 2.0 + lineage_index % 3,
                "target_observed_count": 1.0 + (lineage_index + int(horizon)) % 3,
                "target_detected": 1, "feature__observed_count": 2.0 + lineage_index % 3,
                "feature__module_a": float(lineage_index % 2),
                "feature__source_ct_cells": 2.0 + lineage_index % 3,
                "source_state_prob_0": source_state[0], "source_state_prob_1": source_state[1],
                "target_state_prob_0": target_state[0], "target_state_prob_1": target_state[1],
            })
    cases = pd.DataFrame(rows)
    contract = FeatureContract(("observed_count", "module_a", "source_ct_cells"))
    model = PersistIOTModel(contract, state_count=2, hmm_epochs=20, iot_epochs=20).fit(cases)
    original = model.predict(cases).sort_values("case_id").reset_index(drop=True)
    changed = cases.copy()
    for column in changed:
        if column.startswith("target_"):
            changed[column] = 999.0 if column != "target_detected" else 0
    modified = model.predict(changed).sort_values("case_id").reset_index(drop=True)
    numeric = [column for column in original.select_dtypes(include=[np.number]) if column != "horizon_days"]
    assert np.allclose(original[numeric], modified[numeric])


def test_public_log_sinkhorn_respects_marginals_and_gauge():
    source = np.array([0.4, 0.35, 0.25])
    target = np.array([0.3, 0.45, 0.25])
    cost = np.array([[0.1, 0.5, 0.9], [0.4, 0.2, 0.8], [0.7, 0.5, 0.1]])
    projected = gauge_project_cost(cost)
    plan, diagnostics = log_sinkhorn(source, target, cost)
    assert np.allclose(projected.mean(axis=0), 0.0)
    assert np.allclose(projected.mean(axis=1), 0.0)
    assert np.allclose(plan.sum(axis=1), source, atol=1e-7)
    assert np.allclose(plan.sum(axis=0), target, atol=1e-7)
    assert diagnostics.marginal_residual < 1e-7
