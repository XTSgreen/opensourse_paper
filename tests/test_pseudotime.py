"""Unit tests for the pseudotime readout and evaluation helpers."""

from __future__ import annotations

import numpy as np

from iot_benchmark.pseudotime import (
    chain_case,
    diffusion_pseudotime,
    fit_synthetic_method,
    hitting_time_pseudotime,
    rank_correlation,
    row_normalize_transition,
    state_coupling_from_clone,
    time_separation_auc,
)


def test_hitting_time_on_deterministic_chain() -> None:
    transition = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    values = hitting_time_pseudotime(transition, root=2)
    assert np.allclose(values, [2.0, 1.0, 0.0])


def test_zero_row_fallback_normalizes() -> None:
    matrix = np.array([[1.0, 1.0], [0.0, 0.0]])
    output = row_normalize_transition(matrix)
    assert np.allclose(output.sum(axis=1), 1.0)
    assert np.all(output >= 0)


def test_diffusion_pseudotime_is_monotone_on_a_line() -> None:
    embedding = np.arange(60, dtype=float)[:, None]
    values = diffusion_pseudotime(embedding, root=0, neighbors=8)
    assert rank_correlation(values, embedding[:, 0]) > 0.95


def test_chain_case_shapes_and_truth() -> None:
    phi, theta, train, held, truth = chain_case(states=5, sample_count=300, seed=7)
    assert phi.shape == (5, 5, 5)
    assert theta.shape == (5,)
    assert len(train) == 5 and len(held) == 3
    assert np.allclose(truth, np.arange(5))


def test_state_coupling_aggregation_matches_manual() -> None:
    source = np.array([[1.0, 0.0], [0.0, 1.0]])
    target = np.array([[0.5, 0.5], [0.0, 1.0]])
    clone = np.array([[1.0, 0.0], [0.0, 1.0]])
    aggregated = state_coupling_from_clone(clone, source, target)
    assert np.isclose(aggregated.sum(), 1.0)
    manual = source.T @ clone @ target
    manual = manual / manual.sum()
    assert np.allclose(aggregated, manual)


def test_fit_synthetic_soft_recovers_order() -> None:
    result = fit_synthetic_method(mode="soft_iot", sample_count=300, seed=11, restarts=2)
    assert np.isfinite(result["ordering"])
    assert result["ordering"] > 0.5
    assert len(result["restart_pseudotime"]) == 2


def test_time_separation_auc_bounds() -> None:
    pseudotime = np.array([0.0, 1.0, 2.0])
    source = np.array([[1.0, 0.0, 0.0]])
    target = np.array([[0.0, 0.0, 1.0]])
    assert np.isclose(time_separation_auc(pseudotime, source, target), 1.0)
    assert np.isclose(time_separation_auc(pseudotime[::-1], source, target), 0.0)
    assert np.isclose(time_separation_auc(np.array([1.0, 1.0, 1.0]), source, target), 0.5)
