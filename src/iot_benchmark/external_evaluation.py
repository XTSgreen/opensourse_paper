"""Evaluate normalized external-method artifacts against common lineage panels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import (
    coupling_relative_frobenius,
    cross_entropy,
    fate_correlations,
    multiclass_brier,
    sinkhorn_divergence,
    top1_accuracy,
    transition_weighted_l1,
)


def evaluate_prospective_population_artifact(
    truth_path: str | Path,
    artifact_path: str | Path,
    state_cost: np.ndarray,
) -> dict[str, float | str]:
    """Evaluate a prediction artifact against a physically separate future-truth file."""

    truth_file = np.load(truth_path, allow_pickle=False)
    artifact = np.load(artifact_path, allow_pickle=False)
    prediction = np.asarray(artifact["predicted_population"], dtype=float).reshape(1, -1)
    truth_counts = np.asarray(truth_file["target_counts"], dtype=float).sum(axis=0, keepdims=True)
    truth = truth_counts / truth_counts.sum(axis=1, keepdims=True)
    confidence = float(prediction.max())
    predicted_class = int(prediction.argmax())
    return {
        "truth": str(Path(truth_path).resolve()),
        "artifact": str(Path(artifact_path).resolve()),
        "cross_entropy": cross_entropy(prediction, truth),
        "sinkhorn_divergence": sinkhorn_divergence(prediction[0], truth[0], state_cost),
        "top1_accuracy": top1_accuracy(prediction, truth),
        "multiclass_brier": multiclass_brier(prediction, truth),
        "predicted_confidence": confidence,
        "observed_probability_of_predicted_class": float(truth[0, predicted_class]),
    }


def lineage_truth_coupling(panel: np.lib.npyio.NpzFile) -> np.ndarray:
    source_counts = np.asarray(panel["source_counts"], dtype=float)
    target_counts = np.asarray(panel["target_counts"], dtype=float)
    weights = np.sqrt(source_counts.sum(axis=1) * target_counts.sum(axis=1))
    weights = weights / weights.sum()
    return np.diag(weights)


def aggregate_state_coupling(
    clone_coupling: np.ndarray,
    source_composition: np.ndarray,
    target_composition: np.ndarray,
) -> np.ndarray:
    clone_coupling = np.asarray(clone_coupling, dtype=float)
    source_composition = np.asarray(source_composition, dtype=float)
    target_composition = np.asarray(target_composition, dtype=float)
    if clone_coupling.shape != (len(source_composition), len(target_composition)):
        raise ValueError("clone coupling shape does not match panel rows")
    state_coupling = source_composition.T @ clone_coupling @ target_composition
    total = state_coupling.sum()
    if total <= 0:
        raise ValueError("state coupling has zero mass")
    return state_coupling / total


def clone_fate_probabilities(clone_coupling: np.ndarray, target_composition: np.ndarray) -> np.ndarray:
    clone_coupling = np.asarray(clone_coupling, dtype=float)
    target_composition = np.asarray(target_composition, dtype=float)
    row_sums = clone_coupling.sum(axis=1, keepdims=True)
    conditional = clone_coupling / np.maximum(row_sums, 1e-300)
    fate = conditional @ target_composition
    return fate / np.maximum(fate.sum(axis=1, keepdims=True), 1e-300)


def evaluate_external_artifact(panel_path: str | Path, artifact_path: str | Path) -> dict[str, float | str]:
    panel = np.load(panel_path, allow_pickle=False)
    artifact = np.load(artifact_path, allow_pickle=False)
    estimate = np.asarray(artifact["coupling"], dtype=float)
    truth = lineage_truth_coupling(panel)
    source_composition = np.asarray(panel["source_composition"], dtype=float)
    target_composition = np.asarray(panel["target_composition"], dtype=float)
    estimated_state = aggregate_state_coupling(estimate, source_composition, target_composition)
    true_state = aggregate_state_coupling(truth, source_composition, target_composition)
    estimated_fate = clone_fate_probabilities(estimate, target_composition)
    true_fate = clone_fate_probabilities(truth, target_composition)
    result: dict[str, float | str] = {
        "panel": str(Path(panel_path).resolve()),
        "artifact": str(Path(artifact_path).resolve()),
        "coupling_relative_frobenius": coupling_relative_frobenius(estimate, truth),
        "transition_weighted_l1": transition_weighted_l1(estimated_state, true_state),
    }
    result.update(fate_correlations(estimated_fate, true_fate))
    return result


def write_external_evaluation(panel_path: str | Path, artifact_path: str | Path, output: str | Path) -> Path:
    with np.load(artifact_path, allow_pickle=False) as artifact:
        keys = set(artifact.files)
    if "coupling" in keys:
        result = evaluate_external_artifact(panel_path, artifact_path)
    elif "fate_probabilities" in keys:
        result = evaluate_fate_artifact(panel_path, artifact_path)
    elif "predicted_population" in keys:
        result = evaluate_population_artifact(panel_path, artifact_path)
    else:
        raise ValueError("external artifact has neither coupling nor fate probabilities")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def evaluate_fate_artifact(panel_path: str | Path, artifact_path: str | Path) -> dict[str, float | str]:
    panel = np.load(panel_path, allow_pickle=False)
    artifact = np.load(artifact_path, allow_pickle=False)
    estimate = np.asarray(artifact["fate_probabilities"], dtype=float)
    estimate_names = np.asarray(artifact["fate_names"]).astype(str)
    state_names = np.asarray(panel["state_names"]).astype(str)
    target = np.asarray(panel["target_composition"], dtype=float)
    aligned = np.zeros((len(estimate), len(state_names)), dtype=float)
    for index, name in enumerate(estimate_names):
        matches = np.flatnonzero(state_names == name)
        if len(matches) == 1:
            aligned[:, matches[0]] = estimate[:, index]
    aligned /= np.maximum(aligned.sum(axis=1, keepdims=True), 1e-300)
    result: dict[str, float | str] = {
        "panel": str(Path(panel_path).resolve()),
        "artifact": str(Path(artifact_path).resolve()),
        "coupling_relative_frobenius": "NA-by-design",
        "transition_weighted_l1": "NA-by-design",
    }
    result.update(fate_correlations(aligned, target))
    return result


def evaluate_population_artifact(panel_path: str | Path, artifact_path: str | Path) -> dict[str, float | str]:
    panel = np.load(panel_path, allow_pickle=False)
    artifact = np.load(artifact_path, allow_pickle=False)
    prediction = np.asarray(artifact["predicted_population"], dtype=float).reshape(1, -1)
    target_counts = np.asarray(panel["target_counts"], dtype=float)
    truth = (target_counts.sum(axis=0) / target_counts.sum()).reshape(1, -1)
    state_features = np.eye(prediction.shape[1])
    cost = np.sum((state_features[:, None, :] - state_features[None, :, :]) ** 2, axis=2)
    return {
        "panel": str(Path(panel_path).resolve()),
        "artifact": str(Path(artifact_path).resolve()),
        "cross_entropy": cross_entropy(prediction, truth),
        "sinkhorn_divergence": sinkhorn_divergence(prediction[0], truth[0], cost),
        "top1_accuracy": top1_accuracy(prediction, truth),
        "multiclass_brier": multiclass_brier(prediction, truth),
        "calibration": "NA-by-design-single-population",
    }
