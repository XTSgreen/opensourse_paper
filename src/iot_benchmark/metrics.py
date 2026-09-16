"""Reference metrics shared by all benchmark adapters."""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _probabilities(values: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array < 0):
        raise ValueError("probabilities or masses must be finite and non-negative")
    totals = array.sum(axis=axis, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("probability or mass totals must be positive")
    return array / totals


def coupling_relative_frobenius(estimate: np.ndarray, truth: np.ndarray, eps: float = 1e-12) -> float:
    estimate_mass = _probabilities(np.asarray(estimate, dtype=float).reshape(-1))
    truth_mass = _probabilities(np.asarray(truth, dtype=float).reshape(-1))
    if estimate_mass.shape != truth_mass.shape:
        raise ValueError("coupling shapes must match")
    return float(np.linalg.norm(estimate_mass - truth_mass) / (np.linalg.norm(truth_mass) + eps))


def transition_matrix(coupling: np.ndarray) -> np.ndarray:
    coupling = np.asarray(coupling, dtype=float)
    if coupling.ndim != 2 or np.any(~np.isfinite(coupling)) or np.any(coupling < 0):
        raise ValueError("coupling must be a finite non-negative matrix")
    totals = coupling.sum(axis=1, keepdims=True)
    return coupling / np.maximum(totals, 1e-300)


def transition_weighted_l1(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if estimate.shape != truth.shape or estimate.ndim != 2:
        raise ValueError("transition couplings must be equal two-dimensional arrays")
    weights = truth.sum(axis=1)
    weights = weights / weights.sum()
    row_errors = np.abs(transition_matrix(estimate) - transition_matrix(truth)).sum(axis=1)
    return float(np.dot(weights, row_errors))


def fate_correlations(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if estimate.shape != truth.shape:
        raise ValueError("fate matrices must have equal shapes")
    x, y = estimate.reshape(-1), truth.reshape(-1)
    return {
        "fate_pearson": float(pearsonr(x, y).statistic),
        "fate_spearman": float(spearmanr(x, y).statistic),
        "fate_top1_accuracy": float(np.mean(estimate.argmax(axis=1) == truth.argmax(axis=1))),
    }


def cross_entropy(prediction: np.ndarray, truth: np.ndarray, floor: float = 1e-12) -> float:
    prediction = _probabilities(prediction, axis=1)
    truth = _probabilities(truth, axis=1)
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth shapes must match")
    return float(-np.mean(np.sum(truth * np.log(np.clip(prediction, floor, 1.0)), axis=1)))


def multiclass_brier(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = _probabilities(prediction, axis=1)
    truth = _probabilities(truth, axis=1)
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth shapes must match")
    return float(np.mean(np.sum((prediction - truth) ** 2, axis=1)))


def top1_accuracy(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = _probabilities(prediction, axis=1)
    truth = _probabilities(truth, axis=1)
    return float(np.mean(prediction.argmax(axis=1) == truth.argmax(axis=1)))


def expected_calibration_error(prediction: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    prediction = _probabilities(prediction, axis=1)
    truth = _probabilities(truth, axis=1)
    confidence = prediction.max(axis=1)
    predicted_class = prediction.argmax(axis=1)
    observed = truth[np.arange(len(truth)), predicted_class]
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (confidence < upper if index < bins - 1 else confidence <= upper)
        if np.any(mask):
            error += float(mask.mean()) * abs(float(confidence[mask].mean() - observed[mask].mean()))
    return float(error)


def _sinkhorn_cost(a: np.ndarray, b: np.ndarray, cost: np.ndarray, epsilon: float, iterations: int) -> float:
    a = _probabilities(np.asarray(a, dtype=float).reshape(-1))
    b = _probabilities(np.asarray(b, dtype=float).reshape(-1))
    cost = np.asarray(cost, dtype=float)
    if cost.shape != (len(a), len(b)):
        raise ValueError("cost shape must match distributions")
    kernel = np.exp(-cost / float(epsilon))
    kernel = np.maximum(kernel, np.finfo(float).tiny)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(iterations):
        u = a / np.maximum(kernel @ v, np.finfo(float).tiny)
        v = b / np.maximum(kernel.T @ u, np.finfo(float).tiny)
    plan = (u[:, None] * kernel) * v[None, :]
    return float(np.sum(plan * cost))


def sinkhorn_divergence(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    *,
    epsilon: float = 0.05,
    iterations: int = 1000,
) -> float:
    ab = _sinkhorn_cost(a, b, cost, epsilon, iterations)
    aa = _sinkhorn_cost(a, a, cost, epsilon, iterations)
    bb = _sinkhorn_cost(b, b, cost, epsilon, iterations)
    return float(max(0.0, ab - 0.5 * aa - 0.5 * bb))


def coefficient_recovery(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    estimate = np.asarray(estimate, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    if estimate.shape != truth.shape:
        raise ValueError("coefficient vectors must match")
    estimate_norm = estimate / max(np.linalg.norm(estimate), 1e-12)
    truth_norm = truth / max(np.linalg.norm(truth), 1e-12)
    nonzero_truth = np.abs(truth) > 1e-8
    sign_accuracy = (
        float(np.mean(np.sign(estimate[nonzero_truth]) == np.sign(truth[nonzero_truth])))
        if np.any(nonzero_truth)
        else float("nan")
    )
    top_k = max(1, int(np.count_nonzero(nonzero_truth)))
    estimate_top = set(np.argsort(np.abs(estimate))[-top_k:].tolist())
    truth_top = set(np.argsort(np.abs(truth))[-top_k:].tolist())
    top_k_overlap = len(estimate_top & truth_top) / top_k
    return {
        "coefficient_cosine": float(np.dot(estimate_norm, truth_norm)),
        "coefficient_spearman": float(spearmanr(estimate, truth).statistic),
        "coefficient_sign_accuracy": sign_accuracy,
        "coefficient_top_k_overlap": float(top_k_overlap),
        "coefficient_relative_l2": float(np.linalg.norm(estimate_norm - truth_norm)),
    }


def direction_stability(directions: np.ndarray) -> dict[str, float]:
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or len(directions) < 2:
        raise ValueError("at least two direction vectors are required")
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("zero directions are invalid")
    normalized = directions / norms
    cosines = [float(np.dot(normalized[i], normalized[j])) for i, j in combinations(range(len(normalized)), 2)]
    return {
        "direction_pairwise_cosine": float(np.mean(cosines)),
        "direction_pairwise_cosine_min": float(np.min(cosines)),
        "direction_seed_variance": float(np.mean(np.var(normalized, axis=0, ddof=1))),
        "direction_sign_agreement": float(np.mean(np.abs(np.mean(np.sign(normalized), axis=0)))),
    }


def directional_curvature(
    objective: Callable[[np.ndarray], float],
    point: np.ndarray,
    direction: np.ndarray,
    *,
    step: float = 1e-4,
) -> float:
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    f0 = float(objective(point))
    fp = float(objective(point + step * direction))
    fm = float(objective(point - step * direction))
    return float((fp - 2.0 * f0 + fm) / (step**2))
