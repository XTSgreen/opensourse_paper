"""Pseudotime readout and evaluation for state-transition operators.

The module turns a state-level transition operator into a random-walk hitting-time
pseudotime, provides stability and ordering metrics, a diffusion-pseudotime
baseline, and a synthetic chain with known progression for known-truth checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from .synthetic import Observation, _normalize, soft_uot_plan


def row_normalize_transition(matrix: np.ndarray, fallback: str = "incoming") -> np.ndarray:
    matrix = np.maximum(np.asarray(matrix, dtype=float), 0.0)
    totals = matrix.sum(axis=1, keepdims=True)
    output = matrix / np.maximum(totals, 1e-300)
    zero = totals[:, 0] <= 0
    if np.any(zero):
        if fallback == "incoming":
            incoming = matrix.sum(axis=0)
            weights = incoming / incoming.sum() if incoming.sum() > 0 else np.ones(matrix.shape[1])
        elif fallback == "uniform":
            weights = np.ones(matrix.shape[1])
        else:
            raise ValueError(f"unknown fallback: {fallback}")
        output[zero] = weights / weights.sum()
    return output


def hitting_time_pseudotime(transition: np.ndarray, root: int) -> np.ndarray:
    transition = np.asarray(transition, dtype=float)
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition must be square")
    states = transition.shape[0]
    if not 0 <= root < states:
        raise ValueError("root out of range")
    keep = [index for index in range(states) if index != root]
    if len(keep) == 0:
        return np.zeros(states)
    local = transition[np.ix_(keep, keep)]
    system = np.eye(len(keep)) - local
    values = np.linalg.solve(system, np.ones(len(keep)))
    pseudotime = np.zeros(states)
    pseudotime[keep] = np.clip(values, 0.0, None)
    return pseudotime


def state_coupling_from_clone(
    clone_coupling: np.ndarray,
    source_composition: np.ndarray,
    target_composition: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_composition, dtype=float)
    target = np.asarray(target_composition, dtype=float)
    coupling = np.asarray(clone_coupling, dtype=float)
    if coupling.shape != (len(source), len(target)):
        raise ValueError("clone coupling shape does not match panel rows")
    aggregated = source.T @ coupling @ target
    total = float(aggregated.sum())
    if total <= 0:
        raise ValueError("state coupling has zero mass")
    return aggregated / total


def state_expected_time(
    source_counts: np.ndarray,
    target_counts: np.ndarray,
    source_time: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_counts, dtype=float)
    target = np.asarray(target_counts, dtype=float)
    source_t = _numeric_time(source_time)
    target_t = _numeric_time(target_time)
    numerator = source.T @ source_t + target.T @ target_t
    denominator = source.sum(axis=0) + target.sum(axis=0)
    return numerator / np.maximum(denominator, 1e-300)


def _numeric_time(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.number):
        return values.astype(float)
    unique = sorted(set(map(str, values)))
    mapping = {name: float(index) for index, name in enumerate(unique)}
    return np.array([mapping[str(value)] for value in values], dtype=float)


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if len(first) < 3 or np.all(first == first[0]) or np.all(second == second[0]):
        return float("nan")
    return float(spearmanr(first, second).statistic)


def kendall_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if len(first) < 3:
        return float("nan")
    return float(kendalltau(first, second).statistic)


def pairwise_stability(vectors: list[np.ndarray]) -> float:
    values = [np.asarray(vector, dtype=float) for vector in vectors]
    scores = [
        rank_correlation(values[i], values[j])
        for i in range(len(values))
        for j in range(i + 1, len(values))
    ]
    scores = [score for score in scores if np.isfinite(score)]
    return float(np.mean(scores)) if scores else float("nan")


def time_separation_auc(
    pseudotime: np.ndarray,
    source_counts: np.ndarray,
    target_counts: np.ndarray,
) -> float:
    """Weighted AUC of state pseudotime separating target-time from source-time mass."""
    pseudotime = np.asarray(pseudotime, dtype=float)
    source_weights = np.asarray(source_counts, dtype=float).sum(axis=0)
    target_weights = np.asarray(target_counts, dtype=float).sum(axis=0)
    if source_weights.sum() <= 0 or target_weights.sum() <= 0:
        return float("nan")
    labels = np.r_[np.zeros(len(pseudotime)), np.ones(len(pseudotime))]
    scores = np.r_[pseudotime, pseudotime]
    weights = np.r_[source_weights, target_weights]
    return float(roc_auc_score(labels, scores, sample_weight=weights))


def jackknife_pseudotime_stability(
    state_couplings: list[np.ndarray],
    weights: list[np.ndarray],
    root: int,
) -> float:
    full = _aggregate_couplings(state_couplings, weights)
    reference = hitting_time_pseudotime(row_normalize_transition(full), root)
    scores = []
    for index in range(len(state_couplings)):
        kept_couplings = [item for i, item in enumerate(state_couplings) if i != index]
        kept_weights = [item for i, item in enumerate(weights) if i != index]
        if not kept_couplings:
            continue
        one_out = _aggregate_couplings(kept_couplings, kept_weights)
        candidate = hitting_time_pseudotime(row_normalize_transition(one_out), root)
        scores.append(rank_correlation(reference, candidate))
    scores = [score for score in scores if np.isfinite(score)]
    return float(np.mean(scores)) if scores else float("nan")


def _aggregate_couplings(state_couplings: list[np.ndarray], weights: list[np.ndarray]) -> np.ndarray:
    total = np.zeros_like(np.asarray(state_couplings[0], dtype=float))
    for coupling, weight in zip(state_couplings, weights):
        total += float(weight) * np.asarray(coupling, dtype=float)
    return total


def diffusion_pseudotime(
    embedding: np.ndarray,
    root: int,
    *,
    neighbors: int = 15,
    alpha: float = 0.5,
    diffusion_time: int = 1,
) -> np.ndarray:
    """Diffusion-pseudotime-style baseline computed on a kNN cell graph."""
    embedding = np.asarray(embedding, dtype=float)
    count = len(embedding)
    if not 0 <= root < count:
        raise ValueError("root out of range")
    k = min(neighbors, count - 1)
    model = NearestNeighbors(n_neighbors=k + 1).fit(embedding)
    distances, indices = model.kneighbors(embedding)
    distances, indices = distances[:, 1:], indices[:, 1:]
    sigma = np.maximum(distances[:, -1], 1e-12)
    weights = np.exp(-((distances / sigma[:, None]) ** 2))
    affinity = np.zeros((count, count), dtype=float)
    rows = np.repeat(np.arange(count), k)
    affinity[rows, indices.reshape(-1)] = weights.reshape(-1)
    affinity = np.maximum(affinity, affinity.T)
    degree = affinity.sum(axis=1, keepdims=True)
    markov = affinity / np.maximum(degree, 1e-300)
    lazy = alpha * markov + (1.0 - alpha) * np.eye(count)
    for _ in range(max(0, diffusion_time - 1)):
        lazy = lazy @ lazy
    return hitting_time_pseudotime(lazy, root)


def chain_case(
    *,
    states: int = 6,
    sample_count: int = 2000,
    seed: int = 20260916,
    observations: int = 5,
    heldout: int = 3,
    features: int = 5,
    dropout: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[Observation], list[Observation], np.ndarray]:
    rng = np.random.default_rng(seed)
    index = np.arange(states, dtype=float)
    phi = np.zeros((states, states, features), dtype=float)
    phi[:, :, 0] = index[None, :]
    phi[:, :, 1] = index[:, None]
    phi[:, :, 2] = index[:, None] * index[None, :] / max(states - 1, 1)
    phi[:, :, 3] = rng.normal(size=(states, states))
    phi[:, :, 4] = rng.normal(size=states)[None, :]
    flat = phi.reshape(-1, features)
    phi = ((flat - flat.mean(axis=0)) / np.maximum(flat.std(axis=0), 1e-12)).reshape(phi.shape)
    theta = np.array([1.4, 0.3, 0.8, -0.4, 0.25])
    theta = theta / np.linalg.norm(theta)
    weights = np.exp(-0.6 * index)
    early = _normalize(weights)
    late = _normalize(weights[::-1])
    train: list[Observation] = []
    held: list[Observation] = []
    for item in range(observations + heldout):
        progress = item / max(observations + heldout - 1, 1)
        source = _normalize(early * rng.uniform(0.6, 1.4, size=states) ** (1.0 - 0.5 * progress))
        target_reference = _normalize(late * rng.uniform(0.6, 1.4, size=states) ** (0.5 + 0.5 * progress))
        plan = soft_uot_plan(phi, theta, source, target_reference)
        counts = rng.multinomial(sample_count, _normalize(plan.reshape(-1))).reshape(plan.shape).astype(float)
        if dropout > 0:
            mask = rng.uniform(size=counts.shape) >= dropout
            counts *= mask
            if counts.sum() <= 0:
                counts.flat[int(np.argmax(plan))] = 1.0
        observed = counts / counts.sum()
        record = Observation(source, target_reference, observed, plan, context=0)
        (train if item < observations else held).append(record)
    truth = index
    return phi, theta, train, held, truth


@dataclass
class PanelRecord:
    dataset: str
    method: str
    seed: int
    state_pseudotime: np.ndarray
    state_time: np.ndarray
    agreement: float
    stability: float
    root: int


def fit_synthetic_method(
    *,
    mode: str,
    sample_count: int = 2000,
    seed: int = 20260916,
    restarts: int = 3,
) -> dict[str, object]:
    from .synthetic import fit_inverse

    phi, theta_true, train, held, truth = chain_case(sample_count=sample_count, seed=seed)
    theta_hat, loss, restart_vectors = fit_inverse(phi, train, mode, seed + 1000, restarts=restarts)
    plans = []
    for observation in train:
        if mode == "soft_iot":
            plan = soft_uot_plan(phi, theta_hat, observation.source, observation.target_reference)
        else:
            from .synthetic import hard_ot_plan

            plan = hard_ot_plan(phi, theta_hat, observation.source, observation.observed_plan.sum(axis=0))
        plans.append(plan)
    total = np.sum(plans, axis=0)
    transition = row_normalize_transition(total)
    root = 0
    pseudotime = hitting_time_pseudotime(transition, root)
    restart_pseudotime = []
    for vector in restart_vectors:
        restart_plans = []
        for observation in train:
            if mode == "soft_iot":
                plan = soft_uot_plan(phi, vector, observation.source, observation.target_reference)
            else:
                from .synthetic import hard_ot_plan

                plan = hard_ot_plan(phi, vector, observation.source, observation.observed_plan.sum(axis=0))
            restart_plans.append(plan)
        restart_transition = row_normalize_transition(np.sum(restart_plans, axis=0))
        restart_pseudotime.append(hitting_time_pseudotime(restart_transition, root))
    normalized = np.asarray(restart_vectors) / np.maximum(
        np.linalg.norm(restart_vectors, axis=1, keepdims=True), 1e-12
    )
    return {
        "method": mode,
        "sample_count": sample_count,
        "seed": seed,
        "truth": truth,
        "pseudotime": pseudotime,
        "restart_pseudotime": restart_pseudotime,
        "ordering": rank_correlation(pseudotime, truth),
        "ordering_kendall": kendall_correlation(pseudotime, truth),
        "restart_stability": pairwise_stability(restart_pseudotime),
        "direction_dispersion": float(np.mean(np.var(normalized, axis=0, ddof=1))) if len(normalized) > 1 else float("nan"),
        "theta_hat": theta_hat,
    }
