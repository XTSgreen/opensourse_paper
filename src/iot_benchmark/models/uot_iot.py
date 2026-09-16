"""Low-capacity public UOT-IOT implementation used by the unified benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..synthetic import hard_ot_plan, soft_uot_plan


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    totals = values.sum(axis=1, keepdims=True)
    output = values / np.maximum(totals, 1e-300)
    zero = totals[:, 0] <= 0
    if np.any(zero):
        output[zero] = 1.0 / values.shape[1]
    return output


def transition_features(states: int) -> tuple[np.ndarray, tuple[str, ...]]:
    phi = np.zeros((states, states, states + 1), dtype=float)
    for target in range(states):
        phi[:, target, target] = 1.0
    phi[:, :, -1] = np.eye(states)
    names = tuple([f"target_state_{index}" for index in range(states)] + ["source_target_persistence"])
    return phi, names


def observed_state_coupling(
    source_composition: np.ndarray,
    target_composition: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    source = _normalize_rows(source_composition)
    target = _normalize_rows(target_composition)
    if weights is None:
        weights = np.ones(len(source), dtype=float)
    weights = np.asarray(weights, dtype=float)
    coupling = np.einsum("n,ni,nj->ij", weights, source, target)
    return coupling / coupling.sum()


def _gauge_direction(theta: np.ndarray, states: int) -> tuple[np.ndarray, float]:
    theta = np.asarray(theta, dtype=float).copy()
    theta[:states] -= theta[:states].mean()
    magnitude = float(np.linalg.norm(theta))
    if magnitude <= 0:
        return np.zeros_like(theta), 0.0
    return theta / magnitude, magnitude


@dataclass
class UOTIOTBenchmarkModel:
    epsilon: float = 1.0
    mu: float = 0.5
    restarts: int = 4
    maxiter: int = 400
    theta_: np.ndarray | None = None
    target_reference_: np.ndarray | None = None
    phi_: np.ndarray | None = None
    feature_names_: tuple[str, ...] | None = None
    objective_: float | None = None

    def _loss(self, theta: np.ndarray, observations: list[dict[str, np.ndarray]]) -> float:
        total = 0.0
        for observation in observations:
            plan = soft_uot_plan(
                self.phi_, theta, observation["source"], observation["target"],
                epsilon=self.epsilon, mu=self.mu,
            )
            estimate = _normalize_rows(plan)
            truth = _normalize_rows(observation["coupling"])
            row_weights = observation["coupling"].sum(axis=1)
            row_weights = row_weights / np.maximum(row_weights.sum(), 1e-300)
            total += float(-np.sum(row_weights[:, None] * truth * np.log(np.maximum(estimate, 1e-300))))
        return total / len(observations)

    def fit(
        self,
        source_composition: np.ndarray,
        target_composition: np.ndarray,
        groups: np.ndarray,
        source_counts: np.ndarray | None = None,
        target_counts: np.ndarray | None = None,
        *,
        seed: int = 20260916,
    ) -> "UOTIOTBenchmarkModel":
        source = _normalize_rows(source_composition)
        target = _normalize_rows(target_composition)
        groups = np.asarray(groups).astype(str)
        states = source.shape[1]
        self.phi_, self.feature_names_ = transition_features(states)
        if source_counts is not None and target_counts is not None:
            weights = np.sqrt(np.asarray(source_counts).sum(axis=1) * np.asarray(target_counts).sum(axis=1))
        else:
            weights = np.ones(len(source), dtype=float)
        observations = []
        for group in np.unique(groups):
            mask = groups == group
            coupling = observed_state_coupling(source[mask], target[mask], weights[mask])
            observations.append(
                {
                    "source": coupling.sum(axis=1),
                    "target": coupling.sum(axis=0),
                    "coupling": coupling,
                }
            )
        self.target_reference_ = np.average(target, axis=0, weights=weights)
        self.target_reference_ /= self.target_reference_.sum()
        rng = np.random.default_rng(seed)
        fits = []
        for _ in range(self.restarts):
            start = rng.normal(0.0, 0.1, size=states + 1)
            result = minimize(
                self._loss,
                start,
                args=(observations,),
                method="L-BFGS-B",
                options={"maxiter": self.maxiter, "ftol": 1e-12, "gtol": 1e-7},
            )
            fits.append(result)
        best = min(fits, key=lambda result: float(result.fun))
        self.theta_ = np.asarray(best.x, dtype=float)
        self.objective_ = float(best.fun)
        return self

    def predict_state(self, source_composition: np.ndarray) -> np.ndarray:
        if self.theta_ is None or self.target_reference_ is None or self.phi_ is None:
            raise RuntimeError("model is not fitted")
        source = _normalize_rows(source_composition)
        predictions = []
        for row in source:
            plan = soft_uot_plan(
                self.phi_, self.theta_, row, self.target_reference_,
                epsilon=self.epsilon, mu=self.mu,
            )
            predictions.append(plan.sum(axis=0))
        return _normalize_rows(np.asarray(predictions))

    def clone_coupling(
        self,
        source_composition: np.ndarray,
        target_composition: np.ndarray,
        source_mass: np.ndarray | None = None,
        target_mass: np.ndarray | None = None,
    ) -> np.ndarray:
        predicted = self.predict_state(source_composition)
        target = _normalize_rows(target_composition)
        cost = np.sum((predicted[:, None, :] - target[None, :, :]) ** 2, axis=2)
        scale = np.median(cost[cost > 0]) if np.any(cost > 0) else 1.0
        phi = np.zeros((len(predicted), len(target), 1), dtype=float)
        phi[:, :, 0] = -cost / max(scale, 1e-12)
        source_mass = np.ones(len(predicted)) if source_mass is None else source_mass
        target_mass = np.ones(len(target)) if target_mass is None else target_mass
        return hard_ot_plan(phi, np.array([1.0]), source_mass, target_mass, epsilon=0.05)

    def direction(self) -> tuple[np.ndarray, float]:
        if self.theta_ is None or self.target_reference_ is None:
            raise RuntimeError("model is not fitted")
        return _gauge_direction(self.theta_, len(self.target_reference_))
