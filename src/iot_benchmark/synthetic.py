"""B1 known-truth benchmark for prediction-identifiability separation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import coefficient_recovery, coupling_relative_frobenius, directional_curvature


SCENARIOS = (
    "identifiable_soft_marginal",
    "hard_marginal_flat_direction",
    "equivalent_fit_multiple_parameters",
    "shared_plus_context_offset",
    "observation_noise_and_dropout",
    "nonlinear_or_hidden_confounding_misspecification",
)
SAMPLE_SIZES = {"small": 400, "medium": 2000, "large": 10000}


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("mass must be positive")
    return values / total


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(np.clip(logits, -700.0, 700.0))
    return values / values.sum(axis=1, keepdims=True)


def cost_from_features(phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return -np.einsum("ijf,f->ij", np.asarray(phi, dtype=float), np.asarray(theta, dtype=float))


def soft_uot_plan(
    phi: np.ndarray,
    theta: np.ndarray,
    source: np.ndarray,
    target_reference: np.ndarray,
    *,
    epsilon: float = 1.0,
    mu: float = 0.5,
    iterations: int = 3000,
    tolerance: float = 1e-12,
) -> np.ndarray:
    source = _normalize(source)
    target_reference = _normalize(target_reference)
    cost = cost_from_features(phi, theta)
    column = target_reference.copy()
    for _ in range(iterations):
        logits = -cost / epsilon + (mu / epsilon) * np.log(
            np.maximum(target_reference[None, :], 1e-300) / np.maximum(column[None, :], 1e-300)
        )
        conditional = _softmax_rows(logits)
        updated = conditional.T @ source
        if float(np.max(np.abs(updated - column))) <= tolerance:
            column = updated
            break
        column = 0.5 * column + 0.5 * updated
        column = _normalize(column)
    logits = -cost / epsilon + (mu / epsilon) * np.log(
        np.maximum(target_reference[None, :], 1e-300) / np.maximum(column[None, :], 1e-300)
    )
    return source[:, None] * _softmax_rows(logits)


def hard_ot_plan(
    phi: np.ndarray,
    theta: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    *,
    epsilon: float = 1.0,
    iterations: int = 1000,
    tolerance: float = 1e-11,
) -> np.ndarray:
    source = _normalize(source)
    target = _normalize(target)
    kernel = np.exp(np.clip(-cost_from_features(phi, theta) / epsilon, -700.0, 700.0))
    kernel = np.maximum(kernel, np.finfo(float).tiny)
    left = np.ones_like(source)
    right = np.ones_like(target)
    for iteration in range(iterations):
        previous_left = left
        previous_right = right
        left = source / np.maximum(kernel @ right, np.finfo(float).tiny)
        right = target / np.maximum(kernel.T @ left, np.finfo(float).tiny)
        if iteration % 10 == 0 and max(
            float(np.max(np.abs(left - previous_left))),
            float(np.max(np.abs(right - previous_right))),
        ) <= tolerance:
            break
    return (left[:, None] * kernel) * right[None, :]


def make_feature_tensor(rng: np.random.Generator, states: int = 7, features: int = 6) -> tuple[np.ndarray, list[str]]:
    if features < 4:
        raise ValueError("at least four features are required")
    phi = np.zeros((states, states, features), dtype=float)
    phi[:, :, 0] = rng.normal(size=(states, states))
    source_axis = rng.normal(size=states)
    target_axis = rng.normal(size=states)
    phi[:, :, 1] = source_axis[:, None] * target_axis[None, :]
    for feature in range(2, features):
        column = rng.normal(size=states)
        column -= column.mean()
        column /= max(column.std(), 1e-12)
        phi[:, :, feature] = column[None, :]
    flat = phi.reshape(-1, features)
    phi = ((flat - flat.mean(axis=0)) / np.maximum(flat.std(axis=0), 1e-12)).reshape(phi.shape)
    names = ["interaction_random", "interaction_bilinear"] + [f"pure_column_{i}" for i in range(features - 2)]
    return phi, names


@dataclass
class Observation:
    source: np.ndarray
    target_reference: np.ndarray
    observed_plan: np.ndarray
    noiseless_plan: np.ndarray
    context: int


def _sample_plan(plan: np.ndarray, count: int, rng: np.random.Generator, dropout: float = 0.0) -> np.ndarray:
    probabilities = _normalize(plan.reshape(-1))
    counts = rng.multinomial(count, probabilities).reshape(plan.shape).astype(float)
    if dropout > 0:
        mask = rng.uniform(size=counts.shape) >= dropout
        counts *= mask
    if counts.sum() <= 0:
        counts.flat[int(np.argmax(probabilities))] = 1.0
    return counts / counts.sum()


def generate_case(
    scenario: str,
    sample_count: int,
    seed: int,
    *,
    observations: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[Observation], list[Observation], list[str]]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    rng = np.random.default_rng(seed)
    phi, names = make_feature_tensor(rng)
    theta = rng.normal(0.0, 0.8, size=phi.shape[-1])
    theta[2:] += np.array([1.1, -0.9, 0.7, -0.6])
    if scenario == "equivalent_fit_multiple_parameters":
        # With only target-column features, the observed target marginal lets a
        # hard-marginal model reproduce the coupling while leaving every true
        # direction coefficient structurally unidentified.
        theta[:2] = 0.0
    theta /= np.linalg.norm(theta)
    context_offset = np.array([0.0, 0.0, 0.35, -0.25, 0.2, -0.15])

    train: list[Observation] = []
    heldout: list[Observation] = []
    for index in range(observations + 3):
        source = rng.dirichlet(np.ones(phi.shape[0]) * 2.0)
        target_reference = rng.dirichlet(np.ones(phi.shape[0]) * 2.0)
        context = index % 2
        active_theta = theta.copy()
        if scenario == "shared_plus_context_offset":
            active_theta = active_theta + context * context_offset
        plan = soft_uot_plan(phi, active_theta, source, target_reference)
        if scenario == "nonlinear_or_hidden_confounding_misspecification":
            hidden = np.sin(phi[:, :, 0] * 1.7) + 0.35 * phi[:, :, 1] ** 2
            logits = np.log(np.maximum(plan, 1e-300)) + 0.35 * hidden
            distorted = np.exp(logits - logits.max())
            plan = distorted / distorted.sum()
        dropout = 0.18 if scenario == "observation_noise_and_dropout" else 0.0
        observed = _sample_plan(plan, sample_count, rng, dropout=dropout)
        item = Observation(source, target_reference, observed, plan, context)
        (train if index < observations else heldout).append(item)
    return phi, theta, train, heldout, names


def _objective(theta: np.ndarray, phi: np.ndarray, observations: list[Observation], mode: str) -> float:
    loss = 0.0
    for observation in observations:
        if mode == "soft_iot":
            estimate = soft_uot_plan(phi, theta, observation.source, observation.target_reference)
        elif mode == "hard_ot":
            estimate = hard_ot_plan(phi, theta, observation.source, observation.observed_plan.sum(axis=0))
        else:
            raise ValueError(mode)
        target = observation.observed_plan
        loss += float(-np.sum(target * np.log(np.maximum(estimate, 1e-300))))
    return loss / len(observations)


def fit_inverse(
    phi: np.ndarray,
    observations: list[Observation],
    mode: str,
    seed: int,
    *,
    restarts: int = 4,
) -> tuple[np.ndarray, float, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(restarts):
        start = rng.normal(0.0, 0.25, size=phi.shape[-1])
        result = minimize(
            _objective,
            start,
            args=(phi, observations, mode),
            method="L-BFGS-B",
            options={"maxiter": 250, "ftol": 1e-12, "gtol": 1e-7},
        )
        fits.append((np.asarray(result.x, dtype=float), float(result.fun)))
    best = min(fits, key=lambda item: item[1])
    return best[0], best[1], [item[0] for item in fits]


def run_case(scenario: str, size_name: str, sample_count: int, seed: int) -> list[dict[str, object]]:
    phi, theta_true, train, heldout, feature_names = generate_case(scenario, sample_count, seed)
    rows: list[dict[str, object]] = []
    for method in ("soft_iot", "hard_ot"):
        started = time.perf_counter()
        theta_hat, train_loss, restarts = fit_inverse(phi, train, method, seed + 1000)
        runtime = time.perf_counter() - started
        recovery = coefficient_recovery(theta_hat, theta_true)
        restart_directions = np.asarray(restarts)
        norms = np.linalg.norm(restart_directions, axis=1, keepdims=True)
        normalized = restart_directions / np.maximum(norms, 1e-12)
        restart_variance = float(np.mean(np.var(normalized, axis=0, ddof=1)))
        curvatures = []
        for index in range(phi.shape[-1]):
            direction = np.zeros(phi.shape[-1])
            direction[index] = 1.0
            curvatures.append(
                directional_curvature(lambda value: _objective(value, phi, train, method), theta_hat, direction, step=1e-3)
            )
        held_errors = []
        for observation in heldout:
            if method == "soft_iot":
                estimate = soft_uot_plan(phi, theta_hat, observation.source, observation.target_reference)
            else:
                estimate = hard_ot_plan(phi, theta_hat, observation.source, observation.observed_plan.sum(axis=0))
            held_errors.append(coupling_relative_frobenius(estimate, observation.noiseless_plan))
        common = {
            "benchmark": "B1_known_truth",
            "scenario": scenario,
            "sample_size": size_name,
            "sample_count": sample_count,
            "seed": seed,
            "method": method,
            "runtime_seconds": runtime,
            "train_cross_entropy": train_loss,
            "heldout_coupling_relative_frobenius": float(np.mean(held_errors)),
            "minimum_curvature": float(np.min(curvatures)),
            "pure_column_minimum_curvature": float(np.min(curvatures[2:])),
            "restart_direction_variance": restart_variance,
            "theta_true": json.dumps(theta_true.tolist()),
            "theta_hat": json.dumps(theta_hat.tolist()),
            "feature_names": json.dumps(feature_names),
        }
        common.update(recovery)
        rows.append(common)
    return rows


def run_suite(output_dir: str | Path, *, smoke: bool = False) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = SCENARIOS[:2] if smoke else SCENARIOS
    sizes = {"small": SAMPLE_SIZES["small"]} if smoke else SAMPLE_SIZES
    seeds = [20260916] if smoke else [20260916, 20260917, 20260918, 20260919, 20260920]
    all_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for scenario in scenarios:
        for size_name, sample_count in sizes.items():
            for seed in seeds:
                all_rows.extend(run_case(scenario, size_name, sample_count, seed))
    frame = pd.DataFrame(all_rows)
    frame.to_csv(output_dir / "b1_all_runs.csv", index=False)
    summary = (
        frame.groupby(["scenario", "sample_size", "method"], as_index=False)
        .agg(
            heldout_coupling_mean=("heldout_coupling_relative_frobenius", "mean"),
            coefficient_cosine_mean=("coefficient_cosine", "mean"),
            coefficient_sign_accuracy_mean=("coefficient_sign_accuracy", "mean"),
            pure_column_minimum_curvature_mean=("pure_column_minimum_curvature", "mean"),
            restart_direction_variance_mean=("restart_direction_variance", "mean"),
            runtime_seconds=("runtime_seconds", "sum"),
        )
    )
    summary.to_csv(output_dir / "b1_summary.csv", index=False)
    manifest = {
        "benchmark": "B1_known_truth",
        "mode": "smoke" if smoke else "full",
        "scenarios": list(scenarios),
        "sample_sizes": sizes,
        "seeds": seeds,
        "rows": int(len(frame)),
        "elapsed_seconds": float(time.perf_counter() - started),
        "outputs": ["b1_all_runs.csv", "b1_summary.csv"],
    }
    (output_dir / "b1_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
