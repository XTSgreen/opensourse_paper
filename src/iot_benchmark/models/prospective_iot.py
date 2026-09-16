"""Prospective continuous-time IOT/UOT state generator.

This implementation is intentionally restricted to train-history transitions.
During prediction it receives a source clone state distribution, a known future
interval and a treatment condition already seen in training.  It never accepts
a target cohort marginal, target state centres or target clone graph.  The
target marginal used by the entropic coupling is generated from the fitted
continuous-time training generator itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class SinkhornDiagnostics:
    marginal_residual: float
    minimum_plan_entry: float
    cost_rank: int
    cost_condition_number: float
    gauge_row_mean_abs_max: float
    gauge_column_mean_abs_max: float
    finite: bool


def gauge_project_cost(cost: np.ndarray) -> np.ndarray:
    """Remove additive row/column ambiguity from an OT cost matrix."""
    matrix = np.asarray(cost, dtype=np.float64)
    return matrix - matrix.mean(axis=0, keepdims=True) - matrix.mean(axis=1, keepdims=True) + matrix.mean()


def log_sinkhorn(
    source_marginal: np.ndarray,
    target_marginal: np.ndarray,
    cost: np.ndarray,
    *,
    epsilon: float = 0.5,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, SinkhornDiagnostics]:
    """Stable balanced Sinkhorn plan and diagnostics for a fixed cost.

    The input marginals are clipped only by a numerical floor and normalized;
    this floor is recorded through the plan diagnostics rather than silently
    changing the scientific interpretation of a zero-probability state.
    """
    if epsilon <= 0:
        raise ValueError("epsilon 必须为正")
    source = np.asarray(source_marginal, dtype=np.float64).clip(1e-12)
    target = np.asarray(target_marginal, dtype=np.float64).clip(1e-12)
    source /= source.sum()
    target /= target.sum()
    projected_cost = gauge_project_cost(np.asarray(cost, dtype=np.float64))
    log_kernel = -projected_cost / epsilon
    log_source, log_target = np.log(source), np.log(target)
    log_u = np.zeros_like(source)
    log_v = np.zeros_like(target)
    for _ in range(max_iterations):
        old_u = log_u.copy()
        log_u = log_source - np.logaddexp.reduce(log_kernel + log_v[None, :], axis=1)
        log_v = log_target - np.logaddexp.reduce(log_kernel.T + log_u[None, :], axis=1)
        if np.max(np.abs(log_u - old_u)) < tolerance:
            break
    log_plan = log_u[:, None] + log_kernel + log_v[None, :]
    plan = np.exp(np.clip(log_plan, -745.0, 50.0))
    residual = max(float(np.abs(plan.sum(axis=1) - source).sum()), float(np.abs(plan.sum(axis=0) - target).sum()))
    singular_values = np.linalg.svd(projected_cost, compute_uv=False)
    positive_singular = singular_values[singular_values > 1e-12]
    condition = float(positive_singular.max() / positive_singular.min()) if positive_singular.size else float("inf")
    diagnostics = SinkhornDiagnostics(
        marginal_residual=residual,
        minimum_plan_entry=float(plan.min()),
        cost_rank=int(np.linalg.matrix_rank(projected_cost)),
        cost_condition_number=condition,
        gauge_row_mean_abs_max=float(np.abs(projected_cost.mean(axis=1)).max()),
        gauge_column_mean_abs_max=float(np.abs(projected_cost.mean(axis=0)).max()),
        finite=bool(np.isfinite(plan).all() and np.isfinite(projected_cost).all()),
    )
    if not diagnostics.finite:
        raise FloatingPointError("Sinkhorn 计划出现非有限值")
    return plan, diagnostics


def _build_generator(raw_off_diagonal: torch.Tensor, state_count: int) -> torch.Tensor:
    rates = torch.nn.functional.softplus(raw_off_diagonal)
    generator = torch.zeros((state_count, state_count), dtype=rates.dtype, device=rates.device)
    position = 0
    for row in range(state_count):
        for column in range(state_count):
            if row == column:
                continue
            generator[row, column] = rates[position]
            position += 1
        generator[row, row] = -generator[row].sum()
    return generator


class ProspectiveIOTGenerator:
    """Low-capacity continuous-time state generator with a Sinkhorn projection.

    ``fit`` is called inside a training fold.  The learned generator supplies a
    future marginal from training history; ``predict`` applies a row-conditional
    Sinkhorn transition to a clone's source state mix and reports numerical
    diagnostics.  An optional UOT mass trend is estimated separately from
    source/target lineage counts and is never used to redefine persistence.
    """

    def __init__(self, state_count: int, *, epsilon: float = 0.5, l2: float = 1e-3, epochs: int = 500) -> None:
        self.state_count = int(state_count)
        self.epsilon = float(epsilon)
        self.l2 = float(l2)
        self.epochs = int(epochs)
        self.generator_: np.ndarray | None = None
        self.training_source_marginal_: np.ndarray | None = None
        self.base_cost_: np.ndarray | None = None
        self.mass_intercept_: float = 0.0
        self.mass_slope_: float = 0.0
        self.fit_diagnostics_: dict[str, Any] = {}

    def fit(
        self,
        source_state_probs: np.ndarray,
        target_state_probs: np.ndarray,
        interval_days: np.ndarray,
        *,
        source_counts: np.ndarray | None = None,
        target_counts: np.ndarray | None = None,
    ) -> "ProspectiveIOTGenerator":
        source = np.asarray(source_state_probs, dtype=np.float64)
        target = np.asarray(target_state_probs, dtype=np.float64)
        interval = np.asarray(interval_days, dtype=np.float64).reshape(-1)
        if source.ndim != 2 or target.shape != source.shape or source.shape[1] != self.state_count or interval.shape[0] != source.shape[0]:
            raise ValueError("state 概率和时间间隔的形状不符合生成器合同")
        valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1) & np.isfinite(interval) & (interval > 0)
        if valid.sum() < self.state_count:
            raise ValueError("可用于连续时间 IOT 生成器的训练对过少")
        source = np.clip(source[valid], 1e-8, None)
        target = np.clip(target[valid], 1e-8, None)
        source /= source.sum(axis=1, keepdims=True)
        target /= target.sum(axis=1, keepdims=True)
        interval = interval[valid]
        torch.manual_seed(20260826)
        source_t = torch.as_tensor(source, dtype=torch.float64)
        target_t = torch.as_tensor(target, dtype=torch.float64)
        interval_t = torch.as_tensor(interval, dtype=torch.float64)
        raw = torch.nn.Parameter(torch.full((self.state_count * (self.state_count - 1),), -2.0, dtype=torch.float64))
        optimizer = torch.optim.Adam([raw], lr=0.05)
        best_loss = float("inf")
        best_raw: np.ndarray | None = None
        unique_intervals, inverse_interval = np.unique(interval, return_inverse=True)
        inverse_interval_t = torch.as_tensor(inverse_interval, dtype=torch.long)
        interval_values_t = torch.as_tensor(unique_intervals, dtype=torch.float64)
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            generator = _build_generator(raw, self.state_count)
            # Time points recur across many clone cases.  Computing one matrix
            # exponential per case would be both needlessly slow and prone to
            # implementation-dependent numerical variation.  Grouping only by
            # known interval retains exactly the same continuous-time model.
            transitions = torch.stack([torch.matrix_exp(generator * value) for value in interval_values_t])
            predictions = torch.bmm(source_t.unsqueeze(1), transitions[inverse_interval_t]).squeeze(1)
            predictions = predictions.clamp_min(1e-10)
            loss = -(target_t * torch.log(predictions)).sum(dim=1).mean() + self.l2 * raw.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("连续时间状态生成器出现非有限损失")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], max_norm=5.0)
            optimizer.step()
            current = float(loss.detach().cpu())
            if current < best_loss:
                best_loss = current
                best_raw = raw.detach().cpu().numpy().copy()
        if best_raw is None:
            raise RuntimeError("连续时间状态生成器没有有效参数")
        with torch.no_grad():
            generator = _build_generator(torch.as_tensor(best_raw, dtype=torch.float64), self.state_count).cpu().numpy()
        self.generator_ = generator
        self.training_source_marginal_ = source.mean(axis=0)
        # The reference cost is generated from the fitted state dynamics, not
        # from any test target marginal.  It is gauge-projected before each
        # Sinkhorn evaluation.
        reference = torch.matrix_exp(torch.as_tensor(generator, dtype=torch.float64)).cpu().numpy().clip(1e-12, 1.0)
        self.base_cost_ = -np.log(reference)
        if source_counts is not None and target_counts is not None:
            source_count = np.asarray(source_counts, dtype=float).reshape(-1)[valid]
            target_count = np.asarray(target_counts, dtype=float).reshape(-1)[valid]
            ratios = np.log1p(target_count) - np.log1p(source_count)
            design = np.column_stack([np.ones_like(interval), interval])
            coefficients, *_ = np.linalg.lstsq(design, ratios, rcond=None)
            self.mass_intercept_, self.mass_slope_ = map(float, coefficients)
        self.fit_diagnostics_ = {
            "train_pairs": int(source.shape[0]),
            "state_count": self.state_count,
            "cross_entropy_plus_l2": best_loss,
            "generator_row_sum_abs_max": float(np.abs(generator.sum(axis=1)).max()),
            "generator_max_off_diagonal": float(np.max(generator - np.diag(np.diag(generator)))),
            "training_source_marginal_min": float(self.training_source_marginal_.min()),
        }
        return self

    def _require_fit(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.generator_ is None or self.training_source_marginal_ is None or self.base_cost_ is None:
            raise RuntimeError("必须先在训练折调用 fit")
        return self.generator_, self.training_source_marginal_, self.base_cost_

    def predict(self, source_state_probs: np.ndarray, interval_days: float) -> tuple[np.ndarray, dict[str, float]]:
        generator, source_marginal, base_cost = self._require_fit()
        source = np.asarray(source_state_probs, dtype=np.float64).clip(1e-8)
        source /= source.sum()
        if not np.isfinite(interval_days) or interval_days < 0:
            raise ValueError("预测时间间隔必须为有限非负数")
        transition_reference = torch.matrix_exp(torch.as_tensor(generator * float(interval_days), dtype=torch.float64)).cpu().numpy()
        target_marginal_prior = source_marginal @ transition_reference
        target_marginal_prior = np.clip(target_marginal_prior, 1e-12, None)
        target_marginal_prior /= target_marginal_prior.sum()
        time_cost = -np.log(np.clip(transition_reference, 1e-12, 1.0))
        cost = 0.5 * base_cost + 0.5 * time_cost
        plan, diagnostics = log_sinkhorn(source_marginal, target_marginal_prior, cost, epsilon=self.epsilon)
        conditional = plan / source_marginal[:, None]
        predicted = source @ conditional
        predicted /= predicted.sum()
        entropy = float(-(predicted * np.log(np.maximum(predicted, 1e-12))).sum())
        expected_cost = float(np.sum((source[:, None] * conditional) * gauge_project_cost(cost)))
        result = {
            "iot_transition_entropy": entropy,
            "iot_expected_cost": expected_cost,
            "iot_sinkhorn_marginal_residual": diagnostics.marginal_residual,
            "iot_plan_minimum": diagnostics.minimum_plan_entry,
            "iot_cost_rank": float(diagnostics.cost_rank),
            "iot_cost_condition_number": diagnostics.cost_condition_number,
            "iot_gauge_row_mean_abs_max": diagnostics.gauge_row_mean_abs_max,
            "iot_gauge_column_mean_abs_max": diagnostics.gauge_column_mean_abs_max,
            "iot_reliability": float(
                diagnostics.finite
                and diagnostics.marginal_residual <= 1e-6
                and diagnostics.cost_condition_number < 1e8
            ),
            "uot_relative_mass_multiplier": float(np.exp(self.mass_intercept_ + self.mass_slope_ * float(interval_days))),
        }
        return predicted, result

    def predict_batch(
        self, source_state_probs: np.ndarray, interval_days: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Predict many clone cases without reading a target cohort marginal.

        The result is algebraically identical to calling :meth:`predict` for
        each row.  It evaluates the continuous-time generator and Sinkhorn
        plan once per distinct *known* horizon, which makes outer-fold and
        external evaluation feasible without changing the information set.
        """
        generator, source_marginal, base_cost = self._require_fit()
        source = np.asarray(source_state_probs, dtype=np.float64)
        horizons = np.asarray(interval_days, dtype=np.float64).reshape(-1)
        if source.ndim != 2 or source.shape[1] != self.state_count or source.shape[0] != horizons.size:
            raise ValueError("批量 IOT 预测的状态概率或时间间隔形状不正确")
        if not np.isfinite(source).all() or not np.isfinite(horizons).all() or np.any(horizons < 0):
            raise ValueError("批量 IOT 预测只能使用有限、非负的源状态和时间间隔")
        source = np.clip(source, 1e-8, None)
        source /= source.sum(axis=1, keepdims=True)
        output = np.empty_like(source)
        entropy = np.empty(horizons.size, dtype=float)
        expected_cost = np.empty(horizons.size, dtype=float)
        reliability = np.empty(horizons.size, dtype=float)
        mass_multiplier = np.empty(horizons.size, dtype=float)
        residual = np.empty(horizons.size, dtype=float)
        condition_number = np.empty(horizons.size, dtype=float)
        unique_horizons, inverse_horizon = np.unique(horizons, return_inverse=True)
        for horizon_index, horizon in enumerate(unique_horizons):
            indices = np.flatnonzero(inverse_horizon == horizon_index)
            transition_reference = torch.matrix_exp(
                torch.as_tensor(generator * float(horizon), dtype=torch.float64)
            ).cpu().numpy()
            target_marginal_prior = source_marginal @ transition_reference
            target_marginal_prior = np.clip(target_marginal_prior, 1e-12, None)
            target_marginal_prior /= target_marginal_prior.sum()
            time_cost = -np.log(np.clip(transition_reference, 1e-12, 1.0))
            cost = 0.5 * base_cost + 0.5 * time_cost
            plan, diagnostics = log_sinkhorn(source_marginal, target_marginal_prior, cost, epsilon=self.epsilon)
            conditional = plan / source_marginal[:, None]
            predicted = source[indices] @ conditional
            predicted /= predicted.sum(axis=1, keepdims=True)
            output[indices] = predicted
            entropy[indices] = -(predicted * np.log(np.maximum(predicted, 1e-12))).sum(axis=1)
            expected_cost[indices] = np.einsum("bi,ij,ij->b", source[indices], conditional, gauge_project_cost(cost))
            reliability_value = float(
                diagnostics.finite
                and diagnostics.marginal_residual <= 1e-6
                and diagnostics.cost_condition_number < 1e8
            )
            reliability[indices] = reliability_value
            mass_multiplier[indices] = np.exp(self.mass_intercept_ + self.mass_slope_ * float(horizon))
            residual[indices] = diagnostics.marginal_residual
            condition_number[indices] = diagnostics.cost_condition_number
        return output, {
            "iot_transition_entropy": entropy,
            "iot_expected_cost": expected_cost,
            "iot_reliability": reliability,
            "uot_relative_mass_multiplier": mass_multiplier,
            "iot_sinkhorn_marginal_residual": residual,
            "iot_cost_condition_number": condition_number,
        }
