"""Observation-corrected absorbing persistence model.

This module separates a latent absorbing persistence state from an observed
sequencing count.  It intentionally accepts only source/history features,
known time intervals and known planned sample depth.  It has no parameter or
method argument that accepts target labels, target population margins or a
target clone graph during prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from scipy.stats import nbinom


@dataclass(frozen=True)
class PersistencePredictions:
    p_persist: np.ndarray
    p_detect: np.ndarray
    abundance_q10: np.ndarray
    abundance_q50: np.ndarray
    abundance_q90: np.ndarray


def _negative_binomial_log_probability(count: torch.Tensor, mean: torch.Tensor, dispersion: torch.Tensor) -> torch.Tensor:
    """NB2 log PMF with mean ``mean`` and positive dispersion ``dispersion``."""
    mean = mean.clamp_min(1e-8)
    dispersion = dispersion.clamp_min(1e-5)
    probability = (dispersion / (dispersion + mean)).clamp(1e-8, 1.0 - 1e-8)
    return (
        torch.lgamma(count + dispersion)
        - torch.lgamma(dispersion)
        - torch.lgamma(count + 1.0)
        + dispersion * torch.log(probability)
        + count * torch.log1p(-probability)
    )


class ObservationCorrectedPersistenceHMM(torch.nn.Module):
    """Two-state forward model with an absorbing inactive state.

    ``A=1`` denotes latent persistence.  The active-to-active transition is
    ``exp(-softplus(rate) * delta_days)``, giving a non-increasing prior
    persistence curve at all horizons.  The active emission is zero-inflated
    negative-binomial, so a zero observation need not mean inactivation.
    """

    def __init__(self, n_features: int, *, l2: float = 1e-3, false_positive_rate: float = 1e-6) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.l2 = float(l2)
        self.false_positive_rate = float(false_positive_rate)
        self.hazard_weight = torch.nn.Parameter(torch.zeros(n_features))
        self.hazard_intercept = torch.nn.Parameter(torch.tensor(-2.0))
        self.abundance_weight = torch.nn.Parameter(torch.zeros(n_features))
        self.abundance_intercept = torch.nn.Parameter(torch.tensor(-1.0))
        self.dropout_intercept = torch.nn.Parameter(torch.tensor(-1.0))
        self.dropout_depth_weight = torch.nn.Parameter(torch.tensor(-0.2))
        self.log_dispersion = torch.nn.Parameter(torch.tensor(0.0))

    def _parameters_for_horizon(
        self, source_features: torch.Tensor, planned_depth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        linear_hazard = source_features @ self.hazard_weight + self.hazard_intercept
        hazard_rate = torch.nn.functional.softplus(linear_hazard).clamp_max(10.0)
        log_depth = torch.log1p(planned_depth.clamp_min(0.0))
        log_mean = source_features @ self.abundance_weight + self.abundance_intercept + log_depth
        mean = torch.exp(log_mean.clamp(-12.0, 12.0))
        dropout = torch.sigmoid(self.dropout_intercept + self.dropout_depth_weight * log_depth)
        dispersion = torch.nn.functional.softplus(self.log_dispersion) + 1e-4
        return hazard_rate, mean, dropout, dispersion

    def active_log_emission(self, observed_count: torch.Tensor, mean: torch.Tensor, dropout: torch.Tensor, dispersion: torch.Tensor) -> torch.Tensor:
        nb_log = _negative_binomial_log_probability(observed_count, mean, dispersion)
        is_zero = observed_count <= 0
        zero_probability = dropout + (1.0 - dropout) * torch.exp(_negative_binomial_log_probability(torch.zeros_like(mean), mean, dispersion))
        return torch.where(is_zero, torch.log(zero_probability.clamp_min(1e-12)), torch.log1p(-dropout).clamp_min(-30.0) + nb_log)

    def inactive_log_emission(self, observed_count: torch.Tensor) -> torch.Tensor:
        # The false-positive component is deliberately small and fixed.  It
        # represents barcode collision/background, not inferred biological
        # reactivation, which remains impossible in this absorbing model.
        rate = torch.as_tensor(self.false_positive_rate, dtype=observed_count.dtype, device=observed_count.device)
        return -rate + observed_count * torch.log(rate.clamp_min(1e-12)) - torch.lgamma(observed_count + 1.0)

    def negative_log_likelihood(
        self,
        source_features: torch.Tensor,
        future_counts: torch.Tensor,
        interval_days: torch.Tensor,
        planned_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Return average sequence NLL using the exact two-state forward pass.

        Shapes are ``[lineage, feature]`` and ``[lineage, horizon]``.  Missing
        future measurements must be encoded as NaN; they are skipped without
        providing their count or any target-derived summary to the model.
        """
        if future_counts.ndim != 2 or interval_days.shape != future_counts.shape or planned_depth.shape != future_counts.shape:
            raise ValueError("future_counts、interval_days 和 planned_depth 必须同形状 [lineage, horizon]")
        if source_features.ndim not in (2, 3) or source_features.shape[0] != future_counts.shape[0] or source_features.shape[-1] != self.n_features:
            raise ValueError("source_features 必须为 [lineage, feature] 或 [lineage, horizon, feature]")
        if source_features.ndim == 3 and source_features.shape[1] != future_counts.shape[1]:
            raise ValueError("三维 source_features 的 horizon 维度必须与 future_counts 一致")
        active_log_prob = torch.zeros(future_counts.shape[0], dtype=source_features.dtype, device=source_features.device)
        inactive_log_prob = torch.full_like(active_log_prob, -torch.inf)
        for horizon in range(future_counts.shape[1]):
            count = future_counts[:, horizon]
            observed = torch.isfinite(count)
            safe_count = torch.nan_to_num(count, nan=0.0)
            features_horizon = source_features if source_features.ndim == 2 else source_features[:, horizon, :]
            hazard_rate, mean, dropout, dispersion = self._parameters_for_horizon(features_horizon, planned_depth[:, horizon])
            active_stay = torch.exp(-hazard_rate * interval_days[:, horizon].clamp_min(0.0))
            transition_active = torch.log(active_stay.clamp(1e-12, 1.0))
            # Padded, unobserved horizons have interval 0.  Evaluating
            # ``log1p(-1)`` there produces an infinite intermediate whose
            # derivative can become NaN even though the horizon is masked.
            # Clipping the probability before the logarithm keeps the absent
            # transition numerically inert and leaves all positive intervals
            # unchanged to machine precision.
            transition_inactive = torch.log((1.0 - active_stay).clamp(1e-12, 1.0))
            next_active = active_log_prob + transition_active
            next_inactive = torch.logaddexp(inactive_log_prob, active_log_prob + transition_inactive)
            active_emission = self.active_log_emission(safe_count, mean, dropout, dispersion)
            inactive_emission = self.inactive_log_emission(safe_count)
            active_log_prob = torch.where(observed, next_active + active_emission, next_active)
            inactive_log_prob = torch.where(observed, next_inactive + inactive_emission, next_inactive)
            normalizer = torch.where(observed, torch.logaddexp(active_log_prob, inactive_log_prob), torch.zeros_like(active_log_prob))
            active_log_prob = active_log_prob - normalizer
            inactive_log_prob = inactive_log_prob - normalizer
        # The per-step normalizers are omitted after normalization above; run a
        # second unnormalised pass so the returned value is the actual log
        # likelihood rather than only a filtered posterior.  This avoids a
        # numerically unstable product of tiny probabilities.
        return self._unnormalised_sequence_nll(source_features, future_counts, interval_days, planned_depth)

    def _unnormalised_sequence_nll(
        self, source_features: torch.Tensor, future_counts: torch.Tensor, interval_days: torch.Tensor, planned_depth: torch.Tensor
    ) -> torch.Tensor:
        active_log_prob = torch.zeros(future_counts.shape[0], dtype=source_features.dtype, device=source_features.device)
        inactive_log_prob = torch.full_like(active_log_prob, -torch.inf)
        for horizon in range(future_counts.shape[1]):
            count = future_counts[:, horizon]
            observed = torch.isfinite(count)
            safe_count = torch.nan_to_num(count, nan=0.0)
            features_horizon = source_features if source_features.ndim == 2 else source_features[:, horizon, :]
            hazard_rate, mean, dropout, dispersion = self._parameters_for_horizon(features_horizon, planned_depth[:, horizon])
            active_stay = torch.exp(-hazard_rate * interval_days[:, horizon].clamp_min(0.0))
            next_active = active_log_prob + torch.log(active_stay.clamp(1e-12, 1.0))
            next_inactive = torch.logaddexp(
                inactive_log_prob,
                active_log_prob + torch.log((1.0 - active_stay).clamp(1e-12, 1.0)),
            )
            active_log_prob = torch.where(observed, next_active + self.active_log_emission(safe_count, mean, dropout, dispersion), next_active)
            inactive_log_prob = torch.where(observed, next_inactive + self.inactive_log_emission(safe_count), next_inactive)
        log_likelihood = torch.logaddexp(active_log_prob, inactive_log_prob)
        regularizer = self.l2 * (
            self.hazard_weight.square().sum()
            + self.abundance_weight.square().sum()
            + self.hazard_intercept.square()
            + self.abundance_intercept.square()
        )
        return -log_likelihood.mean() + regularizer

    @torch.no_grad()
    def predict(
        self, source_features: torch.Tensor, interval_days: torch.Tensor, planned_depth: torch.Tensor
    ) -> PersistencePredictions:
        """Produce unconditioned prospective predictions for future horizons."""
        if interval_days.ndim != 2 or planned_depth.shape != interval_days.shape:
            raise ValueError("interval_days 和 planned_depth 必须同形状 [lineage, horizon]")
        if source_features.ndim not in (2, 3) or source_features.shape[0] != interval_days.shape[0] or source_features.shape[-1] != self.n_features:
            raise ValueError("source_features 必须为 [lineage, feature] 或 [lineage, horizon, feature]")
        if source_features.ndim == 3 and source_features.shape[1] != interval_days.shape[1]:
            raise ValueError("三维 source_features 的 horizon 维度必须与 interval_days 一致")
        cumulative_active = torch.ones(interval_days.shape[0], dtype=source_features.dtype, device=source_features.device)
        persistence, detection, lower, median, upper = [], [], [], [], []
        for horizon in range(interval_days.shape[1]):
            features_horizon = source_features if source_features.ndim == 2 else source_features[:, horizon, :]
            hazard_rate, mean, dropout, dispersion = self._parameters_for_horizon(features_horizon, planned_depth[:, horizon])
            cumulative_active = cumulative_active * torch.exp(-hazard_rate * interval_days[:, horizon].clamp_min(0.0))
            nb_zero = torch.exp(_negative_binomial_log_probability(torch.zeros_like(mean), mean, dispersion))
            active_detection = (1.0 - (dropout + (1.0 - dropout) * nb_zero)).clamp(0.0, 1.0)
            inactive_detection = torch.full_like(active_detection, self.false_positive_rate)
            detection.append(cumulative_active * active_detection + (1.0 - cumulative_active) * inactive_detection)
            persistence.append(cumulative_active)
            mean_np = mean.detach().cpu().numpy()
            dispersion_np = float(dispersion.detach().cpu().numpy())
            probability = dispersion_np / (dispersion_np + mean_np)
            lower.append(torch.as_tensor(nbinom.ppf(0.10, dispersion_np, probability), device=mean.device, dtype=mean.dtype))
            median.append(torch.as_tensor(nbinom.ppf(0.50, dispersion_np, probability), device=mean.device, dtype=mean.dtype))
            upper.append(torch.as_tensor(nbinom.ppf(0.90, dispersion_np, probability), device=mean.device, dtype=mean.dtype))
        return PersistencePredictions(
            p_persist=torch.stack(persistence, dim=1).cpu().numpy(),
            p_detect=torch.stack(detection, dim=1).cpu().numpy(),
            abundance_q10=torch.stack(lower, dim=1).cpu().numpy(),
            abundance_q50=torch.stack(median, dim=1).cpu().numpy(),
            abundance_q90=torch.stack(upper, dim=1).cpu().numpy(),
        )

    def fit_model(
        self,
        source_features: np.ndarray,
        future_counts: np.ndarray,
        interval_days: np.ndarray,
        planned_depth: np.ndarray,
        *,
        learning_rate: float = 0.03,
        max_epochs: int = 1000,
        tolerance: float = 1e-7,
    ) -> list[float]:
        """Deterministic Adam fit used only on a training partition."""
        torch.manual_seed(20260826)
        features = torch.as_tensor(source_features, dtype=torch.float64)
        counts = torch.as_tensor(future_counts, dtype=torch.float64)
        intervals = torch.as_tensor(interval_days, dtype=torch.float64)
        depth = torch.as_tensor(planned_depth, dtype=torch.float64)
        self.to(dtype=torch.float64)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        history: list[float] = []
        best = float("inf")
        stale = 0
        for _ in range(max_epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = self.negative_log_likelihood(features, counts, intervals, depth)
            if not torch.isfinite(loss):
                raise FloatingPointError("HMM 似然出现非有限值")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
            optimizer.step()
            current = float(loss.detach().cpu())
            history.append(current)
            if best - current > tolerance:
                best = current
                stale = 0
            else:
                stale += 1
                if stale >= 80:
                    break
        return history


def make_interval_matrix(horizons_days: Iterable[float], n_lineages: int) -> np.ndarray:
    """Convert cumulative horizons to incremental intervals with validation."""
    horizons = np.asarray(list(horizons_days), dtype=float)
    if horizons.ndim != 1 or horizons.size == 0 or np.any(np.diff(horizons) < 0) or np.any(horizons < 0):
        raise ValueError("horizons_days 必须是非负、递增的一维数组")
    increments = np.diff(np.concatenate(([0.0], horizons)))
    return np.repeat(increments[None, :], n_lineages, axis=0)
