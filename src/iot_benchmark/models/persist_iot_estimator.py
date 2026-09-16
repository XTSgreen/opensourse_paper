"""Fold-local PERSIST-IOT estimator built from prospective components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .persist_features import FeatureContract, feature_matrix, source_state_matrix, target_state_matrix
from .persist_hmm import ObservationCorrectedPersistenceHMM
from .prospective_iot import ProspectiveIOTGenerator


@dataclass
class RobustFeatureScaler:
    median: np.ndarray | None = None
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "RobustFeatureScaler":
        values = np.asarray(values, dtype=float)
        median = np.nanmedian(values, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isfinite(values), values, median[None, :])
        mean = filled.mean(axis=0)
        scale = filled.std(axis=0)
        self.median = median
        self.mean = mean
        self.scale = np.where(scale > 1e-8, scale, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.median is None or self.mean is None or self.scale is None:
            raise RuntimeError("特征标准化器尚未拟合")
        values = np.asarray(values, dtype=float)
        filled = np.where(np.isfinite(values), values, self.median[None, :])
        return (filled - self.mean[None, :]) / self.scale[None, :]


class PersistIOTEstimator:
    """Continuous-time IOT features plus an observation-corrected HMM.

    Every `fit` call is fold-local.  `predict` forms its feature tensors from
    source snapshots and known horizons only; it deliberately ignores all
    columns beginning with `target_`, including target count and state labels.
    """

    def __init__(
        self,
        feature_contract: FeatureContract,
        *,
        use_iot: bool = True,
        state_count: int = 6,
        l2: float = 1e-3,
        planned_depth: float = 1000.0,
        hmm_epochs: int = 350,
        iot_epochs: int = 180,
    ) -> None:
        self.feature_contract = feature_contract
        self.use_iot = bool(use_iot)
        self.state_count = int(state_count)
        self.l2 = float(l2)
        self.planned_depth = float(planned_depth)
        self.hmm_epochs = int(hmm_epochs)
        self.iot_epochs = int(iot_epochs)
        self.scaler = RobustFeatureScaler()
        self.iot_by_condition: dict[str, ProspectiveIOTGenerator] = {}
        self.hmm: ObservationCorrectedPersistenceHMM | None = None
        self.max_horizon_days: float = 0.0
        self.fit_metadata: dict[str, Any] = {}

    def _fit_iot(self, cases: pd.DataFrame) -> None:
        if not self.use_iot:
            return
        source_state = source_state_matrix(cases)
        target_state = target_state_matrix(cases)
        valid_state = (
            cases["target_observed_count"].to_numpy(dtype=float) > 0
        ) & np.isfinite(source_state).all(axis=1) & np.isfinite(target_state).all(axis=1)
        for condition in sorted(cases["condition_id"].astype(str).unique()):
            mask = (cases["condition_id"].astype(str).to_numpy() == condition) & valid_state
            if mask.sum() < max(self.state_count * 3, 24):
                continue
            model = ProspectiveIOTGenerator(self.state_count, epsilon=0.5, l2=self.l2, epochs=self.iot_epochs)
            model.fit(
                source_state[mask],
                target_state[mask],
                cases.loc[mask, "horizon_days"].to_numpy(dtype=float),
                source_counts=cases.loc[mask, "source_observed_count"].to_numpy(dtype=float),
                target_counts=cases.loc[mask, "target_observed_count"].to_numpy(dtype=float),
            )
            self.iot_by_condition[condition] = model

    def _case_tensor(self, cases: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return per-case model features, IOT state prediction and reliability."""
        source_base = feature_matrix(cases, self.feature_contract, include_horizon=False)
        base = self.scaler.transform(source_base)
        horizon = np.log1p(cases["horizon_days"].to_numpy(dtype=float))[:, None]
        source_state = source_state_matrix(cases)
        state_prediction = source_state.copy()
        reliability = np.zeros(cases.shape[0], dtype=float)
        iot_features = np.zeros((cases.shape[0], self.state_count + 4), dtype=float)
        if self.use_iot:
            condition_values = cases["condition_id"].astype(str).to_numpy()
            horizon_values = cases["horizon_days"].to_numpy(dtype=float)
            for condition, model in self.iot_by_condition.items():
                indices = np.flatnonzero(condition_values == condition)
                predicted_state, diagnostics = model.predict_batch(source_state[indices], horizon_values[indices])
                state_prediction[indices] = predicted_state
                reliability[indices] = diagnostics["iot_reliability"]
                iot_features[indices, : self.state_count] = predicted_state
                iot_features[indices, self.state_count :] = np.column_stack(
                    [
                        diagnostics["iot_transition_entropy"],
                        diagnostics["iot_expected_cost"],
                        diagnostics["iot_reliability"],
                        np.log1p(np.maximum(diagnostics["uot_relative_mass_multiplier"], 0.0)),
                    ]
                )
        return np.column_stack([base, horizon, iot_features]) if self.use_iot else np.column_stack([base, horizon]), state_prediction, reliability

    @staticmethod
    def _snapshot_ids(cases: pd.DataFrame) -> pd.Series:
        return (
            cases["dataset_id"].astype(str)
            + "|"
            + cases["replicate_id"].astype(str)
            + "|"
            + cases["condition_id"].astype(str)
            + "|"
            + cases["lineage_id"].astype(str)
            + "|"
            + cases["source_time"].astype(str)
        )

    def _assemble_sequences(
        self,
        cases: pd.DataFrame,
        case_features: np.ndarray,
        *,
        include_observed_counts: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
        identifiers = self._snapshot_ids(cases)
        groups: list[np.ndarray] = [np.flatnonzero(identifiers.to_numpy() == snapshot) for snapshot in identifiers.drop_duplicates()]
        max_horizons = max(len(group) for group in groups)
        n_features = case_features.shape[1]
        features = np.zeros((len(groups), max_horizons, n_features), dtype=float)
        counts = np.full((len(groups), max_horizons), np.nan, dtype=float)
        intervals = np.zeros((len(groups), max_horizons), dtype=float)
        depths = np.full((len(groups), max_horizons), self.planned_depth, dtype=float)
        mapping: list[tuple[int, int]] = [(-1, -1) for _ in range(cases.shape[0])]
        for sequence_index, case_indices in enumerate(groups):
            ordered_indices = case_indices[np.argsort(cases.iloc[case_indices]["horizon_days"].to_numpy(dtype=float))]
            previous_horizon = 0.0
            for horizon_index, case_index in enumerate(ordered_indices):
                horizon = float(cases.iloc[case_index]["horizon_days"])
                features[sequence_index, horizon_index] = case_features[case_index]
                intervals[sequence_index, horizon_index] = horizon - previous_horizon
                previous_horizon = horizon
                if include_observed_counts:
                    counts[sequence_index, horizon_index] = float(cases.iloc[case_index]["target_observed_count"])
                mapping[case_index] = (sequence_index, horizon_index)
        return features, counts, intervals, depths, mapping

    def fit(self, cases: pd.DataFrame) -> "PersistIOTEstimator":
        self.feature_contract.assert_safe()
        self.max_horizon_days = float(cases["horizon_days"].max())
        self.scaler.fit(feature_matrix(cases, self.feature_contract, include_horizon=False))
        self._fit_iot(cases)
        case_features, _, _ = self._case_tensor(cases)
        features, counts, intervals, depths, _ = self._assemble_sequences(cases, case_features, include_observed_counts=True)
        self.hmm = ObservationCorrectedPersistenceHMM(case_features.shape[1], l2=self.l2)
        history = self.hmm.fit_model(features, counts, intervals, depths, max_epochs=self.hmm_epochs)
        self.fit_metadata = {
            "train_cases": int(cases.shape[0]),
            "train_source_snapshots": int(features.shape[0]),
            "feature_count": int(case_features.shape[1]),
            "iot_conditions_fitted": sorted(self.iot_by_condition),
            "hmm_epochs_run": len(history),
            "iot_epochs": self.iot_epochs,
            "hmm_final_loss": float(history[-1]),
            "planned_depth": self.planned_depth,
        }
        return self

    def predict(self, cases: pd.DataFrame) -> pd.DataFrame:
        if self.hmm is None:
            raise RuntimeError("PERSIST-IOT 尚未拟合")
        case_features, state_prediction, reliability = self._case_tensor(cases)
        features, _, intervals, depths, mapping = self._assemble_sequences(cases, case_features, include_observed_counts=False)
        predictions = self.hmm.predict(
            torch_as_float64(features),
            torch_as_float64(intervals),
            torch_as_float64(depths),
        )
        output = cases.loc[:, ["case_id", "dataset_id", "replicate_id", "condition_id", "lineage_id", "source_time", "target_time", "horizon_days"]].copy()
        row_indices = np.arange(cases.shape[0])
        sequence_index = np.asarray([mapping[index][0] for index in row_indices], dtype=int)
        horizon_index = np.asarray([mapping[index][1] for index in row_indices], dtype=int)
        output["p_persist"] = predictions.p_persist[sequence_index, horizon_index]
        output["p_detect"] = predictions.p_detect[sequence_index, horizon_index]
        output["abundance_q10"] = predictions.abundance_q10[sequence_index, horizon_index]
        output["abundance_q50"] = predictions.abundance_q50[sequence_index, horizon_index]
        output["abundance_q90"] = predictions.abundance_q90[sequence_index, horizon_index]
        output["iot_reliability"] = reliability
        for state_index in range(state_prediction.shape[1]):
            output[f"state_prob_{state_index}"] = state_prediction[:, state_index]
        source_size = cases.get("feature__source_ct_cells", cases["source_observed_count"]).to_numpy(dtype=float)
        reasons = []
        for size, horizon, reliability_value in zip(source_size, output["horizon_days"], reliability):
            codes = []
            if size < 2:
                codes.append("too_few_source_cells")
            if horizon > self.max_horizon_days:
                codes.append("horizon_out_of_range")
            if self.use_iot and reliability_value < 0.5:
                codes.append("iot_ill_conditioned")
            reasons.append("|".join(codes))
        output["reason_codes"] = reasons
        output["abstain"] = output["reason_codes"].str.len().gt(0).astype(int)
        return output


def torch_as_float64(values: np.ndarray):
    """Keep torch import local so panel building remains light-weight."""
    import torch

    return torch.as_tensor(values, dtype=torch.float64)
