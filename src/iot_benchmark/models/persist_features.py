"""Convert explicit-zero lineage panels into leakage-safe forecast cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


FORBIDDEN_FEATURE_FRAGMENTS = (
    "future_",
    "long_absent",
    "target_",
    "label",
    "outcome",
    "de_novo",
)
IDENTIFIER_COLUMNS = {
    "dataset_id",
    "replicate_id",
    "condition_id",
    "lineage_id",
    "lineage_instance_id",
    "time",
    "sample_ids",
    "observed_conditions",
    "first_observed_time",
}


@dataclass(frozen=True)
class FeatureContract:
    columns: tuple[str, ...]

    def assert_safe(self) -> None:
        unsafe = [column for column in self.columns if any(fragment in column.lower() for fragment in FORBIDDEN_FEATURE_FRAGMENTS)]
        if unsafe:
            raise ValueError("特征合同包含未来或标签字段: " + ", ".join(unsafe))


def select_source_features(panel: pd.DataFrame) -> FeatureContract:
    """Select only numeric variables observable at a prediction cutoff."""
    allowed: list[str] = []
    for column in panel.columns:
        lower = column.lower()
        if column in IDENTIFIER_COLUMNS or column in {"time_index", "time_days", "has_future_measurement", "detected"}:
            continue
        if any(fragment in lower for fragment in FORBIDDEN_FEATURE_FRAGMENTS):
            continue
        if lower.startswith("state_count_"):
            # State proportions carry composition but clone cell count is
            # represented once by observed_count, avoiding duplicated size.
            continue
        if pd.api.types.is_numeric_dtype(panel[column]):
            allowed.append(column)
    preferred = [
        "observed_count",
        "past_detected_count",
        "past_observed_count",
        "consecutive_prior_absences",
        "source_ct_cells",
        "source_gex_cells",
    ]
    unique = set(allowed)
    ordered = [column for column in preferred if column in unique] + sorted(unique.difference(preferred))
    contract = FeatureContract(tuple(ordered))
    contract.assert_safe()
    return contract


def build_forecast_cases(panel: pd.DataFrame, *, feature_contract: FeatureContract | None = None) -> tuple[pd.DataFrame, FeatureContract]:
    """Create one case for every source snapshot and later planned timepoint.

    The feature values are copied only from the source panel row.  The target
    row is used strictly for labels (`target_observed_count`,
    `target_detected`, `target_state_prob_*`) and is never part of the returned
    feature contract.
    """
    needed = {"dataset_id", "replicate_id", "condition_id", "lineage_id", "time_index", "time_days", "observed_count", "detected"}
    missing = sorted(needed.difference(panel.columns))
    if missing:
        raise ValueError("谱系面板缺少预测案例字段: " + ", ".join(missing))
    features = feature_contract or select_source_features(panel)
    features.assert_safe()
    state_columns = sorted([column for column in panel.columns if column.startswith("state_prob_")], key=lambda name: int(name.rsplit("_", 1)[-1]))
    grouping = ["dataset_id", "replicate_id", "condition_id", "lineage_id"]
    cases: list[dict] = []
    for group_id, trajectory in panel.groupby(grouping, sort=False):
        ordered = trajectory.sort_values("time_index")
        rows = list(ordered.itertuples(index=False))
        column_index = {column: index for index, column in enumerate(ordered.columns)}
        for source_position, source in enumerate(rows[:-1]):
            source_values = source._asdict()
            for target_position in range(source_position + 1, len(rows)):
                target = rows[target_position]._asdict()
                horizon_days = float(target["time_days"] - source_values["time_days"])
                if horizon_days <= 0:
                    raise ValueError("同一谱系轨迹的时间必须严格递增")
                record = {
                    "case_id": f"{group_id[0]}|{group_id[1]}|{group_id[2]}|{group_id[3]}|{source_values['time']}->{target['time']}",
                    "dataset_id": group_id[0],
                    "replicate_id": group_id[1],
                    "condition_id": group_id[2],
                    "lineage_id": group_id[3],
                    "source_time": source_values["time"],
                    "target_time": target["time"],
                    "source_time_index": int(source_values["time_index"]),
                    "target_time_index": int(target["time_index"]),
                    "horizon_days": horizon_days,
                    "source_observed_count": float(source_values["observed_count"]),
                    "target_observed_count": float(target["observed_count"]),
                    "target_detected": int(target["detected"]),
                    "target_sample_depth_observed_only": float(
                        target.get("sample_gex_cell_depth", target.get("sample_lineaged_cell_depth", np.nan))
                    ),
                }
                for column in features.columns:
                    record[f"feature__{column}"] = source_values[column]
                for state_column in state_columns:
                    record[f"source_{state_column}"] = source_values[state_column]
                    record[f"target_{state_column}"] = target[state_column]
                cases.append(record)
    output = pd.DataFrame(cases)
    if output.empty:
        raise ValueError("没有可构建的预测案例")
    # The checks below make the information boundary executable rather than a
    # convention: every model feature must originate from a source row.
    model_columns = [column.removeprefix("feature__") for column in output.columns if column.startswith("feature__")]
    if tuple(model_columns) != features.columns:
        raise AssertionError("案例特征列与冻结合同不一致")
    if any("target_" in column or "future_" in column for column in model_columns):
        raise AssertionError("案例模型特征泄漏未来字段")
    return output, features


def feature_matrix(cases: pd.DataFrame, feature_contract: FeatureContract, *, include_horizon: bool = True) -> np.ndarray:
    columns = [f"feature__{column}" for column in feature_contract.columns]
    frame = cases.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    values = frame.to_numpy(dtype=float)
    if include_horizon:
        values = np.column_stack([values, np.log1p(cases["horizon_days"].to_numpy(dtype=float))])
    return values


def source_state_matrix(cases: pd.DataFrame) -> np.ndarray:
    columns = sorted([column for column in cases.columns if column.startswith("source_state_prob_")], key=lambda name: int(name.rsplit("_", 1)[-1]))
    if not columns:
        raise ValueError("预测案例缺少源状态概率")
    values = cases.loc[:, columns].to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0)
    sums = values.sum(axis=1, keepdims=True)
    return np.divide(values, sums, out=np.full_like(values, 1.0 / values.shape[1]), where=sums > 0)


def target_state_matrix(cases: pd.DataFrame) -> np.ndarray:
    columns = sorted([column for column in cases.columns if column.startswith("target_state_prob_")], key=lambda name: int(name.rsplit("_", 1)[-1]))
    if not columns:
        raise ValueError("预测案例缺少目标状态概率")
    values = cases.loc[:, columns].to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0)
    sums = values.sum(axis=1, keepdims=True)
    return np.divide(values, sums, out=np.full_like(values, 1.0 / values.shape[1]), where=sums > 0)
