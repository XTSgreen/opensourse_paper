import numpy as np
import pytest

from iot_benchmark.schemas import BenchmarkRow, DirectionArtifact, PredictionArtifact, gauge_fix_direction


def test_benchmark_row_enforces_failure_reason_and_finite_success():
    row = BenchmarkRow(
        dataset="synthetic",
        benchmark="B1_known_truth",
        task="coupling",
        split="seed0",
        repeat=0,
        seed=1,
        method="UOT-IOT",
        method_version="0.1",
        metric="coupling_relative_frobenius",
        value=0.1,
        status="success",
        runtime_seconds=0.5,
    )
    assert row.to_dict()["value"] == 0.1
    with pytest.raises(ValueError):
        BenchmarkRow(
            dataset="synthetic", benchmark="B1_known_truth", task="coupling", split="seed0",
            repeat=0, seed=1, method="bad", method_version="0", metric="loss", value=None,
            status="failed", runtime_seconds=0.1,
        )


def test_prediction_and_direction_contracts():
    PredictionArtifact(
        sample_ids=("a", "b"),
        state_names=("s1", "s2"),
        probabilities=np.array([[0.7, 0.3], [0.1, 0.9]]),
        information_mode="prospective_prediction",
    )
    direction, magnitude = gauge_fix_direction(np.array([1.0, 2.0, 4.0]))
    artifact = DirectionArtifact(
        feature_names=("a", "b", "c"), direction=direction, magnitude=magnitude,
        gauge="center_then_l2", source="native_parameter",
    )
    assert np.isclose(np.linalg.norm(artifact.direction), 1.0)
