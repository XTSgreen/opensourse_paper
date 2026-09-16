"""Stable machine-readable contracts for all benchmark outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np


RunStatus = Literal["success", "failed", "not_applicable", "timeout"]


@dataclass(frozen=True)
class BenchmarkRow:
    dataset: str
    benchmark: str
    task: str
    split: str
    repeat: int
    seed: int
    method: str
    method_version: str
    metric: str
    value: float | None
    status: RunStatus
    runtime_seconds: float
    peak_memory_gb: float | None = None
    failure_reason: str | None = None
    information_mode: str = "retrospective_reconstruction"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.repeat < 0:
            raise ValueError("repeat must be non-negative")
        if self.runtime_seconds < 0:
            raise ValueError("runtime_seconds must be non-negative")
        if self.status == "success":
            if self.value is None or not np.isfinite(self.value):
                raise ValueError("successful benchmark rows require a finite value")
            if self.failure_reason:
                raise ValueError("successful benchmark rows cannot have a failure reason")
        elif not self.failure_reason:
            raise ValueError("non-success benchmark rows require a failure reason")
        if self.information_mode not in {
            "retrospective_reconstruction",
            "prospective_prediction",
            "oracle_upper_bound",
        }:
            raise ValueError("unknown information mode")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionArtifact:
    sample_ids: tuple[str, ...]
    state_names: tuple[str, ...]
    probabilities: np.ndarray
    information_mode: str

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=float)
        if probabilities.shape != (len(self.sample_ids), len(self.state_names)):
            raise ValueError("probability shape does not match sample and state labels")
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError("probabilities must be finite and non-negative")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("each probability row must sum to one")


@dataclass(frozen=True)
class DirectionArtifact:
    feature_names: tuple[str, ...]
    direction: np.ndarray
    magnitude: float
    gauge: str
    source: Literal["native_parameter", "perturbation_sensitivity", "posthoc_probe"]

    def __post_init__(self) -> None:
        direction = np.asarray(self.direction, dtype=float)
        if direction.shape != (len(self.feature_names),):
            raise ValueError("direction length does not match feature names")
        if np.any(~np.isfinite(direction)):
            raise ValueError("direction must be finite")
        norm = float(np.linalg.norm(direction))
        if not np.isclose(norm, 1.0, atol=1e-8):
            raise ValueError("gauge-fixed direction must have unit L2 norm")
        if not np.isfinite(self.magnitude) or self.magnitude < 0:
            raise ValueError("magnitude must be finite and non-negative")


def gauge_fix_direction(values: np.ndarray, *, center: bool = True) -> tuple[np.ndarray, float]:
    direction = np.asarray(values, dtype=float).reshape(-1)
    if np.any(~np.isfinite(direction)):
        raise ValueError("direction contains non-finite values")
    if center:
        direction = direction - direction.mean()
    magnitude = float(np.linalg.norm(direction))
    if magnitude <= 0:
        raise ValueError("direction is zero after gauge centering")
    return direction / magnitude, magnitude
