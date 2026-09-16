"""Stage 2 canonical interface: Gated Hierarchical IOT (GH-IOT).

The archived Gated-IOT implementation remains the numerical backend.  This
adapter exposes adaptive-gate and global-gate configurations under one formal
model name and records the legacy name for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .gh_backend import GatedIOTModel, normalize_rows


@dataclass
class GHIOTModel:
    """Unified Stage 2 model for contextual and state-composition prediction.

    ``gate_mode="adaptive"`` is the historical Gated-IOT-DM configuration.
    ``gate_mode="global"`` is the historical Gated-IOT-global-alpha
    configuration.  ``loss_mode="square"`` is retained only for ablation and
    is never treated as a separate formal model.
    """

    gate_mode: str = "adaptive"
    loss_mode: str = "cross_entropy"
    lam: float = 3.0
    gate_l2: float = 1.0
    backend_: Optional[GatedIOTModel] = None
    global_alpha_: Optional[float] = None
    fit_metadata_: Dict[str, Any] = field(default_factory=dict)
    cost_: Optional[np.ndarray] = None

    model_name: str = "GH-IOT"
    stage: int = 2

    def __post_init__(self) -> None:
        if self.gate_mode not in {"adaptive", "global"}:
            raise ValueError("gate_mode must be 'adaptive' or 'global'")
        if self.loss_mode not in {"cross_entropy", "square"}:
            raise ValueError("loss_mode must be 'cross_entropy' or 'square'")

    def fit(
        self,
        x: np.ndarray,
        source_counts: np.ndarray,
        target_counts: np.ndarray,
        q: np.ndarray,
        cost: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> "GHIOTModel":
        """Fit a fold-local contextual IOT state predictor."""

        self.backend_ = GatedIOTModel(
            lam=float(self.lam),
            gate_l2=float(self.gate_l2),
            loss_mode=str(self.loss_mode),
        ).fit(
            np.asarray(x, dtype=float),
            np.asarray(source_counts, dtype=float),
            np.asarray(target_counts, dtype=float),
            np.asarray(q, dtype=float),
            np.asarray(cost, dtype=float),
            weights,
        )
        self.cost_ = np.asarray(cost, dtype=float)
        _, alpha_train = self.backend_.predict(
            np.asarray(x, dtype=float), np.asarray(q, dtype=float), self.cost_
        )
        self.global_alpha_ = float(np.mean(alpha_train))
        self.fit_metadata_ = {
            "gate_mode": str(self.gate_mode),
            "loss_mode": str(self.loss_mode),
            "lam": float(self.lam),
            "gate_l2": float(self.gate_l2),
            "global_alpha": float(self.global_alpha_),
            "legacy_name": (
                "Gated-IOT-DM"
                if self.gate_mode == "adaptive" and self.loss_mode == "cross_entropy"
                else "Gated-IOT-global-alpha"
                if self.gate_mode == "global" and self.loss_mode == "cross_entropy"
                else "Gated-IOT-square-loss"
            ),
        }
        return self

    def predict_state(
        self,
        x: np.ndarray,
        q: np.ndarray,
        cost: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return state composition and the gate used for each case."""

        if self.backend_ is None or self.global_alpha_ is None:
            raise RuntimeError("GH-IOT 尚未拟合")
        cost_value = self.cost_ if cost is None else np.asarray(cost, dtype=float)
        q_value = normalize_rows(np.asarray(q, dtype=float))
        x_value = np.asarray(x, dtype=float)
        if self.gate_mode == "adaptive":
            prediction, alpha = self.backend_.predict(x_value, q_value, cost_value)
            return normalize_rows(prediction), np.asarray(alpha, dtype=float)

        uniform = np.full_like(q_value, 1.0 / q_value.shape[1])
        residual = self.backend_.supervised_.predict(x_value, uniform)
        prediction = self.global_alpha_ * q_value + (1.0 - self.global_alpha_) * residual
        return normalize_rows(prediction), np.full(x_value.shape[0], self.global_alpha_)

    def predict(
        self,
        x: np.ndarray,
        q: np.ndarray,
        cost: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generic alias for ``predict_state``."""

        return self.predict_state(x, q, cost)

    def explain(self) -> Dict[str, Any]:
        """Return the gate configuration and fitted backend metadata."""

        if self.backend_ is None:
            raise RuntimeError("GH-IOT 尚未拟合")
        return {
            "model": self.model_name,
            "stage": self.stage,
            "fit_metadata": dict(self.fit_metadata_),
            "kappa": float(self.backend_.kappa_),
        }
