"""Stage 1 canonical interface: UOT-IOT.

This adapter deliberately delegates to the archived low-capacity IOT solver.
It supplies a stable ``fit``/``predict_state``/``explain`` interface while
keeping legacy configurations such as ``pure`` and ``balanced`` as metadata.
No historical implementation is deleted or rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .gh_backend import fit_iot_q, normalize_rows


@dataclass
class UOTIOTModel:
    """Canonical Stage 1 model backed by the archived ``fit_iot_q`` solver.

    Parameters
    ----------
    variant:
        A legacy configuration label.  The canonical default is ``"uot"``;
        labels such as ``"pure"`` or ``"balanced"`` are retained for audit
        and do not silently change the archived solver.
    epsilon:
        Entropic regularisation passed to the historical Sinkhorn solver.
    """

    variant: str = "uot"
    epsilon: float = 0.5
    q_: Optional[np.ndarray] = None
    fit_metadata_: Dict[str, Any] = field(default_factory=dict)

    model_name: str = "UOT-IOT"
    stage: int = 1

    def fit(
        self,
        source_counts: np.ndarray,
        target_counts: np.ndarray,
        cost: np.ndarray,
    ) -> "UOTIOTModel":
        """Fit a training-fold IOT transition matrix."""

        q, metadata = fit_iot_q(
            np.asarray(source_counts, dtype=float),
            np.asarray(target_counts, dtype=float),
            np.asarray(cost, dtype=float),
            eps=float(self.epsilon),
        )
        self.q_ = normalize_rows(q)
        self.fit_metadata_ = {
            "variant": str(self.variant),
            "epsilon": float(self.epsilon),
            **metadata,
        }
        return self

    def predict_state(
        self,
        source_counts: np.ndarray,
        q: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict the next state composition from source composition."""

        transition = self.q_ if q is None else np.asarray(q, dtype=float)
        if transition is None:
            raise RuntimeError("UOT-IOT 尚未拟合")
        source = normalize_rows(np.asarray(source_counts, dtype=float))
        return normalize_rows(source @ normalize_rows(transition))

    def predict(self, source_counts: np.ndarray) -> np.ndarray:
        """Alias used by generic three-stage callers."""

        return self.predict_state(source_counts)

    def explain(self) -> Dict[str, Any]:
        """Return a serialisable description of the fitted transport layer."""

        if self.q_ is None:
            raise RuntimeError("UOT-IOT 尚未拟合")
        return {
            "model": self.model_name,
            "stage": self.stage,
            "variant": str(self.variant),
            "transition": self.q_.tolist(),
            "fit_metadata": dict(self.fit_metadata_),
        }
