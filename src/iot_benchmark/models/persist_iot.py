"""Stage 3 canonical interface: PERSIST-IOT.

The persistence implementation has a deliberately separate code tree because
it owns the observation-corrected HMM and prospective IOT generator.  This
wrapper lazily loads that implementation so importing the three-stage registry
does not require the heavy longitudinal dependencies.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from .persist_iot_estimator import PersistIOTEstimator


class PersistIOTModel:
    """Canonical Stage 3 wrapper around the archived PERSIST-IOT estimator."""

    model_name = "PERSIST-IOT"
    stage = 3

    def __init__(
        self,
        feature_contract: Any,
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
        self.backend_: Optional[Any] = None

    def fit(self, cases: pd.DataFrame) -> "PersistIOTModel":
        self.backend_ = PersistIOTEstimator(
            self.feature_contract,
            use_iot=self.use_iot,
            state_count=self.state_count,
            l2=self.l2,
            planned_depth=self.planned_depth,
            hmm_epochs=self.hmm_epochs,
            iot_epochs=self.iot_epochs,
        ).fit(cases)
        return self

    def predict(self, cases: pd.DataFrame) -> pd.DataFrame:
        if self.backend_ is None:
            raise RuntimeError("PERSIST-IOT 尚未拟合")
        return self.backend_.predict(cases)

    def explain(self) -> dict[str, Any]:
        if self.backend_ is None:
            raise RuntimeError("PERSIST-IOT 尚未拟合")
        return {
            "model": self.model_name,
            "stage": self.stage,
            "use_iot": self.use_iot,
            "fit_metadata": dict(self.backend_.fit_metadata),
            "outputs": ["p_persist", "p_detect", "state_prob_*"],
        }
