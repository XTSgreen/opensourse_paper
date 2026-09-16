"""Fold-local beta calibration for prospective detection probabilities."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class BetaCalibrator:
    """Beta-calibration map fitted only on a held-out development partition."""

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.constant: float | None = None

    @staticmethod
    def _features(probability: np.ndarray) -> np.ndarray:
        value = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
        return np.column_stack([np.log(value), -np.log1p(-value)])

    def fit(self, probability: np.ndarray, labels: np.ndarray) -> "BetaCalibrator":
        probability = np.asarray(probability, dtype=float)
        labels = np.asarray(labels, dtype=int)
        mask = np.isfinite(probability) & np.isfinite(labels)
        probability, labels = probability[mask], labels[mask]
        if probability.size == 0:
            self.constant = 0.5
            return self
        if np.unique(labels).size < 2:
            self.constant = float(labels.mean())
            return self
        self.model = LogisticRegression(C=1e3, max_iter=2000, random_state=20260826).fit(self._features(probability), labels)
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        probability = np.asarray(probability, dtype=float)
        if self.constant is not None:
            return np.full(probability.shape, self.constant, dtype=float)
        if self.model is None:
            raise RuntimeError("校准器尚未拟合")
        return self.model.predict_proba(self._features(probability))[:, 1]
