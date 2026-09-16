import numpy as np

from scripts.analyze_b3_calibration import _ece


def test_statewise_ece_is_zero_for_calibrated_composition():
    probabilities = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3]])
    assert _ece(probabilities, probabilities, bins=5) == 0.0


def test_statewise_ece_is_positive_for_miscalibrated_composition():
    probabilities = np.array([[0.8, 0.1, 0.1]])
    observed = np.array([[0.4, 0.3, 0.3]])
    assert _ece(probabilities, observed, bins=5) > 0.0
