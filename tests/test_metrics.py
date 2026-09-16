import numpy as np

from iot_benchmark.metrics import (
    coefficient_recovery,
    coupling_relative_frobenius,
    cross_entropy,
    directional_curvature,
    direction_stability,
    expected_calibration_error,
    fate_correlations,
    multiclass_brier,
    sinkhorn_divergence,
    top1_accuracy,
    transition_weighted_l1,
)


def test_exact_predictions_have_optimal_reference_metrics():
    coupling = np.array([[0.4, 0.1], [0.2, 0.3]])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9]])
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.isclose(coupling_relative_frobenius(coupling, coupling), 0.0)
    assert np.isclose(transition_weighted_l1(coupling, coupling), 0.0)
    assert np.isclose(multiclass_brier(probabilities, probabilities), 0.0)
    assert np.isclose(top1_accuracy(probabilities, probabilities), 1.0)
    assert np.isclose(sinkhorn_divergence(probabilities[0], probabilities[0], cost), 0.0)
    assert cross_entropy(probabilities, probabilities) > 0.0
    assert expected_calibration_error(probabilities, probabilities) >= 0.0
    assert fate_correlations(probabilities, probabilities)["fate_top1_accuracy"] == 1.0


def test_direction_recovery_stability_and_curvature():
    truth = np.array([-1.0, 0.5, 2.0])
    recovered = truth * 3.0
    metrics = coefficient_recovery(recovered, truth)
    assert np.isclose(metrics["coefficient_cosine"], 1.0)
    stability = direction_stability(np.vstack([truth, recovered, truth * 0.5]))
    assert np.isclose(stability["direction_pairwise_cosine"], 1.0)
    objective = lambda x: float(np.sum(x**2))
    curvature = directional_curvature(objective, np.zeros(3), np.array([1.0, 0.0, 0.0]))
    assert np.isclose(curvature, 2.0, atol=1e-6)
