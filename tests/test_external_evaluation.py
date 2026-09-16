import numpy as np

from iot_benchmark.external_evaluation import aggregate_state_coupling, clone_fate_probabilities


def test_clone_coupling_aggregation_and_fate_are_probabilities():
    coupling = np.diag([0.4, 0.6])
    source = np.array([[1.0, 0.0], [0.0, 1.0]])
    target = np.array([[0.8, 0.2], [0.1, 0.9]])
    state = aggregate_state_coupling(coupling, source, target)
    fate = clone_fate_probabilities(coupling, target)
    assert np.isclose(state.sum(), 1.0)
    assert np.allclose(fate.sum(axis=1), 1.0)
    assert np.allclose(fate, target)
