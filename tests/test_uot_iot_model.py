import numpy as np

from iot_benchmark.models import UOTIOTBenchmarkModel


def test_uot_iot_model_outputs_probabilities_and_unit_direction():
    rng = np.random.default_rng(4)
    source = rng.dirichlet(np.ones(3), size=12)
    target = 0.7 * source + 0.3 * np.roll(source, 1, axis=1)
    groups = np.repeat(["a", "b", "c"], 4)
    counts = np.full_like(source, 10.0)
    model = UOTIOTBenchmarkModel(restarts=2, maxiter=50).fit(
        source, target, groups, counts, counts, seed=7
    )
    prediction = model.predict_state(source)
    direction, magnitude = model.direction()
    assert np.allclose(prediction.sum(axis=1), 1.0)
    assert np.isclose(np.linalg.norm(direction), 1.0)
    assert magnitude > 0
