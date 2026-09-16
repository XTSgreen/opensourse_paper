import numpy as np

from iot_benchmark.synthetic import (
    generate_case,
    hard_ot_plan,
    make_feature_tensor,
    soft_uot_plan,
)


def test_soft_and_hard_plans_respect_required_margins():
    rng = np.random.default_rng(7)
    phi, _ = make_feature_tensor(rng, states=5, features=6)
    theta = rng.normal(size=6)
    source = rng.dirichlet(np.ones(5))
    target = rng.dirichlet(np.ones(5))
    soft = soft_uot_plan(phi, theta, source, target)
    hard = hard_ot_plan(phi, theta, source, target)
    assert np.allclose(soft.sum(axis=1), source, atol=1e-8)
    assert np.allclose(hard.sum(axis=1), source, atol=1e-8)
    assert np.allclose(hard.sum(axis=0), target, atol=1e-8)
    assert not np.allclose(soft.sum(axis=0), target)


def test_generated_case_has_five_train_and_three_heldout_observations():
    phi, theta, train, heldout, names = generate_case(
        "identifiable_soft_marginal", sample_count=300, seed=11
    )
    assert phi.shape == (7, 7, 6)
    assert theta.shape == (6,)
    assert len(train) == 5
    assert len(heldout) == 3
    assert len(names) == 6
