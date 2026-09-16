import numpy as np

from iot_benchmark.statistics import paired_bootstrap_interval


def test_paired_bootstrap_detects_uniform_positive_difference():
    mean, lower, upper = paired_bootstrap_interval(np.array([0.2, 0.3, 0.4, 0.5]), replicates=1000)
    assert mean > 0
    assert lower > 0
    assert upper >= lower
