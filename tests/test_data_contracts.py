import numpy as np

from iot_benchmark.data_contracts import _normalize_rows


def test_state_compositions_are_normalized():
    values = np.array([[1.0, 2.0], [4.0, 1.0]])
    normalized = _normalize_rows(values)
    assert np.allclose(normalized.sum(axis=1), 1.0)
    assert np.all(normalized >= 0)
