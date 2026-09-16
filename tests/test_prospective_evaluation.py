from pathlib import Path

import numpy as np

from iot_benchmark.external_evaluation import evaluate_prospective_population_artifact


def test_prospective_population_evaluation(tmp_path: Path):
    truth = tmp_path / "truth.npz"
    artifact = tmp_path / "artifact.npz"
    np.savez_compressed(truth, target_counts=np.asarray([[8.0, 2.0], [4.0, 1.0]]))
    np.savez_compressed(artifact, predicted_population=np.asarray([0.8, 0.2]))
    result = evaluate_prospective_population_artifact(truth, artifact, np.asarray([[0.0, 1.0], [1.0, 0.0]]))
    assert result["top1_accuracy"] == 1.0
    assert result["cross_entropy"] > 0.0
    assert result["multiclass_brier"] < 1e-12
